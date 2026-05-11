import pandas as pd

from src.processing.prompt_utils import (
    replace_prompt_entities,
    apply_prompt_repetition,
    save_partial_results,
    should_partial_save,
)
from src.processing.result_utils import create_result


def load_candidate_groups(pairs_df, target_side, logger):
    """
    Group candidate pairs by target entity.

    Args:
        pairs_df: DataFrame with candidate pairs (two columns)
        target_side: 'd1' or 'd2' - which side contains target entities
        logger: Logger instance

    Returns:
        Dict mapping target_id -> list of candidate_ids, plus column names
    """
    col_names = pairs_df.columns.tolist()

    if target_side == 'd2':
        target_col = col_names[1]
        candidate_col = col_names[0]
    else:
        target_col = col_names[0]
        candidate_col = col_names[1]

    logger.info(f"Target column: {target_col}, Candidate column: {candidate_col}")

    groups = {}
    for _, row in pairs_df.iterrows():
        target_id = row[target_col]
        candidate_id = row[candidate_col]

        if target_id not in groups:
            groups[target_id] = []
        groups[target_id].append(candidate_id)

    logger.info(f"Found {len(groups)} target entities with candidates")
    return groups, target_col, candidate_col


def process_candidate_selection(llm, processor, prompt_template, schema_class, task_config, logger,
                                start_index=0, count=None, partial_save_interval=None, partial_save_path=None,
                                repetitions=0, repetition_style="basic"):
    """
    Process candidate selection: for each target entity, find the matching candidate.
    """
    results = []
    target_side = task_config.get("target_side", "d2")

    candidate_groups, target_col, candidate_col = load_candidate_groups(
        processor.pairs_df, target_side, logger
    )

    target_ids = list(candidate_groups.keys())
    total_targets = len(target_ids)

    if count is None:
        count = total_targets - start_index
    end_index = min(start_index + count, total_targets)

    logger.info(f"Processing targets {start_index + 1} to {end_index} (total: {total_targets})")

    if target_side == 'd2':
        target_df = processor.dataset2_df
        candidate_df = processor.dataset1_df
        col1_name = processor.col1_name  # candidates
        col2_name = processor.col2_name  # targets
    else:
        target_df = processor.dataset1_df
        candidate_df = processor.dataset2_df
        col1_name = processor.col1_name  # targets
        col2_name = processor.col2_name  # candidates

    pairs_processed = 0

    for target_idx in range(start_index, end_index):
        target_id = target_ids[target_idx]
        candidate_ids = candidate_groups[target_id]

        logger.info(f"Processing target {target_idx + 1}/{total_targets}: ID={target_id} with {len(candidate_ids)} candidates")

        prompt = None

        try:
            target_entity = target_df.loc[target_id].to_dict()
            target_entity_clean = {k: v for k, v in target_entity.items() if k.lower() != 'id'}

            candidates_data = []
            for cand_id in candidate_ids:
                cand_entity = candidate_df.loc[cand_id].to_dict()
                cand_entity_clean = {k: v for k, v in cand_entity.items() if k.lower() != 'id'}
                candidates_data.append({
                    'id': cand_id,
                    'entity': cand_entity_clean
                })

            if not candidates_data:
                raise ValueError(f"No valid candidates found for target {target_id}")

            candidates_str = "\n".join([
                f"{i+1}. {cand['entity']}"
                for i, cand in enumerate(candidates_data)
            ])

            entity_values = {
                "target_entity": str(target_entity_clean),
                "candidates": candidates_str
            }
            prompt = replace_prompt_entities(prompt_template, entity_values)
            prompt = apply_prompt_repetition(prompt, repetitions, repetition_style)

            logger.debug(f"Generated prompt: {prompt[:400]}{'...' if len(prompt) > 400 else ''}")

            response = llm.generate_structured(prompt, schema_class)
            selected_idx = response.selected_candidate  # 1-based, or 0 for no match

            logger.info(f"Target {target_id}: selected candidate index = {selected_idx}")

            for i, cand_data in enumerate(candidates_data):
                cand_id = cand_data['id']
                is_match = 1 if (selected_idx == i + 1) else 0

                if target_side == 'd2':
                    pair_data = {
                        'pair_index': pairs_processed,
                        f'{col1_name}_index': cand_id,
                        f'{col2_name}_index': target_id,
                        'entity1_raw': cand_data['entity'],
                        'entity2_raw': target_entity_clean,
                    }
                else:
                    pair_data = {
                        'pair_index': pairs_processed,
                        f'{col1_name}_index': target_id,
                        f'{col2_name}_index': cand_id,
                        'entity1_raw': target_entity_clean,
                        'entity2_raw': cand_data['entity'],
                    }

                result = create_result(
                    pair_data=pair_data,
                    col1_name=col1_name,
                    col2_name=col2_name,
                    success=True,
                    prompt=prompt,
                    response=response,
                    target_id=target_id,
                    candidate_id=cand_id,
                    candidate_position=i + 1,
                    total_candidates=len(candidates_data),
                    selected_candidate=selected_idx,
                    match=is_match
                )
                results.append(result)
                pairs_processed += 1

            logger.info(f"Target {target_id} processed: {len(candidates_data)} pairs created")

        except Exception as e:
            logger.error(f"Failed to process target {target_id}: {e}")

            for i, cand_id in enumerate(candidate_ids):
                if target_side == 'd2':
                    pair_data = {
                        'pair_index': pairs_processed,
                        f'{col1_name}_index': cand_id,
                        f'{col2_name}_index': target_id,
                        'entity1_raw': f"candidate_{cand_id}",
                        'entity2_raw': f"target_{target_id}",
                    }
                else:
                    pair_data = {
                        'pair_index': pairs_processed,
                        f'{col1_name}_index': target_id,
                        f'{col2_name}_index': cand_id,
                        'entity1_raw': f"target_{target_id}",
                        'entity2_raw': f"candidate_{cand_id}",
                    }

                result = create_result(
                    pair_data=pair_data,
                    col1_name=col1_name,
                    col2_name=col2_name,
                    success=False,
                    prompt=prompt if prompt else "Error generating prompt",
                    error=e,
                    target_id=target_id,
                    candidate_id=cand_id,
                    candidate_position=i + 1,
                    total_candidates=len(candidate_ids)
                )
                results.append(result)
                pairs_processed += 1

        if should_partial_save(pairs_processed, partial_save_interval, partial_save_path):
            save_partial_results(results, partial_save_path, pairs_processed,
                                 sum(len(candidate_groups[t]) for t in target_ids[start_index:end_index]), logger)

    return pd.DataFrame(results)

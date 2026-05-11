import pandas as pd

from src.processing.prompt_utils import (
    replace_prompt_entities,
    apply_prompt_repetition,
    save_partial_results,
    should_partial_save,
)
from src.processing.result_utils import create_result


def process_entity_pairs_single(llm, processor, entities_list, prompt_template, schema_class, logger,
                                start_index=0, count=None, partial_save_interval=None, partial_save_path=None,
                                repetitions=0, repetition_style="basic"):
    """Process entity pairs one at a time (original behavior)."""
    results = []

    total_pairs = len(processor.pairs_df)
    if count is None:
        count = total_pairs - start_index

    end_index = min(start_index + count, total_pairs)

    col1_name = processor.col1_name
    col2_name = processor.col2_name

    for pair_idx in range(start_index, end_index):
        logger.info(f"Processing pair {pair_idx + 1}/{total_pairs} (batch: {pair_idx - start_index + 1}/{count})")

        pair_data = None
        prompt = None

        try:
            pair_data = processor.get_pair_by_index(pair_idx)

            entity_values = {}
            if len(entities_list) >= 2:
                entity1_clean = {k: v for k, v in pair_data['entity1_raw'].items() if k != 'id'}
                entity2_clean = {k: v for k, v in pair_data['entity2_raw'].items() if k != 'id'}

                entity_values[entities_list[0]] = str(entity1_clean)
                entity_values[entities_list[1]] = str(entity2_clean)

            prompt = replace_prompt_entities(prompt_template, entity_values)
            prompt = apply_prompt_repetition(prompt, repetitions, repetition_style)

            logger.debug(f"Generated prompt: {prompt[:200]}{'...' if len(prompt) > 200 else ''}")

            response = llm.generate_structured(prompt, schema_class)

            result = create_result(
                pair_data=pair_data,
                col1_name=col1_name,
                col2_name=col2_name,
                success=True,
                prompt=prompt,
                response=response
            )

            results.append(result)
            logger.info(f"Pair {pair_idx + 1} processed successfully")

        except Exception as e:
            logger.error(f"Failed to process pair {pair_idx + 1}: {e}")

            if pair_data is None:
                result = create_result(
                    pair_data=None,
                    col1_name=col1_name,
                    col2_name=col2_name,
                    success=False,
                    error=e,
                    row_id=pair_idx + 1,
                    pair_index=pair_idx
                )
            else:
                result = create_result(
                    pair_data=pair_data,
                    col1_name=col1_name,
                    col2_name=col2_name,
                    success=False,
                    prompt=prompt if prompt else "Error generating prompt",
                    error=e
                )

            results.append(result)

        pairs_processed = pair_idx - start_index + 1
        if should_partial_save(pairs_processed, partial_save_interval, partial_save_path):
            save_partial_results(results, partial_save_path, pairs_processed, count, logger)

    return pd.DataFrame(results)


def process_entity_pairs_multi(llm, processor, entities_list, prompt_template, schema_class, pairs_per_prompt, logger,
                               start_index=0, count=None, partial_save_interval=None, partial_save_path=None,
                               repetitions=0, repetition_style="basic"):
    """Process multiple entity pairs in a single prompt using static schema."""
    results = []

    total_pairs = len(processor.pairs_df)
    if count is None:
        count = total_pairs - start_index

    end_index = min(start_index + count, total_pairs)

    col1_name = processor.col1_name
    col2_name = processor.col2_name

    batch_num = 0
    for batch_start in range(start_index, end_index, pairs_per_prompt):
        batch_end = min(batch_start + pairs_per_prompt, end_index)
        actual_pairs_in_batch = batch_end - batch_start
        batch_num += 1

        results_start = 0
        if actual_pairs_in_batch < pairs_per_prompt and batch_start > start_index:
            results_start = pairs_per_prompt - actual_pairs_in_batch
            batch_start = batch_end - pairs_per_prompt
            actual_pairs_in_batch = pairs_per_prompt
            logger.info(f"Shifted batch {batch_num} back to start at pair {batch_start + 1} (recording from position {results_start + 1})")

        logger.info(f"Processing batch {batch_num} with pairs {batch_start + 1}-{batch_end} ({actual_pairs_in_batch} pairs)")

        prompt = None

        try:
            batch_pairs = []
            entity_values = {}

            for i, pair_idx in enumerate(range(batch_start, batch_end)):
                pair_data = processor.get_pair_by_index(pair_idx)
                batch_pairs.append(pair_data)

                entity1_clean = {k: v for k, v in pair_data['entity1_raw'].items() if k != 'id'}
                entity2_clean = {k: v for k, v in pair_data['entity2_raw'].items() if k != 'id'}

                pair_num = i + 1
                entity_values[f"entity_{pair_num}a"] = str(entity1_clean)
                entity_values[f"entity_{pair_num}b"] = str(entity2_clean)

            prompt = replace_prompt_entities(prompt_template, entity_values)
            prompt = apply_prompt_repetition(prompt, repetitions, repetition_style)

            logger.debug(f"Generated prompt for batch: {prompt[:300]}{'...' if len(prompt) > 300 else ''}")

            response = llm.generate_structured(prompt, schema_class)

            response_dict = response.model_dump() if hasattr(response, 'model_dump') else {}

            for i, pair_data in enumerate(batch_pairs):
                if i < results_start:
                    continue
                pair_num = i + 1
                match_field = f"pair_{pair_num}_match"
                match_value = response_dict.get(match_field, None)

                result = create_result(
                    pair_data=pair_data,
                    col1_name=col1_name,
                    col2_name=col2_name,
                    success=True,
                    prompt=prompt,
                    response=response,
                    batch_num=batch_num,
                    pairs_in_batch=actual_pairs_in_batch,
                    match=match_value
                )

                results.append(result)
                logger.info(f"Pair {pair_data['pair_index'] + 1} processed successfully (match: {match_value})")

        except Exception as e:
            logger.error(f"Failed to process batch {batch_num} (pairs {batch_start + 1}-{batch_end}): {e}")

            for idx_in_batch, pair_idx in enumerate(range(batch_start, batch_end)):
                if idx_in_batch < results_start:
                    continue
                pair_data = None
                try:
                    pair_data = processor.get_pair_by_index(pair_idx)
                except Exception:
                    pass

                if pair_data is None:
                    result = create_result(
                        pair_data=None,
                        col1_name=col1_name,
                        col2_name=col2_name,
                        success=False,
                        error=e,
                        row_id=pair_idx + 1,
                        pair_index=pair_idx,
                        batch_num=batch_num,
                        pairs_in_batch=actual_pairs_in_batch
                    )
                else:
                    result = create_result(
                        pair_data=pair_data,
                        col1_name=col1_name,
                        col2_name=col2_name,
                        success=False,
                        prompt=prompt if prompt else "Error generating prompt",
                        error=e,
                        batch_num=batch_num,
                        pairs_in_batch=actual_pairs_in_batch
                    )

                results.append(result)

        pairs_processed = batch_end - start_index
        if should_partial_save(pairs_processed, partial_save_interval, partial_save_path):
            save_partial_results(results, partial_save_path, pairs_processed, count, logger)

    return pd.DataFrame(results)

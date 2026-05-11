import pandas as pd


def replace_prompt_entities(prompt_template, entity_values):
    """Replace entity placeholders in prompt template.

    Supports both [entity] and {{entity}} format.
    """
    prompt = prompt_template
    for entity, value in entity_values.items():
        prompt = prompt.replace(f"[{entity}]", value)
        prompt = prompt.replace(f"{{{{{entity}}}}}", value)
    return prompt


def apply_prompt_repetition(prompt, repetitions, style="basic"):
    """
    Repeat the prompt instruction within a single prompt string.
    Based on Leviathan et al. (2024) - prompt repetition improves non-reasoning LLMs.

    Args:
        prompt: The fully-constructed prompt string.
        repetitions: Number of extra copies to append (0 = no change).
        style: 'basic' (space-separated) or 'verbose' (bridging phrases between copies).
    """
    if repetitions <= 0:
        return prompt
    bridges = ["Let me repeat that: ", "Let me repeat that one more time: "]
    parts = [prompt]
    for i in range(repetitions):
        bridge = (bridges[i] if i < len(bridges) else "Let me repeat: ") if style == "verbose" else ""
        parts.append(bridge + prompt)
    return " ".join(parts)


def save_partial_results(results, partial_save_path, pairs_processed, total_count, logger):
    """Save partial results to CSV."""
    partial_df = pd.DataFrame(results)
    partial_file = partial_save_path / "partial_results.csv"
    partial_df.to_csv(partial_file, index=False)
    logger.info(f"Partial save completed: {pairs_processed} pairs saved to {partial_file}")
    print(f"Partial save: {pairs_processed}/{total_count} pairs processed")


def should_partial_save(pairs_processed, partial_save_interval, partial_save_path):
    """Check if it's time to do a partial save."""
    return (partial_save_interval and partial_save_path and
            pairs_processed % partial_save_interval == 0)

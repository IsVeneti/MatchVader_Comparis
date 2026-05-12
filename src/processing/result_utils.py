def create_result(pair_data, col1_name, col2_name, success=True, prompt=None,
                  response=None, error=None, **extra_fields):
    """
    Create a standardized result dictionary for entity matching.

    Args:
        pair_data: Dictionary with pair information (can be None for errors)
        col1_name: Name of first column
        col2_name: Name of second column
        success: Whether processing succeeded
        prompt: The prompt used (optional)
        response: The LLM response object (optional)
        error: Error message if failed (optional)
        **extra_fields: Additional fields to include in result

    Returns:
        Dictionary with standardized result structure
    """
    result = {
        "success": success,
    }

    if pair_data:
        result.update({
            "row_id": pair_data['pair_index'] + 1,
            "pair_index": pair_data['pair_index'],
            f"{col1_name}_index": pair_data[f'{col1_name}_index'],
            f"{col2_name}_index": pair_data[f'{col2_name}_index'],
            "entity1": str(pair_data['entity1_raw']),
            "entity2": str(pair_data['entity2_raw']),
        })

    if prompt is not None:
        result["prompt"] = prompt

    if response is not None:
        result["response"] = str(response)
        if hasattr(response, 'model_dump'):
            result.update(response.model_dump())

    if error is not None:
        result["error"] = str(error)

    result.update(extra_fields)

    return result

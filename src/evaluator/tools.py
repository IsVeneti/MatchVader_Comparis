import os
import pandas as pd
import json

def merge_csv(llm_response_csv, entity_mappings_csv, output_path):
    """
    Merges two CSV files by extracting the "yes" or "no" responses from csv1,
    converting them to binary (1 for "yes", 0 for "no"), and appending them as
    a new column to csv2.

    Parameters:
    llm_response_csv (str): File path to the first CSV containing LLM responses.
    entity_mappings_csv (str): File path to the second CSV containing entity mappings.
    output_path (str): File path where the merged CSV will be saved.

    Returns:
    None: Saves the merged CSV to the specified output path.
    """
    # Read csv1
    df1 = pd.read_csv(llm_response_csv)

    # Extract responses and convert JSON strings to dictionaries
    df1['Response'] = df1['Response'].apply(lambda x: json.loads(x)['response'])

    # Convert "yes" to 1 and "no" to 0
    df1['match'] = df1['Response'].map({'yes': 1, 'no': 0})

    # Read csv2
    df2 = pd.read_csv(entity_mappings_csv)

    # Ensure index alignment
    if len(df1) != len(df2):
        raise ValueError("llm_response_csv and entity_mappings_csv do not have the same number of rows.")

    # Merge data by adding the match column to df2
    df2['match'] = df1['match']

    # Save to a new CSV
    df2.to_csv(output_path, index=False)
    print(f"Merged CSV saved to {output_path}")



# def add_prediction_to_mapping(llm_response_csv, entity_mappings_csv, output_path):
#     """_summary_

#     Args:
#         llm_response_csv (str): File path to the CSV containing LLM responses.
#         entity_mappings_csv (str): File path to the second CSV containing entity mappings.
#         output_path (str): _description_
#     """

# Example usage
# print(os.listdir())
merge_csv('prompt_results_chatc_think_qa.csv', 'data/d1_pairs.csv', 'output.csv')

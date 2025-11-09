
# importing pandas
import pandas as pd
 

# Creating Dictionary
d1 = {'id': [1, 2, 9, 12],
    'val1': ['a', 'b', 'f', 'd'],
    'res': [1,1,0,0]}
 
df1 = pd.DataFrame(d1)
 

# Creating dictionary
d2 = {'id': [1, 2, 9, 8,9],
    'val2': ['p', 'b', 'r', 's','f'],
    'match':[1,1,1,1,1]}
df2 = pd.DataFrame(d2)
print(df2.columns[1])

import pandas as pd


def merge_dataframes(pairs: pd.DataFrame, ground_truth: pd.DataFrame, 
                     join_type="inner",
                     pairs_cols=("id1", "id2"),
                     gt_cols=("D1", "D2"),
                     gt_sep=','):
    """
    Merges two dataframes based on specified id columns using the given join type.
    
    Args:
        pairs_csv (str): Path to the first CSV file (e.g., candidate pairs with scores).
        ground_truth_csv (str): Path to the ground truth CSV file (with matches).
        join_type (str): Type of join to perform. Options: "inner", "outer", "left", "right".
        pairs_cols (tuple): Column names for IDs in pairs CSV, e.g., ("id1", "id2").
        gt_cols (tuple): Column names for IDs in ground truth CSV, e.g., ("D1", "D2").
        gt_sep (str): Separator for ground truth CSV (default: '|').
    
    Returns:
        pd.DataFrame: Merged dataframe.
    """
    # Validate join type
    if join_type not in ["inner", "outer", "left", "right"]:
        raise ValueError("Invalid join type. Choose from 'inner', 'outer', 'left', or 'right'.")
    pairs_df = pairs.copy()
    ground_truth_df = ground_truth.copy()

    # Rename columns to common names for merging
    pairs_df = pairs_df.rename(columns={
        pairs_cols[0]: 'id1', 
        pairs_cols[1]: 'id2'
    })
    print(gt_cols[0],gt_cols[1])
    ground_truth_df = ground_truth_df.rename(columns={gt_cols[0]: 'id1', gt_cols[1]: 'id2'})
    print("Ground truth columns:", ground_truth_df.columns)
    # Merge DataFrames
    merged_df = pd.merge(
        pairs_df, 
        ground_truth_df, 
        on=['id1', 'id2'], 
        how=join_type, 
        suffixes=('_pairs', '_gt')
    )
    
    # Fill missing numeric values with 0 (skip id columns)
    for col in merged_df.columns:
        if col not in ['id1', 'id2']:
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce').fillna(0).astype(int)
    
    return merged_df

# TODO: I fucked up my results, so this is a quick fix to merge based on existing 'match' column
def merge_response_pairs(llm_response_csv, entity_mappings_csv):
    """
    Merges two CSV files by extracting the match values from csv1
    and appending them as a new column to csv2.
    
    Parameters:
    llm_response_csv (str): File path to the first CSV containing LLM responses.
    entity_mappings_csv (str): File path to the second CSV containing entity mappings.
    
    Returns:
    None: Saves the merged CSV to the specified output path.
    """
    # Read csv1
    df1 = pd.read_csv(llm_response_csv)
    
    # Check if 'match' column exists
    if 'match' not in df1.columns:
        raise ValueError("The 'match' column does not exist in llm_response_csv.")
    
    # Read csv2
    df2 = pd.read_csv(entity_mappings_csv)
    
    # Ensure index alignment
    if len(df1) != len(df2):
        raise ValueError("llm_response_csv and entity_mappings_csv do not have the same number of rows.")
    
    # Merge data by adding the match column to df2
    df2['match'] = df1['match'].values
    
    return df2

# TODO: This works for binary pairs, may have problems with multiples - check alternatives
def filter_csv_columns(input_csv, columns):
    """
    Reads a CSV file and saves a new CSV with only the specified columns.
    
    Parameters:
    input_csv (str): File path to the input CSV.
    columns (list): List of column names to keep in the output CSV.
    
    Returns:
        pd.DataFrame: DataFrame containing only the specified columns.
    """
    # Read the CSV
    df = pd.read_csv(input_csv)
    
    # Check if all specified columns exist
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The following columns do not exist in the CSV: {missing_cols}")
    
    # Filter to keep only specified columns
    df_filtered = df[columns]
    
    return df_filtered


if __name__ == "__main__":
    # get_paired_ids_no_fullfile("output.csv","./data/rest1clean.csv","./data/rest2clean.csv")
    # get_paired_ids("paired_data_with_indexes.csv")
    merged_inner = merge_dataframes('data/d1_data/d1_pairs_with_ids.csv', 'data/d1_data/gtclean_evaluator_comma.csv', 'outer',pairs_cols=("id1","id2"),gt_cols=("D1","D2"))
    # merge_dataframes('paired_data_id.csv', 'data/d1_data/gtclean_evaluator_comma.csv', 'outer',pairs_cols=("rest1_index","rest2_index"))
    # merge_dataframes('paired_data_id.csv', 'data/d1_data/gtclean_evaluator_comma.csv', 'left',pairs_cols=("rest1_index","rest2_index"))
    print("Hello")



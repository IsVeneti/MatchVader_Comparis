# importing pandas
import csv
import pandas as pd


def load_csv_with_separator_detection(csv_path,skiprows=None,header=0):
    """
    Load CSV with automatic separator detection using csv.Sniffer.
    
    Args:
        csv_path (str): Path to CSV file
        separator (str, optional): Separator to use. If None, will auto-detect.
    
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    # Use provided separator or detect it
    with open(csv_path, 'r') as f:
        sample = f.read(4096)  # Read first 4KB for detection
        sniffer = csv.Sniffer()
        separator = sniffer.sniff(sample).delimiter
    
    # Load the CSV
    if skiprows is not None:
        df = pd.read_csv(csv_path, sep=separator,header=header)
    else:
        df = pd.read_csv(csv_path, sep=separator, skiprows=skiprows,header=header)
    
    return df


def load_ground_truth(ground_truth_path):
    """
    Load ground truth CSV with automatic separator detection and add match column.
    
    Args:
        ground_truth_path (str): Path to ground truth CSV file
    
    Returns:
        pd.DataFrame: Ground truth dataframe with 'match' column set to 1
    """
    # Load the ground truth using separator detection
    gt_df = load_csv_with_separator_detection(ground_truth_path)
    
    # Add match column with all 1s (these are true matches)
    gt_df['match'] = 1
    
    return gt_df



def merge_dataframes(pairs: pd.DataFrame, ground_truth: pd.DataFrame, 
                     join_type="inner"):
    """
    Merges two dataframes based on their first two columns using the given join type.
    Automatically renames the first two columns of each DataFrame to 'id1' and 'id2'.
    
    Args:
        pairs (pd.DataFrame): DataFrame with candidate pairs (can include scores).
        ground_truth (pd.DataFrame): Ground truth DataFrame (with matches).
        join_type (str): Type of join to perform. Options: "inner", "outer", "left", "right".
    
    Returns:
        pd.DataFrame: Merged dataframe.
    """
    # Validate join type
    if join_type not in ["inner", "outer", "left", "right"]:
        raise ValueError("Invalid join type. Choose from 'inner', 'outer', 'left', or 'right'.")
    
    pairs_df = pairs.copy()
    ground_truth_df = ground_truth.copy()

    # Simply rename first two columns to id1, id2
    pairs_df = pairs_df.rename(columns={pairs_df.columns[0]: "id1", pairs_df.columns[1]: "id2"})
    ground_truth_df = ground_truth_df.rename(columns={ground_truth_df.columns[0]: "id1", ground_truth_df.columns[1]: "id2"})

    # DEBUG - Check data types and sample values
    print("Pairs DataFrame:")
    print(f"  Columns: {pairs_df.columns.tolist()}")
    print(f"  id1 dtype: {pairs_df['id1'].dtype}, id2 dtype: {pairs_df['id2'].dtype}")
    print(f"  Sample: {pairs_df[['id1', 'id2']].head(3).to_dict('records')}")
    
    print("\nGround Truth DataFrame:")
    print(f"  Columns: {ground_truth_df.columns.tolist()}")
    print(f"  id1 dtype: {ground_truth_df['id1'].dtype}, id2 dtype: {ground_truth_df['id2'].dtype}")
    print(f"  Sample: {ground_truth_df[['id1', 'id2']].head(3).to_dict('records')}")
    
    
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


def merge_response_pairs(llm_response_csv, entity_mappings_csv):
    """
    Merges two CSV files by extracting the match values from csv1
    and appending them as a new column to csv2.
    
    Parameters:
        llm_response_csv (str): File path to the first CSV containing LLM responses.
        entity_mappings_csv (str): File path to the second CSV containing entity mappings.
    
    Returns:
        pd.DataFrame: DataFrame with match column added from llm_response_csv.
    """
    # Read csv1
    df1 = load_csv_with_separator_detection(llm_response_csv)
    
    # Check if 'match' column exists
    if 'match' not in df1.columns:
        raise ValueError("The 'match' column does not exist in llm_response_csv.")
    
    # Read csv2
    df2 = load_csv_with_separator_detection(entity_mappings_csv)
    
    # Ensure index alignment
    if len(df1) != len(df2):
        raise ValueError("llm_response_csv and entity_mappings_csv do not have the same number of rows.")
    
    # Merge data by adding the match column to df2
    df2['match'] = df1['match'].values
    
    return df2


def filter_csv_columns(input_csv, columns):
    """
    Reads a CSV file and returns a DataFrame with only the specified columns.
    
    Parameters:
        input_csv (str): File path to the input CSV.
        columns (list): List of column names to keep in the output.
    
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
    # Example usage:
    # gt_df = load_ground_truth('data/ground_truth.csv')  # Auto-detects separator, adds match column
    # pairs_df = pd.read_csv('data/pairs.csv')
    # merged = merge_dataframes(pairs_df, gt_df, 'outer')  # First 2 columns automatically used
    
    print("Module loaded successfully.")
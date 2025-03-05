import csv
from typing import List, Tuple
import pandas as pd
import openpyxl

def read_csv_to_dataframe(file_name: str, delimiter: str) -> pd.DataFrame:
    """
    Reads a CSV file into a pandas DataFrame using the specified delimiter.

    Parameters:
        file_name (str): The name of the CSV file to read.
        delimiter (str): The delimiter used in the CSV file.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the file data.
    """
    return pd.read_csv(file_name, delimiter=delimiter)

def read_paired_indexes(file_name: str, delimiter: str = ',') -> List[Tuple[int, int]]:
    """
    Reads a CSV file with paired indexes and returns a list of index pairs.

    Parameters:
        file_name (str): The name of the CSV file containing paired indexes.
        delimiter (str): The delimiter used in the CSV file. Default is ','.

    Returns:
        List[Tuple[int, int]]: A list of tuples containing paired indexes.
    """
    df = pd.read_csv(file_name, delimiter=delimiter)
    return list(df.itertuples(index=False, name=None))

def get_data_for_paired_indexes(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    paired_indexes: List[Tuple[int, int]]
) -> pd.DataFrame:
    """
    Fetches data for each index pair from the two datasets and returns it as a DataFrame.

    Parameters:
        df1 (pd.DataFrame): DataFrame of data from the first CSV file.
        df2 (pd.DataFrame): DataFrame of data from the second CSV file.
        paired_indexes (List[Tuple[int, int]]): List of index pairs.

    Returns:
        pd.DataFrame: A DataFrame containing data from both datasets for each index pair.
    """
    paired_data = []

    for index1, index2 in paired_indexes:
        # Fetch data for the paired indexes from both datasets
        data_row1 = df1.iloc[index1].tolist() if index1 < len(df1) else [None] * len(df1.columns)
        data_row2 = df2.iloc[index2].tolist() if index2 < len(df2) else [None] * len(df2.columns)

        paired_data.append(data_row1 + data_row2)

    # Combine into a DataFrame
    columns = list(df1.columns) + list(df2.columns)
    return pd.DataFrame(paired_data, columns=columns)

def get_data_with_indexes(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    paired_indexes: List[Tuple[int, int]]
) -> pd.DataFrame:
    """
    Fetches data for each index pair from the two datasets, including the indexes, and returns it as a DataFrame.

    Parameters:
        df1 (pd.DataFrame): DataFrame of data from the first CSV file.
        df2 (pd.DataFrame): DataFrame of data from the second CSV file.
        paired_indexes (List[Tuple[int, int]]): List of index pairs.

    Returns:
        pd.DataFrame: A DataFrame containing indexes and data from both datasets for each index pair.
    """
    paired_data = []

    for index1, index2 in paired_indexes:
        # Fetch data for the paired indexes from both datasets
        data_row1 = df1.iloc[index1].tolist() if index1 < len(df1) else [None] * len(df1.columns)
        data_row2 = df2.iloc[index2].tolist() if index2 < len(df2) else [None] * len(df2.columns)

        paired_data.append([index1] + data_row1 + [index2] + data_row2)

    # Combine into a DataFrame
    columns = ['Index1'] + list(df1.columns) + ['Index2'] + list(df2.columns)
    return pd.DataFrame(paired_data, columns=columns)

def save_dataframe(df: pd.DataFrame, file_name: str, file_type: str = 'csv') -> None:
    """
    Saves a pandas DataFrame to the specified file type.

    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        file_name (str): The name of the output file.
        file_type (str): The type of file to save ('csv' or 'excel'). Default is 'csv'.

    Returns:
        None
    """
    if file_type == 'csv':
        df.to_csv(file_name, index=False)
    elif file_type == 'excel':
        df.to_excel(file_name, index=False)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def main():
    """
    Main function to execute the script logic.
    """
    # Specify the CSV file names
    file1 = 'data/rest1clean.csv'
    file2 = 'data/rest2clean.csv'
    paired_file = 'data/d1_pairs.csv'

    # Read CSV files into DataFrames
    df1 = read_csv_to_dataframe(file1, delimiter='|')
    df2 = read_csv_to_dataframe(file2, delimiter='|')

    # Read paired indexes from the CSV file
    paired_indexes = read_paired_indexes(paired_file, delimiter=',')

    # Get data for paired indexes
    paired_data = get_data_for_paired_indexes(df1, df2, paired_indexes)
    print("Paired data:")
    print(paired_data)

    # Get data with indexes
    data_with_indexes = get_data_with_indexes(df1, df2, paired_indexes)
    print("\nData with indexes:")
    print(data_with_indexes)

    save_dataframe(data_with_indexes, 'data_with_indexes.csv', 'csv')
    save_dataframe(data_with_indexes, 'data_with_indexes.xlsx', 'excel')

if __name__ == "__main__":
    main()

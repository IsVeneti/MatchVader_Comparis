import json
import pandas as pd
import os
print(os.getcwd())  # Prints the current working directory

def process_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path,sep='|')

    # Find the column where "Index2" starts
    split_index = df.columns.get_loc("Index2")

    # Extract columns dynamically
    data1_columns = [col for col in df.columns[:split_index] if "Index" not in col and "id" not in col]
    data2_columns = [col for col in df.columns[split_index:] if "Index" not in col and "id" not in col]

    # Extract data
    data1 = df[data1_columns]
    data2 = df[data2_columns]

    # Display the results
    print(data1)
    print(data2)
    return data1,data2



def generate_comparison_prompt(data1, data2):
    """
    Creates a comparison table with column names and a prompt asking if each row is the same.
    """
    # Ensure both dataframes have the same index for alignment
    data1 = data1.reset_index(drop=True)
    data2 = data2.reset_index(drop=True)

    # Create a new DataFrame with column names included
    comparison_rows = []
    for i in range(len(data1)):
        row_data = {
            "Row": f"Row {i}",
            "Data 1": "\n".join(f"{col}: {val}" for col, val in data1.iloc[i].dropna().items()),
            "Data 2": "\n".join(f"{col}: {val}" for col, val in data2.iloc[i].dropna().items()),
            "Prompt": "Is row1 the same as row2?"
        }
        comparison_rows.append(row_data)

    # Convert to DataFrame
    df_comparison = pd.DataFrame(comparison_rows)

    return df_comparison  # This returns the DataFrame instead of displaying it


def generate_comparison_prompt_json(data1, data2, prompt, include_nans=True):
    """
    Creates a comparison DataFrame where each row's data is stored as a JSON string.
    The JSON format has 'entity1' and 'entity2' as top-level keys.
    Optionally includes NaN values as empty strings if include_nans=True.
    Allows customization of the prompt.
    """
    # Ensure both dataframes have the same index for alignment
    data1 = data1.reset_index(drop=True)
    data2 = data2.reset_index(drop=True)

    # Convert rows to structured JSON format
    def row_to_json(row1, row2):
        entity1_data = {col: ("" if pd.isna(val) else val) for col, val in row1.items()} if include_nans else {col: val for col, val in row1.dropna().items()}
        entity2_data = {col: ("" if pd.isna(val) else val) for col, val in row2.items()} if include_nans else {col: val for col, val in row2.dropna().items()}
        print(entity1_data)
        return json.dumps({"entity1": entity1_data, "entity2": entity2_data})

    df_comparison = pd.DataFrame({
        "Comparison": [row_to_json(data1.iloc[i], data2.iloc[i]) for i in range(len(data1))],
        "Prompt": prompt  # Customizable prompt
    })

    return df_comparison  # Returning a DataFrame with JSON strings


# Example usage:
# df_comparison = generate_comparison_prompt(data1, data2)
# print(df_comparison)  # Prints to console



# Example usage
data1, data2 = process_csv("paired_data_with_indexes.csv")
prompt_table = generate_comparison_prompt_json(data1,data2,"Do these entities match? Answer yes or no")
prompt_table.to_csv("prompt_attempt2.csv", sep='|')
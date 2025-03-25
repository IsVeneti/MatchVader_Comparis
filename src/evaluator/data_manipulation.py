
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

def merge_dataframes(pairs_csv, ground_truth_csv, output_csv, join_type="inner"):
    """
    Merges two CSV files based on id1 and id2 using the specified join type.

    Args:
        pairs_csv (str): Path to the first CSV file.
        ground_truth_csv (str): Path to the second CSV file.
        output_csv (str): Path to save the merged CSV file.
        join_type (str): Type of join to perform. Options: "inner", "outer", "left", "right".
    """
    # Load DataFrames
    pairs_df = pd.read_csv(pairs_csv)
    print("Pairs DataFrame:\n", pairs_df)

    ground_truth_df = pd.read_csv(ground_truth_csv)
    ground_truth_df.columns = pairs_df.columns  # Ensure matching column names
    print("Ground Truth DataFrame:\n", ground_truth_df)

    # Validate join type
    if join_type not in ["inner", "outer", "left"]:
        raise ValueError("Invalid join type. Choose from 'inner', 'outer', or 'left'.")

    # Merge DataFrames based on the selected join type
    merged_df = pd.merge(pairs_df, ground_truth_df, on=["id1", "id2"], how=join_type)
    
    # For outer or left joins, fill missing values with 0
    if join_type in ["outer", "left","right"]:
        print("in here")
        merged_df.iloc[:, 2:] = merged_df.iloc[:, 2:].fillna(0).astype(int)

    
    print(f"{join_type.capitalize()} Join Result:\n", merged_df)
    
    # Save to CSV
    merged_df.to_csv(output_csv, index=False)


# TODO: Maybe i need to add a column name param?
def get_paired_ids(full_paired_file):
    """Get the full file with pairs (data with indexes) and return the id based pairs

    Args:
        full_paired_file (str): _description_
    """
    full_paired_df = pd.read_csv(full_paired_file,sep='|')
    # print(full_paired_df.columns)

    paired_df = full_paired_df[["id","id.1"]]
    print(paired_df)
    return paired_df

def get_paired_ids_no_fullfile(pair_csv,data1,data2):
    pair_df = pd.read_csv(pair_csv)
    data1_df = pd.read_csv(data1,sep='|')
    data2_df = pd.read_csv(data2,sep='|')
    # Match values
    matched_df1 = data1_df.loc[pair_df.iloc[:,0]].reset_index()
    print(f"matched_df1\n {matched_df1}")
    matched_df2 = data2_df.loc[pair_df.iloc[:,1]].reset_index()
    print(f"matched_df2\n {matched_df2}")
    print(pair_df)
    # Concatenating results
    # result = pd.concat([matched_df1.iloc[:,[0,1]], matched_df2.iloc[:,[0,1]],pair_df.iloc[:,2]], axis=1)

    # Result without the old index
    result = pd.concat([matched_df1.iloc[:,1].rename('id1'), matched_df2.iloc[:,1].rename('id2'),pair_df.iloc[:,2]], axis=1)

    result.to_csv("paired_data_id.csv",index=False)
    # print(result)

# get_paired_ids_no_fullfile("output.csv","./data/rest1clean.csv","./data/rest2clean.csv")
# get_paired_ids("paired_data_with_indexes.csv")
merge_dataframes('paired_data_id.csv', 'data/gtclean_evaluator_comma.csv', 'output_merge_inner.csv','inner')
merge_dataframes('paired_data_id.csv', 'data/gtclean_evaluator_comma.csv', 'output_merge_outer.csv','outer')
merge_dataframes('paired_data_id.csv', 'data/gtclean_evaluator_comma.csv', 'output_merge_left.csv','left')




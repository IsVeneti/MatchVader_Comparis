
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

def inner_merge(pairs_csv, ground_truth_csv, output_csv):
    """_summary_

    Args:
        pairs_csv (_type_): _description_
        ground_truth_csv (_type_): _description_
        output_csv (_type_): _description_
    """
    pairs_df = pd.read_csv(pairs_csv)
    print("pairs_df\n", pairs_df)
    ground_truth_df = pd.read_csv(ground_truth_csv)

    ground_truth_df.columns=pairs_df.columns
    print("ground_truth_df\n", ground_truth_df)
    # inner join
    inner_df = pd.merge(pairs_df, ground_truth_df, on=["id1","id2"], how='inner')
    print(inner_df)
    inner_df.to_csv(output_csv, index=False)

def outer_merge(pairs_csv, ground_truth_csv, output_csv):
    """_summary_

    Args:
        pairs_csv (_type_): _description_
        ground_truth_csv (_type_): _description_
        output_csv (_type_): _description_
    """
    pairs_df = pd.read_csv(pairs_csv)
    print("pairs_df\n", pairs_df)
    ground_truth_df = pd.read_csv(ground_truth_csv)

    ground_truth_df.columns=pairs_df.columns
    print("ground_truth_df\n", ground_truth_df)
    # inner join
    inner_df = pd.merge(pairs_df, ground_truth_df, on=["id1","id2"], how='inner')
    print(inner_df)
    inner_df.to_csv(output_csv, index=False)

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

get_paired_ids_no_fullfile("output.csv","./data/rest1clean.csv","./data/rest2clean.csv")
# get_paired_ids("paired_data_with_indexes.csv")
inner_merge('paired_data_id.csv', 'data/gtclean_evaluator_comma.csv', 'output_inner.csv')


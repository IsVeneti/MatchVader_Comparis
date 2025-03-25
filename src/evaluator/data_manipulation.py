
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
    inner_df = pd.merge(pairs_df, ground_truth_df, on=["rest1clean.csv","rest2clean.csv"], how='inner')
    print(inner_df)
    inner_df.to_csv(output_csv, index=False)

inner_merge('output.csv', 'data/gtclean_evaluator_comma.csv', 'output_inner.csv')


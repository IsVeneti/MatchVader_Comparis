import os
import pandas as pd

def get_or_create_id_pairs(pairs_csv, data1_csv, data2_csv,
                           output_csv='d1_pairs_with_ids.csv',
                           data_sep='|', pairs_sep=',',
                           force_recreate=False):
    """
    Get ID-based pairs - either load existing file or create it.
    
    Args:
        pairs_csv: Original index-based pairs file
        data1_csv: First dataset with ID column
        data2_csv: Second dataset with ID column
        output_csv: Where to save/load the ID-based pairs
        data_sep: Separator for data files
        pairs_sep: Separator for pairs file
        force_recreate: If True, recreate even if file exists
    
    Returns:
        DataFrame with ID-based pairs
    """
    # Check if already exists
    if os.path.exists(output_csv) and not force_recreate:
        print(f"✓ Loading existing ID-based pairs from '{output_csv}'")
        return pd.read_csv(output_csv)
    
    # Need to create it
    print(f"→ Converting pairs to IDs (this may take a moment)...")
    
    # Load files
    pairs_df = pd.read_csv(pairs_csv, sep=pairs_sep, header=None, skiprows=1)
    data1_df = pd.read_csv(data1_csv, sep=data_sep)
    data2_df = pd.read_csv(data2_csv, sep=data_sep)
    
    # Get indices from pairs
    indices1 = pairs_df.iloc[:, 0]
    indices2 = pairs_df.iloc[:, 1]
    
    # Look up the actual IDs
    ids1 = data1_df.iloc[indices1]['id'].reset_index(drop=True)
    ids2 = data2_df.iloc[indices2]['id'].reset_index(drop=True)
    
    # Build result with IDs
    result = pd.DataFrame({
        'id1': ids1,
        'id2': ids2
    })
    
    # Add score if exists
    if pairs_df.shape[1] > 2:
        result['score'] = pairs_df.iloc[:, 2].values
    
    
    return result

if __name__ == "__main__":
    # === ONE-TIME CONVERSION ===
    print(os.getcwd())
    output_csv = 'data/d1_data/d1_pairs_with_ids.csv'

    pairs_with_ids = get_or_create_id_pairs(
        'data/d1_data/d1_pairs.csv',
        'data/d1_data/rest1clean.csv',
        'data/d1_data/rest2clean.csv',
        output_csv=output_csv
    )
    # Save for next time
    pairs_with_ids.to_csv(output_csv, index=False)
    print(f"✓ Saved ID-based pairs to '{output_csv}' ({len(pairs_with_ids)} pairs)")

    print(f"Converted pairs to IDs: {len(pairs_with_ids)} pairs")

    
    # # Simple merge!
    # merged = merge_dataframes('d1_pairs_with_ids.csv', 'gtclean.csv')
    # print(f"Merged: {len(merged)} matching pairs found")
import pandas as pd
import json
import sys

input_file = "result/phi_try_o/results.csv"
output_file = input_file.replace(".csv", "_fixed.csv")

df = pd.read_csv(input_file)

def extract_match(response):
    try:
        data = json.loads(str(response))
        return data.get("match", None)
    except (json.JSONDecodeError, TypeError):
        return None

df["match"] = df["response"].apply(extract_match)

df.to_csv(output_file, index=False)
print(f"Saved to {output_file}")
print(f"Match values: {df['match'].value_counts().to_dict()}")
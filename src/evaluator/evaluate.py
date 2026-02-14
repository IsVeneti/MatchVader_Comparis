import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def evaluate_results(merged_data: pd.DataFrame):
    """Returns evaluation metrics as a dictionary"""
    df = merged_data
    y_pred = df.iloc[:, -2]
    y_true = df.iloc[:, -1]
    
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    results = {
        'accuracy': float(acc),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'confusion_matrix': conf_matrix.tolist()
    }
    
    return results

def save_results_as_json(results_dict, filename='evaluation_results.json'):
    """
    Save evaluation results as JSON file
    
    Args:
        results_dict (dict): Dictionary containing evaluation results for different datasets
        filename (str): Output JSON filename
    """
    with open(filename, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"Results saved to {filename}")

# TODO: This has an error float object is not subscriptable
def save_results_as_csv(results_dict, filename='evaluation_results.csv'):
    """
    Save evaluation results as CSV file
    
    Args:
        results_dict (dict): Dictionary containing evaluation results for different datasets
        filename (str): Output CSV filename
    """
    rows = []
    for dataset, metrics in results_dict.items():
        result_row = {
            'dataset': dataset,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score']
        }
        
        # Flatten confusion matrix into separate columns
        conf_matrix = metrics['confusion_matrix']
        for i, cm_row in enumerate(conf_matrix):
            for j, value in enumerate(cm_row):
                result_row[f'cm_{i}_{j}'] = value
        
        rows.append(result_row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    print("outer")
    evaluate_results("output_merge_outer.csv")
    print("inner")
    evaluate_results("output_merge_inner.csv")

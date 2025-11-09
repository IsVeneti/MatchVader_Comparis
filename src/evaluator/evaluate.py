import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def evaluate_results(merged_data: pd.DataFrame):
    """
    Evaluates the classification results in the last two columns of the merged CSV using sklearn metrics.

    Args:
        merged_csv (pd.DataFrame): Merged DataFrame containing predictions and ground truth in the last two columns.
    
    Returns:
        None (Prints evaluation metrics and confusion matrix)
    """
    # Load the merged results
    df = merged_data
    
    # Get the last two columns (assumed to be prediction and ground truth)
    y_pred = df.iloc[:, -2]  # Second last column (Predictions)
    y_true = df.iloc[:, -1]  # Last column (Ground truth)

    print("\nEvaluation Results:")

    # Classification Metrics
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)

if __name__ == "__main__":
    print("outer")
    evaluate_results("output_merge_outer.csv")
    print("inner")
    evaluate_results("output_merge_inner.csv")

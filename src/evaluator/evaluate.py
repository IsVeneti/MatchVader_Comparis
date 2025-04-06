import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def evaluate_results(merged_csv):
    """
    Evaluates the classification results in the last two columns of the merged CSV using sklearn metrics.

    Args:
        merged_csv (str): Path to the merged CSV file.
    
    Returns:
        None (Prints evaluation metrics and confusion matrix)
    """
    # Load the merged results
    df = pd.read_csv(merged_csv)
    
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


evaluate_results("output_merge_outer.csv")
print("inner")
evaluate_results("output_merge_inner.csv")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

def evaluate_model(y_true, y_pred):
    """
    Calculate evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Ground truth labels.

    y_pred : array-like
        Predicted labels.

    Returns
    -------
    dict
        Dictionary containing evaluation metrics.
    """

    results = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-score": f1_score(y_true, y_pred)
    }

    return results

def metrics_to_dataframe(results):
    """
    Convert metric dictionary to DataFrame.
    """

    return pd.DataFrame(
        results.items(),
        columns=["Metric", "Score"]
    )

def get_classification_report(y_true, y_pred):
    """
    Return classification report.
    """

    return classification_report(
        y_true,
        y_pred
    )

def get_confusion_matrix(
    y_true,
    y_pred
):
    """
    Compute confusion matrix.
    """

    return confusion_matrix(
        y_true,
        y_pred
    )

def plot_confusion_matrix(cm, save_path=None):
    """
    Plot confusion matrix.
    """

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.5
    )

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    if save_path:
        plt.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    plt.show()

def save_metrics(
    metrics_df,
    filepath
):
    """
    Save metrics to CSV.
    """

    metrics_df.to_csv(
        filepath,
        index=False
    )
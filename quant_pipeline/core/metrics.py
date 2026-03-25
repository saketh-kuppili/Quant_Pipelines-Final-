"""
Evaluation metrics for classification models.
"""

from sklearn.metrics import accuracy_score


def compute_accuracy(preds, labels):
    """
    Compute classification accuracy.

    Parameters
    ----------
    preds : list[int]
        Predicted labels.
    labels : list[int]
        Ground truth labels.

    Returns
    -------
    float
        Accuracy in [0, 1].
    """
    return accuracy_score(labels, preds)
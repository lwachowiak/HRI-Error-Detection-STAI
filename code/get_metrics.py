import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_metrics(y_pred, y_true, tolerance=0):
    """
    Calculate macro-averaged accuracy, precision, recall, F1-score, and apply a tolerance margin
    around each prediction for multiclass problems.

    Args:
        y_pred (list or np.array): List or array of predicted values.
        y_true (list or np.array): List or array of true values.
        tolerance (int): Number of samples around each prediction where predictions
                         are considered correct.
                         NOTE: tolerance will be applied to an interval of 0.5s around each prediction. The number of samples may vary depending on the sampling rate.

    Returns:
        dict: Dictionary containing the calculated metrics.

    """
    # Convert inputs to NumPy arrays
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Get unique classes
    classes = np.unique(np.concatenate([y_true, y_pred]))

    # Create a copy of y_pred for the tolerant predictions
    y_pred_tolerant = y_pred.copy()

    # Apply tolerance margin
    if tolerance > 0:
        for i in range(len(y_pred)):
            start = max(0, i - tolerance)
            end = min(len(y_pred), i + tolerance + 1)
            if y_true[i] in y_pred[start:end]:
                y_pred_tolerant[i] = y_true[i]


    # Calculate base metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=classes, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, labels=classes, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=classes, average='macro', zero_division=0)


    # Calculate tolerant metrics
    accuracy_tolerant = accuracy_score(y_true, y_pred_tolerant)
    precision_tolerant = precision_score(y_true, y_pred_tolerant, labels=classes, average='macro', zero_division=0)
    recall_tolerant = recall_score(y_true, y_pred_tolerant, labels=classes, average='macro', zero_division=0)
    f1_tolerant = f1_score(y_true, y_pred_tolerant, labels=classes, average='macro', zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy_tolerant': accuracy_tolerant,
        'precision_tolerant': precision_tolerant,
        'recall_tolerant': recall_tolerant,
        'f1_tolerant': f1_tolerant
    }



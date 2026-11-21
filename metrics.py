"""
From-scratch classification metrics built only with NumPy (no scikit-learn).
Includes Accuracy, Precision, Recall, F1-Score, Log-Loss, ROC Curve, and ROC-AUC.
"""

import numpy as np


def confusion_counts(y_true, y_pred):
    """Return (TN, FP, FN, TP) for binary labels (0 = Away Win, 1 = Home Win)."""
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return tn, fp, fn, tp


def accuracy(y_true, y_pred):
    """Fraction of all games predicted correctly."""
    return float(np.mean(y_true == y_pred))


def precision(y_true, y_pred):
    """Of all games where we predicted a Home Win, how many were right?"""
    _, fp, _, tp = confusion_counts(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    """Of all actual Home Wins, how many did we catch?"""
    _, _, fn, tp = confusion_counts(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    """Harmonic mean that balances Precision and Recall."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0


def log_loss(y_true, y_prob):
    """
    Measures how well-calibrated the predicted probabilities are (lower is better).
    Confident wrong predictions receive a much larger penalty than uncertain ones.
    """
    p = np.clip(y_prob, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1.0 - p)))


def roc_curve(y_true, y_prob):
    """
    Sweep probability thresholds from high to low and compute:
      - FPR (False Positive Rate): away wins mistakenly predicted as home wins
      - TPR (True Positive Rate) : home wins correctly predicted as home wins
    """
    # Test every unique predicted probability as a cutoff, plus endpoints
    thresholds = np.unique(y_prob)[::-1]
    fpr_list = [0.0]
    tpr_list = [0.0]

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    for thresh in thresholds:
        preds = (y_prob >= thresh).astype(np.int64)
        tp = int(np.sum((y_true == 1) & (preds == 1)))
        fp = int(np.sum((y_true == 0) & (preds == 1)))
        tpr_list.append(tp / n_pos if n_pos > 0 else 0.0)
        fpr_list.append(fp / n_neg if n_neg > 0 else 0.0)

    fpr_list.append(1.0)
    tpr_list.append(1.0)
    return np.array(fpr_list), np.array(tpr_list)


def roc_auc(y_true, y_prob):
    """Area under the ROC curve using the trapezoid rule (0.5 = random, 1.0 = perfect)."""
    fpr, tpr = roc_curve(y_true, y_prob)
    widths = fpr[1:] - fpr[:-1]
    avg_heights = (tpr[1:] + tpr[:-1]) / 2.0
    return float(np.sum(widths * avg_heights))


def evaluate_model(y_true, y_pred, y_prob):
    """Compute all Milestone 3 metrics and return them in one dictionary."""
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


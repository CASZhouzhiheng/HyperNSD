"""Metrics for OOD and misclassification detection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def fpr_at_recall(labels: np.ndarray, scores: np.ndarray, recall: float = 0.95) -> float:
    """Return the false-positive rate when recall first reaches the target."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if np.unique(labels).size < 2:
        return float("nan")
    order = np.argsort(scores)[::-1]
    labels = labels[order]
    positives = max(int((labels == 1).sum()), 1)
    negatives = max(int((labels == 0).sum()), 1)
    true_positives = np.cumsum(labels == 1)
    false_positives = np.cumsum(labels == 0)
    reached = np.flatnonzero(true_positives / positives >= recall)
    return float(false_positives[reached[0]] / negatives) if reached.size else 1.0


def binary_detection_metrics(positive_scores: np.ndarray, negative_scores: np.ndarray) -> dict[str, float]:
    """Compute AUROC, two AUPR variants, and FPR95 for high-score positives."""
    positive_scores = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    negative_scores = np.asarray(negative_scores, dtype=np.float64).reshape(-1)
    labels = np.concatenate((np.ones_like(positive_scores), np.zeros_like(negative_scores)))
    scores = np.concatenate((positive_scores, negative_scores))
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "aupr_pos": float(average_precision_score(labels, scores)),
        "aupr_neg": float(average_precision_score(1 - labels, -scores)),
        "fpr95": fpr_at_recall(labels, scores),
    }


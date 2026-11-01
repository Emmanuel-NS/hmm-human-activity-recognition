"""Classification metrics for HMM decoded sequences."""

from __future__ import annotations

import numpy as np
import pandas as pd


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_sensitivity_specificity(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, class_names: list[str]
) -> pd.DataFrame:
    """
    Sensitivity (recall) = TP / (TP + FN)
    Specificity = TN / (TN + FP)
    Overall accuracy is the same for all rows (global).
    """
    cm = confusion_matrix(y_true, y_pred, n_classes)
    total = cm.sum()
    accuracy = float(np.trace(cm) / total) if total else 0.0

    rows = []
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        rows.append(
            {
                "State (Activity)": name,
                "Number of Samples": int(cm[i, :].sum()),
                "Sensitivity": round(sens, 4),
                "Specificity": round(spec, 4),
                "Overall Accuracy": round(accuracy, 4),
            }
        )
    return pd.DataFrame(rows)

"""Metrics: 3-class Accuracy/F1 (fixed labels, zero_division=0), MAE, Pearson."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def cls_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "MacroF1": float(f1_score(y_true, y_pred, labels=[0, 1, 2],
                                  average="macro", zero_division=0)),
        "WeightedF1": float(f1_score(y_true, y_pred, labels=[0, 1, 2],
                                     average="weighted", zero_division=0)),
    }


def reg_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred = np.clip(y_pred, -3.0, 3.0)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    return {"MAE": mae, "Pearson": pearson}


def head_agreement(y_cls_pred: np.ndarray, y_reg_pred: np.ndarray) -> float:
    """Fraction where the classifier's class matches the regressor's sign class."""
    reg_sign = np.where(y_reg_pred < 0, 0, np.where(y_reg_pred > 0, 2, 1))
    return float(np.mean(y_cls_pred == reg_sign))

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module | None = None,
    device: torch.device | None = None,
):
    """Evaluate a model without updating its parameters."""
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    total_loss = 0.0
    correct = 0
    n = 0
    all_y = []
    all_pred = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)

        if criterion is not None:
            total_loss += criterion(logits, y).item() * y.size(0)

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        n += y.size(0)

        all_y.append(y.cpu())
        all_pred.append(pred.cpu())

    return {
        "loss": total_loss / n if criterion is not None else np.nan,
        "accuracy": correct / n,
        "y_true": torch.cat(all_y).numpy(),
        "y_pred": torch.cat(all_pred).numpy(),
    }


def class_metrics(y_true, y_pred, n_classes: int = 6):
    """Return per-class precision, recall, F1 and support."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(n_classes),
        zero_division=0,
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "accuracy": accuracy_score(y_true, y_pred),
    }


def normalized_confusion(y_true, y_pred, n_classes: int = 6):
    """Row-normalized confusion matrix; each true-class row sums to 1."""
    return confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(n_classes),
        normalize="true",
    )

from __future__ import annotations

from copy import deepcopy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .evaluation import evaluate
from .model import AcousticCNN1D


BASE_LR = 0.0005


def set_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def paper_lr(epoch: int, total_epochs: int, base_lr: float = BASE_LR) -> float:
    """Learning-rate schedule used in the reference experiments."""
    p = epoch / total_epochs
    return base_lr / ((1 + 10 * p) ** 0.75)


def train_model(
    train_loader,
    val_loader,
    *,
    seed: int = 0,
    epochs: int = 50,
    n_classes: int = 6,
    device: torch.device | None = None,
    verbose: bool = True,
):
    """Train the CNN and restore the best source-validation checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)

    model = AcousticCNN1D(n_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR)

    best_val_acc = -1.0
    best_epoch = -1
    best_state = None
    history = []

    for epoch in range(epochs):
        lr = paper_lr(epoch, epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr

        model.train()
        loss_sum = 0.0
        correct = 0
        n = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += y.size(0)

        train_loss = loss_sum / n
        train_acc = correct / n
        val_metrics = evaluate(model, val_loader, criterion, device=device)

        history.append({
            "epoch": epoch + 1,
            "lr": lr,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
        })

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())

        if verbose and (
            epoch == 0
            or (epoch + 1) % 5 == 0
            or epoch + 1 == epochs
        ):
            print(
                f"Epoch {epoch + 1:02d}/{epochs} | "
                f"train={train_acc:.4f} | "
                f"val={val_metrics['accuracy']:.4f} | "
                f"lr={lr:.6f}"
            )

    model.load_state_dict(best_state)

    return model, pd.DataFrame(history), best_epoch, best_val_acc

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


TASK_TO_DOMAIN = {9: 0, 10: 1, 11: 2, 12: 3}
DOMAIN_TO_SPEED = {0: "5 Hz", 1: "10 Hz", 2: "20 Hz", 3: "30 Hz"}

CLASS_NAMES = [
    "Healthy",
    "Bearing fault",
    "Bowed rotor",
    "Broken rotor bars",
    "Rotor misalignment",
    "Voltage unbalance",
]


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 256,
    shuffle: bool = False,
    device: torch.device | None = None,
    num_workers: int = 2,
) -> DataLoader:
    """Convert (N, 1024) acoustic windows to Conv1d-ready mini-batches."""
    X_tensor = torch.from_numpy(X).float().unsqueeze(1)
    y_tensor = torch.from_numpy(y.astype(np.int64)).long()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return DataLoader(
        TensorDataset(X_tensor, y_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )


def load_acoustic_task(data_path: str | Path, target_domain: int):
    """Load one domain-generalization task through Genti's helper module.

    The preprocessing repository must be importable and provide
    preprocess_hustmotor.load_dg_task.
    """
    try:
        from preprocess_hustmotor import load_dg_task
    except ImportError as exc:
        raise ImportError(
            "Could not import preprocess_hustmotor.load_dg_task. "
            "Clone https://github.com/Genti24-bit/ISCF_Project and add it "
            "to PYTHONPATH, or use the public Colab notebook."
        ) from exc

    task = load_dg_task(str(data_path), target_domain=target_domain)
    _, Xa_tr, y_tr = task["train"]
    _, Xa_va, y_va = task["val"]
    _, Xa_te, y_te = task["target"]

    return (Xa_tr, y_tr), (Xa_va, y_va), (Xa_te, y_te)

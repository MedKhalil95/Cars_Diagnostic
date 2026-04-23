import numpy as np
import torch
from torch.utils.data import Dataset


class CarDataset(Dataset):
    """
    PyTorch Dataset for tabular car diagnostic data.

    Args:
        X: Feature matrix, shape (N, F) — numpy array or compatible.
        y: Integer class labels, shape (N,).
    """

    def __init__(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if len(X) != len(y):
            raise ValueError(f"X and y length mismatch: {len(X)} vs {len(y)}")

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
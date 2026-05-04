"""Tensor dataset loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class EventTensorDataset(Dataset):
    def __init__(self, feature_path: str | Path, split: int | None = None) -> None:
        arrays = np.load(feature_path, allow_pickle=True)
        indices = np.arange(arrays["x_pre"].shape[0])
        if split is not None:
            indices = indices[arrays["split"] == split]
        self.x_pre = torch.as_tensor(arrays["x_pre"][indices], dtype=torch.float32)
        self.event_embedding = torch.as_tensor(
            arrays["event_embedding"][indices], dtype=torch.float32
        )
        self.metadata = torch.as_tensor(arrays["metadata"][indices], dtype=torch.float32)
        self.y = torch.as_tensor(arrays["y"][indices], dtype=torch.float32)
        self.is_event = torch.as_tensor(arrays["is_event"][indices], dtype=torch.float32)
        self.indices = indices

    def __len__(self) -> int:
        return int(self.x_pre.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x_pre": self.x_pre[idx],
            "event_embedding": self.event_embedding[idx],
            "metadata": self.metadata[idx],
            "y": self.y[idx],
            "is_event": self.is_event[idx],
            "index": torch.as_tensor(self.indices[idx], dtype=torch.long),
        }

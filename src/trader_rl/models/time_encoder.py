"""Temporal feature encoder used by policy networks."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEncoder(nn.Module):
    """Encode per-instrument temporal context via lightweight convolutions."""

    def __init__(self, feature_dim: int, out_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(feature_dim, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.proj = nn.Linear(64, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``x`` shaped as ``[B, N, T, F]`` into ``[B, N, out_dim]``."""
        batch, instruments, window, features = x.shape
        out = x.reshape(batch * instruments, window, features).transpose(1, 2)
        out = F.gelu(self.conv1(out))
        out = F.gelu(self.conv2(out))
        out = out.mean(dim=-1)
        out = self.proj(out)
        out = self.norm(out)
        return out.reshape(batch, instruments, -1)

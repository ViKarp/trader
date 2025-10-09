"""PPO agent configuration."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PPOConfig:
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    lr: float = 3e-4
    update_epochs: int = 6
    minibatch_size: int = 64
    rollout_steps: int = 1024
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    device: str = "cpu"

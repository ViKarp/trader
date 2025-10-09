"""Environment configuration objects."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EnvConfig:
    window_size: int = 64
    initial_equity: float = 1_000_000.0
    commission_bp: float = 2.0
    borrow_fee_bp_day: float = 20.0
    spread_bp: float = 10.0
    impact_coef: float = 2e-4
    max_leverage: float = 2.0
    gamma: float = 0.999
    feature_set: str = "minimal"

"""Utility functions for feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(series: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute the Relative Strength Index for a 1D price series."""
    diff = np.diff(series, prepend=series[0])
    up = np.clip(diff, 0, None)
    dn = np.clip(-diff, 0, None)
    roll_up = pd.Series(up).rolling(period).mean().values
    roll_dn = pd.Series(dn).rolling(period).mean().values
    rs = np.divide(roll_up, roll_dn + 1e-12)
    out = 100.0 - (100.0 / (1.0 + rs))
    out[np.isnan(out)] = 50.0
    return out


def ema(series: np.ndarray, span: int = 12) -> np.ndarray:
    """Exponential moving average."""
    return pd.Series(series).ewm(span=span, adjust=False).mean().values


def zscore(values: np.ndarray, window: int = 64) -> np.ndarray:
    """Rolling z-score with zero fill for NaNs."""
    series = pd.Series(values)
    mean = series.rolling(window).mean().values
    std = series.rolling(window).std().values
    normalized = (values - mean) / (std + 1e-8)
    normalized[np.isnan(normalized)] = 0.0
    return normalized

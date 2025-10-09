"""Technical indicator utilities used across trading environments."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(series: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute the Relative Strength Index for a closing price series."""
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
    """Exponential moving average with pandas semantics."""
    return pd.Series(series).ewm(span=span, adjust=False).mean().values


def zscore(series: np.ndarray, window: int = 64) -> np.ndarray:
    """Rolling z-score normalisation."""
    s = pd.Series(series)
    m = s.rolling(window).mean().values
    v = s.rolling(window).std().values
    z = (series - m) / (v + 1e-8)
    z[np.isnan(z)] = 0.0
    return z

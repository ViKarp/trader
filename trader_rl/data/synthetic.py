"""Synthetic datasets used for quick experiments."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd


def make_or_load_sample_csv(path: str, n_symbols: int = 4, bars: int = 60_000) -> pd.DataFrame:
    """Create or load a synthetic OHLCV dataset."""
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["timestamp"])

    rng = np.random.RandomState(0)
    base_ts = pd.date_range("2024-01-01", periods=bars, freq="15min", tz="UTC")
    rows = []
    for s in range(n_symbols):
        px = 100 + np.cumsum(rng.randn(bars) * 0.2)
        px = np.maximum(px, 1.0)
        vol = rng.lognormal(mean=12, sigma=0.3, size=bars).astype(np.float64)
        for t in range(bars):
            close = px[t]
            open_ = close * (1 + rng.randn() * 0.0005)
            high = max(open_, close) * (1 + abs(rng.randn()) * 0.001)
            low = min(open_, close) * (1 - abs(rng.randn()) * 0.001)
            rows.append(
                {
                    "timestamp": base_ts[t],
                    "symbol": f"S{s + 1}",
                    "open": float(open_),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close),
                    "volume": float(vol[t]),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df

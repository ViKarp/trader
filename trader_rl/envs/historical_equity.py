"""Environment tailored for the historical US equities dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from trader_rl.config import EnvConfig
from trader_rl.data.us_equities import load_us_equities

from .multi_asset import MultiAssetTradingEnv


class HistoricalEquityTradingEnv(MultiAssetTradingEnv):
    """Multi-asset environment preconfigured for the Kaggle US equities dataset."""

    def __init__(
        self,
        data_root: Union[str, Path],
        symbols: Optional[Sequence[str]] = None,
        *,
        asset_folder: str = "Stocks",
        freq: str = "1D",
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
        fill_method: str = "ffill",
        min_bars: int = 64,
        cfg: EnvConfig = EnvConfig(),
    ) -> None:
        dataset = load_us_equities(
            root=data_root,
            symbols=symbols,
            asset_folder=asset_folder,
            freq=freq,
            start=start,
            end=end,
            fill_method=fill_method,
            min_bars=min_bars,
        )
        super().__init__(df=dataset, symbols=symbols, cfg=cfg)

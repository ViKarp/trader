"""Helpers for loading the shared US equities dataset."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

import pandas as pd

PRICE_COLS = ["open", "high", "low", "close"]
VOL_COLS = ["volume"]


def _resolve_file(directory: Path, symbol: str) -> Path:
    """Return the path to the file containing data for ``symbol``.

    The Kaggle dataset distributes one ``.txt`` file per ticker. Depending on
    the platform the filename can be e.g. ``AAPL.txt`` or ``aapl.txt``. We try
    to locate the file using several common patterns and raise a clear error if
    nothing matches.
    """

    candidates = [directory / f"{symbol}.txt", directory / f"{symbol.upper()}.txt"]
    for path in candidates:
        if path.exists():
            return path

    matches = list(directory.glob(f"{symbol}*.txt"))
    if matches:
        return matches[0]

    matches = list(directory.glob(f"{symbol.upper()}*.txt"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Не найден файл с историческими данными для тикера '{symbol}' в '{directory}'"
    )


def _normalise_frame(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df.copy()
    lower_to_original = {c.lower(): c for c in df.columns}

    rename_pairs = {
        lower_to_original.get("date", "date"): "timestamp",
        lower_to_original.get("open", "open"): "open",
        lower_to_original.get("high", "high"): "high",
        lower_to_original.get("low", "low"): "low",
        lower_to_original.get("close", "close"): "close",
        lower_to_original.get("volume", "volume"): "volume",
    }
    df = df.rename(columns=rename_pairs)

    for col in PRICE_COLS + VOL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset="timestamp", keep="last")
    df["symbol"] = symbol
    return df[["timestamp", "symbol", *PRICE_COLS, *VOL_COLS]].reset_index(drop=True)


def _resample(df: pd.DataFrame, freq: str, fill_method: str) -> pd.DataFrame:
    df = df.set_index("timestamp").sort_index()

    if freq.upper() in ("1D", "D"):
        full_index = pd.date_range(df.index.min(), df.index.max(), freq="1D", tz="UTC")
        df = df.reindex(full_index)
    else:
        df = df.resample(freq).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )

    if fill_method == "ffill":
        df[PRICE_COLS] = df[PRICE_COLS].ffill().bfill()
        df[VOL_COLS] = df[VOL_COLS].fillna(0.0)
    elif fill_method == "drop":
        df = df.dropna(subset=PRICE_COLS)
        df[VOL_COLS] = df[VOL_COLS].fillna(0.0)
    else:
        raise ValueError("fill_method должен быть 'ffill' или 'drop'")

    df = df.dropna(subset=PRICE_COLS)
    df[VOL_COLS] = df[VOL_COLS].fillna(0.0)
    df.index.name = "timestamp"
    df = df.reset_index()
    return df


def load_us_equities(
    root: Union[str, Path],
    symbols: Optional[Sequence[str]] = None,
    asset_folder: str = "Stocks",
    freq: str = "1D",
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    fill_method: str = "ffill",
    min_bars: int = 64,
) -> pd.DataFrame:
    """Load a subset of the US equities dataset into a single DataFrame.

    Parameters
    ----------
    root:
        Path to the directory containing the extracted ``Data`` folder from the
        dataset.
    symbols:
        Iterable of tickers to load. If ``None`` we load every file in the
        ``asset_folder`` directory.
    asset_folder:
        Subdirectory name. The dataset contains ``Stocks`` and ``ETFs``.
    freq:
        Resampling frequency (pandas offset alias). ``"1D"`` keeps the original
        daily bars, while values such as ``"1W"`` or ``"1M"`` aggregate data to
        weekly or monthly bars respectively.
    start, end:
        Optional date boundaries. We keep rows ``start <= timestamp <= end``.
    fill_method:
        How to handle missing bars after resampling. ``"ffill"`` performs
        forward/backward filling for prices and replaces missing volume with
        zeros. ``"drop"`` removes incomplete rows.
    min_bars:
        Minimum number of bars required to keep a symbol. This prevents loading
        illiquid assets with extremely short histories.
    """

    data_root = Path(root)
    directory = data_root / asset_folder
    if not directory.exists():
        raise FileNotFoundError(
            f"Папка '{asset_folder}' не найдена внутри '{data_root}'. Проверьте путь к датасету."
        )

    if symbols is None:
        symbols = sorted(path.stem for path in directory.glob("*.txt"))

    fill_method = fill_method.lower()
    frames: List[pd.DataFrame] = []
    for symbol in symbols:
        file_path = _resolve_file(directory, symbol)
        try:
            raw = pd.read_csv(file_path)
            normalised = _normalise_frame(raw, symbol)
            if start is not None:
                start_ts = pd.Timestamp(start, tz="UTC")
                normalised = normalised[normalised["timestamp"] >= start_ts]
            if end is not None:
                end_ts = pd.Timestamp(end, tz="UTC")
                normalised = normalised[normalised["timestamp"] <= end_ts]

            if normalised.empty:
                continue

            resampled = _resample(normalised, freq=freq, fill_method=fill_method)
            if len(resampled) < min_bars:
                continue

            resampled["symbol"] = symbol
            frames.append(resampled)
        except:
            continue

    if not frames:
        raise ValueError("Не удалось загрузить ни одного символа из датасета")

    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined = combined.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return combined

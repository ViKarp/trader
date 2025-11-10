"""Helpers for loading the shared US equities dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

PRICE_COLS = ["open", "high", "low", "close"]
VOL_COLS = ["volume"]
RAW_REQUIRED_COLS = {"date", "open", "high", "low", "close", "volume"}


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


def _validate_raw_frame(df: pd.DataFrame, symbol: str) -> None:
    """Ensure the raw dataframe conforms to the expected Kaggle schema."""

    lower = {c.lower(): c for c in df.columns}
    missing = RAW_REQUIRED_COLS - set(lower)
    if missing:
        raise ValueError(
            "У тикера '{}' отсутствуют необходимые колонки: {}".format(
                symbol, ", ".join(sorted(missing))
            )
        )

    date_col = lower["date"]
    if df[date_col].isna().any():
        raise ValueError(
            f"В файле с тикером '{symbol}' есть пропуски в колонке даты."
        )

    duplicated = df[date_col].duplicated().sum()
    if duplicated:
        raise ValueError(
            f"В файле для '{symbol}' найдено {duplicated} дублирующихся дат."
        )

    for name in RAW_REQUIRED_COLS - {"date"}:
        col = lower[name]
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.isna().all():
            raise ValueError(
                f"Колонку '{col}' у '{symbol}' не удалось преобразовать в числовой формат."
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
            if col in PRICE_COLS:
                # Treat non-positive prices as missing values so they can be
                # forward/backward filled during resampling. Some tickers in the
                # dataset contain 0 or negative placeholders around corporate
                # actions which would otherwise propagate ``-inf`` values in the
                # log-return based features.
                df.loc[df[col] <= 0, col] = pd.NA
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

    mask_finite = np.isfinite(df[PRICE_COLS + VOL_COLS])
    if not mask_finite.to_numpy().all():
        df = df[mask_finite.all(axis=1)]

    df.index.name = "timestamp"
    df = df.reset_index()
    return df


def _assert_valid_bars(df: pd.DataFrame, symbol: str, stage: str) -> None:
    """Validate that the processed bars are finite and consistent."""

    if df.empty:
        raise ValueError(f"После этапа '{stage}' у тикера '{symbol}' не осталось строк")

    if not df["timestamp"].is_monotonic_increasing:
        raise ValueError(
            f"На этапе '{stage}' временные метки у '{symbol}' неотсортированы по возрастанию"
        )

    duplicated = df["timestamp"].duplicated().sum()
    if duplicated:
        raise ValueError(
            f"На этапе '{stage}' у '{symbol}' встречено {duplicated} дублирующихся временных меток"
        )

    numeric_cols: Iterable[str] = list(PRICE_COLS) + list(VOL_COLS)
    values = df[numeric_cols].to_numpy()
    if not np.isfinite(values).all():
        bad_idx = np.where(~np.isfinite(values))
        raise ValueError(
            "На этапе '{}' у '{}' обнаружены нечисловые значения (пример: строка {}, колонка {}).".format(
                stage, symbol, int(bad_idx[0][0]), numeric_cols[int(bad_idx[1][0])]
            )
        )

    if (df[VOL_COLS] < 0).any().any():
        raise ValueError(f"На этапе '{stage}' у '{symbol}' объемы не должны быть отрицательными")

    if (df[PRICE_COLS] <= 0).any().any():
        raise ValueError(
            f"На этапе '{stage}' у '{symbol}' цены должны быть строго положительными"
        )

    if (df["high"] < df["low"]).any():
        raise ValueError(f"На этапе '{stage}' у '{symbol}' high ниже low")

    bad_open = (df["open"] > df["high"]) | (df["open"] < df["low"])
    bad_close = (df["close"] > df["high"]) | (df["close"] < df["low"])
    if bad_open.any():
        raise ValueError(
            f"На этапе '{stage}' у '{symbol}' значения open выходят за диапазон high/low"
        )
    if bad_close.any():
        raise ValueError(
            f"На этапе '{stage}' у '{symbol}' значения close выходят за диапазон high/low"
        )


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
        raw = pd.read_csv(file_path)
        _validate_raw_frame(raw, symbol)
        normalised = _normalise_frame(raw, symbol)
        _assert_valid_bars(normalised, symbol, stage="normalise")
        if start is not None:
            start_ts = pd.Timestamp(start, tz="UTC")
            normalised = normalised[normalised["timestamp"] >= start_ts]
        if end is not None:
            end_ts = pd.Timestamp(end, tz="UTC")
            normalised = normalised[normalised["timestamp"] <= end_ts]

        if normalised.empty:
            continue

        resampled = _resample(normalised, freq=freq, fill_method=fill_method)
        _assert_valid_bars(resampled, symbol, stage="resample")
        if len(resampled) < min_bars:
            continue

        resampled["symbol"] = symbol
        frames.append(resampled)

    if not frames:
        raise ValueError("Не удалось загрузить ни одного символа из датасета")

    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined = combined.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    for sym, frame in combined.groupby("symbol", sort=False):
        _assert_valid_bars(frame.reset_index(drop=True), symbol=sym, stage="final")

    numeric_cols = PRICE_COLS + VOL_COLS
    values = combined[numeric_cols].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Объединенный датафрейм содержит нечисловые значения после всех проверок")

    return combined

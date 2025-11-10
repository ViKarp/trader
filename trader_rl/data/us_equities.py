"""Helpers for loading the shared US equities dataset."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

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


def _align_to_joint_range(
    df: pd.DataFrame,
    joint_index: pd.DatetimeIndex,
    fill_method: str,
) -> pd.DataFrame:
    df = df.set_index(["timestamp", "symbol"]).sort_index()

    def _align_symbol(group: pd.DataFrame) -> pd.DataFrame:
        symbol = group.index.get_level_values("symbol").unique()[0]
        values = group.droplevel("symbol").reindex(joint_index).copy()
        values.index.name = "timestamp"

        if fill_method == "drop":
            values = values.dropna(subset=PRICE_COLS)
            values[VOL_COLS] = values[VOL_COLS].fillna(0.0)
        elif fill_method == "ffill":
            values[PRICE_COLS] = values[PRICE_COLS].ffill().bfill()
            values = values.dropna(subset=PRICE_COLS)
            values[VOL_COLS] = values[VOL_COLS].fillna(0.0)
        elif fill_method == "bfill":
            values[PRICE_COLS] = values[PRICE_COLS].bfill().ffill()
            values = values.dropna(subset=PRICE_COLS)
            values[VOL_COLS] = values[VOL_COLS].fillna(0.0)
        else:
            raise ValueError("fill_method должен быть 'ffill', 'bfill' или 'drop'")

        values["symbol"] = symbol
        return values.reset_index()

    aligned = df.groupby(level="symbol", group_keys=False).apply(_align_symbol)
    return aligned.reset_index(drop=True)


def _resample_aligned(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq.upper() in ("1D", "D"):
        return df

    df = df.set_index(["timestamp", "symbol"]).sort_index()

    def _resample_symbol(group: pd.DataFrame) -> pd.DataFrame:
        symbol = group.index.get_level_values("symbol").unique()[0]
        values = group.droplevel("symbol")
        resampled = values.resample(freq).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        resampled = resampled.dropna(subset=PRICE_COLS)
        resampled[VOL_COLS] = resampled[VOL_COLS].fillna(0.0)
        resampled["symbol"] = symbol
        return resampled.reset_index()

    resampled = df.groupby(level="symbol", group_keys=False).apply(_resample_symbol)
    return resampled.reset_index(drop=True)


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
    fill_method: str = "bfill",
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
        How to handle missing bars after resampling. ``"bfill"`` (по умолчанию)
        выполняет обратное заполнение с дополнительным ``ffill`` для внутренних
        разрывов и заменяет пропуски объема нулями. ``"ffill"`` сохраняет
        прежнее поведение прямого заполнения, ``"drop"`` удаляет неполные строки.
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
    per_symbol: Dict[str, pd.DataFrame] = {}
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
        per_symbol[symbol] = normalised.assign(symbol=symbol)

    if not per_symbol:
        raise ValueError("Не удалось загрузить ни одного символа из датасета")

    active_symbols = list(per_symbol.keys())

    while True:
        active_frames = [per_symbol[s] for s in active_symbols]
        starts = [frame["timestamp"].min() for frame in active_frames]
        ends = [frame["timestamp"].max() for frame in active_frames]

        joint_start = max(starts)
        joint_end = min(ends)
        if pd.isna(joint_start) or pd.isna(joint_end) or joint_start > joint_end:
            raise ValueError(
                "Не удалось определить общий временной диапазон для выбранных тикеров"
            )

        joint_index = pd.date_range(joint_start, joint_end, freq="1D", tz="UTC")

        combined = pd.concat(active_frames, axis=0, ignore_index=True)
        combined = combined[
            (combined["timestamp"] >= joint_start) & (combined["timestamp"] <= joint_end)
        ]
        combined = combined.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

        aligned = _align_to_joint_range(combined, joint_index=joint_index, fill_method=fill_method)
        for sym, frame in aligned.groupby("symbol", sort=False):
            _assert_valid_bars(frame.reset_index(drop=True), symbol=sym, stage="aligned")

        resampled = _resample_aligned(aligned, freq=freq)
        resampled = resampled.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

        valid_symbols: List[str] = []
        for sym, frame in resampled.groupby("symbol", sort=False):
            _assert_valid_bars(frame.reset_index(drop=True), symbol=sym, stage="resample")
            if len(frame) >= min_bars:
                valid_symbols.append(sym)

        if not valid_symbols:
            raise ValueError(
                "После обработки не осталось тикеров, удовлетворяющих требованиям"
            )

        if set(valid_symbols) == set(active_symbols):
            final = resampled.reset_index(drop=True)
            break

        active_symbols = valid_symbols

    numeric_cols = PRICE_COLS + VOL_COLS
    values = final[numeric_cols].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Объединенный датафрейм содержит нечисловые значения после всех проверок")

    return final

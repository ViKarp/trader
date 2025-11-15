"""Experiment management utilities."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates


class ExperimentLogger:
    """Utility class that keeps track of experiment metadata and artifacts.

    The logger streams human-friendly messages to the console while keeping
    a persistent log file. Once the experiment finishes it serialises the
    collected metrics, produces diagnostic plots and stores the model
    weights as artifacts.
    """

    _VERBOSITY_LEVELS = {"all", "updates"}

    def __init__(
        self,
        root_dir: Optional[Path | str] = None,
        name: Optional[str] = None,
        overwrite: bool = False,
        verbosity: str = "all",
    ) -> None:
        self.root_dir = Path(root_dir or "experiments")
        self.root_dir.mkdir(parents=True, exist_ok=True)

        if name is None:
            name = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.experiment_dir = self.root_dir / name
        if self.experiment_dir.exists() and not overwrite:
            raise FileExistsError(
                f"Experiment directory '{self.experiment_dir}' already exists. "
                "Pass overwrite=True to reuse it."
            )
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        logger_name = f"trader_rl.experiment.{self.experiment_dir.name}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        # Remove handlers from a previous run with the same name.
        if self.logger.handlers:
            for handler in list(self.logger.handlers):
                self.logger.removeHandler(handler)

        formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(stream_handler)

        file_handler = logging.FileHandler(self.experiment_dir / "experiment.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        if verbosity not in self._VERBOSITY_LEVELS:
            raise ValueError(
                f"Unsupported verbosity '{verbosity}'. "
                f"Choose from: {sorted(self._VERBOSITY_LEVELS)}"
            )
        self._verbosity = verbosity

        self._history: Dict[str, list[float]] = {}
        self._updates: list[int] = []
        self._metadata: Dict[str, Any] = {}
        self._trades: list[Dict[str, Any]] = []
        self._price_data: Optional[Dict[str, Any]] = None

        self.logger.info("Experiment directory: %s", self.experiment_dir.resolve())

    # ------------------------------------------------------------------
    # Metadata helpers
    def log_config(self, **configs: Any) -> None:
        """Persist experiment configurations to a JSON file."""

        payload: Dict[str, Any] = {}
        for name, cfg in configs.items():
            payload[name] = self._convert(cfg)
        target = self.experiment_dir / "config.json"
        with target.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        self.logger.info("Saved configuration to %s", target.name)

    def add_metadata(self, **metadata: Any) -> None:
        """Store additional metadata that will be serialized on finalize."""

        self._metadata.update({k: self._convert(v) for k, v in metadata.items()})

    def log_message(self, message: str) -> None:
        """Log an arbitrary informational message."""

        self.logger.info(message)

    def set_price_data(
        self,
        *,
        symbols: Iterable[str],
        timestamps: Iterable[str],
        closes: Iterable[Iterable[float]],
    ) -> None:
        """Provide historical price data for plotting trade timelines."""

        self._price_data = {
            "symbols": list(symbols),
            "timestamps": list(timestamps),
            "closes": [list(series) for series in closes],
        }

    def log_step(self, event: Mapping[str, Any]) -> None:
        """Log detailed information about a single trading action."""

        symbol = event.get("symbol", "?")
        side = event.get("side", "?")
        qty = float(event.get("qty", 0.0))
        price = float(event.get("price", float("nan")))
        position = float(event.get("position", 0.0))
        target = float(event.get("target_fraction", 0.0))
        equity = float(event.get("equity", float("nan")))
        timestamp = event.get("timestamp", "?")
        step_idx = event.get("step", None)
        prefix = f"Step {step_idx:06d} | " if step_idx is not None else ""
        message = (
            f"{prefix}{timestamp} | {symbol} {side} qty={qty:.4f} "
            f"price={price:.4f} pos={position:.4f} target={target:.4f} equity={equity:.2f}"
        )
        if self._verbosity == "all":
            self.logger.info(message)
        self._trades.append({k: self._convert(v) for k, v in event.items()})

    # ------------------------------------------------------------------
    # Metrics collection
    def log_update(self, update: int, metrics: Mapping[str, Any]) -> None:
        """Log metrics for a training update and keep them for later analysis."""

        self._updates.append(update)
        segments: list[str] = []
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._history.setdefault(key, []).append(float(value))
                segments.append(f"{key}={value:.4f}")
            else:
                # Persist non-numeric metrics separately but still surface in the console.
                self._history.setdefault(key, []).append(float("nan"))
                segments.append(f"{key}={value}")
        joined = " | ".join(segments)
        self.logger.info("Update %03d | %s", update, joined)

    # ------------------------------------------------------------------
    # Finalisation
    def finalize(self, agent: Optional[Any] = None) -> None:
        """Persist collected metrics, plots and optionally model weights."""

        self._dump_history()
        self._plot_equity_curve()
        self._plot_trade_charts()
        self._dump_metadata()
        if agent is not None and hasattr(agent, "save_weights"):
            weights_path = self.experiment_dir / "model_weights.pt"
            agent.save_weights(weights_path)
            self.logger.info("Saved model weights to %s", weights_path.name)

    # ------------------------------------------------------------------
    # Internal helpers
    def _convert(self, value: Any) -> Any:
        if is_dataclass(value):
            return asdict(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {k: self._convert(v) for k, v in value.items()}
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            return [self._convert(v) for v in value]
        return value

    def _dump_history(self) -> None:
        if not self._updates:
            return
        metrics_file = self.experiment_dir / "metrics.csv"
        keys = sorted(self._history.keys())
        with metrics_file.open("w", encoding="utf-8") as f:
            header = ["update", *keys]
            f.write(",".join(header) + "\n")
            for idx, update in enumerate(self._updates):
                row = [str(update)]
                for key in keys:
                    values = self._history.get(key, [])
                    if idx < len(values) and values[idx] == values[idx]:
                        row.append(f"{values[idx]:.6f}")
                    else:
                        row.append("")
                f.write(",".join(row) + "\n")
        self.logger.info("Saved metrics to %s", metrics_file.name)

    def _plot_equity_curve(self) -> None:
        equity = self._history.get("last_equity")
        if not equity:
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self._updates, equity, marker="o", linewidth=1.5)
        ax.set_title("Equity progression")
        ax.set_xlabel("Update")
        ax.set_ylabel("Equity")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        plot_path = self.experiment_dir / "equity.png"
        fig.savefig(plot_path)
        plt.close(fig)
        self.logger.info("Saved equity plot to %s", plot_path.name)

    def _plot_trade_charts(self) -> None:
        if not self._price_data or not self._trades:
            return

        symbols = self._price_data["symbols"]
        timestamps = self._price_data["timestamps"]
        closes = self._price_data["closes"]
        if not symbols or not timestamps:
            return

        parsed_ts = [datetime.fromisoformat(ts) for ts in timestamps]
        dates = mdates.date2num(parsed_ts)

        trades_by_symbol: Dict[str, list[Dict[str, Any]]] = {symbol: [] for symbol in symbols}
        for trade in self._trades:
            sym = trade.get("symbol")
            if sym in trades_by_symbol:
                trades_by_symbol[sym].append(trade)

        for idx, symbol in enumerate(symbols):
            price_series = closes[idx] if idx < len(closes) else None
            if price_series is None:
                continue
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot_date(dates, price_series, "-", linewidth=1.2, label="Close")

            buy_plotted = False
            sell_plotted = False
            for trade in trades_by_symbol[symbol]:
                try:
                    dt = datetime.fromisoformat(str(trade.get("timestamp")))
                    ts_num = mdates.date2num(dt)
                    price = float(trade.get("price", float("nan")))
                    side = str(trade.get("side", "")).upper()
                    qty = float(trade.get("qty", 0.0))
                except Exception:
                    continue
                if not side or price != price:
                    continue
                color = "green" if side == "BUY" else "red"
                marker = "^" if side == "BUY" else "v"
                label = None
                if side == "BUY" and not buy_plotted:
                    label = "Buy"
                    buy_plotted = True
                elif side == "SELL" and not sell_plotted:
                    label = "Sell"
                    sell_plotted = True
                ax.scatter(ts_num, price, color=color, marker=marker, s=60, label=label)
                ax.annotate(
                    f"{side[0]} {qty:.2f}",
                    (ts_num, price),
                    textcoords="offset points",
                    xytext=(0, 6 if side == "BUY" else -10),
                    ha="center",
                    fontsize=8,
                    color=color,
                )

            ax.set_title(f"{symbol} trades")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(loc="best")
            fig.autofmt_xdate()
            fig.tight_layout()
            plot_path = self.experiment_dir / f"{symbol.lower()}_trades.png"
            fig.savefig(plot_path)
            plt.close(fig)
            self.logger.info("Saved trade plot to %s", plot_path.name)

    def _dump_metadata(self) -> None:
        if not self._metadata:
            return
        meta_file = self.experiment_dir / "metadata.json"
        with meta_file.open("w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2, ensure_ascii=False)
        self.logger.info("Saved metadata to %s", meta_file.name)

"""Experiment management utilities."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import matplotlib.pyplot as plt


class ExperimentLogger:
    """Utility class that keeps track of experiment metadata and artifacts.

    The logger streams human-friendly messages to the console while keeping
    a persistent log file. Once the experiment finishes it serialises the
    collected metrics, produces diagnostic plots and stores the model
    weights as artifacts.
    """

    def __init__(
        self,
        root_dir: Optional[Path | str] = None,
        name: Optional[str] = None,
        overwrite: bool = False,
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

        self._history: Dict[str, list[float]] = {}
        self._updates: list[int] = []
        self._metadata: Dict[str, Any] = {}

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

    def _dump_metadata(self) -> None:
        if not self._metadata:
            return
        meta_file = self.experiment_dir / "metadata.json"
        with meta_file.open("w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2, ensure_ascii=False)
        self.logger.info("Saved metadata to %s", meta_file.name)

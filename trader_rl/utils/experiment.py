"""Experiment logging utilities for training runs."""
from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_serializable(obj: Any) -> Any:
    """Recursively convert dataclasses and path-like objects to serializable forms."""

    if is_dataclass(obj):
        return {k: _ensure_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _ensure_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_ensure_serializable(v) for v in obj]
    return obj


class ExperimentLogger:
    """Lightweight logger that stores metrics and artifacts for experiments."""

    def __init__(self, root: Path | str = "artifacts", run_name: str = "experiment") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base_dir = self.root / f"{timestamp}_{run_name}"
        counter = 1
        run_dir = base_dir
        while run_dir.exists():
            run_dir = self.root / f"{timestamp}_{run_name}_{counter}"
            counter += 1
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=False)

        self.metrics: List[Dict[str, Any]] = []
        self._logger = logging.getLogger(f"trader_rl.experiment.{self.run_dir.name}")
        self._logger.setLevel(logging.INFO)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
            self._logger.addHandler(handler)
        self._logger.propagate = False

        self._logger.info('Эксперимент "%s" инициализирован', self.run_dir.name)

    # ------------------------------------------------------------------
    # Logging helpers
    def log_message(self, message: str) -> None:
        """Log a free-form message to the console."""

        self._logger.info(message)

    def log_update(self, metrics: Dict[str, Any]) -> None:
        """Store metrics for a single training update and print them."""

        record: Dict[str, Any] = {}
        for key, value in metrics.items():
            if hasattr(value, "item"):
                try:
                    record[key] = float(value.item())
                    continue
                except Exception:  # pragma: no cover - defensive
                    record[key] = value
                    continue
            if isinstance(value, (float, int)):
                record[key] = float(value)
            else:
                record[key] = value
        self.metrics.append(record)

        update = record.get("update")
        equity = record.get("equity")
        reward = record.get("reward_mean")
        message_parts = []
        if update is not None:
            message_parts.append(f"итерация={int(update):03d}")
        if equity is not None and isinstance(equity, (float, int)):
            message_parts.append(f"equity={equity:,.2f}")
        if reward is not None and isinstance(reward, (float, int)):
            message_parts.append(f"avg_reward={reward:,.4f}")
        if not message_parts:
            message_parts.append(str(record))
        self._logger.info(" | ".join(message_parts))

    # ------------------------------------------------------------------
    # Artifact handling
    def _save_metrics_files(self) -> None:
        if not self.metrics:
            self._logger.warning("Нет метрик для сохранения")
            return

        metrics_path = self.run_dir / "metrics.csv"
        fieldnames: Sequence[str] = sorted({key for row in self.metrics for key in row.keys()})
        with metrics_path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.metrics:
                writer.writerow(row)
        self._logger.info("Метрики сохранены в %s", metrics_path)

        json_path = self.run_dir / "metrics.json"
        with json_path.open("w", encoding="utf-8") as fp:
            json.dump(self.metrics, fp, ensure_ascii=False, indent=2)
        self._logger.info("Метрики сохранены в %s", json_path)

    def _save_equity_plot(self) -> None:
        if not self.metrics:
            return

        updates: List[float] = []
        equities: List[float] = []
        rewards: List[float] = []
        for row in self.metrics:
            update = row.get("update")
            equity = row.get("equity")
            reward = row.get("reward_mean")
            if update is None or equity is None:
                continue
            try:
                updates.append(float(update))
                equities.append(float(equity))
                if reward is not None:
                    rewards.append(float(reward))
                else:
                    rewards.append(float("nan"))
            except (TypeError, ValueError):
                continue

        if not updates:
            return

        fig, ax1 = plt.subplots(figsize=(8, 4.5), dpi=120)
        ax1.plot(updates, equities, label="Equity", color="#1f77b4")
        ax1.set_xlabel("Итерация обновления")
        ax1.set_ylabel("Equity", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")

        if any(math.isfinite(value) for value in rewards):
            ax2 = ax1.twinx()
            ax2.plot(updates, rewards, label="Средняя награда", color="#ff7f0e", linestyle="--")
            ax2.set_ylabel("Средняя награда", color="#ff7f0e")
            ax2.tick_params(axis="y", labelcolor="#ff7f0e")

        fig.tight_layout()
        plot_path = self.run_dir / "training_dynamics.png"
        fig.savefig(plot_path)
        plt.close(fig)
        self._logger.info("График динамики обучения сохранён в %s", plot_path)

    def finalize(self, agent: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Persist collected artifacts and save model weights."""

        if metadata:
            serialized = _ensure_serializable(metadata)
            meta_path = self.run_dir / "metadata.json"
            with meta_path.open("w", encoding="utf-8") as fp:
                json.dump(serialized, fp, ensure_ascii=False, indent=2)
            self._logger.info("Метаданные сохранены в %s", meta_path)

        self._save_metrics_files()
        self._save_equity_plot()

        if hasattr(agent, "save_weights"):
            weights_path = self.run_dir / "policy.pt"
            try:
                agent.save_weights(weights_path, metadata=metadata)
                self._logger.info("Веса модели сохранены в %s", weights_path)
            except Exception as exc:  # pragma: no cover - defensive
                self._logger.error("Не удалось сохранить веса модели: %s", exc)

        self._logger.info("Артефакты эксперимента доступны в %s", self.run_dir.resolve())

    @property
    def artifacts_path(self) -> Path:
        """Return the root path where artifacts are stored."""

        return self.run_dir

    def architecture_signature(self, params: Iterable[Tuple[str, Any]]) -> List[Tuple[str, Tuple[int, ...]]]:
        """Helper to expose consistent architecture signatures in logs."""

        signature: List[Tuple[str, Tuple[int, ...]]] = []
        for name, tensor in params:
            try:
                shape = tuple(int(dim) for dim in tensor.shape)
            except AttributeError:
                shape = tuple()
            signature.append((name, shape))
        return signature

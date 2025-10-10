"""Minimal training loop that demonstrates the PPO agent on synthetic data."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch

from trader_rl.agents import PPOAgent
from trader_rl.config import EnvConfig, PPOConfig
from trader_rl.data import make_or_load_sample_csv
from trader_rl.envs import MultiAssetTradingEnv
from trader_rl.utils import ExperimentLogger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Demo PPO training run")
    parser.add_argument("--data-path", type=Path, default=Path("market_sample.csv"))
    parser.add_argument("--total-updates", type=int, default=30, help="Количество обновлений PPO")
    parser.add_argument("--log-dir", type=Path, default=Path("artifacts"), help="Каталог для артефактов")
    parser.add_argument("--run-name", type=str, default="ppo_demo", help="Имя эксперимента")
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Путь к начальному чекпойнту модели (проверяется совместимость архитектуры)",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logger = ExperimentLogger(root=args.log_dir, run_name=args.run_name)
    logger.log_message("Загрузка данных и подготовка среды")

    df = make_or_load_sample_csv(args.data_path)
    symbols = sorted(df["symbol"].unique().tolist())[:4]

    cfg_env = EnvConfig(
        window_size=64,
        initial_equity=1_000_000.0,
        commission_bp=2.0,
        borrow_fee_bp_day=20.0,
        spread_bp=8.0,
        impact_coef=2e-4,
        max_leverage=2.0,
        feature_set="tech",
    )

    env = MultiAssetTradingEnv(df, symbols=symbols, cfg=cfg_env)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_ppo = PPOConfig(device=device, rollout_steps=1_024, update_epochs=6, minibatch_size=128, lr=3e-4)

    logger.log_message(f"Используется устройство: {device}")
    ppo = PPOAgent(
        env,
        cfg_ppo,
        logger=logger,
        initial_weights=str(args.weights) if args.weights is not None else None,
    )

    logger.log_message("Начало обучения агента")
    ppo.train(total_updates=args.total_updates)

    metadata = {
        "device": device,
        "symbols": symbols,
        "env_config": asdict(cfg_env),
        "ppo_config": asdict(cfg_ppo),
        "initial_weights": str(args.weights) if args.weights is not None else None,
    }
    logger.finalize(ppo, metadata=metadata)


if __name__ == "__main__":
    main()

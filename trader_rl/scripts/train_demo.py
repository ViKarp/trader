"""Minimal training loop that demonstrates the PPO agent on synthetic data."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from trader_rl.agents import PPOAgent
from trader_rl.config import EnvConfig, PPOConfig
from trader_rl.data import make_or_load_sample_csv
from trader_rl.envs import MultiAssetTradingEnv
from trader_rl.utils import ExperimentLogger


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the demo PPO agent on synthetic data.")
    parser.add_argument(
        "--total-updates",
        type=int,
        default=None,
        help="Number of PPO updates to run (default: cover full dataset once)",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("experiments"),
        help="Directory where experiment artifacts will be stored.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Custom experiment name (defaults to timestamp).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing experiment directory.",
    )
    parser.add_argument(
        "--initial-weights",
        type=Path,
        default=None,
        help="Optional path to a checkpoint with initial policy weights.",
    )
    args = parser.parse_args()

    logger = ExperimentLogger(
        root_dir=args.experiment_root,
        name=args.experiment_name,
        overwrite=args.overwrite,
    )

    data_path = "market_sample.csv"
    df = make_or_load_sample_csv(data_path)
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
    episode_steps = max(len(env.timestamps) - cfg_env.window_size - 1, 1)
    cfg_ppo = PPOConfig(
        device=device,
        rollout_steps=episode_steps,
        update_epochs=6,
        minibatch_size=128,
        lr=3e-4,
    )
    ppo = PPOAgent(env, cfg_ppo, initial_weights=args.initial_weights)

    logger.log_message(f"Device: {device}")
    logger.log_config(env=cfg_env, ppo=cfg_ppo, symbols=symbols)
    logger.add_metadata(data_path=str(Path(data_path).resolve()))
    logger.set_price_data(
        symbols=symbols,
        timestamps=[ts.isoformat() for ts in env.timestamps],
        closes=env.closes,
    )

    total_updates = args.total_updates if args.total_updates is not None else 1
    ppo.train(total_updates=total_updates, log_fn=logger.log_update, step_log_fn=logger.log_step)
    logger.finalize(ppo)


if __name__ == "__main__":
    main()

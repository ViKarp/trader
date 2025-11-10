"""Training entry point for the historical US equities dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from trader_rl.agents import PPOAgent
from trader_rl.config import EnvConfig, PPOConfig
from trader_rl.envs import HistoricalEquityTradingEnv
from trader_rl.utils import ExperimentLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the PPO agent on the historical US equities dataset."
    )
    parser.add_argument(
        "data_root",
        type=Path,
        help="Path to the directory that contains the extracted 'Data' folder from the dataset.",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help="Optional list of ticker symbols to trade. Loads all symbols if omitted.",
    )
    parser.add_argument(
        "--asset-folder",
        type=str,
        default="Stocks",
        help="Sub-folder inside the dataset to load (e.g. 'Stocks' or 'ETFs').",
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="1D",
        help="Resampling frequency (pandas offset alias).",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Optional inclusive start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional inclusive end date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--fill-method",
        type=str,
        default="ffill",
        choices=("ffill", "drop"),
        help="How to handle missing bars after resampling.",
    )
    parser.add_argument(
        "--min-bars",
        type=int,
        default=256,
        help="Minimum number of bars required per symbol.",
    )
    parser.add_argument(
        "--total-updates",
        type=int,
        default=200,
        help="Number of PPO updates to execute.",
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
    parser.add_argument(
        "--window-size",
        type=int,
        default=128,
        help="Number of timesteps in the observation window.",
    )
    parser.add_argument(
        "--feature-set",
        type=str,
        default="tech",
        choices=("minimal", "tech"),
        help="Feature set to construct for each asset.",
    )
    parser.add_argument(
        "--commission-bp",
        type=float,
        default=1.0,
        help="Commission in basis points applied to each trade.",
    )
    parser.add_argument(
        "--borrow-fee-bp-day",
        type=float,
        default=10.0,
        help="Daily borrow fee in basis points for short positions.",
    )
    parser.add_argument(
        "--spread-bp",
        type=float,
        default=5.0,
        help="Bid/ask spread in basis points.",
    )
    parser.add_argument(
        "--impact-coef",
        type=float,
        default=1e-4,
        help="Linear market impact coefficient.",
    )
    parser.add_argument(
        "--max-leverage",
        type=float,
        default=3.0,
        help="Maximum portfolio leverage.",
    )
    parser.add_argument(
        "--ppo-rollout-steps",
        type=int,
        default=2_048,
        help="Number of environment steps per PPO update.",
    )
    parser.add_argument(
        "--ppo-update-epochs",
        type=int,
        default=8,
        help="Number of epochs per PPO update.",
    )
    parser.add_argument(
        "--ppo-minibatch-size",
        type=int,
        default=256,
        help="Minibatch size for PPO optimisation.",
    )
    parser.add_argument(
        "--ppo-lr",
        type=float,
        default=2e-4,
        help="Learning rate for PPO optimiser.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger = ExperimentLogger(
        root_dir=args.experiment_root,
        name=args.experiment_name,
        overwrite=args.overwrite,
    )

    cfg_env = EnvConfig(
        window_size=args.window_size,
        initial_equity=1_000_000.0,
        commission_bp=args.commission_bp,
        borrow_fee_bp_day=args.borrow_fee_bp_day,
        spread_bp=args.spread_bp,
        impact_coef=args.impact_coef,
        max_leverage=args.max_leverage,
        feature_set=args.feature_set,
    )

    env = HistoricalEquityTradingEnv(
        data_root=args.data_root,
        symbols=args.symbols,
        asset_folder=args.asset_folder,
        freq=args.freq,
        start=args.start,
        end=args.end,
        fill_method=args.fill_method,
        min_bars=args.min_bars,
        cfg=cfg_env,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_ppo = PPOConfig(
        device=device,
        rollout_steps=args.ppo_rollout_steps,
        update_epochs=args.ppo_update_epochs,
        minibatch_size=args.ppo_minibatch_size,
        lr=args.ppo_lr,
    )

    agent = PPOAgent(env, cfg_ppo, initial_weights=args.initial_weights)

    logger.log_message(f"Device: {device}")
    logger.log_config(env=cfg_env, ppo=cfg_ppo, symbols=env.symbols)
    logger.add_metadata(
        data_root=str(Path(args.data_root).resolve()),
        asset_folder=args.asset_folder,
        freq=args.freq,
        start=args.start,
        end=args.end,
        fill_method=args.fill_method,
        min_bars=args.min_bars,
        symbols=list(env.symbols),
        total_updates=args.total_updates,
    )

    agent.train(total_updates=args.total_updates, log_fn=logger.log_update)
    logger.finalize(agent)


if __name__ == "__main__":
    main()

"""Command-line helpers for running demo training loops."""
from __future__ import annotations

import torch

from trader_rl.agents.ppo import PPOAgent, PPOConfig
from trader_rl.data.synthetic import make_or_load_sample_csv
from trader_rl.env.config import EnvConfig
from trader_rl.env.multi_asset import MultiAssetTradingEnv


def run_demo(data_path: str = "market_sample.csv", total_updates: int = 30) -> None:
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
    agent = PPOAgent(env, PPOConfig(device=device, rollout_steps=1024, update_epochs=6, minibatch_size=128, lr=3e-4))
    agent.train(total_updates=total_updates)


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()

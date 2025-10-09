"""Minimal training loop that demonstrates the PPO agent on synthetic data."""
from __future__ import annotations

import torch

from trader_rl.agents import PPOAgent
from trader_rl.config import EnvConfig, PPOConfig
from trader_rl.data import make_or_load_sample_csv
from trader_rl.envs import MultiAssetTradingEnv


def main() -> None:
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
    ppo = PPOAgent(
        env,
        PPOConfig(device=device, rollout_steps=1_024, update_epochs=6, minibatch_size=128, lr=3e-4),
    )
    ppo.train(total_updates=30)


if __name__ == "__main__":
    main()

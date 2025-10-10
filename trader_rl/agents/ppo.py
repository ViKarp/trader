"""Proximal Policy Optimization agent."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn

from trader_rl.config import PPOConfig
from trader_rl.envs import MultiAssetTradingEnv
from trader_rl.models import PolicyNet


class PPOAgent:
    """Lightweight PPO implementation for the trading environment."""

    def __init__(
        self,
        env: MultiAssetTradingEnv,
        cfg: PPOConfig,
        initial_weights: Optional[str | Path] = None,
    ) -> None:
        self.env = env
        obs, _ = env.reset()
        f_time = obs["time"].shape[-1]
        p_in = obs["pos"].shape[-1]
        g_in = obs["global"].shape[-1]
        self.N = obs["time"].shape[0]
        self.model = PolicyNet(f_time=f_time, p_in=p_in, g_in=g_in).to(cfg.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.cfg = cfg
        self.device = cfg.device
        self._global_step = 0
        self._arch_signature = {
            "model": self.model.__class__.__name__,
            "num_assets": self.N,
            "time_features": f_time,
            "position_features": p_in,
            "global_features": g_in,
        }

        if initial_weights is not None:
            self.load_weights(initial_weights)

    # ------------------------------------------------------------------
    # Helpers
    def _to_torch_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {
            "time": torch.tensor(obs["time"], dtype=torch.float32, device=self.device).unsqueeze(0),
            "pos": torch.tensor(obs["pos"], dtype=torch.float32, device=self.device).unsqueeze(0),
            "global": torch.tensor(obs["global"], dtype=torch.float32, device=self.device).unsqueeze(0),
        }

    def _stack_obs(self, obs_batch: Dict[str, List[np.ndarray]]) -> Dict[str, torch.Tensor]:
        x_time = torch.tensor(np.stack(obs_batch["time"], axis=0), dtype=torch.float32, device=self.device)
        x_pos = torch.tensor(np.stack(obs_batch["pos"], axis=0), dtype=torch.float32, device=self.device)
        x_global = torch.tensor(
            np.stack(obs_batch["global"], axis=0), dtype=torch.float32, device=self.device
        )
        return {"time": x_time, "pos": x_pos, "global": x_global}

    # ------------------------------------------------------------------
    # Rollout and update
    def collect_rollout(self):
        env = self.env
        obs, _ = env.reset()
        batch = {
            "obs_time": [],
            "obs_pos": [],
            "obs_global": [],
            "disc": [],
            "cont": [],
            "logp": [],
            "value": [],
            "reward": [],
            "done": [],
        }
        steps = 0
        episode_returns: List[float] = []
        current_episode_return = 0.0
        trade_events: List[Dict[str, Any]] = []

        while steps < self.cfg.rollout_steps:
            tobs = self._to_torch_obs(obs)
            with torch.no_grad():
                disc, cont, logp, value = self.model.act(tobs)
            disc_np = disc.squeeze(0).cpu().numpy().astype(np.int64)
            cont_np = cont.squeeze(0).cpu().numpy().astype(np.float32)
            action = {"disc": disc_np, "target": cont_np}
            next_obs, reward, term, trunc, info = env.step(action)

            batch["obs_time"].append(obs["time"])
            batch["obs_pos"].append(obs["pos"])
            batch["obs_global"].append(obs["global"])
            batch["disc"].append(disc_np)
            batch["cont"].append(cont_np)
            batch["logp"].append(logp.item())
            batch["value"].append(value.item())
            batch["reward"].append(reward)
            batch["done"].append(float(term or trunc))

            trade_events.extend(self._extract_trade_events(info, disc_np, cont_np))

            obs = next_obs
            current_episode_return += reward
            steps += 1
            if term or trunc:
                episode_returns.append(current_episode_return)
                current_episode_return = 0.0
                obs, _ = env.reset()

        if steps >= self.cfg.rollout_steps and current_episode_return:
            episode_returns.append(current_episode_return)

        b = len(batch["reward"])
        rewards = torch.tensor(batch["reward"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch["done"], dtype=torch.float32, device=self.device)
        values = torch.tensor(batch["value"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            adv = torch.zeros(b, dtype=torch.float32, device=self.device)
            lastgaelam = 0.0
            next_value = 0.0
            for t in reversed(range(b)):
                nextnonterminal = 1.0 - dones[t]
                delta = rewards[t] + self.cfg.gamma * next_value * nextnonterminal - values[t]
                lastgaelam = (
                    delta + self.cfg.gamma * self.cfg.gae_lambda * nextnonterminal * lastgaelam
                )
                adv[t] = lastgaelam
                next_value = values[t]
            returns = adv + values

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_batch = {"time": batch["obs_time"], "pos": batch["obs_pos"], "global": batch["obs_global"]}
        obs_t = self._stack_obs(obs_batch)
        disc_t = torch.tensor(np.stack(batch["disc"], axis=0), dtype=torch.int64, device=self.device)
        cont_t = torch.tensor(np.stack(batch["cont"], axis=0), dtype=torch.float32, device=self.device)
        logp_t = torch.tensor(batch["logp"], dtype=torch.float32, device=self.device)

        stats = {
            "rollout_steps": float(b),
            "mean_reward": float(rewards.mean().item()),
            "total_reward": float(rewards.sum().item()),
            "episodes_completed": float(len(episode_returns)),
            "mean_episode_reward": float(np.mean(episode_returns))
            if episode_returns
            else float("nan"),
        }
        return (
            obs_t,
            disc_t,
            cont_t,
            logp_t,
            values.detach(),
            returns.detach(),
            adv.detach(),
            stats,
            trade_events,
        )

    def update(
        self,
        obs_t: Dict[str, torch.Tensor],
        disc_t: torch.Tensor,
        cont_t: torch.Tensor,
        logp_old: torch.Tensor,
        values_old: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Dict[str, float]:
        b = returns.shape[0]
        idx = np.arange(b)
        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        batches = 0
        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, b, self.cfg.minibatch_size):
                end = start + self.cfg.minibatch_size
                mb_idx = idx[start:end]
                mb_obs = {
                    "time": obs_t["time"][mb_idx],
                    "pos": obs_t["pos"][mb_idx],
                    "global": obs_t["global"][mb_idx],
                }
                mb_disc = disc_t[mb_idx]
                mb_cont = cont_t[mb_idx]
                mb_logp_old = logp_old[mb_idx]
                mb_returns = returns[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_values_old = values_old[mb_idx]

                logp, entropy, value = self.model.evaluate_actions(mb_obs, mb_disc, mb_cont)
                ratio = torch.exp(logp - mb_logp_old)

                pg1 = ratio * mb_adv
                pg2 = torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef) * mb_adv
                pg_loss = -torch.min(pg1, pg2).mean()

                v_loss_unclipped = (value - mb_returns).pow(2)
                v_clipped = mb_values_old + torch.clamp(
                    value - mb_values_old, -self.cfg.clip_coef, self.cfg.clip_coef
                )
                v_loss_clipped = (v_clipped - mb_returns).pow(2)
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                ent = entropy.mean()
                loss = pg_loss + self.cfg.vf_coef * v_loss - self.cfg.ent_coef * ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.opt.step()
                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += ent.item()
                batches += 1

        if batches == 0:
            return {"policy_loss": float("nan"), "value_loss": float("nan"), "entropy": float("nan")}

        return {
            "policy_loss": total_pg_loss / batches,
            "value_loss": total_v_loss / batches,
            "entropy": total_entropy / batches,
        }

    def train(
        self,
        total_updates: int = 50,
        log_fn: Optional[Callable[[int, Mapping[str, float]], None]] = None,
        step_log_fn: Optional[Callable[[Mapping[str, Any]], None]] = None,
    ) -> None:
        for update in range(1, total_updates + 1):
            (
                obs_t,
                disc_t,
                cont_t,
                logp_old,
                values_old,
                returns,
                adv,
                rollout_stats,
                trade_events,
            ) = self.collect_rollout()
            update_stats = self.update(obs_t, disc_t, cont_t, logp_old, values_old, returns, adv)
            eq = float(self.env.info_last.get("equity", float("nan")))
            metrics = {**rollout_stats, **update_stats, "last_equity": eq}
            if step_log_fn is not None:
                for event in trade_events:
                    step_log_fn(event)
            if log_fn is not None:
                log_fn(update, metrics)
            else:
                formatted = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"Update {update:03d}: {formatted}")

    def _extract_trade_events(
        self, info: Mapping[str, Any], disc: np.ndarray, cont: np.ndarray
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        if not info:
            return events
        symbols = getattr(self.env, "symbols", [f"asset_{i}" for i in range(self.N)])
        timestamp = info.get("timestamp")
        order_qty = np.asarray(info.get("order_qty", np.zeros(self.N)), dtype=float)
        fill_price = np.asarray(info.get("fill_price", np.zeros(self.N)), dtype=float)
        position_qty = np.asarray(info.get("position_qty", np.zeros(self.N)), dtype=float)
        current_frac = np.asarray(info.get("current_fraction", np.zeros(self.N)), dtype=float)
        post_frac = np.asarray(info.get("post_fraction", np.zeros(self.N)), dtype=float)
        close = np.asarray(info.get("close", np.zeros(self.N)), dtype=float)
        equity = float(info.get("equity_after_trade", info.get("equity", float("nan"))))

        for idx, symbol in enumerate(symbols):
            qty = order_qty[idx] if idx < len(order_qty) else 0.0
            if np.isclose(qty, 0.0):
                continue
            side = "BUY" if qty > 0 else "SELL"
            event = {
                "step": self._global_step,
                "timestamp": timestamp,
                "symbol": symbol,
                "side": side,
                "qty": float(qty),
                "price": float(fill_price[idx]) if idx < len(fill_price) else float("nan"),
                "position": float(position_qty[idx]) if idx < len(position_qty) else float("nan"),
                "target_fraction": float(post_frac[idx]) if idx < len(post_frac) else float("nan"),
                "previous_fraction": float(current_frac[idx]) if idx < len(current_frac) else float("nan"),
                "close_price": float(close[idx]) if idx < len(close) else float("nan"),
                "equity": equity,
                "disc_action": int(disc[idx]) if idx < len(disc) else int(0),
                "cont_action": float(cont[idx]) if idx < len(cont) else float("nan"),
            }
            events.append(event)
        self._global_step += 1
        return events

    # ------------------------------------------------------------------
    # Serialization helpers
    def save_weights(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict(),
            "architecture": self._arch_signature,
            "config": {
                "ppo": self.cfg.__dict__,
            },
        }
        torch.save(payload, target)

    def load_weights(self, path: str | Path) -> None:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Weights file '{source}' does not exist")
        payload = torch.load(source, map_location=self.device)
        if isinstance(payload, dict) and "state_dict" in payload:
            state_dict = payload["state_dict"]
            arch = payload.get("architecture")
        else:
            state_dict = payload
            arch = None

        if arch is not None and arch != self._arch_signature:
            raise ValueError(
                "Loaded weights architecture does not match the current model. "
                f"Expected {self._arch_signature}, got {arch}."
            )

        self._validate_state_dict(state_dict)
        self.model.load_state_dict(state_dict)

    def _validate_state_dict(self, state_dict: Mapping[str, torch.Tensor]) -> None:
        model_state = self.model.state_dict()
        missing = set(model_state.keys()) - set(state_dict.keys())
        unexpected = set(state_dict.keys()) - set(model_state.keys())
        if missing or unexpected:
            raise ValueError(
                "State dict structure mismatch: "
                f"missing={sorted(missing)} unexpected={sorted(unexpected)}"
            )
        for key, tensor in model_state.items():
            other = state_dict[key]
            if tensor.shape != other.shape:
                raise ValueError(
                    f"Tensor shape mismatch for '{key}': expected {tuple(tensor.shape)}, "
                    f"got {tuple(other.shape)}"
                )

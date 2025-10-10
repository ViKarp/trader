"""Proximal Policy Optimization agent."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from trader_rl.config import PPOConfig
from trader_rl.envs import MultiAssetTradingEnv
from trader_rl.models import PolicyNet
from trader_rl.utils import ExperimentLogger


class PPOAgent:
    """Lightweight PPO implementation for the trading environment."""

    def __init__(
        self,
        env: MultiAssetTradingEnv,
        cfg: PPOConfig,
        *,
        logger: Optional[ExperimentLogger] = None,
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
        self.logger = logger

        self._architecture_signature = self._build_signature(self.model.state_dict().items())

        if initial_weights is not None:
            self.load_weights(initial_weights)
            if self.logger:
                self.logger.log_message(f"Начальные веса загружены из {initial_weights}")

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
        while steps < self.cfg.rollout_steps:
            tobs = self._to_torch_obs(obs)
            with torch.no_grad():
                disc, cont, logp, value = self.model.act(tobs)
            disc_np = disc.squeeze(0).cpu().numpy().astype(np.int64)
            cont_np = cont.squeeze(0).cpu().numpy().astype(np.float32)
            action = {"disc": disc_np, "target": cont_np}
            next_obs, reward, term, trunc, _ = env.step(action)

            batch["obs_time"].append(obs["time"])
            batch["obs_pos"].append(obs["pos"])
            batch["obs_global"].append(obs["global"])
            batch["disc"].append(disc_np)
            batch["cont"].append(cont_np)
            batch["logp"].append(logp.item())
            batch["value"].append(value.item())
            batch["reward"].append(reward)
            batch["done"].append(float(term or trunc))

            obs = next_obs
            steps += 1
            if term or trunc:
                obs, _ = env.reset()

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
        return (
            obs_t,
            disc_t,
            cont_t,
            logp_t,
            values.detach(),
            returns.detach(),
            adv.detach(),
            rewards.detach(),
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
    ) -> None:
        b = returns.shape[0]
        idx = np.arange(b)
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

    def train(self, total_updates: int = 50) -> None:
        for update in range(1, total_updates + 1):
            (
                obs_t,
                disc_t,
                cont_t,
                logp_old,
                values_old,
                returns,
                adv,
                rewards,
            ) = self.collect_rollout()
            self.update(obs_t, disc_t, cont_t, logp_old, values_old, returns, adv)
            eq = self.env.info_last.get("equity", float("nan"))
            metrics = {
                "update": update,
                "equity": float(eq),
                "reward_sum": float(rewards.sum().item()),
                "reward_mean": float(rewards.mean().item()),
                "adv_mean": float(adv.mean().item()),
                "adv_std": float(adv.std().item()),
            }
            if self.logger:
                self.logger.log_update(metrics)
            else:
                print(f"Update {update:03d}: last_equity={eq:,.2f}")

        if self.logger:
            self.logger.log_message("Обучение завершено")

    # ------------------------------------------------------------------
    # Weight management helpers
    def _build_signature(
        self, params: Iterable[Tuple[str, torch.Tensor]]
    ) -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
        signature: List[Tuple[str, Tuple[int, ...]]] = []
        for name, tensor in params:
            shape = tuple(int(dim) for dim in tensor.shape)
            signature.append((name, shape))
        return tuple(signature)

    def _normalize_signature(
        self, signature: Sequence[Tuple[str, Sequence[int]]]
    ) -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
        normalized: List[Tuple[str, Tuple[int, ...]]] = []
        for name, shape in signature:
            normalized.append((str(name), tuple(int(dim) for dim in shape)))
        return tuple(normalized)

    def save_weights(
        self,
        path: str | Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.model.state_dict(),
            "optimizer_state": self.opt.state_dict(),
            "architecture": list(self._architecture_signature),
            "metadata": metadata or {},
        }
        torch.save(payload, path)
        if self.logger:
            self.logger.log_message(f"Веса модели сохранены в {path}")
        return path

    def load_weights(self, path: str | Path, strict: bool = True) -> None:
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        architecture = checkpoint.get("architecture")
        if architecture is None:
            raise ValueError(
                f"Файл {path} не содержит сигнатуру архитектуры и не может быть использован"
            )
        normalized_arch = self._normalize_signature(architecture)
        if normalized_arch != self._architecture_signature:
            raise ValueError(
                "Сигнатура архитектуры сохранённых весов не совпадает с текущей моделью"
            )

        state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            raise ValueError(f"В файле {path} отсутствует состояние модели")
        self.model.load_state_dict(state_dict, strict=strict)

        opt_state = checkpoint.get("optimizer_state")
        if opt_state is not None:
            try:
                self.opt.load_state_dict(opt_state)
            except ValueError:
                if self.logger:
                    self.logger.log_message(
                        "Оптимизатор не удалось инициализировать из сохранённого состояния"
                    )

        if self.logger:
            self.logger.log_message(f"Веса модели загружены из {path}")

"""Proximal Policy Optimization agent tailored for the trading environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from trader_rl.env.multi_asset import MultiAssetTradingEnv
from trader_rl.models.time_encoder import TimeEncoder


@dataclass
class PPOConfig:
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    lr: float = 3e-4
    update_epochs: int = 6
    minibatch_size: int = 64
    rollout_steps: int = 1024
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    device: str = "cpu"


class PolicyNet(nn.Module):
    """Instrument-independent policy-head with shared temporal encoder."""

    def __init__(self, f_time: int, p_in: int, g_in: int, d_model: int = 128):
        super().__init__()
        self.time_enc = TimeEncoder(f_time, out_dim=64)
        self.pos_mlp = nn.Sequential(nn.Linear(p_in, 64), nn.GELU(), nn.Linear(64, 32))
        self.g_proj = nn.Linear(g_in, 32)

        self.backbone = nn.Sequential(
            nn.Linear(64 + 32 + 32, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.head_disc = nn.Linear(d_model, 3)
        self.head_mean = nn.Linear(d_model, 1)
        self.log_std = nn.Parameter(torch.zeros(1))
        self.value_head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, x_time: torch.Tensor, x_pos: torch.Tensor, x_global: torch.Tensor):
        batch, instruments, _, _ = x_time.shape
        h_time = self.time_enc(x_time)
        h_pos = self.pos_mlp(x_pos)
        g = F.gelu(self.g_proj(x_global))
        g = g.unsqueeze(1).repeat(1, instruments, 1)
        z = torch.cat([h_time, h_pos, g], dim=-1)
        z = self.backbone(z)

        logits = self.head_disc(z)
        mean = torch.tanh(self.head_mean(z)).squeeze(-1)
        value = self.value_head(z.mean(dim=1)).squeeze(-1)
        std = torch.exp(self.log_std) + 1e-6
        return logits, mean, std, value

    def act(self, obs: Dict[str, torch.Tensor]):
        x_time, x_pos, x_global = obs["time"], obs["pos"], obs["global"]
        logits, mean, std, value = self.forward(x_time, x_pos, x_global)
        cat = Categorical(logits=logits)
        disc = cat.sample()
        logp_disc = cat.log_prob(disc).sum(dim=-1)

        base = Normal(mean, std)
        y = base.rsample()
        cont = torch.tanh(y)
        logp_cont = base.log_prob(y) - torch.log(1.0 - cont.pow(2) + 1e-8)
        logp_cont = logp_cont.sum(dim=-1)

        return disc, cont, logp_disc + logp_cont, value

    def evaluate_actions(self, obs: Dict[str, torch.Tensor], disc_actions: torch.Tensor, cont_actions: torch.Tensor):
        x_time, x_pos, x_global = obs["time"], obs["pos"], obs["global"]
        logits, mean, std, value = self.forward(x_time, x_pos, x_global)
        cat = Categorical(logits=logits)
        logp_disc = cat.log_prob(disc_actions).sum(dim=-1)
        entropy_disc = cat.entropy().sum(dim=-1)

        atanh = 0.5 * (torch.log1p(cont_actions + 1e-8) - torch.log1p(-cont_actions + 1e-8))
        base = Normal(mean, std)
        logp_cont = base.log_prob(atanh) - torch.log(1.0 - cont_actions.pow(2) + 1e-8)
        logp_cont = logp_cont.sum(dim=-1)
        entropy_cont = (0.5 * (1.0 + np.log(2 * np.pi)) + torch.log(std)).sum(dim=-1)

        logp = logp_disc + logp_cont
        entropy = entropy_disc + entropy_cont
        return logp, entropy, value


class PPOAgent:
    """Minimal yet extendable PPO agent implementation."""

    def __init__(self, env: MultiAssetTradingEnv, cfg: PPOConfig):
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

    def _to_torch_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {
            "time": torch.tensor(obs["time"], dtype=torch.float32, device=self.device).unsqueeze(0),
            "pos": torch.tensor(obs["pos"], dtype=torch.float32, device=self.device).unsqueeze(0),
            "global": torch.tensor(obs["global"], dtype=torch.float32, device=self.device).unsqueeze(0),
        }

    def _stack_obs(self, obs_batch: Dict[str, List[np.ndarray]]) -> Dict[str, torch.Tensor]:
        x_time = torch.tensor(np.stack(obs_batch["time"], axis=0), dtype=torch.float32, device=self.device)
        x_pos = torch.tensor(np.stack(obs_batch["pos"], axis=0), dtype=torch.float32, device=self.device)
        x_glob = torch.tensor(np.stack(obs_batch["global"], axis=0), dtype=torch.float32, device=self.device)
        return {"time": x_time, "pos": x_pos, "global": x_glob}

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

        B = len(batch["reward"])
        rewards = torch.tensor(batch["reward"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch["done"], dtype=torch.float32, device=self.device)
        values = torch.tensor(batch["value"], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            adv = torch.zeros(B, dtype=torch.float32, device=self.device)
            lastgaelam = 0.0
            next_value = 0.0
            for t in reversed(range(B)):
                nextnonterminal = 1.0 - dones[t]
                delta = rewards[t] + self.cfg.gamma * next_value * nextnonterminal - values[t]
                lastgaelam = delta + self.cfg.gamma * self.cfg.gae_lambda * nextnonterminal * lastgaelam
                adv[t] = lastgaelam
                next_value = values[t]
            returns = adv + values

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs_batch = {"time": batch["obs_time"], "pos": batch["obs_pos"], "global": batch["obs_global"]}
        obs_t = self._stack_obs(obs_batch)
        disc_t = torch.tensor(np.stack(batch["disc"], axis=0), dtype=torch.int64, device=self.device)
        cont_t = torch.tensor(np.stack(batch["cont"], axis=0), dtype=torch.float32, device=self.device)
        logp_t = torch.tensor(batch["logp"], dtype=torch.float32, device=self.device)
        return obs_t, disc_t, cont_t, logp_t, values.detach(), returns.detach(), adv.detach()

    def update(self, obs_t, disc_t, cont_t, logp_old, values_old, returns, advantages) -> None:
        B = returns.shape[0]
        idx = np.arange(B)
        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(idx)
            for start in range(0, B, self.cfg.minibatch_size):
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
                v_clipped = mb_values_old + torch.clamp(value - mb_values_old, -self.cfg.clip_coef, self.cfg.clip_coef)
                v_loss_clipped = (v_clipped - mb_returns).pow(2)
                v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                ent = entropy.mean()
                loss = pg_loss + self.cfg.vf_coef * v_loss - self.cfg.ent_coef * ent

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.opt.step()

    def train(self, total_updates: int = 50) -> None:
        for up in range(1, total_updates + 1):
            obs_t, disc_t, cont_t, logp_old, values_old, returns, adv = self.collect_rollout()
            self.update(obs_t, disc_t, cont_t, logp_old, values_old, returns, adv)
            eq = self.env.info_last.get("equity", float("nan"))
            print(f"Update {up:03d}: last_equity={eq:,.2f}")

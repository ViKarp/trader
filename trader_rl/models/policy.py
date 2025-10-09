"""Neural network policy used by the PPO agent."""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class TimeEncoder(nn.Module):
    """Temporal encoder for per-asset price windows."""

    def __init__(self, f_in: int, d_out: int = 64) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(f_in, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.proj = nn.Linear(64, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, t, f_t = x.shape
        x = x.reshape(b * n, t, f_t).transpose(1, 2)
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.mean(dim=-1)
        x = self.proj(x)
        x = self.norm(x)
        return x.reshape(b, n, -1)


class PolicyNet(nn.Module):
    """Combined actor-critic network for multi-asset trading."""

    def __init__(self, f_time: int, p_in: int, g_in: int, d_model: int = 128) -> None:
        super().__init__()
        self.time_enc = TimeEncoder(f_time, d_out=64)
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
        logits, mean, std, value = self._forward_impl(x_time, x_pos, x_global)
        return logits, mean, std, value

    def _forward_impl(self, x_time, x_pos, x_global):
        b, n, _, _ = x_time.shape
        h_time = self.time_enc(x_time)
        h_pos = self.pos_mlp(x_pos)
        g = F.gelu(self.g_proj(x_global))
        g = g.unsqueeze(1).repeat(1, n, 1)
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

    def evaluate_actions(
        self,
        obs: Dict[str, torch.Tensor],
        disc_actions: torch.Tensor,
        cont_actions: torch.Tensor,
    ):
        x_time, x_pos, x_global = obs["time"], obs["pos"], obs["global"]
        logits, mean, std, value = self.forward(x_time, x_pos, x_global)
        cat = Categorical(logits=logits)
        logp_disc = cat.log_prob(disc_actions).sum(dim=-1)
        entropy_disc = cat.entropy().sum(dim=-1)

        atanh = 0.5 * (torch.log1p(cont_actions + 1e-8) - torch.log1p(-cont_actions + 1e-8))
        base = Normal(mean, std)
        logp_cont = base.log_prob(atanh) - torch.log(1.0 - cont_actions.pow(2) + 1e-8)
        logp_cont = logp_cont.sum(dim=-1)
        entropy_cont = (0.5 * (1.0 + math.log(2 * math.pi)) + torch.log(std)).sum(dim=-1)

        logp = logp_disc + logp_cont
        entropy = entropy_disc + entropy_cont
        return logp, entropy, value

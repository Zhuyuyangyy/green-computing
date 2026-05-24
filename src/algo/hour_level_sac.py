"""
Hour-Level SAC Agent for Carbon-Aware Procurement
对齐 papers/02/ method.tex Section 2.2

论文公式: Eq.(eq:sac_objective)
- SAC自动温度参数 alpha
- 动作空间: [carbon_quota, renewable_ratio] in [0,1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Tuple, Optional


class SACActor(nn.Module):
    """SAC Actor: 输出连续动作的均值和标准差"""

    def __init__(self, latent_dim: int, action_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden // 2, action_dim)
        self.log_std = nn.Linear(hidden // 2, action_dim)

        self.action_dim = action_dim
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def forward(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(latent)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std.exp()

    def sample(self, latent: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, std = self.forward(latent)
        dist = Normal(mu, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)  # squashed
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True)


class SACCritic(nn.Module):
    """Twin Q-network"""

    def __init__(self, latent_dim: int, action_dim: int = 2, hidden: int = 256):
        super().__init__()
        self.q1_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([latent, action], dim=-1)
        return self.q1_net(x), self.q2_net(x)


class HourLevelSAC:
    """
    Hour-level SAC agent for carbon-aware procurement
    对齐 method.tex Section 2.2: Eq.(eq:sac_objective)
    """

    def __init__(
        self,
        latent_dim: int = 32,
        action_dim: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.actor = SACActor(latent_dim, action_dim).to(device)
        self.critic = SACCritic(latent_dim, action_dim).to(device)
        self.critic_target = SACCritic(latent_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # 自动温度参数
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -action_dim

    def update(self, batch: dict) -> dict:
        latent = batch["latent"].to(self.device)
        action = batch["action"].to(self.device)
        reward = batch["reward"].to(self.device)
        next_latent = batch["next_latent"].to(self.device)
        done = batch["done"].to(self.device)

        # ── Critic update ───────────────────────────────────────────
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_latent)
            q1_target, q2_target = self.critic_target(next_latent, next_action)
            q_target = torch.min(q1_target, q2_target)
            v_target = q_target - self.alpha * next_log_prob
            q_backup = reward + self.gamma * (1 - done.float()) * v_target

        q1, q2 = self.critic(latent, action)
        critic_loss = F.mse_loss(q1, q_backup) + F.mse_loss(q2, q_backup)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ── Actor update ────────────────────────────────────────────
        new_action, log_prob = self.actor.sample(latent)
        q1_new, q2_new = self.critic(latent, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ── Alpha (entropy) update ───────────────────────────────────
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = self.log_alpha.exp().item()

        # ── Target network update ────────────────────────────────────
        for mp, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.mul_(1 - self.tau)
            tp.data.add_(self.tau * mp.data)

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "alpha": self.alpha,
        }

    @torch.no_grad()
    def select_action(self, latent: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        self.actor.eval()
        latent = latent.to(self.device)
        if deterministic:
            mu, _ = self.actor(latent)
            action = torch.tanh(mu)
        else:
            action, _ = self.actor.sample(latent)
        return action.cpu().numpy()

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha = ckpt["log_alpha"]

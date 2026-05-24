"""
Minute-Level TD3 Agent for Load Balancing
对齐 papers/02/ method.tex Section 2.3

论文公式: Eq.(eq:td3_critic)
- Twin Delayed DDPG (TD3) 用于跨节点负载迁移
- 动作: n_gpus 维迁移比例向量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class TD3Actor(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, action_dim),
            nn.Tanh(),  # 归一化到 [-1, 1]
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class TD3Critic(nn.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        # Q1
        self.q1_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # Q2
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


class MinuteLevelTD3:
    """
    Minute-level TD3 agent for cross-node workload migration
    对齐 method.tex Section 2.3: Eq.(eq:td3_critic)
    """

    def __init__(
        self,
        latent_dim: int = 32,
        action_dim: int = 8,  # n_gpus
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.device = device
        self.total_it = 0

        self.actor = TD3Actor(latent_dim, action_dim).to(device)
        self.actor_target = TD3Actor(latent_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = TD3Critic(latent_dim, action_dim).to(device)
        self.critic_target = TD3Critic(latent_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def update(self, batch: dict) -> dict:
        self.total_it += 1
        latent = batch["latent"].to(self.device)
        action = batch["action"].to(self.device)
        reward = batch["reward"].to(self.device)
        next_latent = batch["next_latent"].to(self.device)
        done = batch["done"].to(self.device)

        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_latent) + noise).clamp(-1, 1)

            # TD3: min of two target Q values
            q1_target, q2_target = self.critic_target(next_latent, next_action)
            q_target = torch.min(q1_target, q2_target)
            q_backup = reward + self.gamma * (1 - done.float()) * q_target

        # Critic update
        q1, q2 = self.critic(latent, action)
        critic_loss = F.mse_loss(q1, q_backup) + F.mse_loss(q2, q_backup)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor update (delayed)
        actor_loss = None
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic.q1_net(torch.cat([latent, self.actor(latent)], dim=-1)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Target update
            for mp, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.mul_(1 - self.tau)
                tp.data.add_(self.tau * mp.data)

            for mp, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.mul_(1 - self.tau)
                tp.data.add_(self.tau * mp.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item() if actor_loss else 0.0,
        }

    @torch.no_grad()
    def select_action(self, latent: torch.Tensor, deterministic: bool = True) -> np.ndarray:
        self.actor.eval()
        action = self.actor(latent.to(self.device)).cpu().numpy()
        return action

    def save(self, path: str):
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])

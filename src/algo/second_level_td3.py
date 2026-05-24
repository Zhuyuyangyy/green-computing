"""
Second-Level TD3 Agent for GPU Frequency and Cooling Control
对齐 papers/02/ method.tex Section 2.4

动作空间: [freq_1..freq_N, flow_1..flow_N] 每个GPU的频率+流速
每秒执行一次操作

论文公式: Section 2.4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class SecondLevelTD3(nn.Module):
    """Second-level TD3: 控制GPU频率+冷却流速"""

    def __init__(self, latent_dim: int, n_gpus: int, hidden: int = 256):
        super().__init__()
        self.n_gpus = n_gpus
        self.action_dim = n_gpus * 2  # freq + flow per GPU

        self.actor = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, self.action_dim),
            nn.Sigmoid(),  # [0, 1] 输出
        )

        self.actor_target = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, self.action_dim),
            nn.Sigmoid(),
        )
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_q1 = nn.Sequential(
            nn.Linear(latent_dim + self.action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.critic_q2 = nn.Sequential(
            nn.Linear(latent_dim + self.action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.critic_target_q1 = nn.Sequential(
            nn.Linear(latent_dim + self.action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.critic_target_q2 = nn.Sequential(
            nn.Linear(latent_dim + self.action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.critic_target_q1.load_state_dict(self.critic_q1.state_dict())
        self.critic_target_q2.load_state_dict(self.critic_q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_opt = torch.optim.Adam(
            list(self.critic_q1.parameters()) + list(self.critic_q2.parameters()), lr=3e-4
        )

    def forward(self, latent, action):
        x = torch.cat([latent, action], dim=-1)
        return self.critic_q1(x), self.critic_q2(x)

    def update(self, batch, gamma=0.99, tau=0.005, policy_noise=0.1, noise_clip=0.3, policy_delay=2):
        self.total_it = getattr(self, "total_it", 0) + 1

        device = next(self.critic_q1.parameters()).device
        latent = batch["latent"].to(device).float()
        action = batch["action"].to(device).float()
        reward = batch["reward"].to(device).float()
        next_latent = batch["next_latent"].to(device).float()
        done = batch["done"].to(device).float()

        with torch.no_grad():
            noise = (torch.randn_like(action) * policy_noise).clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_latent) + noise).clamp(0, 1)
            q1_t, q2_t = self.critic_target_q1(torch.cat([next_latent, next_action], -1)), \
                          self.critic_target_q2(torch.cat([next_latent, next_action], -1))
            q_target = torch.min(q1_t, q2_t)
            # Ensure proper [B, 1] shape for broadcasting with q_target
            reward_t = reward.unsqueeze(1) if reward.dim() == 1 else reward
            done_t = done.unsqueeze(1) if done.dim() == 1 else done
            q_backup = reward_t + gamma * (1 - done_t.float()) * q_target

        q1, q2 = self.critic_q1(torch.cat([latent, action], -1)), \
                 self.critic_q2(torch.cat([latent, action], -1))
        critic_loss = F.mse_loss(q1, q_backup) + F.mse_loss(q2, q_backup)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = None
        if self.total_it % policy_delay == 0:
            actor_loss = -self.critic_q1(torch.cat([latent, self.actor(latent)], -1)).mean()
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            for mp, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.mul_(1 - tau)
                tp.data.add_(tau * mp.data)
            for mp, tp in zip(self.critic_q1.parameters(), self.critic_target_q1.parameters()):
                tp.data.mul_(1 - tau)
                tp.data.add_(tau * mp.data)
            for mp, tp in zip(self.critic_q2.parameters(), self.critic_target_q2.parameters()):
                tp.data.mul_(1 - tau)
                tp.data.add_(tau * mp.data)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item() if actor_loss else 0.0}

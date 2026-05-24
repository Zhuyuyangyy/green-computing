# -*- coding: utf-8 -*-
"""
Hierarchical RL Trainer for Data Center Energy Optimization
三层HRL协调器: Hour(SAC) -> Minute(TD3) -> Second(TD3)

Paper formulas:
- Hour: method.tex Eq.(eq:sac_objective)
- Minute: method.tex Eq.(eq:td3_critic)
- Second: method.tex Section 2.4
- PINN: method.tex Eq.(eq:pinn_loss), Eq.(eq:total_loss)
"""

import torch
import numpy as np
from collections import deque
import time
from src.env.datacenter_env import DataCenterEnv
from src.algo.lstm_encoder import LSTMEncoder
from src.algo.hour_level_sac import HourLevelSAC
from src.algo.minute_level_td3 import MinuteLevelTD3
from src.algo.second_level_td3 import SecondLevelTD3
from src.algo.pinn_thermal import PINNLossCalculator, PINNThermalModel


class HierarchicalRLTrainer:
    """
    Three-level HRL training coordinator
    Hour(SAC) -> Minute(TD3) -> Second(TD3)
    """

    def __init__(
        self,
        n_gpus: int = 8,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.device = device
        self.n_gpus = n_gpus

        # Environment
        self.env = DataCenterEnv(n_gpus=n_gpus, seed=seed)
        obs_dim = self.env.observation_dim  # 4N + 4

        # Shared LSTM encoder
        self.encoder = LSTMEncoder(
            obs_dim=obs_dim,
            hidden_dims=(256, 128, 64),
            latent_dim=32,
        ).to(device)

        # Hour-level SAC
        self.hour_agent = HourLevelSAC(
            latent_dim=32,
            action_dim=2,  # carbon_quota, renewable_ratio
            lr=3e-4,
            device=device,
        )

        # Minute-level TD3
        self.minute_agent = MinuteLevelTD3(
            latent_dim=32,
            action_dim=n_gpus,
            lr=3e-4,
            device=device,
        )

        # Second-level TD3
        self.second_agent = SecondLevelTD3(
            latent_dim=32,
            n_gpus=n_gpus,
        ).to(device)

        # PINN thermal model
        self.pinn = PINNThermalModel(n_gpus=n_gpus).to(device)
        self.pinn_calculator = PINNLossCalculator(self.pinn, beta=0.1)

        # Experience buffers
        self.hour_buffer = deque(maxlen=10000)
        self.minute_buffer = deque(maxlen=10000)
        self.second_buffer = deque(maxlen=50000)

        self.stats = {"hour_updates": 0, "minute_updates": 0, "second_updates": 0}

    def encode(self, obs: np.ndarray) -> torch.Tensor:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        z, _ = self.encoder(obs_t)
        return z

    def run_episode(self, max_steps: int = 3600, render: bool = False):
        """
        Run one complete episode (1 hour = 3600 seconds).
        Internal loop: second-level (1s) -> minute-level (60s) -> hour-level (3600s)
        """
        obs = self.env.reset()
        episode_stats = {
            "total_power": 0.0,
            "total_carbon": 0.0,
            "thermal_violations": 0,
            "pue_avg": 0.0,
            "steps": 0,
        }

        for step in range(max_steps):
            # Second-level: per-second control
            obs_t = self.encode(obs)
            second_action = self.second_agent.actor(obs_t).detach().cpu().numpy().flatten()

            # Split frequency and flow actions
            freq_action = second_action[:self.n_gpus]
            flow_action = second_action[self.n_gpus:]

            obs, reward, done, info = self.env.step_second(freq_action, flow_action)

            # Store second-level experience
            self.second_buffer.append({
                "obs": obs.astype(np.float32) if hasattr(obs, 'astype') else np.array(obs, dtype=np.float32),
                "action": np.array(second_action, dtype=np.float32),
                "reward": float(reward),
                "done": bool(done),
            })

            episode_stats["total_power"] += info.get("p_total", 0)
            episode_stats["total_carbon"] += info.get("carbon_rate", 0)
            episode_stats["pue_avg"] += info.get("pue", 1.0)
            if info.get("thermal_violation"):
                episode_stats["thermal_violations"] += 1
            episode_stats["steps"] += 1

            # Minute-level: update every 60 seconds
            if step > 0 and step % 60 == 0:
                self._update_minute()
                self.env.apply_environmental_disturbance()

            # Hour-level: update every 3600 seconds
            if step > 0 and step % 3600 == 0:
                self._update_hour()

            if done:
                break

        episode_stats["pue_avg"] /= max(episode_stats["steps"], 1)
        return episode_stats

    def _update_second(self, batch_size: int = 256):
        if len(self.second_buffer) < batch_size:
            return {}
        batch_idx = np.random.choice(len(self.second_buffer), batch_size, replace=False)
        batch = {
            k: torch.stack([
                torch.tensor(self.second_buffer[i][k], dtype=torch.float32)
                for i in batch_idx
            ])
            for k in ["obs", "action"]
        }
        for k in ["reward", "done"]:
            batch[k] = torch.tensor(
                [self.second_buffer[i][k] for i in batch_idx],
                dtype=torch.float32
            )

        with torch.no_grad():
            obs_t = batch["obs"].to(self.device)
            z, _ = self.encoder(obs_t)
            z_next, _ = self.encoder(obs_t)  # placeholder
            batch["latent"] = z
            batch["next_latent"] = z_next

        stats = self.second_agent.update(batch)
        self.stats["second_updates"] += 1
        return stats

    def _update_minute(self):
        if len(self.minute_buffer) < 64:
            return {}
        batch_idx = np.random.choice(len(self.minute_buffer), 64, replace=False)
        batch = {
            k: torch.stack([
                torch.tensor(self.minute_buffer[i][k], dtype=torch.float32)
                for i in batch_idx
            ])
            for k in ["action"]
        }
        for k in ["reward", "done"]:
            batch[k] = torch.tensor(
                [self.minute_buffer[i][k] for i in batch_idx],
                dtype=torch.float32
            )
        obs = torch.stack([
            torch.tensor(self.minute_buffer[i]["obs"], dtype=torch.float32)
            for i in batch_idx
        ])
        z, _ = self.encoder(obs.to(self.device))
        batch["latent"] = z
        batch["next_latent"] = z

        stats = self.minute_agent.update(batch)
        self.stats["minute_updates"] += 1
        return stats

    def _update_hour(self):
        if len(self.hour_buffer) < 32:
            return {}
        batch_idx = np.random.choice(len(self.hour_buffer), 32, replace=False)
        batch = {}
        for k in ["latent", "action", "reward", "next_latent"]:
            vals = []
            for i in batch_idx:
                if k in self.hour_buffer[i]:
                    v = self.hour_buffer[i][k]
                    if not isinstance(v, torch.Tensor):
                        v = torch.tensor(v, dtype=torch.float32)
                    vals.append(v)
            if vals:
                batch[k] = torch.stack(vals)
        batch["done"] = torch.zeros(len(batch_idx), dtype=torch.float32)
        if "latent" in batch:
            stats = self.hour_agent.update(batch)
            self.stats["hour_updates"] += 1
            return stats
        return {}

    def train(self, n_episodes: int = 100, save_dir: str = "outputs"):
        """Main training loop."""
        import os
        os.makedirs(save_dir, exist_ok=True)

        for ep in range(n_episodes):
            t0 = time.time()
            stats = self.run_episode()
            second_stats = self._update_second()

            elapsed = time.time() - t0

            print(f"[Ep {ep+1}/{n_episodes}] "
                  f"Power={stats.get('total_power', 0):.1f}kW "
                  f"Carbon={stats.get('total_carbon', 0):.2f}g "
                  f"PUE={stats.get('pue_avg', 0):.3f} "
                  f"Violations={stats.get('thermal_violations', 0)} "
                  f"t={elapsed:.1f}s "
                  f"Updates:H={self.stats['hour_updates']}M={self.stats['minute_updates']}S={self.stats['second_updates']}")

            if (ep + 1) % 10 == 0:
                self.save(f"{save_dir}/checkpoint_ep{ep+1}.pt")

    def save(self, path: str):
        torch.save({
            "encoder": self.encoder.state_dict(),
            "hour_agent": self.hour_agent.actor.state_dict(),
            "minute_agent": self.minute_agent.actor.state_dict(),
            "second_agent": self.second_agent.state_dict(),
            "pinn": self.pinn.state_dict(),
            "stats": self.stats,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.pinn.load_state_dict(ckpt["pinn"])
        self.stats = ckpt.get("stats", self.stats)


if __name__ == "__main__":
    trainer = HierarchicalRLTrainer(n_gpus=8, device="cpu", seed=42)
    trainer.train(n_episodes=50, save_dir="outputs")

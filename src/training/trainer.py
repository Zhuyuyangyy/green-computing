# -*- coding: utf-8 -*-
"""
Hierarchical RL Trainer for Data Center Energy Optimization
三层HRL协调器: Hour(SAC) -> Minute(TD3) -> Second(TD3)

Paper formulas:
- Hour: method.tex Eq.(eq:sac_objective)
- Minute: method.tex Eq.(eq:td3_critic)
- Second: method.tex Section 2.4
- PINN: method.tex Eq.(eq:pinn_loss), Eq.(eq:total_loss)

DISCLAIMER: All training runs against a SIMPLIFIED SIMULATION environment.
Reported metrics (PUE, carbon emissions, power) are SYNTHETIC — they
originate from the simulated data center, NOT from real hardware.  The
trained policy has NOT been validated on actual data center infrastructure.
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
        self.pinn_optimizer = torch.optim.Adam(self.pinn.parameters(), lr=3e-4)

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

        All three replay buffers are populated during the episode:
        - second_buffer: every 1s step (with next_obs for proper next-state latent)
        - minute_buffer: every 60s boundary (latent from encoder)
        - hour_buffer: at episode end (latent from encoder)
        """
        obs = self.env.reset()
        prev_obs = None
        minute_start_step = 0
        hour_start_step = 0
        minute_reward = 0.0
        hour_reward = 0.0
        episode_stats = {
            "total_power": 0.0,
            "total_carbon": 0.0,
            "thermal_violations": 0,
            "pue_avg": 0.0,
            "steps": 0,
        }

        for step in range(max_steps):
            # Second-level: per-second control
            cur_obs = np.array(obs, dtype=np.float32)
            obs_t = self.encode(cur_obs)
            second_action = self.second_agent.actor(obs_t).detach().cpu().numpy().flatten()

            # Split frequency and flow actions
            freq_action = second_action[:self.n_gpus]
            flow_action = second_action[self.n_gpus:]

            obs, reward, done, info = self.env.step_second(freq_action, flow_action)
            next_obs = np.array(obs, dtype=np.float32)

            # Accumulate hierarchical rewards
            minute_reward += float(reward)
            hour_reward += float(reward)

            # Store second-level experience (with next_obs for proper TD target)
            self.second_buffer.append({
                "obs": cur_obs,
                "next_obs": next_obs,
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

            # Minute-level: store transition and update every 60 seconds
            if step > 0 and step % 60 == 0:
                z_start = self.encode(
                    np.array(self.env.get_observation(), dtype=np.float32)
                ).detach().cpu().numpy().flatten()
                z_now = obs_t.detach().cpu().numpy().flatten()

                # Get minute-level action from agent
                minute_action = self.minute_agent.actor(
                    torch.FloatTensor(z_start).unsqueeze(0).to(self.device)
                ).detach().cpu().numpy().flatten()

                self.minute_buffer.append({
                    "latent": z_start,
                    "next_latent": z_now,
                    "action": minute_action.astype(np.float32),
                    "reward": minute_reward,
                    "done": bool(done),
                })

                self._update_minute()
                self.env.apply_environmental_disturbance()
                minute_reward = 0.0

            # Hour-level: store transition every 3600 seconds
            if step > 0 and step % 3600 == 0:
                z_hour_start = self.encode(
                    np.array(self.env.get_observation(), dtype=np.float32)
                ).detach().cpu().numpy().flatten()
                z_hour_now = obs_t.detach().cpu().numpy().flatten()

                hour_action = self.hour_agent.actor(
                    torch.FloatTensor(z_hour_start).unsqueeze(0).to(self.device)
                )[0].detach().cpu().numpy().flatten()

                self.hour_buffer.append({
                    "latent": z_hour_start,
                    "next_latent": z_hour_now,
                    "action": hour_action.astype(np.float32),
                    "reward": hour_reward,
                    "done": bool(done),
                })

                self._update_hour()
                hour_reward = 0.0

            if done:
                break

        # Store final hour-level transition at episode end (without resetting env)
        if episode_stats["steps"] > 0:
            z_final = obs_t.detach().cpu().numpy().flatten()
            hour_action = self.hour_agent.actor(
                torch.FloatTensor(z_final).unsqueeze(0).to(self.device)
            )[0].detach().cpu().numpy().flatten()
            self.hour_buffer.append({
                "latent": z_final,
                "next_latent": z_final,  # terminal state
                "action": hour_action.astype(np.float32),
                "reward": hour_reward,
                "done": True,
            })

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
            for k in ["obs", "next_obs", "action"]
        }
        for k in ["reward", "done"]:
            batch[k] = torch.tensor(
                [self.second_buffer[i][k] for i in batch_idx],
                dtype=torch.float32
            )

        obs_t = batch["obs"].to(self.device)
        next_obs_t = batch["next_obs"].to(self.device)
        z, _ = self.encoder(obs_t)
        # FIX: compute next-state latent from ACTUAL next observation
        z_next, _ = self.encoder(next_obs_t)
        batch["latent"] = z
        batch["next_latent"] = z_next

        # ── PINN thermal loss integration (method.tex Eq.(eq:total_loss)) ──
        # L_total = L_RL + beta * L_PINN
        # Extract junction temperatures from observations (positions n_gpus:2*n_gpus)
        n = self.n_gpus
        t_j_current = obs_t[:, n:2 * n]           # current junction temps
        t_j_next = next_obs_t[:, n:2 * n]         # next junction temps (ground truth)
        p_gpu = batch["action"][:, :n]             # GPU frequency proxy

        # PINN loss: physics-consistency penalty on temperature transitions
        # Passed to agent.update() for joint backward with critic loss
        pinn_loss = self.pinn.pinn_loss(t_j_next, t_j_current, p_gpu)

        stats = self.second_agent.update(
            batch,
            pinn_loss=pinn_loss,
            pinn_optimizer=self.pinn_optimizer,
        )
        self.stats["second_updates"] += 1
        return stats

    def _update_minute(self):
        if len(self.minute_buffer) < 64:
            return {}
        batch_idx = np.random.choice(len(self.minute_buffer), 64, replace=False)

        # Buffer now stores latent and next_latent directly (pre-computed
        # at minute boundaries in run_episode), so no encoder call needed.
        batch = {}
        for k in ["latent", "next_latent", "action"]:
            batch[k] = torch.stack([
                torch.tensor(self.minute_buffer[i][k], dtype=torch.float32)
                for i in batch_idx
            ])
        for k in ["reward", "done"]:
            batch[k] = torch.tensor(
                [self.minute_buffer[i][k] for i in batch_idx],
                dtype=torch.float32
            )

        stats = self.minute_agent.update(batch)
        self.stats["minute_updates"] += 1
        return stats

    def _update_hour(self):
        if len(self.hour_buffer) < 32:
            return {}
        batch_idx = np.random.choice(len(self.hour_buffer), 32, replace=False)
        batch = {}
        for k in ["latent", "next_latent", "action"]:
            batch[k] = torch.stack([
                torch.tensor(self.hour_buffer[i][k], dtype=torch.float32)
                for i in batch_idx
            ])
        for k in ["reward", "done"]:
            batch[k] = torch.tensor(
                [self.hour_buffer[i][k] for i in batch_idx],
                dtype=torch.float32
            )
        stats = self.hour_agent.update(batch)
        self.stats["hour_updates"] += 1
        return stats

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

    def train_epoch(self, epoch: int = 0):
        """Single-epoch training step for main.py interface.

        DISCLAIMER: All results are synthetic. This trains on a simulated
        data center environment with simplified physics models.  No real
        hardware is controlled and no real energy savings are achieved.
        """
        t0 = time.time()
        stats = self.run_episode()
        self._update_second()
        elapsed = time.time() - t0

        return {
            "epoch": epoch,
            "pue": stats.get("pue_avg", 1.0),
            "carbon_kg": stats.get("total_carbon", 0.0) / 1000.0,
            "total_power": stats.get("total_power", 0.0),
            "thermal_violations": stats.get("thermal_violations", 0),
            "elapsed": elapsed,
            "hour_updates": self.stats["hour_updates"],
            "minute_updates": self.stats["minute_updates"],
            "second_updates": self.stats["second_updates"],
        }

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

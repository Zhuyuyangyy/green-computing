"""Tests for MinuteLevelTD3 - TD3 agent for load balancing."""
import pytest
import torch
import numpy as np
from src.algo.minute_level_td3 import MinuteLevelTD3, TD3Actor, TD3Critic


class TestTD3Actor:
    def test_forward_output_shape(self):
        actor = TD3Actor(latent_dim=32, action_dim=8)
        latent = torch.randn(4, 32)
        out = actor(latent)
        assert out.shape == (4, 8)

    def test_output_range_tanh(self):
        actor = TD3Actor(latent_dim=32, action_dim=8)
        latent = torch.randn(100, 32)
        out = actor(latent)
        assert torch.all(out >= -1.0)
        assert torch.all(out <= 1.0)


class TestTD3Critic:
    def test_forward_output_shape(self):
        critic = TD3Critic(latent_dim=32, action_dim=8)
        latent = torch.randn(4, 32)
        action = torch.randn(4, 8)
        q1, q2 = critic(latent, action)
        assert q1.shape == (4, 1)
        assert q2.shape == (4, 1)


class TestMinuteLevelTD3:
    def test_init(self, latent_dim, device):
        agent = MinuteLevelTD3(latent_dim=latent_dim, action_dim=8, device=device)
        assert agent.gamma == 0.99

    def test_select_action(self, latent_dim, device, random_latent_batch):
        agent = MinuteLevelTD3(latent_dim=latent_dim, action_dim=8, device=device)
        action = agent.select_action(random_latent_batch)
        assert isinstance(action, np.ndarray)

    def test_update_returns_dict(self, latent_dim, device):
        agent = MinuteLevelTD3(latent_dim=latent_dim, action_dim=8, device=device)
        batch = {
            "latent": torch.randn(16, latent_dim),
            "action": torch.randn(16, 8),
            "reward": torch.randn(16),
            "next_latent": torch.randn(16, latent_dim),
            "done": torch.zeros(16),
        }
        result = agent.update(batch)
        assert "critic_loss" in result
        assert "actor_loss" in result

    def test_save_load(self, latent_dim, device, tmp_path):
        agent = MinuteLevelTD3(latent_dim=latent_dim, action_dim=8, device=device)
        path = str(tmp_path / "td3_test.pt")
        agent.save(path)
        agent2 = MinuteLevelTD3(latent_dim=latent_dim, action_dim=8, device=device)
        agent2.load(path)

    def test_delayed_policy_update(self, latent_dim, device):
        agent = MinuteLevelTD3(latent_dim=latent_dim, action_dim=8, device=device, policy_delay=2)
        batch = {
            "latent": torch.randn(16, latent_dim),
            "action": torch.randn(16, 8),
            "reward": torch.randn(16),
            "next_latent": torch.randn(16, latent_dim),
            "done": torch.zeros(16),
        }
        r1 = agent.update(batch)
        assert r1["actor_loss"] == 0.0  # not updated on odd step
        r2 = agent.update(batch)
        assert r2["actor_loss"] != 0.0  # updated on even step

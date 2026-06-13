"""Tests for HourLevelSAC - SAC agent for carbon-aware procurement."""
import pytest
import torch
import numpy as np
from src.algo.hour_level_sac import HourLevelSAC, SACActor, SACCritic


class TestSACActor:
    def test_init(self):
        actor = SACActor(latent_dim=32, action_dim=2)
        assert actor.action_dim == 2

    def test_forward_output_shape(self):
        actor = SACActor(latent_dim=32, action_dim=2)
        latent = torch.randn(4, 32)
        mu, std = actor(latent)
        assert mu.shape == (4, 2)
        assert std.shape == (4, 2)

    def test_std_positive(self):
        actor = SACActor(latent_dim=32, action_dim=2)
        latent = torch.randn(4, 32)
        _, std = actor(latent)
        assert torch.all(std > 0)

    def test_sample_output_shape(self):
        actor = SACActor(latent_dim=32, action_dim=2)
        latent = torch.randn(4, 32)
        action, log_prob = actor.sample(latent)
        assert action.shape == (4, 2)
        assert log_prob.shape == (4, 1)

    def test_sample_action_range(self):
        actor = SACActor(latent_dim=32, action_dim=2)
        latent = torch.randn(100, 32)
        action, _ = actor.sample(latent)
        assert torch.all(action >= -1.0)
        assert torch.all(action <= 1.0)


class TestSACCritic:
    def test_forward_output_shape(self):
        critic = SACCritic(latent_dim=32, action_dim=2)
        latent = torch.randn(4, 32)
        action = torch.randn(4, 2)
        q1, q2 = critic(latent, action)
        assert q1.shape == (4, 1)
        assert q2.shape == (4, 1)


class TestHourLevelSAC:
    def test_init(self, latent_dim, device):
        agent = HourLevelSAC(latent_dim=latent_dim, device=device)
        assert agent.gamma == 0.99
        assert agent.tau == 0.005

    def test_select_action_deterministic(self, latent_dim, device, random_latent):
        agent = HourLevelSAC(latent_dim=latent_dim, device=device)
        action = agent.select_action(random_latent, deterministic=True)
        assert isinstance(action, np.ndarray)
        assert action.shape == (1, 2)

    def test_select_action_stochastic(self, latent_dim, device, random_latent):
        agent = HourLevelSAC(latent_dim=latent_dim, device=device)
        action = agent.select_action(random_latent, deterministic=False)
        assert isinstance(action, np.ndarray)
        assert action.shape == (1, 2)

    def test_update_returns_dict(self, latent_dim, device):
        agent = HourLevelSAC(latent_dim=latent_dim, device=device)
        batch = {
            "latent": torch.randn(8, latent_dim),
            "action": torch.randn(8, 2),
            "reward": torch.randn(8),
            "next_latent": torch.randn(8, latent_dim),
            "done": torch.zeros(8),
        }
        result = agent.update(batch)
        assert "actor_loss" in result
        assert "critic_loss" in result
        assert "alpha" in result

    def test_save_load(self, latent_dim, device, tmp_path):
        agent = HourLevelSAC(latent_dim=latent_dim, device=device)
        path = str(tmp_path / "sac_test.pt")
        agent.save(path)
        agent2 = HourLevelSAC(latent_dim=latent_dim, device=device)
        agent2.load(path)
        # Verify loaded model produces same output
        latent = torch.randn(1, latent_dim)
        a1 = agent.select_action(latent, deterministic=True)
        a2 = agent2.select_action(latent, deterministic=True)
        np.testing.assert_array_almost_equal(a1, a2)

"""Tests for SecondLevelTD3 - TD3 agent for GPU frequency and cooling control."""
import pytest
import torch
import numpy as np
from src.algo.second_level_td3 import SecondLevelTD3


class TestSecondLevelTD3:
    def test_init(self, latent_dim, n_gpus):
        agent = SecondLevelTD3(latent_dim=latent_dim, n_gpus=n_gpus)
        assert agent.n_gpus == n_gpus
        assert agent.action_dim == n_gpus * 2

    def test_forward_output_shape(self, latent_dim, n_gpus):
        agent = SecondLevelTD3(latent_dim=latent_dim, n_gpus=n_gpus)
        latent = torch.randn(4, latent_dim)
        action = torch.randn(4, n_gpus * 2)
        q1, q2 = agent(latent, action)
        assert q1.shape == (4, 1)
        assert q2.shape == (4, 1)

    def test_actor_output_shape(self, latent_dim, n_gpus):
        agent = SecondLevelTD3(latent_dim=latent_dim, n_gpus=n_gpus)
        latent = torch.randn(4, latent_dim)
        out = agent.actor(latent)
        assert out.shape == (4, n_gpus * 2)

    def test_actor_output_range(self, latent_dim, n_gpus):
        agent = SecondLevelTD3(latent_dim=latent_dim, n_gpus=n_gpus)
        latent = torch.randn(100, latent_dim)
        out = agent.actor(latent)
        assert torch.all(out >= 0.0)
        assert torch.all(out <= 1.0)

    def test_update_returns_dict(self, latent_dim, n_gpus):
        agent = SecondLevelTD3(latent_dim=latent_dim, n_gpus=n_gpus)
        batch = {
            "latent": torch.randn(16, latent_dim),
            "action": torch.randn(16, n_gpus * 2),
            "reward": torch.randn(16),
            "next_latent": torch.randn(16, latent_dim),
            "done": torch.zeros(16),
        }
        result = agent.update(batch)
        assert "critic_loss" in result
        assert "actor_loss" in result

    def test_state_dict(self, latent_dim, n_gpus):
        agent = SecondLevelTD3(latent_dim=latent_dim, n_gpus=n_gpus)
        sd = agent.state_dict()
        assert isinstance(sd, dict)
        assert len(sd) > 0

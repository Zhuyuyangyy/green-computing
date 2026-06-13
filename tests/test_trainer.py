"""Tests for HierarchicalRLTrainer - HRL training coordinator."""
import pytest
import torch
import numpy as np
from src.training.trainer import HierarchicalRLTrainer


class TestTrainerInit:
    def test_init(self, n_gpus, device, seed):
        trainer = HierarchicalRLTrainer(n_gpus=n_gpus, device=device, seed=seed)
        assert trainer.n_gpus == n_gpus
        assert trainer.device == device

    def test_encoder_created(self, n_gpus, device, seed):
        trainer = HierarchicalRLTrainer(n_gpus=n_gpus, device=device, seed=seed)
        assert trainer.encoder is not None

    def test_agents_created(self, n_gpus, device, seed):
        trainer = HierarchicalRLTrainer(n_gpus=n_gpus, device=device, seed=seed)
        assert trainer.hour_agent is not None
        assert trainer.minute_agent is not None
        assert trainer.second_agent is not None

    def test_pinn_created(self, n_gpus, device, seed):
        trainer = HierarchicalRLTrainer(n_gpus=n_gpus, device=device, seed=seed)
        assert trainer.pinn is not None
        assert trainer.pinn_calculator is not None

    def test_buffers_initialized(self, n_gpus, device, seed):
        trainer = HierarchicalRLTrainer(n_gpus=n_gpus, device=device, seed=seed)
        assert len(trainer.hour_buffer) == 0
        assert len(trainer.minute_buffer) == 0
        assert len(trainer.second_buffer) == 0


class TestTrainerEncode:
    def test_encode_returns_tensor(self, n_gpus, device, seed, random_obs):
        trainer = HierarchicalRLTrainer(n_gpus=n_gpus, device=device, seed=seed)
        z = trainer.encode(random_obs)
        assert isinstance(z, torch.Tensor)
        assert z.shape == (1, 32)

    def test_encode_output_dim(self, n_gpus, device, seed, random_obs):
        trainer = HierarchicalRLTrainer(n_gpus=n_gpus, device=device, seed=seed)
        z = trainer.encode(random_obs)
        assert z.shape[-1] == 32


class TestTrainerEpisode:
    def test_run_episode_returns_stats(self, n_gpus, device, seed):
        trainer = HierarchicalRLTrainer(n_gpus=n_gpus, device=device, seed=seed)
        stats = trainer.run_episode(max_steps=10)
        assert "total_power" in stats
        assert "total_carbon" in stats
        assert "thermal_violations" in stats
        assert "pue_avg" in stats
        assert "steps" in stats

    def test_run_episode_steps_count(self, n_gpus, device, seed):
        trainer = HierarchicalRLTrainer(n_gpus=n_gpus, device=device, seed=seed)
        stats = trainer.run_episode(max_steps=50)
        assert stats["steps"] == 50

    def test_run_episode_fills_buffer(self, n_gpus, device, seed):
        trainer = HierarchicalRLTrainer(n_gpus=n_gpus, device=device, seed=seed)
        trainer.run_episode(max_steps=10)
        assert len(trainer.second_buffer) == 10


class TestTrainerUpdate:
    def test_update_second_insufficient_buffer(self, n_gpus, device, seed):
        trainer = HierarchicalRLTrainer(n_gpus=n_gpus, device=device, seed=seed)
        result = trainer._update_second(batch_size=256)
        assert result == {}

    def test_update_minute_insufficient_buffer(self, n_gpus, device, seed):
        trainer = HierarchicalRLTrainer(n_gpus=n_gpus, device=device, seed=seed)
        result = trainer._update_minute()
        assert result == {}

    def test_update_hour_insufficient_buffer(self, n_gpus, device, seed):
        trainer = HierarchicalRLTrainer(n_gpus=n_gpus, device=device, seed=seed)
        result = trainer._update_hour()
        assert result == {}

    def test_update_second_with_data(self, n_gpus, device, seed):
        trainer = HierarchicalRLTrainer(n_gpus=n_gpus, device=device, seed=seed)
        trainer.run_episode(max_steps=300)
        result = trainer._update_second(batch_size=64)
        assert "critic_loss" in result

    def test_update_minute_with_data(self, n_gpus, device, seed):
        trainer = HierarchicalRLTrainer(n_gpus=n_gpus, device=device, seed=seed)
        latent_dim = 32
        for _ in range(100):
            trainer.minute_buffer.append({
                "latent": np.random.randn(latent_dim).astype(np.float32),
                "next_latent": np.random.randn(latent_dim).astype(np.float32),
                "action": np.random.randn(n_gpus).astype(np.float32),
                "reward": 0.0,
                "done": False,
            })
        result = trainer._update_minute()
        assert "critic_loss" in result


class TestTrainerSaveLoad:
    def test_save(self, n_gpus, device, seed, tmp_path):
        trainer = HierarchicalRLTrainer(n_gpus=n_gpus, device=device, seed=seed)
        path = str(tmp_path / "test_save.pt")
        trainer.save(path)
        import os
        assert os.path.exists(path)

    def test_load(self, n_gpus, device, seed, tmp_path):
        trainer = HierarchicalRLTrainer(n_gpus=n_gpus, device=device, seed=seed)
        path = str(tmp_path / "test_load.pt")
        trainer.save(path)
        trainer2 = HierarchicalRLTrainer(n_gpus=n_gpus, device=device, seed=seed)
        trainer2.load(path)

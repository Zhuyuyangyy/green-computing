"""Shared test fixtures for green-computing test suite."""
import sys
import os
import pytest
import numpy as np
import torch

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def n_gpus():
    return 4


@pytest.fixture
def seed():
    return 42


@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def latent_dim():
    return 32


@pytest.fixture
def obs_dim(n_gpus):
    return 4 * n_gpus + 4


@pytest.fixture
def random_obs(obs_dim):
    return np.random.randn(obs_dim).astype(np.float32)


@pytest.fixture
def random_obs_batch(obs_dim):
    return np.random.randn(8, obs_dim).astype(np.float32)


@pytest.fixture
def random_latent(latent_dim):
    return torch.randn(1, latent_dim)


@pytest.fixture
def random_latent_batch(latent_dim):
    return torch.randn(16, latent_dim)

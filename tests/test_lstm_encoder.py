"""Tests for LSTMEncoder - shared temporal feature extractor."""
import pytest
import torch
from src.algo.lstm_encoder import LSTMEncoder


class TestLSTMEncoderInit:
    def test_default_init(self):
        encoder = LSTMEncoder(obs_dim=36)
        assert encoder.obs_dim == 36
        assert encoder.hidden_dims == (256, 128, 64)
        assert encoder.latent_dim == 32

    def test_custom_init(self):
        encoder = LSTMEncoder(obs_dim=20, hidden_dims=(128, 64, 32), latent_dim=16)
        assert encoder.hidden_dims == (128, 64, 32)
        assert encoder.latent_dim == 16

    def test_parameter_count(self):
        encoder = LSTMEncoder(obs_dim=36)
        count = encoder.get_flat_weights()
        assert count > 0


class TestLSTMEncoderForward:
    def test_forward_2d_input(self):
        encoder = LSTMEncoder(obs_dim=36)
        obs = torch.randn(4, 36)
        z, hidden = encoder(obs)
        assert z.shape == (4, 32)
        assert len(hidden) == 3

    def test_forward_3d_input(self):
        encoder = LSTMEncoder(obs_dim=36)
        obs = torch.randn(4, 10, 36)
        z, hidden = encoder(obs)
        assert z.shape == (4, 32)

    def test_forward_single_sample(self):
        encoder = LSTMEncoder(obs_dim=36)
        obs = torch.randn(1, 36)
        z, hidden = encoder(obs)
        assert z.shape == (1, 32)

    def test_forward_batch_consistency(self):
        encoder = LSTMEncoder(obs_dim=36)
        encoder.eval()
        obs = torch.randn(1, 36)
        z1, _ = encoder(obs)
        z2, _ = encoder(obs)
        assert torch.allclose(z1, z2)

    def test_hidden_state_shapes(self):
        encoder = LSTMEncoder(obs_dim=36, hidden_dims=(256, 128, 64))
        obs = torch.randn(2, 36)
        _, (h1, h2, h3) = encoder(obs)
        assert h1.shape == (1, 2, 256)
        assert h2.shape == (1, 2, 128)
        assert h3.shape == (1, 2, 64)


class TestLSTMEncoderGradient:
    def test_backward_pass(self):
        encoder = LSTMEncoder(obs_dim=36)
        obs = torch.randn(4, 36)
        z, _ = encoder(obs)
        loss = z.sum()
        loss.backward()
        for p in encoder.parameters():
            if p.requires_grad:
                assert p.grad is not None

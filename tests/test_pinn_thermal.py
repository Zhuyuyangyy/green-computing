"""Tests for PINN thermal model and loss calculator."""
import pytest
import torch
from src.algo.pinn_thermal import PINNThermalModel, PINNLossCalculator


class TestPINNThermalModel:
    def test_init(self, n_gpus):
        model = PINNThermalModel(n_gpus=n_gpus)
        assert model.n_gpus == n_gpus
        assert model.c_thermal == 1000.0
        assert model.r_jc == 0.15
        assert model.r_sink == 0.08

    def test_physics_forward_shape(self, n_gpus):
        model = PINNThermalModel(n_gpus=n_gpus)
        t_j = torch.randn(4, n_gpus) + 50.0
        p_gpu = torch.randn(4, n_gpus) + 200.0
        t_new = model.physics_forward(t_j, p_gpu)
        assert t_new.shape == (4, n_gpus)

    def test_physics_forward_equation(self, n_gpus):
        model = PINNThermalModel(n_gpus=n_gpus)
        t_j = torch.full((1, n_gpus), 60.0)
        p_gpu = torch.full((1, n_gpus), 200.0)
        t_new = model.physics_forward(t_j, p_gpu, t_coolant=25.0)
        r_total = 0.15 + 0.08
        heat_flow = (60.0 - 25.0) / r_total
        delta_t = 1.0 / 1000.0 * (200.0 - heat_flow)
        expected = 60.0 + delta_t
        assert torch.allclose(t_new, torch.full((1, n_gpus), expected), atol=1e-5)

    def test_forward_shape(self, n_gpus, obs_dim):
        model = PINNThermalModel(n_gpus=n_gpus)
        obs = torch.randn(4, obs_dim)
        p_gpu = torch.randn(4, n_gpus)
        t_pred = model.forward(obs, p_gpu)
        assert t_pred.shape == (4, n_gpus)

    def test_pinn_loss_scalar(self, n_gpus):
        model = PINNThermalModel(n_gpus=n_gpus)
        t_pred = torch.randn(4, n_gpus) + 50.0
        t_true = torch.randn(4, n_gpus) + 50.0
        p_gpu = torch.randn(4, n_gpus) + 200.0
        loss = model.pinn_loss(t_pred, t_true, p_gpu)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_pinn_loss_zero_when_perfect(self, n_gpus):
        model = PINNThermalModel(n_gpus=n_gpus)
        t_true = torch.full((1, n_gpus), 60.0)
        p_gpu = torch.full((1, n_gpus), 200.0)
        t_physics = model.physics_forward(t_true, p_gpu)
        loss = model.pinn_loss(t_physics, t_true, p_gpu)
        assert loss.item() < 1e-6


class TestPINNLossCalculator:
    def test_compute_returns_tuple(self, n_gpus):
        model = PINNThermalModel(n_gpus=n_gpus)
        calc = PINNLossCalculator(model, beta=0.1)
        obs = torch.randn(4, 4 * n_gpus + 4)
        p_gpu = torch.randn(4, n_gpus) + 200.0
        t_true = torch.randn(4, n_gpus) + 50.0
        t_pred = torch.randn(4, n_gpus) + 50.0
        total, pinn = calc.compute(1.0, obs, p_gpu, t_true, t_pred)
        assert isinstance(total, float)
        assert isinstance(pinn, float)

    def test_compute_total_includes_beta(self, n_gpus):
        model = PINNThermalModel(n_gpus=n_gpus)
        calc = PINNLossCalculator(model, beta=0.1)
        obs = torch.randn(4, 4 * n_gpus + 4)
        p_gpu = torch.full((4, n_gpus), 200.0)
        t_true = torch.full((4, n_gpus), 60.0)
        t_pred = torch.full((4, n_gpus), 60.0)
        total, pinn = calc.compute(5.0, obs, p_gpu, t_true, t_pred)
        # total = rl_loss + beta * pinn_loss
        assert total == pytest.approx(5.0 + 0.1 * pinn, abs=1e-3)

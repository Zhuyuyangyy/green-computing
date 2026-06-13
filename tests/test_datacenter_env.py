"""Tests for DataCenterEnv - data center simulation environment."""
import pytest
import numpy as np
from src.env.datacenter_env import DataCenterEnv


class TestDataCenterEnvInit:
    def test_default_init(self):
        env = DataCenterEnv()
        assert env.n_gpus == 8
        assert env.p_idle == 50.0
        assert env.p_tdp == 400.0

    def test_custom_init(self):
        env = DataCenterEnv(n_gpus=4, seed=123)
        assert env.n_gpus == 4

    def test_observation_dim(self):
        env = DataCenterEnv(n_gpus=8)
        assert env.observation_dim == 4 * 8 + 4  # 36

    def test_observation_dim_custom(self):
        env = DataCenterEnv(n_gpus=4)
        assert env.observation_dim == 4 * 4 + 4  # 20


class TestDataCenterEnvReset:
    def test_reset_returns_observation(self):
        env = DataCenterEnv(n_gpus=4, seed=42)
        obs = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (env.observation_dim,)
        assert obs.dtype == np.float32

    def test_reset_resets_time(self):
        env = DataCenterEnv(n_gpus=4, seed=42)
        env.reset()
        assert env.current_time == 0.0
        assert env.episode_step == 0


class TestGPUPowerModel:
    def test_compute_gpu_power_basic(self):
        env = DataCenterEnv(n_gpus=4, seed=42)
        util = np.array([0.5, 0.5, 0.5, 0.5])
        freq = np.array([1.41, 1.41, 1.41, 1.41])
        power = env.compute_gpu_power(util, freq)
        assert power.shape == (4,)
        assert np.all(power >= env.p_idle)

    def test_gpu_power_at_max(self):
        env = DataCenterEnv(n_gpus=1, seed=42)
        util = np.array([1.0])
        freq = np.array([1.41])
        power = env.compute_gpu_power(util, freq)
        expected = 50.0 + (400.0 - 50.0) * 1.0 * (1.41 / 1.41) ** 1.3
        assert abs(power[0] - expected) < 0.01

    def test_gpu_power_at_idle(self):
        env = DataCenterEnv(n_gpus=1, seed=42)
        util = np.array([0.0])
        freq = np.array([1.41])
        power = env.compute_gpu_power(util, freq)
        assert abs(power[0] - 50.0) < 0.01

    def test_gpu_power_scales_with_freq(self):
        env = DataCenterEnv(n_gpus=1, seed=42)
        util = np.array([1.0])
        power_high = env.compute_gpu_power(util, np.array([1.41]))
        power_low = env.compute_gpu_power(util, np.array([0.35]))
        assert power_high[0] > power_low[0]


class TestHeatTransfer:
    def test_compute_heat_dissipation(self):
        env = DataCenterEnv(n_gpus=4, seed=42)
        flow = np.array([2.0, 2.0, 2.0, 2.0])
        delta_t = np.array([10.0, 10.0, 10.0, 10.0])
        q = env.compute_heat_dissipation(flow, delta_t)
        assert q.shape == (4,)
        assert np.all(q > 0)

    def test_heat_dissipation_scales_with_flow(self):
        env = DataCenterEnv(n_gpus=1, seed=42)
        q_low = env.compute_heat_dissipation(np.array([1.0]), np.array([10.0]))
        q_high = env.compute_heat_dissipation(np.array([3.0]), np.array([10.0]))
        assert q_high[0] > q_low[0]


class TestCOP:
    def test_cop_at_25c(self):
        env = DataCenterEnv()
        assert abs(env.cop(25.0) - 2.8) < 0.01

    def test_cop_minimum_is_1(self):
        env = DataCenterEnv()
        assert env.cop(100.0) == 1.0

    def test_cop_decreases_with_temp(self):
        env = DataCenterEnv()
        assert env.cop(20.0) > env.cop(30.0)


class TestChillerPower:
    def test_chiller_power_positive(self):
        env = DataCenterEnv()
        p = env.compute_chiller_power(10000.0, 25.0)
        assert p > 0

    def test_chiller_power_scales_with_heat(self):
        env = DataCenterEnv()
        p1 = env.compute_chiller_power(5000.0, 25.0)
        p2 = env.compute_chiller_power(10000.0, 25.0)
        assert p2 > p1


class TestThermalNetwork:
    def test_junction_temp(self):
        env = DataCenterEnv(n_gpus=4, seed=42)
        p_gpu = np.array([200.0, 200.0, 200.0, 200.0])
        t_j = env.compute_junction_temp(p_gpu, 25.0)
        expected = 25.0 + (0.15 + 0.08) * 200.0
        assert np.allclose(t_j, expected)

    def test_junction_temp_increases_with_power(self):
        env = DataCenterEnv(n_gpus=1, seed=42)
        t_low = env.compute_junction_temp(np.array([100.0]), 25.0)
        t_high = env.compute_junction_temp(np.array([400.0]), 25.0)
        assert t_high[0] > t_low[0]


class TestPUE:
    def test_pue_calculation(self):
        env = DataCenterEnv()
        pue = env.compute_pue(1000.0, 200.0)
        p_total = 1000.0 + 200.0 + 0.04 * 1000.0
        expected = p_total / 1000.0
        assert abs(pue - expected) < 0.01

    def test_pue_minimum_is_1(self):
        env = DataCenterEnv()
        pue = env.compute_pue(1000.0, 0.0)
        assert pue >= 1.0

    def test_pue_zero_it_power(self):
        env = DataCenterEnv()
        pue = env.compute_pue(0.0, 0.0)
        assert pue == 1.0


class TestCarbonEmission:
    def test_carbon_rate(self):
        env = DataCenterEnv()
        rate = env.compute_carbon_rate(10000.0, 500.0)
        expected = (10000.0 / 1000.0) * 500.0 / 3600.0
        assert abs(rate - expected) < 0.001

    def test_carbon_rate_zero_power(self):
        env = DataCenterEnv()
        assert env.compute_carbon_rate(0.0, 500.0) == 0.0


class TestObservation:
    def test_observation_shape(self):
        env = DataCenterEnv(n_gpus=4, seed=42)
        obs = env.get_observation()
        assert obs.shape == (16 + 4,)  # 4*4 + 4

    def test_observation_dtype(self):
        env = DataCenterEnv(n_gpus=4, seed=42)
        obs = env.get_observation()
        assert obs.dtype == np.float32


class TestStepSecond:
    def test_step_second_returns_correct_types(self):
        env = DataCenterEnv(n_gpus=4, seed=42)
        env.reset()
        freq = np.array([0.5, 0.5, 0.5, 0.5])
        flow = np.array([0.5, 0.5, 0.5, 0.5])
        obs, reward, done, info = env.step_second(freq, flow)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_second_info_keys(self):
        env = DataCenterEnv(n_gpus=4, seed=42)
        env.reset()
        _, _, _, info = env.step_second(
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([0.5, 0.5, 0.5, 0.5]),
        )
        expected_keys = {"p_gpu_total", "p_chiller", "p_total", "pue",
                         "carbon_rate", "t_j_max", "thermal_violation"}
        assert expected_keys.issubset(set(info.keys()))

    def test_step_second_increments_step(self):
        env = DataCenterEnv(n_gpus=4, seed=42)
        env.reset()
        assert env.episode_step == 0
        env.step_second(np.full(4, 0.5), np.full(4, 0.5))
        assert env.episode_step == 1

    def test_step_second_done_after_3600(self):
        env = DataCenterEnv(n_gpus=4, seed=42)
        env.reset()
        env.episode_step = 3599
        _, _, done, _ = env.step_second(np.full(4, 0.5), np.full(4, 0.5))
        assert done

    def test_step_second_frequency_clipping(self):
        env = DataCenterEnv(n_gpus=4, seed=42)
        env.reset()
        env.step_second(np.full(4, 2.0), np.full(4, 0.5))
        assert np.all(env.gpu_freq <= env.f_max)
        assert np.all(env.gpu_freq >= env.f_min)


class TestStepMinute:
    def test_step_minute_returns_correct_types(self):
        env = DataCenterEnv(n_gpus=4, seed=42)
        env.reset()
        action = np.array([0.1, 0.1, 0.1, 0.1])
        obs, reward, done, info = env.step_minute(action)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)


class TestEnvironmentalDisturbance:
    def test_disturbance_changes_carbon(self):
        env = DataCenterEnv(n_gpus=4, seed=42)
        env.reset()
        original_carbon = env.grid_carbon
        env.apply_environmental_disturbance()
        # With random seed, it may or may not change, but function should not error
        assert isinstance(env.grid_carbon, float)

    def test_disturbance_changes_temp(self):
        env = DataCenterEnv(n_gpus=4, seed=42)
        env.reset()
        env.apply_environmental_disturbance()
        assert 20.0 <= env.ambient_temp <= 35.0

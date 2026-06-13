"""Smoke tests - basic import and sanity checks."""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_import_main():
    """Verify the main module can be imported."""
    import main
    assert hasattr(main, "main")
    assert hasattr(main, "demo_mode")


def test_import_env():
    """Verify the environment module can be imported."""
    from src.env.datacenter_env import DataCenterEnv
    assert DataCenterEnv is not None


def test_import_algo():
    """Verify all algorithm modules can be imported."""
    from src.algo.lstm_encoder import LSTMEncoder
    from src.algo.hour_level_sac import HourLevelSAC
    from src.algo.minute_level_td3 import MinuteLevelTD3
    from src.algo.second_level_td3 import SecondLevelTD3
    from src.algo.pinn_thermal import PINNThermalModel
    assert all([LSTMEncoder, HourLevelSAC, MinuteLevelTD3, SecondLevelTD3, PINNThermalModel])


def test_import_trainer():
    """Verify the trainer module can be imported."""
    from src.training.trainer import HierarchicalRLTrainer
    assert HierarchicalRLTrainer is not None

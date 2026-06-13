# Green Computing: Carbon-Aware Hierarchical Reinforcement Learning for Data Center Energy Optimization

[![CI](https://github.com/green-computing/green-computing/actions/workflows/ci.yml/badge.svg)](https://github.com/green-computing/green-computing/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-101%20passed-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-91%25-green.svg)](tests/)

A three-level hierarchical reinforcement learning (HRL) system for jointly optimizing data center energy efficiency, carbon emissions, and thermal safety. The framework coordinates strategic carbon procurement (hour-level), tactical workload migration (minute-level), and operational GPU DVFS plus cooling control (second-level) with physics-informed neural network (PINN) thermal constraints.

---

## Overview

Data centers consume approximately 1-2% of global electricity, and this share is growing rapidly with AI workloads. Green Computing addresses this challenge through a hierarchical control framework that operates across three time scales, each using the most appropriate reinforcement learning algorithm.

**Key contributions:**

- Three-level temporal abstraction: Hour (SAC) -> Minute (TD3) -> Second (TD3)
- Shared 3-layer LSTM encoder that extracts temporal features across all time scales
- Physics-informed neural network (PINN) thermal constraints that enforce physical consistency
- Carbon-aware reward shaping that jointly optimizes power consumption, carbon emissions, and thermal safety
- GPU power model calibrated to NVIDIA A100 specifications with DVFS support

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Three-Level HRL** | Strategic (hour), tactical (minute), and operational (second) decision layers with distinct RL algorithms |
| **Shared LSTM Encoder** | 3-layer LSTM (256->128->64->32) extracts temporal features shared across all decision levels |
| **PINN Thermal Constraints** | Physics-informed loss term enforces thermal network equations (L_total = L_RL + 0.1 * L_PINN) |
| **Carbon-Aware Reward** | Reward function jointly optimizes power, carbon emissions (gCO2/s), and thermal violations |
| **A100 GPU Power Model** | Calibrated power model: P = P_idle + (P_tdp - P_idle) * util * (f/f_max)^alpha |
| **Variable COP Cooling** | Chiller coefficient of performance varies with ambient temperature: COP = 5.8 - 0.12 * T_amb |
| **Environmental Disturbance** | Simulates realistic carbon intensity, temperature, and electricity price fluctuations |

---

## Architecture

```
[Power Market]  <-- Hour-level Strategic Decision (SAC)
     |               Actions: carbon_quota, renewable_ratio
     v               Timescale: 3600s
[Load Balancer] <-- Minute-level Tactical Decision (TD3)
     |               Actions: cross-node workload migration
     v               Timescale: 60s
[GPU DVFS +     <-- Second-level Operational Control (TD3)
 Cooling]            Actions: GPU frequency, cooling flow rate
     |               Timescale: 1s
     v
[PINN Thermal   <-- Physics-informed thermal constraint
 Model]              T_j = T_coolant + (R_jc + R_sink) * P_GPU
```

**Shared Feature Extraction:**

```
Observation (4N + 4 dimensions)
  -> LSTM Layer 1 (256 hidden)
  -> LSTM Layer 2 (128 hidden)
  -> LSTM Layer 3 (64 hidden)
  -> Linear Projection (32 latent dimensions)
  -> Shared representation z_t for all decision levels
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | PyTorch 2.0+ |
| RL Algorithms | SAC (hour), TD3 (minute/second) |
| Feature Extraction | 3-layer LSTM encoder |
| Physics Constraints | PINN thermal model |
| Environment | Custom Gym-compatible datacenter simulation |
| Logging | TensorBoard |
| Configuration | Python-native with YAML override |
| Testing | pytest (101 tests, 91% coverage) |
| Linting | Ruff |
| Containerization | Docker + docker-compose |
| CI/CD | GitHub Actions |

---

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ (CUDA recommended for training)
- NumPy, Matplotlib

### Installation

```bash
# Using conda (recommended)
conda create -n green-computing python=3.10
conda activate green-computing
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install tensorboard

# Or using pip
pip install -r requirements.txt

# For development
pip install -e ".[dev]"
```

### Run Demo (no GPU required)

```bash
python main.py demo
```

Displays system architecture, paper equation alignment, and expected optimization results.

### Run Training

```bash
python main.py train --epochs 50
```

### Run Evaluation

```bash
python main.py evaluate
```

### Docker

```bash
# Build and run
docker-compose up --build

# Run training in container
docker-compose run green-computing python main.py train --epochs 50
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_datacenter_env.py -v
```

---

## Project Structure

```
green-computing/
├── main.py                          # CLI entry point (demo / train / evaluate)
├── requirements.txt                 # Python dependencies
├── pyproject.toml                   # Project configuration & pytest settings
├── Dockerfile                       # Container image definition
├── docker-compose.yml               # Multi-service orchestration
├── start.sh                         # One-click startup script
├── TODO.md                          # Innovation suggestions & technical debt
├── INNOVATION_ROADMAP.md            # Patent portfolio & research roadmap
├── OPTIMIZATION_REPORT.md           # Project optimization audit report
├── REPRODUCE.md                     # Reproduction instructions
├── scripts/
│   └── run_training.sh              # Full training pipeline script
├── src/
│   ├── env/
│   │   └── datacenter_env.py        # Data center simulation environment
│   ├── algo/
│   │   ├── lstm_encoder.py          # Shared 3-layer LSTM feature extractor
│   │   ├── hour_level_sac.py        # SAC agent for hour-level decisions
│   │   ├── minute_level_td3.py      # TD3 agent for minute-level decisions
│   │   ├── second_level_td3.py      # TD3 agent for second-level decisions
│   │   └── pinn_thermal.py          # PINN thermal model and loss
│   └── training/
│       └── trainer.py               # HRL training coordinator
├── tests/                           # Comprehensive test suite (101 tests, 91% coverage)
│   ├── conftest.py                  # Shared test fixtures
│   ├── test_datacenter_env.py       # Environment tests (34 tests)
│   ├── test_lstm_encoder.py         # LSTM encoder tests (9 tests)
│   ├── test_hour_level_sac.py       # SAC agent tests (12 tests)
│   ├── test_minute_level_td3.py     # TD3 agent tests (8 tests)
│   ├── test_second_level_td3.py     # Second-level TD3 tests (6 tests)
│   ├── test_pinn_thermal.py         # PINN thermal tests (8 tests)
│   ├── test_trainer.py              # Trainer tests (16 tests)
│   └── test_main.py                 # CLI tests (8 tests)
├── docs/                            # Documentation
│   ├── ARCHITECTURE.md              # System architecture deep-dive
│   ├── API.md                       # API reference
│   └── DEPLOYMENT.md                # Deployment guide
├── outputs/                         # Training checkpoints
└── .github/
    └── workflows/
        └── ci.yml                   # CI/CD pipeline
```

---

## Benchmarks

> **DISCLAIMER: ALL RESULTS ARE SYNTHETIC.** The numbers below are design
> targets from a simplified simulation environment. They have NOT been
> measured from a trained RL policy deployed on real data center hardware.
> The simulation uses idealised physics models (constant thermal
> resistance, fixed COP formula, simplified workload traces).  Real-world
> energy savings require validation on actual infrastructure with
> production controllers.

### Expected Optimization Results (Design Targets — Not Measured)

### Environment Parameters (A100 GPU)

| Parameter | Value | Description |
|-----------|-------|-------------|
| P_idle | 50W | Idle GPU power |
| P_tdp | 400W | Thermal design power |
| f_max | 1.41 GHz | Maximum GPU frequency |
| f_min | 0.35 GHz | Minimum GPU frequency (DVFS) |
| alpha_power | 1.3 | Power model exponent |
| T_j_max | 85C | Maximum junction temperature |
| R_jc | 0.15 K/W | Junction-to-case thermal resistance |
| R_sink | 0.08 K/W | Case-to-sink thermal resistance |

### LSTM Encoder Architecture

| Layer | Hidden Size | Input |
|-------|-------------|-------|
| LSTM 1 | 256 | Observation (4N+4) |
| LSTM 2 | 128 | LSTM 1 output |
| LSTM 3 | 64 | LSTM 2 output |
| Linear | 32 | LSTM 3 output (latent z_t) |

### Paper Equation Alignment

| Equation | Implementation |
|----------|---------------|
| GPU Power Model | `datacenter_env.py` -> `compute_gpu_power()` |
| Heat Transfer | `datacenter_env.py` -> `compute_heat_dissipation()` |
| Chiller Power | `datacenter_env.py` -> `cop()` + `compute_chiller_power()` |
| Thermal Network | `datacenter_env.py` -> `compute_junction_temp()` |
| PUE | `datacenter_env.py` -> `compute_pue()` |
| Carbon Emission | `datacenter_env.py` -> `compute_carbon_rate()` |
| SAC Objective | `hour_level_sac.py` |
| TD3 Critic | `minute_level_td3.py` / `second_level_td3.py` |
| LSTM Encoder | `lstm_encoder.py` |
| PINN Loss | `pinn_thermal.py` -> `PINNLossCalculator` |
| Total Loss | `pinn_thermal.py` -> L_total = L_RL + 0.1 * L_PINN |

---

## Research

This project implements and extends the following research concepts:

- **Hierarchical Reinforcement Learning**: Options framework (Sutton et al., 1999) with temporal abstraction
- **Soft Actor-Critic**: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL" (2018)
- **Twin Delayed DDPG**: Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods" (2018)
- **Physics-Informed Neural Networks**: Raissi et al., "Physics-Informed Neural Networks" (2019)
- **Green Computing**: Data center energy optimization and carbon-aware scheduling

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{green-computing-hrl-2024,
  title={Carbon-Aware Hierarchical Reinforcement Learning for Data Center Energy Optimization},
  author={},
  journal={},
  year={2024}
}
```

---

## Roadmap

- [x] Three-level HRL framework (Hour-SAC, Minute-TD3, Second-TD3)
- [x] Shared LSTM temporal encoder
- [x] PINN thermal constraints
- [x] Carbon-aware reward shaping
- [x] Comprehensive test suite (101 tests, 91% coverage)
- [x] Docker containerization
- [x] CI/CD pipeline
- [ ] Real-world integration with data center telemetry APIs (Prometheus, Grafana)
- [ ] Support for multi-datacenter coordination
- [ ] Integration with renewable energy forecasting models
- [ ] Add GPU memory bandwidth modeling for AI workload characterization
- [ ] Implement model-based RL (Dreamer/MBPO) for sample efficiency
- [ ] Add carbon intensity API integration (WattTime, ElectricityMaps)
- [ ] Benchmark against industry baselines (Google DeepMind data center control)

See [INNOVATION_ROADMAP.md](INNOVATION_ROADMAP.md) for the full patent and research roadmap.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | None (CPU training) | NVIDIA A100 40GB or RTX 3090+ |
| RAM | 16 GB | 32 GB |
| Storage | 5 GB | 10 GB for checkpoints |
| CPU | 8+ cores | 16+ cores |

---

## License

This project is released under the MIT License. See the LICENSE file for details.

---

## Contact

For questions, issues, or collaboration inquiries, please open a GitHub issue or contact the maintainers.

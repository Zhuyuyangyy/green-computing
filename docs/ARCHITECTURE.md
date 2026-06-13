# System Architecture

## Overview

Green Computing implements a three-level hierarchical reinforcement learning (HRL) system for data center energy optimization. The system operates across three temporal scales, each with its own RL agent and decision frequency.

## Three-Level Hierarchy

### Hour-Level: Strategic Carbon Procurement (SAC)

- **Timescale**: 3600 seconds (1 hour)
- **Algorithm**: Soft Actor-Critic (SAC) with automatic entropy tuning
- **Actions**: `carbon_quota`, `renewable_ratio` (continuous, [0, 1])
- **Objective**: Minimize long-term carbon cost while maintaining service availability

The hour-level agent makes strategic decisions about carbon credit procurement and renewable energy ratio. It operates on the slowest timescale and optimizes for long-term carbon cost reduction.

### Minute-Level: Tactical Workload Migration (TD3)

- **Timescale**: 60 seconds (1 minute)
- **Algorithm**: Twin Delayed DDPG (TD3)
- **Actions**: Per-GPU workload migration ratios (continuous, [-1, 1])
- **Objective**: Balance workload across GPUs to minimize hotspots

The minute-level agent performs tactical load balancing by migrating workloads between GPUs. It operates at intermediate frequency and optimizes for even thermal distribution.

### Second-Level: Operational GPU DVFS + Cooling (TD3)

- **Timescale**: 1 second
- **Algorithm**: Twin Delayed DDPG (TD3)
- **Actions**: Per-GPU frequency + cooling flow rate (continuous, [0, 1])
- **Objective**: Minimize power while respecting thermal constraints

The second-level agent performs real-time control of GPU frequency (DVFS) and cooling flow rate. It operates at the fastest timescale and directly controls hardware parameters.

## Shared Feature Extraction

All three decision levels share a common 3-layer LSTM encoder that extracts temporal features from the observation vector.

### Observation Space

The observation vector has dimension `4N + 4` where N is the number of GPUs:

| Index | Dimension | Description |
|-------|-----------|-------------|
| 0..N-1 | N | GPU utilization (0-1) |
| N..2N-1 | N | Workload demand (0-1) |
| 2N..3N-1 | N | Junction temperature (degrees C) |
| 3N | 1 | Coolant inlet temperature |
| 3N+1..4N-1 | N | Cooling flow rate (L/min) |
| 4N | 1 | Grid carbon intensity (gCO2/kWh) |
| 4N+1 | 1 | Electricity price ($/kWh) |
| 4N+2 | 1 | Mean predicted workload |

### LSTM Encoder Architecture

```
Input: (batch, seq_len, 4N+4)
  -> LSTM Layer 1: hidden=256
  -> LSTM Layer 2: hidden=128
  -> LSTM Layer 3: hidden=64
  -> Linear: 64 -> 32
Output: (batch, 32) = z_t (shared latent representation)
```

## Physics-Informed Neural Network (PINN) Thermal Model

The PINN thermal model encodes the physical heat transfer equations as an additional loss term:

### Physical Equations

**Thermal Network (Eq. thermal_network):**
```
T_j(t+1) = T_j(t) + (dt / C_th) * (P_GPU(t) - (T_j(t) - T_coolant) / (R_jc + R_sink))
```

**PINN Loss (Eq. pinn_loss):**
```
L_PINN = (1/N) * ||T_hat(t+1) - f_thermal(T(t), P(t))||^2
```

**Total Loss (Eq. total_loss):**
```
L_total = L_RL + beta * L_PINN,  where beta = 0.1
```

### Thermal Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| C_th | 1000 J/K | Equivalent thermal capacitance |
| R_jc | 0.15 K/W | Junction-to-case thermal resistance |
| R_sink | 0.08 K/W | Case-to-sink thermal resistance |
| T_j_max | 85 C | Maximum junction temperature |

## Data Center Environment

### GPU Power Model

```
P_GPU = P_idle + (P_tdp - P_idle) * utilization * (freq / f_max)^alpha
```

Where:
- P_idle = 50W (idle power)
- P_tdp = 400W (thermal design power for A100)
- f_max = 1.41 GHz
- alpha = 1.3 (power model exponent)

### Cooling System

**Heat Dissipation:**
```
Q = c * rho * Phi * dT
```

**Chiller COP:**
```
COP(T_amb) = max(5.8 - 0.12 * T_amb, 1.0)
```

**Chiller Power:**
```
P_chiller = Q_total / COP
```

### PUE Calculation

```
PUE = (P_GPU + P_chiller + P_loss) / P_GPU
```

Where P_loss = 4% of P_GPU (distribution losses).

### Carbon Emission Rate

```
carbon_rate (gCO2/s) = (P_total_kW) * carbon_intensity (gCO2/kWh) / 3600
```

## Reward Function

The reward function jointly optimizes three objectives:

```
R = R_carbon + R_thermal

R_carbon = -P_total - 0.01 * carbon_intensity * P_total / 1000
R_thermal = -1e4 * sum(max(0, T_j - T_j_max)^2)
```

Where:
- R_carbon penalizes power consumption and carbon emissions
- R_thermal heavily penalizes thermal violations (quadratic penalty)

## Environmental Disturbance

The environment simulates realistic fluctuations in:

1. **Carbon Intensity**: Time-of-day pattern with peak during business hours
2. **Ambient Temperature**: Random variation between 20-35 C
3. **Electricity Price**: Random variation between $0.08-0.18/kWh
4. **Workload Prediction**: Random variation for forecast uncertainty

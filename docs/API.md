# API Reference

## DataCenterEnv

The core simulation environment for data center energy optimization.

### Constructor

```python
DataCenterEnv(
    n_gpus: int = 8,
    dt_hour: float = 3600.0,
    dt_minute: float = 60.0,
    dt_second: float = 1.0,
    seed: int = 42,
)
```

### Methods

#### `reset() -> np.ndarray`

Reset the environment and return the initial observation vector.

**Returns:** Observation vector of shape `(4*n_gpus + 4,)` with dtype `float32`.

#### `get_observation() -> np.ndarray`

Get the current observation vector without stepping the environment.

**Returns:** Observation vector containing GPU utilization, workload, temperatures, flow rates, and external signals.

#### `step_second(freq_action, flow_action) -> Tuple[ndarray, float, bool, Dict]`

Execute one second-level control step.

**Parameters:**
- `freq_action`: ndarray of shape `(n_gpus,)` - Normalized GPU frequencies [0, 1]
- `flow_action`: ndarray of shape `(n_gpus,)` - Normalized cooling flow rates [0, 1]

**Returns:** `(observation, reward, done, info)` tuple

**Info dict keys:**
- `p_gpu_total`: Total GPU power (W)
- `p_chiller`: Chiller power (W)
- `p_total`: Total facility power (W)
- `pue`: Power Usage Effectiveness
- `carbon_rate`: Carbon emission rate (gCO2/s)
- `t_j_max`: Maximum junction temperature (C)
- `thermal_violation`: Boolean flag for thermal violations

#### `step_minute(migrate_action) -> Tuple[ndarray, float, bool, Dict]`

Execute one minute-level workload migration step.

**Parameters:**
- `migrate_action`: ndarray of shape `(n_gpus,)` - Workload migration ratios

**Returns:** `(observation, reward, done, info)` tuple

#### `compute_gpu_power(util, freq) -> np.ndarray`

Compute GPU power consumption using the DVFS power model.

**Parameters:**
- `util`: GPU utilization array [0, 1]
- `freq`: GPU frequency array (GHz)

**Returns:** Power consumption array (W)

#### `compute_heat_dissipation(flow_Lmin, delta_T) -> np.ndarray`

Compute heat dissipation from coolant flow.

#### `cop(t_ambient) -> float`

Compute Coefficient of Performance for the chiller.

#### `compute_chiller_power(total_heat, t_ambient) -> float`

Compute chiller power consumption.

#### `compute_junction_temp(p_gpu, t_coolant) -> np.ndarray`

Compute GPU junction temperatures from the thermal network model.

#### `compute_pue(p_gpu_cluster, p_chiller) -> float`

Compute Power Usage Effectiveness.

#### `compute_carbon_rate(p_total, carbon_intensity) -> float`

Compute carbon emission rate in gCO2/s.

#### `apply_environmental_disturbance()`

Apply random environmental disturbances (carbon intensity, temperature, price).

### Properties

#### `observation_dim -> int`

Returns the observation vector dimension: `4 * n_gpus + 4`

---

## LSTMEncoder

Shared 3-layer LSTM temporal feature extractor.

### Constructor

```python
LSTMEncoder(
    obs_dim: int,
    hidden_dims: Tuple[int, int, int] = (256, 128, 64),
    latent_dim: int = 32,
)
```

### Methods

#### `forward(obs: Tensor) -> Tuple[Tensor, Tuple]`

Encode observation sequence into latent representation.

**Parameters:**
- `obs`: Tensor of shape `(batch, seq_len, obs_dim)` or `(batch, obs_dim)`

**Returns:** `(z_t, hidden_states)` where z_t has shape `(batch, latent_dim)`

#### `get_flat_weights() -> int`

Returns the number of trainable parameters.

---

## HourLevelSAC

SAC agent for hour-level carbon-aware procurement decisions.

### Constructor

```python
HourLevelSAC(
    latent_dim: int = 32,
    action_dim: int = 2,
    lr: float = 3e-4,
    gamma: float = 0.99,
    tau: float = 0.005,
    device: str = "cpu",
)
```

### Methods

#### `update(batch: dict) -> dict`

Perform one gradient update step.

**Batch keys:** `latent`, `action`, `reward`, `next_latent`, `done`

**Returns:** Dict with `actor_loss`, `critic_loss`, `alpha`

#### `select_action(latent, deterministic=False) -> np.ndarray`

Select an action given the latent state.

#### `save(path: str)` / `load(path: str)`

Save/load model checkpoints.

---

## MinuteLevelTD3

TD3 agent for minute-level workload migration decisions.

### Constructor

```python
MinuteLevelTD3(
    latent_dim: int = 32,
    action_dim: int = 8,
    lr: float = 3e-4,
    gamma: float = 0.99,
    tau: float = 0.005,
    policy_noise: float = 0.2,
    noise_clip: float = 0.5,
    policy_delay: int = 2,
    device: str = "cpu",
)
```

### Methods

#### `update(batch: dict) -> dict`

Perform one TD3 update step with delayed policy updates.

#### `select_action(latent, deterministic=True) -> np.ndarray`

Select an action given the latent state.

---

## SecondLevelTD3

TD3 agent for second-level GPU frequency and cooling control.

### Constructor

```python
SecondLevelTD3(
    latent_dim: int,
    n_gpus: int,
    hidden: int = 256,
)
```

### Methods

#### `forward(latent, action) -> Tuple[Tensor, Tensor]`

Compute twin Q-values.

#### `update(batch, ...) -> dict`

Perform one TD3 update step.

---

## PINNThermalModel

Physics-informed neural network for thermal constraint enforcement.

### Constructor

```python
PINNThermalModel(
    n_gpus: int = 8,
    c_thermal: float = 1000.0,
    r_jc: float = 0.15,
    r_sink: float = 0.08,
    device: str = "cpu",
)
```

### Methods

#### `physics_forward(t_j, p_gpu, t_coolant=25.0) -> Tensor`

Compute physics-based temperature prediction using the thermal network equation.

#### `pinn_loss(t_j_pred, t_j_true, p_gpu) -> Tensor`

Compute PINN loss: L_PINN = mean((T_pred - T_physics)^2)

#### `forward(obs, p_gpu) -> Tensor`

Neural network temperature prediction.

---

## PINNLossCalculator

Computes the combined RL + PINN loss.

### Constructor

```python
PINNLossCalculator(pinn_model: PINNThermalModel, beta: float = 0.1)
```

### Methods

#### `compute(rl_loss, obs, p_gpu, t_j_true, t_j_pred) -> Tuple[float, float]`

Returns `(total_loss, pinn_loss)` where total_loss = rl_loss + beta * pinn_loss.

---

## HierarchicalRLTrainer

Training coordinator for the three-level HRL system.

### Constructor

```python
HierarchicalRLTrainer(
    n_gpus: int = 8,
    device: str = "cpu",
    seed: int = 42,
)
```

### Methods

#### `encode(obs: np.ndarray) -> Tensor`

Encode observation into shared latent representation.

#### `run_episode(max_steps=3600, render=False) -> dict`

Run one complete episode and return statistics.

**Returns:** Dict with `total_power`, `total_carbon`, `thermal_violations`, `pue_avg`, `steps`

#### `train(n_episodes=100, save_dir="outputs")`

Main training loop with periodic checkpointing.

#### `save(path: str)` / `load(path: str)`

Save/load full trainer state.

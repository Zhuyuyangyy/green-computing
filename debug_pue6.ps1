import numpy as np
from src.env.datacenter_env import DataCenterEnv

env = DataCenterEnv(n_gpus=8, seed=42)
obs = env.reset()

print("=== Checking initial state ===")
print(f"n_gpus: {env.n_gpus}")
print(f"f_min={env.f_min}, f_max={env.f_max}")
print(f"p_idle={env.p_idle}, p_tdp={env.p_tdp}")
print(f"t_coolant_in={env.t_coolant_in}")
print(f"ambient_temp={env.ambient_temp}")
print(f"cop(25)={env.cop(25)}")

# Check compute_heat_dissipation formula
flow = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # 1 L/min each
delta_T = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])  # 10C delta
q = env.compute_heat_dissipation(flow, delta_T)
print(f"\nHeat with 1L/min and 10C delta per GPU: {q} W")
print(f"Total heat: {np.sum(q)} W")

# Now check what 8 GPUs at full load produce
p_gpu = env.compute_gpu_power(np.ones(8), np.ones(8) * env.f_max)
print(f"\n8 GPUs at max freq and util: P_gpu = {p_gpu} W (total {np.sum(p_gpu)} W)")

# Expected chiller power: total_heat / COP
total_heat_expected = np.sum(p_gpu)  # All heat must be removed
expected_p_chiller = total_heat_expected / env.cop(25)
print(f"Expected p_chiller: {expected_p_chiller:.1f} W")
print(f"Expected PUE: {(np.sum(p_gpu) + expected_p_chiller) / np.sum(p_gpu):.2f}")

# Now with actual step
freq_action = np.random.uniform(0, 1, 8)
flow_action = np.random.uniform(0, 1, 8)
obs, r, done, info = env.step_second(freq_action, flow_action)

print(f"\n=== After one step ===")
print(f"p_gpu_total: {info['p_gpu_total']:.1f} W")
print(f"p_chiller: {info['p_chiller']:.1f} W")
print(f"p_total: {info['p_total']:.1f} W")
print(f"PUE: {info['pue']:.2f}")
print(f"grid_carbon: {env.grid_carbon}")
print(f"ambient_temp: {env.ambient_temp}")

# Compute heat dissipation with current env state
flow_m3s = env.flow_rate / 1000.0 / 60.0
delta_T_current = env.t_outlet - env.t_coolant_in * np.ones(env.n_gpus)
q_current = 4182 * 1050 * flow_m3s * delta_T_current
print(f"\nCurrent flow_m3s[:3]: {flow_m3s[:3]}")
print(f"Current delta_T[:3]: {delta_T_current[:3]}")
print(f"Current q[:3]: {q_current[:3]}")
print(f"Sum of q: {np.sum(q_current):.1f} W")
print(f"Chiller from env: {env.compute_chiller_power(np.sum(q_current), env.ambient_temp):.1f} W")
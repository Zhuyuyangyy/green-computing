import numpy as np
from src.env.datacenter_env import DataCenterEnv

env = DataCenterEnv(n_gpus=8, seed=42)
obs = env.reset()

# Snapshot BEFORE step
t_junction_before = env.t_junction.copy()
t_outlet_before = env.t_outlet.copy()
flow_before = env.flow_rate.copy()

freq_action = np.random.uniform(0, 1, 8)
flow_action = np.random.uniform(0, 1, 8)

# Simulate what step_second computes for heat
new_freq = np.clip(env.f_min + freq_action * (env.f_max - env.f_min), env.f_min, env.f_max)
new_flow = np.clip(0.5 + flow_action * (5.0 - 0.5), 0.5, 5.0)

# Use t_outlet BEFORE update (this is what step_second uses)
delta_T_before = t_outlet_before - env.t_coolant_in * np.ones(env.n_gpus)
flow_m3s_before = flow_before / 1000.0 / 60.0
q_before = 4182 * 1050 * flow_m3s_before * delta_T_before
total_heat_before = np.sum(q_before)
cop = env.cop(env.ambient_temp)
p_chiller_before = total_heat_before / cop

print("=== BEFORE step (what step_second will use) ===")
print(f"t_outlet_before[:3]: {t_outlet_before[:3]}")
print(f"delta_T_before[:3]: {delta_T_before[:3]}")
print(f"flow_before[:3]: {flow_before[:3]}")
print(f"total_heat_before: {total_heat_before}")
print(f"p_chiller (computed with old t_outlet): {p_chiller_before}")

# Now do the step
obs, r, done, info = env.step_second(freq_action, flow_action)

print("\n=== AFTER step ===")
print(f"info p_chiller: {info['p_chiller']}")
print(f"info p_total: {info['p_total']}")
print(f"info pue: {info['pue']}")

# Manually recompute PUE with the OLD t_outlet used in step_second
p_gpu = env.compute_gpu_power(env.gpu_util, env.gpu_freq)
p_gpu_cluster = np.sum(p_gpu)
p_loss = env.p_loss_ratio * p_gpu_cluster
p_total_recomputed = p_gpu_cluster + p_chiller_before + p_loss
pue_recomputed = p_total_recomputed / p_gpu_cluster if p_gpu_cluster > 0 else 1.0

print(f"\nPUE recomputed with old t_outlet: {pue_recomputed}")
print(f"PUE from info: {info['pue']}")
print(f"Match: {abs(pue_recomputed - info['pue']) < 0.01}")
import numpy as np
from src.env.datacenter_env import DataCenterEnv

env = DataCenterEnv(n_gpus=8, seed=42)
obs = env.reset()

# Snapshot BEFORE step
t_junction_before = env.t_junction.copy()
t_outlet_before = env.t_outlet.copy()

freq_action = np.random.uniform(0, 1, 8)
flow_action = np.random.uniform(0, 1, 8)

# What step_second does:
new_freq = np.clip(env.f_min + freq_action * (env.f_max - env.f_min), env.f_min, env.f_max)
new_flow = np.clip(0.5 + flow_action * (5.0 - 0.5), 0.5, 5.0)

# step_second uses NEW flow_rate but OLD t_outlet for heat calculation
delta_T_from_old_outlet = t_outlet_before - env.t_coolant_in * np.ones(env.n_gpus)
flow_m3s_new = new_flow / 1000.0 / 60.0
q_heat_computed = 4182 * 1050 * flow_m3s_new * delta_T_from_old_outlet
total_heat_computed = np.sum(q_heat_computed)
cop = env.cop(env.ambient_temp)
p_chiller_computed = total_heat_computed / cop

print("=== Correct simulation of step_second ===")
print(f"new_flow[:3]: {new_flow[:3]}")
print(f"t_outlet_before[:3]: {t_outlet_before[:3]}")
print(f"delta_T[:3]: {delta_T_from_old_outlet[:3]}")
print(f"flow_m3s_new[:3]: {flow_m3s_new[:3]}")
print(f"q_heat[:3]: {q_heat_computed[:3]}")
print(f"total_heat: {total_heat_computed}")
print(f"COP: {cop}")
print(f"p_chiller: {p_chiller_computed}")

# Now call step
obs, r, done, info = env.step_second(freq_action, flow_action)

print(f"\ninfo p_chiller: {info['p_chiller']}")
print(f"info p_gpu_total: {info['p_gpu_total']}")
print(f"info p_total: {info['p_total']}")
print(f"info pue: {info['pue']}")

# Recompute from info values
p_gpu_cluster = info['p_gpu_total']
p_chiller_info = info['p_chiller']
p_loss = env.p_loss_ratio * p_gpu_cluster
p_total_from_info = p_gpu_cluster + p_chiller_info + p_loss
pue_from_info = p_total_from_info / p_gpu_cluster if p_gpu_cluster > 0 else 1.0

print(f"\nPUE recomputed from info: {pue_from_info}")
print(f"PUE from info['pue']: {info['pue']}")
print(f"Match: {abs(pue_from_info - info['pue']) < 0.001}")
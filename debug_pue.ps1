import numpy as np
from src.env.datacenter_env import DataCenterEnv

env = DataCenterEnv(n_gpus=8, seed=42)
obs = env.reset()

# Trace PUE calculation manually
print("=== Step-by-step PUE trace ===")
freq_action = np.random.uniform(0, 1, 8)
flow_action = np.random.uniform(0, 1, 8)
obs, r, done, info = env.step_second(freq_action, flow_action)

# Manual calculation
p_gpu = env.compute_gpu_power(env.gpu_util, env.gpu_freq)
p_gpu_cluster = np.sum(p_gpu)
flow_m3s = env.flow_rate / 1000.0 / 60.0
delta_T = env.t_outlet - env.t_coolant_in * np.ones(env.n_gpus)
q_heat = env.c_coolant * env.rho_coolant * flow_m3s * delta_T
total_heat = np.sum(q_heat)
cop = env.cop(env.ambient_temp)
p_chiller_manual = total_heat / cop
p_loss = env.p_loss_ratio * p_gpu_cluster
p_total_manual = p_gpu_cluster + p_chiller_manual + p_loss
pue_manual = p_total_manual / p_gpu_cluster if p_gpu_cluster > 0 else 1.0

print(f"env.t_outlet[:3]: {env.t_outlet[:3]}")
print(f"env.t_junction[:3]: {env.t_junction[:3]}")
print(f"delta_T[:3]: {delta_T[:3]}")
print(f"flow_m3s[:3]: {flow_m3s[:3]}")
print(f"q_heat[:3]: {q_heat[:3]}")
print(f"total_heat: {total_heat}")
print(f"COP: {cop}")
print(f"p_chiller_manual: {p_chiller_manual}")
print(f"p_gpu_cluster: {p_gpu_cluster}")
print(f"p_loss: {p_loss}")
print(f"p_total_manual: {p_total_manual}")
print(f"PUE manual: {pue_manual}")
print()
print(f"info from step_second:")
print(f"  p_gpu_total={info['p_gpu_total']}")
print(f"  p_chiller={info['p_chiller']}")
print(f"  p_total={info['p_total']}")
print(f"  pue={info['pue']}")
print(f"  ratio: {info['pue']} vs manual {pue_manual}")
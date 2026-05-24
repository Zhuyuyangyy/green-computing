import numpy as np
from src.env.datacenter_env import DataCenterEnv

env = DataCenterEnv(n_gpus=8, seed=42)
obs = env.reset()

print("=== Initial state ===")
print(f"t_junction[:3]: {env.t_junction[:3]}")
print(f"t_outlet[:3]: {env.t_outlet[:3]}")
print(f"flow_rate[:3]: {env.flow_rate[:3]}")
print(f"gpu_freq[:3]: {env.gpu_freq[:3]}")
print(f"gpu_util[:3]: {env.gpu_util[:3]}")

# Now trace step by step
freq_action = np.random.uniform(0, 1, 8)
flow_action = np.random.uniform(0, 1, 8)

# BEFORE step
print("\n=== BEFORE step_second ===")
print(f"t_junction[:3]: {env.t_junction[:3]}")
print(f"t_outlet[:3]: {env.t_outlet[:3]}")

# Manually trace what step_second will do
env.gpu_freq_test = env.f_min + freq_action * (env.f_max - env.f_min)
env.gpu_freq_test = np.clip(env.gpu_freq_test, env.f_min, env.f_max)
env.flow_rate_test = 0.5 + flow_action * (5.0 - 0.5)
env.flow_rate_test = np.clip(env.flow_rate_test, 0.5, 5.0)
print(f"\nNew freq will be: {env.gpu_freq_test[:3]}")
print(f"New flow will be: {env.flow_rate_test[:3]}")

# Now call step
obs, r, done, info = env.step_second(freq_action, flow_action)

print("\n=== AFTER step_second ===")
print(f"t_junction[:3]: {env.t_junction[:3]}")
print(f"t_outlet[:3]: {env.t_outlet[:3]}")

print("\n=== PUE trace ===")
p_gpu = env.compute_gpu_power(env.gpu_util, env.gpu_freq)
p_gpu_cluster = np.sum(p_gpu)
flow_m3s = env.flow_rate / 1000.0 / 60.0
delta_T = env.t_outlet - env.t_coolant_in * np.ones(env.n_gpus)
print(f"delta_T[:3]: {delta_T[:3]}")
print(f"flow_m3s[:3]: {flow_m3s[:3]}")
q_heat = 4182 * 1050 * flow_m3s * delta_T
total_heat = np.sum(q_heat)
cop = env.cop(env.ambient_temp)
p_chiller_manual = total_heat / cop
p_loss = env.p_loss_ratio * p_gpu_cluster
p_total_manual = p_gpu_cluster + p_chiller_manual + p_loss
pue_manual = p_total_manual / p_gpu_cluster if p_gpu_cluster > 0 else 1.0

print(f"total_heat: {total_heat}")
print(f"COP: {cop}")
print(f"p_chiller_manual: {p_chiller_manual}")
print(f"p_gpu_cluster: {p_gpu_cluster}")
print(f"PUE manual: {pue_manual}")
print(f"PUE from info: {info['pue']}")

# Check: maybe delta_T is computed BEFORE t_outlet is updated in step_second?
# step_second updates t_outlet AFTER computing heat. So maybe the info pue uses OLD t_outlet?
print("\n=== Checking if info uses stale t_outlet ===")
print("info p_chiller:", info['p_chiller'])
print("manual p_chiller:", p_chiller_manual)
print("ratio:", info['p_chiller'] / p_chiller_manual if p_chiller_manual > 0 else 0)
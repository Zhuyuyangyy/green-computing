import numpy as np
from src.env.datacenter_env import DataCenterEnv

env = DataCenterEnv(n_gpus=8, seed=42)
obs = env.reset()

# Snapshot BEFORE step
state_before = {
    't_junction': env.t_junction.copy(),
    't_outlet': env.t_outlet.copy(),
    'flow_rate': env.flow_rate.copy(),
    'gpu_freq': env.gpu_freq.copy(),
    'gpu_util': env.gpu_util.copy(),
}

freq_action = np.random.uniform(0, 1, 8)
flow_action = np.random.uniform(0, 1, 8)

# Try different delta_T combinations
new_freq = np.clip(env.f_min + freq_action * (env.f_max - env.f_min), env.f_min, env.f_max)
new_flow = np.clip(0.5 + flow_action * (5.0 - 0.5), 0.5, 5.0)

for label, dt_arr in [
    ("OLD_t_outlet", state_before['t_outlet'] - 25),
    ("OLD_t_junction", state_before['t_junction'] - 25),
    ("NEW_t_outlet", env.t_outlet - 25),  # this won't exist until after step
    ("NEW_t_junction", env.t_junction - 25),  # this won't exist until after step
]:
    flow_m3s = new_flow / 1000.0 / 60.0
    q = 4182 * 1050 * flow_m3s * dt_arr
    total_heat = np.sum(q)
    cop = env.cop(env.ambient_temp)
    p_chiller = total_heat / cop
    print(f"{label}: total_heat={total_heat:.1f}, p_chiller={p_chiller:.1f}")

# Now call step and see
obs, r, done, info = env.step_second(freq_action, flow_action)
print(f"\ninfo p_chiller: {info['p_chiller']}")
print(f"info p_total: {info['p_total']}")
print(f"info pue: {info['pue']}")
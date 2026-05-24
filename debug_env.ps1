import numpy as np
from src.env.datacenter_env import DataCenterEnv

env = DataCenterEnv(n_gpus=8, seed=42)
obs = env.reset()
done = False
steps = 0
while not done and steps < 10:
    freq_action = np.random.uniform(0, 1, 8)
    flow_action = np.random.uniform(0, 1, 8)
    obs, r, done, info = env.step_second(freq_action, flow_action)
    steps += 1
    print(f'Step {steps}: episode_step={env.episode_step}, done={done}, p_total={info["p_total"]:.1f}W, pue={info["pue"]:.2f}')
print(f'Exited after {steps} steps, done={done}')

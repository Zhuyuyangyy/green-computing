# Carbon-Aware Hierarchical RL Data Center Energy Optimization

三层HRL数据中心能耗优化: Hour-SAC -> Minute-TD3 -> Second-DDPG

Paper: `papers/02_green_computing_carbon_aware_hrl/`

## Project Overview

本项目实现了一个碳感知的分层强化学习(HRL)数据中心能效优化系统,包含三层时域抽象:

| Level | Timescale | Algorithm | Action | Paper Section |
|-------|-----------|-----------|--------|---------------|
| Hour  | 3600s     | SAC       | Carbon procurement / renewable ratio | method.tex Eq.(eq:sac_objective) |
| Minute| 60s       | TD3       | Cross-node workload migration | method.tex Eq.(eq:td3_critic) |
| Second| 1s        | TD3       | GPU DVFS + cooling flow control | method.tex Section 2.4 |

### Key Features
- 三层时域抽象: 战略(小时) -> 战术(分钟) -> 操作(秒)
- 共享3层LSTM编码器提取时序特征 (256->128->64->32)
- PINN热约束保持物理一致性 (beta=0.1)
- 碳强度感知reward: 功率 + 碳排放 + 热违规惩罚
- NVIDIA A100 GPU功率模型 (Eq.(eq:gpu_power))
- 变COP冷却系统模型 (Eq.(eq:chiller_power))

## Environment Dependencies

### Conda Installation
```bash
conda create -n green-computing python=3.10
conda activate green-computing
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install numpy matplotlib -c conda-forge
pip install tensorboard
```

### pip Installation
```bash
pip install torch numpy matplotlib tensorboard
```

## Code Structure

```
src/
  env/
    datacenter_env.py    # 数据中心仿真环境, 对齐 papers/02/ 全部公式
  algo/
    lstm_encoder.py       # 共享3层LSTM编码器: 256->128->64->32
    hour_level_sac.py     # Hour-level SAC (碳采购/可再生能源比例)
    minute_level_td3.py   # Minute-level TD3 (负载均衡)
    second_level_td3.py   # Second-level TD3 (GPU频率+冷却流速)
    pinn_thermal.py       # PINN热模型 L_total = L_RL + 0.1*L_PINN
  training/
    trainer.py            # HRL主训练器,三层协调
```

## Training Pipeline

### Step 1: Train Second-Level (GPU DVFS + Cooling)
```bash
cd /mnt/d/ZYY\ Project/green-computing
python -c "
from src.training.trainer import HierarchicalRLTrainer
trainer = HierarchicalRLTrainer(n_gpus=8, device='cpu', seed=42)
# Focus on second-level updates
for ep in range(50):
    stats = trainer.run_episode(max_steps=3600)
    trainer._update_second(batch_size=256)
    print(f'[Second Ep {ep+1}] Power={stats[\"total_power\"]:.1f}kW PUE={stats[\"pue_avg\"]:.3f}')
trainer.save('outputs/second_level.pt')
"
```

### Step 2: Train Minute-Level (Workload Migration)
```bash
python -c "
from src.training.trainer import HierarchicalRLTrainer
trainer = HierarchicalRLTrainer(n_gpus=8, device='cpu', seed=42)
trainer.load('outputs/second_level.pt') if __import__('os').path.exists('outputs/second_level.pt') else None
for ep in range(50):
    stats = trainer.run_episode(max_steps=3600)
    # Minute-level updates every 60 steps (batched internally)
    if ep % 5 == 0:
        for _ in range(10):
            trainer._update_minute()
    print(f'[Minute Ep {ep+1}] Power={stats[\"total_power\"]:.1f}kW Carbon={stats[\"total_carbon\"]:.2f}g')
trainer.save('outputs/minute_level.pt')
"
```

### Step 3: Train Hour-Level (Carbon Procurement)
```bash
python -c "
from src.training.trainer import HierarchicalRLTrainer
trainer = HierarchicalRLTrainer(n_gpus=8, device='cpu', seed=42)
trainer.load('outputs/minute_level.pt') if __import__('os').path.exists('outputs/minute_level.pt') else None
for ep in range(100):
    stats = trainer.run_episode(max_steps=3600)
    if ep % 10 == 0:
        for _ in range(5):
            trainer._update_hour()
    print(f'[Hour Ep {ep+1}] Power={stats[\"total_power\"]:.1f}kW Carbon={stats[\"total_carbon\"]:.2f}g PUE={stats[\"pue_avg\"]:.3f}')
trainer.save('outputs/full_hrl_model.pt')
"
```

### Full Training (All Levels Simultaneously)
```bash
cd /mnt/d/ZYY\ Project/green-computing
python src/training/trainer.py
```

## Key Parameters

### DataCenterEnv
| Parameter | Default | Description |
|-----------|---------|-------------|
| n_gpus | 8 | GPU数量 |
| p_idle | 50W | A100空闲功率 |
| p_tdp | 400W | A100 TDP |
| f_max | 1.41GHz | 最大GPU频率 |
| f_min | 0.35GHz | 最小GPU频率 |
| alpha_power | 1.3 | GPU功率模型指数 |
| c_coolant | 4182 J/(kg·K) | 冷却液比热容 |
| t_coolant_in | 25°C | 冷却液进口温度 |
| r_th_jc | 0.15 K/W | 结到壳热阻 |
| r_th_sink | 0.08 K/W | 壳到散热器热阻 |
| t_j_max | 85°C | 最大结温 |

### LSTM Encoder
| Parameter | Default | Description |
|-----------|---------|-------------|
| hidden_dims | (256, 128, 64) | 三层LSTM hidden sizes |
| latent_dim | 32 | 输出隐向量维度 |
| obs_dim | 4*N+4 | 输入观测维度 (N=gpu数) |

### Training
| Parameter | Default | Description |
|-----------|---------|-------------|
| gamma | 0.99 | 折扣因子 |
| tau | 0.005 | 目标网络软更新系数 |
| lr | 3e-4 | 学习率 |
| batch_size (second) | 256 | Second级批次大小 |
| batch_size (minute) | 64 | Minute级批次大小 |
| batch_size (hour) | 32 | Hour级批次大小 |

## Expected Outputs

### Training Log Format
```
[Ep 1/50] Power=12543.2kW Carbon=892.34g PUE=1.23 Violations=0 t=12.3s Updates:H=0M=0S=1
[Ep 10/50] Power=11234.5kW Carbon=756.21g PUE=1.18 Violations=0 t=11.8s Updates:H=0M=2S=10
...
```

### Key Metrics
- **Power (kW)**: 总集群功率,越低越好
- **Carbon (g)**: 碳排放速率,越低越好
- **PUE**: 电力使用效率,目标<1.2
- **Thermal Violations**: 结温>85°C次数,应为0

### Checkpoint Files
- `outputs/second_level.pt`: 训练好的Second级
- `outputs/minute_level.pt`: 训练好的Minute级
- `outputs/full_hrl_model.pt`: 完整三层模型

## Hardware Requirements

| Component | Requirement |
|-----------|-------------|
| GPU | NVIDIA A100 (40GB) or equivalent |
| Memory | 16GB+ RAM |
| Storage | 5GB+ for checkpoints |
| CPU | 8+ cores recommended |

### Minimum (CPU-only training)
- 16GB RAM
- 8+ CPU cores
- Training time: ~10-20s per episode

### Recommended (GPU training)
- NVIDIA A100 40GB or RTX 3090+
- 32GB RAM
- Training time: ~5-10s per episode

## Paper Equation Index

| Equation | Implementation |
|----------|---------------|
| Eq.(eq:gpu_power) | `datacenter_env.py compute_gpu_power()` |
| Eq.(eq:heat_transfer) | `datacenter_env.py compute_heat_dissipation()` |
| Eq.(eq:chiller_power) | `datacenter_env.py cop()` + `compute_chiller_power()` |
| Eq.(eq:thermal_network) | `datacenter_env.py compute_junction_temp()` |
| Eq.(eq:pue) | `datacenter_env.py compute_pue()` |
| Eq.(eq:carbon_emission) | `datacenter_env.py compute_carbon_rate()` |
| Eq.(eq:sac_objective) | `hour_level_sac.py` |
| Eq.(eq:td3_critic) | `minute_level_td3.py` |
| Eq.(eq:lstm_encoder) | `lstm_encoder.py` |
| Eq.(eq:pinn_loss) | `pinn_thermal.py PINNLossCalculator` |
| Eq.(eq:total_loss) | `pinn_thermal.py L_total = L_RL + 0.1*L_PINN` |

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

#!/bin/bash
# Hierarchical RL Data Center Training Script
# Three-level training: Hour-SAC -> Minute-TD3 -> Second-DDPG

set -e

PROJECT_DIR="/mnt/d/ZYY Project/green-computing"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

echo "============================================"
echo "Step 1: Train Second-Level (GPU DVFS + Cooling)"
echo "============================================"
python -c "
import torch
import numpy as np
from src.training.trainer import HierarchicalRLTrainer

trainer = HierarchicalRLTrainer(n_gpus=8, device='cpu', seed=42)
print('Second-level training start...')
for ep in range(50):
    stats = trainer.run_episode(max_steps=3600)
    trainer._update_second(batch_size=256)
    print(f'[Second Ep {ep+1}/50] Power={stats[\"total_power\"]:.1f}kW PUE={stats[\"pue_avg\"]:.3f} Violations={stats[\"thermal_violations\"]}')
    if (ep+1) % 10 == 0:
        trainer.save('outputs/second_level.pt')
trainer.save('outputs/second_level.pt')
print('Second-level training complete!')
"

echo ""
echo "============================================"
echo "Step 2: Train Minute-Level (Workload Migration)"
echo "============================================"
python -c "
import torch
import os
from src.training.trainer import HierarchicalRLTrainer

trainer = HierarchicalRLTrainer(n_gpus=8, device='cpu', seed=42)
if os.path.exists('outputs/second_level.pt'):
    trainer.load('outputs/second_level.pt')
    print('Loaded second-level checkpoint')
print('Minute-level training start...')
for ep in range(50):
    stats = trainer.run_episode(max_steps=3600)
    if ep % 5 == 0:
        for _ in range(10):
            trainer._update_minute()
    print(f'[Minute Ep {ep+1}/50] Power={stats[\"total_power\"]:.1f}kW Carbon={stats[\"total_carbon\"]:.2f}g PUE={stats[\"pue_avg\"]:.3f}')
    if (ep+1) % 10 == 0:
        trainer.save('outputs/minute_level.pt')
trainer.save('outputs/minute_level.pt')
print('Minute-level training complete!')
"

echo ""
echo "============================================"
echo "Step 3: Train Hour-Level (Carbon Procurement)"
echo "============================================"
python -c "
import torch
import os
from src.training.trainer import HierarchicalRLTrainer

trainer = HierarchicalRLTrainer(n_gpus=8, device='cpu', seed=42)
if os.path.exists('outputs/minute_level.pt'):
    trainer.load('outputs/minute_level.pt')
    print('Loaded minute-level checkpoint')
print('Hour-level training start...')
for ep in range(100):
    stats = trainer.run_episode(max_steps=3600)
    if ep % 10 == 0:
        for _ in range(5):
            trainer._update_hour()
    print(f'[Hour Ep {ep+1}/100] Power={stats[\"total_power\"]:.1f}kW Carbon={stats[\"total_carbon\"]:.2f}g PUE={stats[\"pue_avg\"]:.3f} Violations={stats[\"thermal_violations\"]}')
    if (ep+1) % 20 == 0:
        trainer.save(f'outputs/full_hrl_ep{ep+1}.pt')
trainer.save('outputs/full_hrl_model.pt')
print('Full HRL training complete! Checkpoints saved to outputs/')
"

echo ""
echo "============================================"
echo "All training complete!"
echo "Checkpoints: outputs/second_level.pt, outputs/minute_level.pt, outputs/full_hrl_model.pt"
echo "============================================"

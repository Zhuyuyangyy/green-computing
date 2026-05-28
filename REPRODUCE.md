# REPRODUCE.md - green-computing

## Prerequisites

- **Python**: 3.10+
- **OS**: Linux / macOS / Windows
- **GPU**: Recommended for training (CUDA)

## Install

```bash
cd green-computing
pip install -r requirements.txt
```

Dependencies: torch, numpy, matplotlib, tensorboard

## Smoke Test

```bash
python main.py --mode demo
```

## Train

```bash
python main.py --mode train
# Or via script:
bash scripts/run_training.sh
```

## Expected Outputs

- Three-layer HRL: Hour-SAC -> Minute-TD3 -> Second-DDPG
- Training checkpoints: `outputs/checkpoint_ep{10,20,30,40,50}.pt`
- TensorBoard logs
- PINN thermal constraints (beta=0.1)

## Known Issues

- `scripts/run_training.sh` contains `/mnt/d/` hardcoded paths
- Training requires significant compute time
- Pre-trained checkpoints available in `outputs/`
- Paper in `papers/02_green_computing_carbon_aware_hrl/`

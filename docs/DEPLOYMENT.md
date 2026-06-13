# Deployment Guide

## Local Development

### Prerequisites

- Python 3.10+
- pip or conda
- (Optional) NVIDIA GPU with CUDA for training acceleration

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd green-computing

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run demo
python main.py demo
```

### Training

```bash
# Quick training (10 epochs)
python main.py train --epochs 10

# Full training pipeline
bash scripts/run_training.sh

# Step-by-step training
# 1. Train second-level (GPU DVFS + cooling)
python -c "
from src.training.trainer import HierarchicalRLTrainer
trainer = HierarchicalRLTrainer(n_gpus=8, device='cpu', seed=42)
for ep in range(50):
    stats = trainer.run_episode(max_steps=3600)
    trainer._update_second(batch_size=256)
trainer.save('outputs/second_level.pt')
"
```

---

## Docker Deployment

### Build Image

```bash
docker build -t green-computing .
```

### Run Container

```bash
# Demo mode
docker run --rm green-computing python main.py demo

# Training mode
docker run --rm green-computing python main.py train --epochs 50

# With GPU support
docker run --rm --gpus all green-computing python main.py train --epochs 50
```

### Docker Compose

```bash
# Start all services
docker-compose up --build

# Run training
docker-compose run green-computing python main.py train --epochs 50

# Run tests
docker-compose run green-computing pytest tests/ -v

# View TensorBoard
# Open http://localhost:6006 in browser
```

---

## Cloud Deployment

### AWS EC2

```bash
# Launch GPU instance (p3.2xlarge for V100)
# SSH into instance
ssh -i key.pem ubuntu@<instance-ip>

# Install Docker
sudo apt update && sudo apt install -y docker.io docker-compose
sudo usermod -aG docker $USER

# Clone and run
git clone <repo-url>
cd green-computing
docker-compose up --build
```

### Google Cloud Platform

```bash
# Create GPU instance
gcloud compute instances create green-computing \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-v100,count=1 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud

# SSH and deploy
gcloud compute ssh green-computing
# Follow Docker deployment steps above
```

### Kubernetes

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: green-computing
spec:
  replicas: 1
  selector:
    matchLabels:
      app: green-computing
  template:
    metadata:
      labels:
        app: green-computing
    spec:
      containers:
      - name: green-computing
        image: green-computing:latest
        command: ["python", "main.py", "train", "--epochs", "100"]
        resources:
          limits:
            nvidia.com/gpu: 1
```

---

## Monitoring

### TensorBoard

TensorBoard logs are saved to `outputs/logs/` by default.

```bash
# Start TensorBoard
tensorboard --logdir=outputs/logs --port=6006

# Access at http://localhost:6006
```

### Metrics

Key metrics to monitor during training:

- **PUE**: Target < 1.2 (lower is better)
- **Carbon Emissions**: gCO2/s trend
- **Thermal Violations**: Should decrease to 0
- **Reward**: Should increase over training
- **Actor/Critic Loss**: Should stabilize

---

## Troubleshooting

### Common Issues

**CUDA out of memory:**
- Reduce batch size in training
- Use CPU training: `device='cpu'`

**Import errors:**
- Ensure `PYTHONPATH` includes the project root
- Run from the project directory: `cd green-computing`

**Training too slow:**
- Reduce `max_steps` in `run_episode()`
- Use GPU: `device='cuda'`
- Reduce `n_gpus` for smaller environment

**Tests failing:**
- Run from project directory: `cd green-computing && pytest tests/ -v`
- Check Python version: `python --version` (need 3.10+)
- Install test dependencies: `pip install pytest pytest-cov`

import sys
sys.path.insert(0, 'src')
from training.trainer import HierarchicalRLTrainer

trainer = HierarchicalRLTrainer(n_gpus=8, seed=42)
stats = trainer.run_episode(max_steps=3600)
print('Episode stats:', stats)

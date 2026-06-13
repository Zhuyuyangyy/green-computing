#!/usr/bin/env python3
"""
Green Computing - 数据中心能效优化系统
=====================================
三层HRL协调器: Hour(SAC) -> Minute(TD3) -> Second(TD3)
论文公式对齐: GPU功率 / 冷却系统 / 热阻网络 / PUE / 碳排放

用法:
  python main.py demo              # 系统架构演示
  python main.py train --epochs 10 # 训练强化学习模型
  python main.py evaluate          # 评估PUE优化效果
"""
import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def print_banner():
    print("=" * 58)
    print("  Green Computing - 数据中心能效优化系统")
    print("  Hierarchical RL: Hour(SAC) → Minute(TD3) → Second(TD3)")
    print("  对齐论文公式: GPU功率 / 冷却 / 热阻 / PUE / 碳排放")
    print("=" * 58)
    print()


def demo_mode():
    """演示模式: 展示系统架构和算法"""
    print("[模式] 演示模式 (demo)")
    print()
    print("系统架构:")
    print("  [电力市场] ← Hour级战略决策 (SAC)")
    print("     ↓")
    print("  [负载均衡] ← Minute级战术决策 (TD3)")
    print("     ↓")
    print("  [GPU频率]   ← Second级操作控制 (TD3)")
    print("     ↓")
    print("  [冷却系统] ← PINN热模型约束")
    print()
    print("论文公式对齐:")
    print("  • GPU功率模型: Eq.(eq:gpu_power) - A100 400W TDP")
    print("  • 冷却系统: Eq.(eq:heat_transfer), Eq.(eq:chiller_power)")
    print("  • 热阻网络: Eq.(eq:thermal_network)")
    print("  • PUE: Eq.(eq:pue) - 目标 < 1.2")
    print("  • 碳排放: Eq.(eq:carbon_emission)")
    print()
    print("=" * 58)
    print("  DISCLAIMER: ALL RESULTS ARE SYNTHETIC")
    print("  The figures below are design targets from a simplified")
    print("  simulation, NOT measured outcomes from a trained policy")
    print("  or real-world deployment.")
    print("=" * 58)
    print()
    print("设计目标 (仿真估算, 非实测):")
    print("  • PUE: 1.45 → 1.18 (节能19%) — 设计目标")
    print("  • 碳排放: -23% (配合绿电调度) — 设计目标")
    print("  • GPU利用率: +15% (负载均衡) — 设计目标")
    print()
    print("=" * 58)
    print("  使用 --train 运行完整训练")
    print("=" * 58)


def train_mode(epochs):
    """训练模式

    DISCLAIMER: Training runs against a simplified simulation environment.
    All reported metrics (PUE, carbon, power) are synthetic — they come
    from the simulated data center, not from real hardware.  The trained
    policy has NOT been validated on actual data center infrastructure.
    """
    print(f"[模式] 训练模式 (epochs={epochs})")
    print("[注意] 所有训练指标来自简化仿真环境, 非真实硬件测量值。")
    print()

    try:
        import torch
        from src.training.trainer import HierarchicalRLTrainer

        print("[训练] 初始化三层HRL协调器...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[训练] 设备: {device}")

        trainer = HierarchicalRLTrainer(
            n_gpus=8,
            device=device,
            seed=42,
        )
        print("[训练] 模型初始化完成")

        print()
        print(f"[训练] 开始训练 {epochs} 个epoch...")
        for epoch in range(epochs):
            result = trainer.train_epoch(epoch)
            pue = result.get('pue', 0)
            carbon = result.get('carbon_kg', 0)
            print(f"  Epoch {epoch+1}/{epochs} | PUE={pue:.3f} | 碳排放={carbon:.1f}kg")

        print()
        print("[训练] 保存模型...")
        trainer.save(os.path.join(os.path.dirname(__file__), 'outputs', 'model.pt'))
        print("[训练] 完成!")

    except ImportError as e:
        print(f"[错误] 缺少依赖: {e}")
        print("[提示] pip install torch numpy")
    except Exception as e:
        print(f"[错误] 训练失败: {e}")


def evaluate_mode():
    """评估模式

    DISCLAIMER: All results below are SYNTHETIC.  The "optimized" PUE is
    computed via a hardcoded multiplier (baseline * 0.81), NOT by a trained
    policy.  This mode only demonstrates the simulation environment; it does
    NOT prove that the RL agent has learned an effective control strategy.
    Real-world energy savings require deployment on actual hardware with
    validated controllers.
    """
    print("[模式] 评估模式 (evaluate)")
    print()
    print("=" * 58)
    print("  DISCLAIMER: ALL RESULTS ARE SYNTHETIC")
    print("  The environment is a simplified simulation.")
    print("  The 'optimized' PUE below uses a hardcoded 19% reduction")
    print("  multiplier (baseline * 0.81), NOT a trained RL policy.")
    print("  No real energy savings have been demonstrated.")
    print("=" * 58)
    print()

    try:
        from src.env.datacenter_env import DataCenterEnv

        env = DataCenterEnv(n_gpus=8, seed=42)
        print(f"[评估] 环境初始化完成 | GPU数量: {env.n_gpus}")

        # 模拟无优化baseline
        state = env.reset()
        total_reward = 0
        for step in range(100):
            freq_action = np.full(env.n_gpus, 0.5)
            flow_action = np.full(env.n_gpus, 0.5)
            state, reward, done, info = env.step_second(freq_action, flow_action)
            total_reward += reward
            if done:
                break

        baseline_pue = info.get('pue', 1.45)
        print(f"[评估] Baseline PUE: {baseline_pue:.3f} (目标 < 1.2)")

        # HARDCODED multiplier — NOT a trained policy
        optimized_pue = baseline_pue * 0.81
        print(f"[评估] 优化后 PUE (硬编码估算): {optimized_pue:.3f}")
        print()
        print("[评估] 关键指标 (均为仿真估算值):")
        print(f"  • GPU功率优化: {(1-optimized_pue/baseline_pue)*100:.1f}% 节能 (估算)")
        print(f"  • 冷却系统: 热阻网络约束 (仿真环境)")
        print(f"  • 碳排放: 基于碳强度实时调度 (仿真环境)")
        print()
        print("[评估] 注意: 上述数值来自简化仿真, 不代表实际部署效果。")
        print("        需要在真实硬件上验证后才能确认有效性。")

    except ImportError as e:
        print(f"[错误] 缺少依赖: {e}")
    except Exception as e:
        print(f"[错误] 评估失败: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Green Computing - 数据中心能效优化系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py demo              # 系统架构演示
  python main.py train --epochs 10 # 训练HRL模型
  python main.py evaluate           # 评估PUE优化效果
        """,
    )
    parser.add_argument(
        'mode',
        nargs='?',
        default='demo',
        choices=['demo', 'train', 'evaluate'],
        help='运行模式',
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=10,
        help='训练轮数 (默认: 10)',
    )

    args = parser.parse_args()
    print_banner()

    if args.mode == 'demo':
        demo_mode()
    elif args.mode == 'train':
        train_mode(args.epochs)
    elif args.mode == 'evaluate':
        evaluate_mode()


if __name__ == '__main__':
    main()
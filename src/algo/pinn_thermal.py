"""
Physics-Informed Neural Network Thermal Model
对齐 papers/02/ method.tex Section 2.5: Eq.(eq:pinn_loss)

将热物理方程编码为PINN loss term:
L_total = L_RL + beta * L_PINN,  beta=0.1

论文公式:
- Eq.(eq:pinn_loss): 温度预测误差
- Eq.(eq:thermal_forward): 物理热模型
"""

import torch
import torch.nn as nn
import numpy as np


class PINNThermalModel(nn.Module):
    """
    PINN热模型: 将热阻网络物理方程编码为神经网络的额外约束

    物理方程 (Eq.(eq:thermal_forward)):
    T_i(t+1) = T_i(t) + (dt / C_th) * (P_i^GPU(t) - (T_i(t) - T_coolant) / (R_jc + R_sink))
    """

    def __init__(
        self,
        n_gpus: int = 8,
        c_thermal: float = 1000.0,   # J/K, 等效热容
        r_jc: float = 0.15,           # K/W
        r_sink: float = 0.08,          # K/W
        device: str = "cpu",
    ):
        super().__init__()
        self.n_gpus = n_gpus
        self.dt = 1.0  # second
        self.c_thermal = c_thermal
        self.r_jc = r_jc
        self.r_sink = r_sink
        self.device = device

        # MLP用于预测温度变化
        self.net = nn.Sequential(
            nn.Linear(n_gpus * 3, 128),  # 输入: [T_j, P_gpu, T_coolant]
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_gpus),
        )

    def physics_forward(
        self,
        t_j: torch.Tensor,      # (batch, n_gpus)
        p_gpu: torch.Tensor,      # (batch, n_gpus)
        t_coolant: float = 25.0,
    ) -> torch.Tensor:
        """
        物理前向传播 Eq.(eq:thermal_forward)
        """
        r_total = self.r_jc + self.r_sink
        heat_flow = (t_j - t_coolant) / r_total
        delta_t = self.dt / self.c_thermal * (p_gpu - heat_flow)
        return t_j + delta_t

    def pinn_loss(
        self,
        t_j_pred: torch.Tensor,  # 神经网络预测的温度
        t_j_true: torch.Tensor,  # 真实温度
        p_gpu: torch.Tensor,     # GPU功率
    ) -> torch.Tensor:
        """
        PINN loss: Eq.(eq:pinn_loss)
        L_PINN = (1/N) * ||T_hat(t+1) - f_thermal(T(t), P(t))||^2
        """
        t_j_physics = self.physics_forward(t_j_true, p_gpu)
        return torch.mean((t_j_pred - t_j_physics) ** 2)

    def forward(self, obs: torch.Tensor, p_gpu: torch.Tensor) -> torch.Tensor:
        """
        预测下一个温度
        obs: (batch, obs_dim) 包含当前温度
        p_gpu: (batch, n_gpus)
        """
        # 从obs中提取温度 (假设在位置 n_gpus:n_gpus*2)
        t_j_current = obs[:, self.n_gpus:self.n_gpus * 2]
        delta_t = self.net(torch.cat([t_j_current, p_gpu, torch.zeros_like(p_gpu)], dim=-1))
        return t_j_current + delta_t


class PINNLossCalculator:
    """
    计算总loss: L_total = L_RL + beta * L_PINN
    对应 method.tex Eq.(eq:total_loss), beta=0.1
    """

    def __init__(self, pinn_model: PINNThermalModel, beta: float = 0.1):
        self.pinn = pinn_model
        self.beta = beta

    def compute(
        self,
        rl_loss: float,
        obs: torch.Tensor,
        p_gpu: torch.Tensor,
        t_j_true: torch.Tensor,
        t_j_pred: torch.Tensor,
    ) -> tuple[float, float]:
        """
        返回 (total_loss, pinn_loss)
        """
        pinn_loss = self.pinn.pinn_loss(t_j_pred, t_j_true, p_gpu)
        total_loss = rl_loss + self.beta * pinn_loss
        return total_loss.item(), pinn_loss.item()

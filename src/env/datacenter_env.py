"""
Data Center Energy Optimization Environment
对齐 papers/02/ sections/system_model.tex 的全部公式

论文公式对应:
- GPU功率模型: Eq.(eq:gpu_power)
- 冷却系统: Eq.(eq:heat_transfer), Eq.(eq:chiller_power)
- 热阻网络: Eq.(eq:thermal_network)
- PUE: Eq.(eq:pue)
- 碳排放: Eq.(eq:carbon_emission)

DISCLAIMER: This is a SIMPLIFIED SIMULATION environment.  All physics
models use idealised parameters (constant thermal resistance, linear COP,
fixed workload distributions).  No real data center hardware is controlled.
Results produced by this environment are SYNTHETIC and should not be cited
as measured energy savings.
"""

import numpy as np
from typing import Dict, Tuple, Optional


class DataCenterEnv:
    """
    数据中心能效优化环境，三层时域抽象:
    - Hour-level:  3600s 战略决策 (电力采购/碳配额)
    - Minute-level: 60s  战术负载均衡
    - Second-level:  1s   操作级GPU频率+冷却控制
    """

    def __init__(
        self,
        n_gpus: int = 8,
        dt_hour: float = 3600.0,
        dt_minute: float = 60.0,
        dt_second: float = 1.0,
        seed: int = 42,
    ):
        self.n_gpus = n_gpus
        self.dt_h = dt_hour
        self.dt_m = dt_minute
        self.dt_s = dt_second

        # ── GPU A100 参数 (system_model.tex Section 2.1) ──────────────────
        self.p_idle = 50.0          # W, Eq.(eq:gpu_power)
        self.p_tdp = 400.0          # W, TDP
        self.f_max = 1.41          # GHz, Eq.(eq:gpu_power)
        self.f_min = 0.35          # GHz, DVFS下限
        self.alpha_power = 1.3     # Eq.(eq:gpu_power)

        # ── 冷却系统参数 (Section 2.2) ─────────────────────────────────
        self.c_coolant = 4182.0     # J/(kg·K), 比热容
        self.rho_coolant = 1050.0   # kg/m³, 密度
        self.t_coolant_in = 25.0    # °C, 进水温度

        # ── 热阻网络 (Section 2.3) ─────────────────────────────────────
        self.r_th_jc = 0.15         # K/W, junction-to-case
        self.r_th_sink = 0.08       # K/W, case-to-sink
        self.t_j_max = 85.0         # °C, 最大结温

        # ── PUE 参数 (Section 2.4) ─────────────────────────────────────
        self.p_loss_ratio = 0.04    # 4% 配电损耗

        # ── COP模型 (Eq.(eq:chiller_power)) ────────────────────────────
        # COP(T_amb) = 5.8 - 0.12 * T_amb

        # ── 内部状态 ───────────────────────────────────────────────────
        np.random.seed(seed)
        self._reset_internal()

    def _reset_internal(self):
        self.gpu_util = np.random.uniform(0.3, 0.8, self.n_gpus)  # 利用率
        self.gpu_freq = np.random.uniform(self.f_min, self.f_max, self.n_gpus)  # GHz
        self.flow_rate = np.random.uniform(0.5, 3.0, self.n_gpus)  # L/min
        self.t_junction = np.random.uniform(40.0, 65.0, self.n_gpus)  # °C
        self.t_outlet = self.t_junction - 5.0  # 出水温度

        # 外部信号
        self.ambient_temp = 25.0     # °C
        self.grid_carbon = 450.0     # gCO2/kWh
        self.electricity_price = 0.12  # $/kWh

        # 负载
        self.workload = np.random.uniform(0.4, 0.9, self.n_gpus)
        self.workload_predicted = self.workload.copy()

        # 时间步
        self.current_time = 0.0
        self.episode_step = 0

    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation. For Gym-compatible interface."""
        self._reset_internal()
        self.current_time = 0.0
        self.episode_step = 0
        return self.get_observation()

    # ── GPU功率模型 Eq.(eq:gpu_power) ──────────────────────────────────
    def compute_gpu_power(self, util: np.ndarray, freq: np.ndarray) -> np.ndarray:
        return (
            self.p_idle
            + (self.p_tdp - self.p_idle) * util * (freq / self.f_max) ** self.alpha_power
        )

    # ── 热传输 Eq.(eq:heat_transfer) ──────────────────────────────────
    def compute_heat_dissipation(
        self, flow_Lmin: np.ndarray, delta_T: np.ndarray
    ) -> np.ndarray:
        """Q_i = c * rho * Phi * dT, 单位W"""
        flow_m3s = flow_Lmin / 1000.0 / 60.0  # L/min → m³/s
        return self.c_coolant * self.rho_coolant * flow_m3s * delta_T

    def compute_delta_T(self, t_inlet: float, t_outlet: float) -> np.ndarray:
        return t_outlet - t_inlet * np.ones(self.n_gpus)

    # ── COP模型 Eq.(eq:chiller_power) ─────────────────────────────────
    def cop(self, t_ambient: float) -> float:
        return max(5.8 - 0.12 * t_ambient, 1.0)

    # ── Chiller功率 Eq.(eq:chiller_power) ─────────────────────────────
    def compute_chiller_power(self, total_heat: float, t_ambient: float) -> float:
        return total_heat / self.cop(t_ambient)

    # ── 热阻网络 Eq.(eq:thermal_network) ──────────────────────────────
    def compute_junction_temp(
        self, p_gpu: np.ndarray, t_coolant: float
    ) -> np.ndarray:
        """T_j = T_coolant + (R_jc + R_sink) * P_GPU"""
        return t_coolant + (self.r_th_jc + self.r_th_sink) * p_gpu

    # ── PUE Eq.(eq:pue) ───────────────────────────────────────────────
    def compute_pue(self, p_gpu_cluster: float, p_chiller: float) -> float:
        p_it = p_gpu_cluster
        p_total = p_gpu_cluster + p_chiller + self.p_loss_ratio * p_gpu_cluster
        return p_total / p_it if p_it > 0 else 1.0

    # ── 碳排放 Eq.(eq:carbon_emission) ───────────────────────────────
    def compute_carbon_rate(self, p_total: float, carbon_intensity: float) -> float:
        """gCO2/s = (kW) * (gCO2/kWh) / 3600"""
        return (p_total / 1000.0) * carbon_intensity / 3600.0

    # ── 观测向量构建 (method.tex Section 2.4 LSTM编码器) ─────────────
    def get_observation(self) -> np.ndarray:
        obs = np.concatenate([
            self.gpu_util,                  # N 维
            self.workload,                  # N 维
            self.t_junction,               # N 维
            np.array([self.t_coolant_in]), # 1 维
            self.flow_rate,                # N 维
            np.array([
                self.grid_carbon,          # 1 维
                self.electricity_price,     # 1 维
                np.mean(self.workload_predicted),  # 1 维
            ]),
        ])
        return obs.astype(np.float32)

    @property
    def observation_dim(self) -> int:
        """4N + 4, 对应 method.tex Section 2.4"""
        return 4 * self.n_gpus + 4

    # ── Second-level 动作: GPU频率 + 冷却流速 ─────────────────────────
    def step_second(
        self, freq_action: np.ndarray, flow_action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Second-level: 1s 操作控制
        freq_action:   n_gpus, 每个GPU的目标频率 (归一化 [0,1])
        flow_action:   n_gpus, 每个GPU的冷却流速 (归一化 [0,1])
        """
        # 动作映射
        self.gpu_freq = self.f_min + freq_action * (self.f_max - self.f_min)
        self.gpu_freq = np.clip(self.gpu_freq, self.f_min, self.f_max)

        self.flow_rate = 0.5 + flow_action * (5.0 - 0.5)  # [0.5, 5.0] L/min
        self.flow_rate = np.clip(self.flow_rate, 0.5, 5.0)

        # 更新利用率 (受负载驱动，有惯性)
        delta_util = 0.1 * (self.workload - self.gpu_util)
        self.gpu_util = np.clip(self.gpu_util + delta_util, 0.0, 1.0)

        # 功率计算 Eq.(eq:gpu_power)
        p_gpu = self.compute_gpu_power(self.gpu_util, self.gpu_freq)
        p_gpu_cluster = np.sum(p_gpu)

        # 热计算
        q_heat = self.compute_heat_dissipation(
            self.flow_rate,
            self.t_outlet - self.t_coolant_in * np.ones(self.n_gpus),
        )
        total_heat = np.sum(q_heat)

        # Chiller功率 Eq.(eq:chiller_power)
        p_chiller = self.compute_chiller_power(total_heat, self.ambient_temp)

        # 总功率
        p_total = p_gpu_cluster + p_chiller + self.p_loss_ratio * p_gpu_cluster

        # 结温更新 Eq.(eq:thermal_network) 离散化
        t_j_new = self.compute_junction_temp(p_gpu, self.t_coolant_in)
        self.t_junction = 0.9 * self.t_junction + 0.1 * t_j_new
        self.t_outlet = self.t_junction - 5.0

        # PUE
        pue = self.compute_pue(p_gpu_cluster, p_chiller)

        # Reward: 碳感知 Eq.(method.tex Eq.(eq:carbon_reward)) + 热惩罚 Eq.(eq:thermal_penalty)
        r_carbon = -p_total - 0.01 * self.grid_carbon * p_total / 1000.0
        thermal_penalty = -1e4 * np.sum(np.maximum(0, self.t_junction - self.t_j_max) ** 2)
        r = r_carbon + thermal_penalty

        # 更新
        self.current_time += self.dt_s
        self.episode_step += 1

        done = self.episode_step >= 3600  # 1小时 = 3600秒

        info = {
            "p_gpu_total": float(p_gpu_cluster),
            "p_chiller": float(p_chiller),
            "p_total": float(p_total),
            "pue": float(pue),
            "carbon_rate": float(self.compute_carbon_rate(p_total, self.grid_carbon)),
            "t_j_max": float(np.max(self.t_junction)),
            "thermal_violation": bool(np.any(self.t_junction > self.t_j_max)),
        }

        return self.get_observation(), r, done, info

    # ── Minute-level 动作: 负载迁移 ─────────────────────────────────
    def step_minute(
        self,
        migrate_action: np.ndarray,
        second_level_actions: Optional[list] = None,
    ) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Minute-level: 60s 负载均衡
        migrate_action: n_gpus, 迁移比例 [0,1]
        second_level_actions: list of (freq_action, flow_action) tuples for
            each sub-second step within this minute. If None, the caller is
            responsible for driving second-level steps externally.  When
            provided, its length determines how many sub-steps are executed
            (typically 60 for one-minute windows).

        NOTE: Previously this method generated random second-level actions
        internally, which broke the hierarchical structure.  The caller must
        now supply the agent's second-level actions explicitly.
        """
        # 简单负载迁移模型
        total_workload = np.sum(self.workload)
        delta = migrate_action * 0.1
        self.workload = np.clip(self.workload + delta, 0.1, 1.0)
        self.workload = self.workload / np.sum(self.workload) * total_workload

        if second_level_actions is not None:
            for freq_action, flow_action in second_level_actions:
                _, _, done_s, _ = self.step_second(freq_action, flow_action)
                if done_s:
                    break

        p_gpu = self.compute_gpu_power(self.gpu_util, self.gpu_freq)
        p_chiller = self.compute_chiller_power(
            np.sum(self.compute_heat_dissipation(
                self.flow_rate,
                self.t_junction - self.t_coolant_in * np.ones(self.n_gpus),
            )),
            self.ambient_temp,
        )
        p_total = np.sum(p_gpu) + p_chiller

        r = -(p_total + 0.01 * self.grid_carbon * p_total / 1000.0)
        thermal_penalty = -1e4 * np.sum(np.maximum(0, self.t_junction - self.t_j_max) ** 2)
        r += thermal_penalty

        done = self.episode_step >= 60  # minute-level episode

        info = {
            "p_gpu_total": float(np.sum(p_gpu)),
            "p_chiller": float(p_chiller),
            "p_total": float(p_total),
            "pue": float(self.compute_pue(np.sum(p_gpu), p_chiller)),
            "thermal_violation": bool(np.any(self.t_junction > self.t_j_max)),
        }

        return self.get_observation(), r, done, info

    # ── 环境扰动: 模拟真实碳强度波动 ───────────────────────────────
    def apply_environmental_disturbance(self):
        """每小时调用一次，模拟碳强度和温度变化"""
        hour = int(self.current_time / self.dt_h) % 24

        # 碳强度: 峰谷特性
        if 6 <= hour < 12:
            ci = np.random.uniform(400, 600)
        elif 12 <= hour < 18:
            ci = np.random.uniform(500, 700)
        elif 18 <= hour < 22:
            ci = np.random.uniform(450, 650)
        else:
            ci = np.random.uniform(300, 450)
        self.grid_carbon = ci

        # 温度扰动
        self.ambient_temp = np.random.uniform(20.0, 35.0)

        # 价格扰动
        self.electricity_price = np.random.uniform(0.08, 0.18)

        # 负载预测
        self.workload_predicted = np.random.uniform(0.3, 0.9, self.n_gpus)

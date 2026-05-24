"""
Shared LSTM Feature Extractor
对齐 papers/02/ method.tex Section 2.4 LSTM编码器

论文公式: Eq.(eq:lstm_encoder)
"""

import torch
import torch.nn as nn
from typing import Tuple


class LSTMEncoder(nn.Module):
    """
    三层LSTM编码器，将观测序列编码为共享隐含表征 z_t

    架构: LSTM1(256) -> LSTM2(128) -> LSTM3(64) -> Linear -> z_t
    对应 method.tex Eq.(eq:lstm_encoder):
        h_t^{(1)} = LSTM^{(1)}(o_t; h_{t-1}^{(1)})
        h_t^{(2)} = LSTM^{(2)}(h_t^{(1)}; h_{t-1}^{(2)})
        h_t^{(3)} = LSTM^{(3)}(h_t^{(2)}; h_{t-1}^{(3)})
        z_t = Linear(h_t^{(3)})
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dims: Tuple[int, int, int] = (256, 128, 64),
        latent_dim: int = 32,
    ):
        super().__init__()
        h1, h2, h3 = hidden_dims

        self.lstm1 = nn.LSTM(obs_dim, h1, batch_first=True)
        self.lstm2 = nn.LSTM(h1, h2, batch_first=True)
        self.lstm3 = nn.LSTM(h2, h3, batch_first=True)
        self.to_latent = nn.Linear(h3, latent_dim)

        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.obs_dim = obs_dim

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, Tuple]:
        """
        obs: (batch, seq_len, obs_dim) 或 (batch, obs_dim)
        returns: z_t, (h1, h2, h3) hidden states
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (batch, 1, obs_dim)

        h1, (h1_c, _) = self.lstm1(obs)
        h2, (h2_c, _) = self.lstm2(h1)
        h3, (h3_c, _) = self.lstm3(h2)

        # 取最后一个 timestep
        z_t = self.to_latent(h3[:, -1, :])

        return z_t, (h1_c, h2_c, h3_c)

    def get_flat_weights(self) -> int:
        """返回可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

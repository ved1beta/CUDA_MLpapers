"""
Short depthwise causal convolution with dilation.

kernel_size=4, dilation=max_ngram_order (e.g. 3)
Effective receptive field covers positions: t, t-3, t-6, t-9
This captures patterns spaced at N-gram boundaries while
remaining strictly causal (no future leakage).
"""

import torch
import torch.nn as nn


class ShortConv(nn.Module):
    """Dilated depthwise causal conv for local context expansion post-gating.

    Operates on [B, T, hc_mult, D] tensors with per-branch RMSNorm.
    Conv is depthwise (groups=total_channels) for minimal FLOPs.
    """

    def __init__(
        self,
        hidden_size: int,
        hc_mult: int = 4,
        kernel_size: int = 4,
        dilation: int = 3,
        norm_eps: float = 1e-5,
        activation: bool = True,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.has_activation = activation

        total_channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            in_channels=total_channels,
            out_channels=total_channels,
            kernel_size=kernel_size,
            groups=total_channels,
            bias=False,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation,
        )
        # Zero-init conv weights so Engram starts as identity at init
        nn.init.zeros_(self.conv.weight)

        self.norms = nn.ModuleList([
            nn.RMSNorm(hidden_size, eps=norm_eps)
            for _ in range(hc_mult)
        ])

        if self.has_activation:
            self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, hc_mult, D]
        Returns:
            [B, T, hc_mult, D]
        """
        B, T, G, C = x.shape

        # Per-branch RMSNorm then concat along channel dim
        normed = torch.cat(
            [self.norms[i](x[:, :, i, :]) for i in range(G)],
            dim=-1,
        )  # [B, T, G*C]

        # Conv expects [B, channels, T]
        y = self.conv(normed.transpose(1, 2))
        y = y[..., :T]  # trim causal padding

        if self.has_activation:
            y = self.act_fn(y)

        # Reshape back to [B, T, G, C]
        y = y.transpose(1, 2).reshape(B, T, G, C).contiguous()
        return y

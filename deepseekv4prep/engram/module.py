"""
Core Engram module: conditional memory via scalable N-gram lookup.

Pipeline per layer:
  1. Hash input_ids -> table indices       (deterministic, CPU or CUDA)
  2. Multi-head embedding gather            (sparse lookup)
  3. Shared value projection W_V            (engram_dim -> hidden_dim)
  4. Per-branch key projections {W_K^(m)}   (engram_dim -> hidden_dim)
  5. Context-aware scalar gating            (sigmoid of normalized dot product)
  6. Short dilated depthwise causal conv    (local receptive field expansion)
  7. Residual add to hidden states

The K and V projections can be fused into a single FP8 matmul in production.
Conv weights are zero-initialized so the module starts as identity.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .config import EngramConfig, BackboneConfig
from .hashing import NgramHashMapping
from .embedding import MultiHeadEmbedding
from .conv import ShortConv


class EngramModule(nn.Module):
    """Full Engram module for a single transformer layer position.

    Integrates with multi-branch (mHC) architectures: one shared embedding
    table and value projection, M separate key projections for branch-specific
    gating. The gating uses sqrt-abs activation from the official demo
    for gradient stability.
    """

    def __init__(
        self,
        engram_cfg: EngramConfig,
        backbone_cfg: BackboneConfig,
        layer_id: int,
        hash_mapping: NgramHashMapping,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.max_n = engram_cfg.max_ngram_size
        self.n_heads = engram_cfg.n_head_per_ngram
        self.hc_mult = backbone_cfg.hc_mult
        self.hidden_size = backbone_cfg.hidden_size
        self.hash_mapping = hash_mapping

        head_dim = engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram
        n_orders = engram_cfg.max_ngram_size - 1  # bigram + trigram
        total_heads = n_orders * self.n_heads

        # Shared multi-head embedding table (all orders + heads in one table)
        vocab_sizes = hash_mapping.flat_vocab_sizes(layer_id)
        self.multi_head_embedding = MultiHeadEmbedding(vocab_sizes, head_dim)

        # Engram hidden dim = n_orders * n_embed_per_ngram
        self.engram_hidden_size = n_orders * engram_cfg.n_embed_per_ngram
        D = backbone_cfg.hidden_size

        # Shared value projection: engram_hidden -> D
        self.proj_v = nn.Linear(self.engram_hidden_size, D, bias=False)

        # Per-branch key projections for gating
        self.proj_k = nn.ModuleList([
            nn.Linear(self.engram_hidden_size, D, bias=False)
            for _ in range(self.hc_mult)
        ])

        # Per-branch RMSNorm for query (hidden state) and key (memory)
        self.norm_q = nn.ModuleList([
            nn.RMSNorm(D) for _ in range(self.hc_mult)
        ])
        self.norm_k = nn.ModuleList([
            nn.RMSNorm(D) for _ in range(self.hc_mult)
        ])

        # Short causal conv for local context refinement
        self.short_conv = ShortConv(
            hidden_size=D,
            hc_mult=self.hc_mult,
            kernel_size=engram_cfg.kernel_size,
            dilation=engram_cfg.max_ngram_size,
        )

        self.inv_sqrt_d = 1.0 / math.sqrt(D)

    def _retrieve_embeddings(self, input_ids: np.ndarray) -> torch.Tensor:
        """Hash and gather N-gram embeddings.

        Args:
            input_ids: [B, T] raw token IDs (numpy).
        Returns:
            [B, T, engram_hidden_size] float tensor on appropriate device.
        """
        hash_ids = self.hash_mapping.hash(input_ids)
        hash_tensor = torch.from_numpy(hash_ids[self.layer_id])

        # Move to same device as embedding table
        device = self.multi_head_embedding.embedding.weight.device
        hash_tensor = hash_tensor.to(device)

        # [B, T, total_heads, head_dim]
        embs = self.multi_head_embedding(hash_tensor)
        # Flatten heads: [B, T, total_heads * head_dim] = [B, T, engram_hidden_size]
        return embs.flatten(start_dim=-2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: np.ndarray,
    ) -> torch.Tensor:
        """Compute Engram residual contribution.

        Args:
            hidden_states: [B, T, hc_mult, D] multi-branch hidden states.
            input_ids:     [B, T] raw token IDs (numpy int64).
        Returns:
            [B, T, hc_mult, D] residual to add to hidden_states.
        """
        # Phase 1: Retrieval
        e_t = self._retrieve_embeddings(input_ids)
        e_t = e_t.to(hidden_states.dtype)

        # Phase 2: Fusion
        # Shared value projection
        v_t = self.proj_v(e_t)  # [B, T, D]

        # Per-branch gating
        gates: list[torch.Tensor] = []
        for m in range(self.hc_mult):
            h_m = hidden_states[:, :, m, :]  # [B, T, D]
            k_m = self.proj_k[m](e_t)        # [B, T, D]

            q_norm = self.norm_q[m](h_m)
            k_norm = self.norm_k[m](k_m)

            # Scalar gate: dot product -> sqrt-abs -> sigmoid
            dot = (q_norm * k_norm).sum(dim=-1) * self.inv_sqrt_d  # [B, T]
            # sqrt-abs activation for gradient stability (from official demo)
            gate = dot.abs().clamp_min(1e-6).sqrt() * dot.sign()
            gate = gate.sigmoid().unsqueeze(-1)  # [B, T, 1]
            gates.append(gate)

        # Stack gates: [B, T, hc_mult, 1]
        gates_t = torch.stack(gates, dim=2)

        # Gated value: broadcast v_t across branches
        # v_t: [B, T, D] -> [B, T, 1, D]
        fused = gates_t * v_t.unsqueeze(2)  # [B, T, hc_mult, D]

        # Phase 3: Short conv refinement + residual
        output = fused + self.short_conv(fused)
        return output


class TransformerBlockWithEngram(nn.Module):
    """Transformer block where Engram is inserted BEFORE attention + MoE.

    Engram's residual is added first, then attention, then MoE.
    Attention and MoE are mocked here -- replace with MLA/DeepSeekMoE.
    """

    def __init__(
        self,
        backbone_cfg: BackboneConfig,
        layer_id: int,
        engram_module: Optional[EngramModule] = None,
    ):
        super().__init__()
        self.engram = engram_module
        # Mock attention and MoE (replace with real implementations)
        self.attn = lambda x: x
        self.moe = lambda x: x

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: np.ndarray,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, T, hc_mult, D]
            input_ids:     [B, T] numpy int64
        """
        if self.engram is not None:
            hidden_states = hidden_states + self.engram(hidden_states, input_ids)
        hidden_states = self.attn(hidden_states) + hidden_states
        hidden_states = self.moe(hidden_states) + hidden_states
        return hidden_states

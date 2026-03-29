"""
Python interface to Engram CUDA kernels with PyTorch autograd integration.

Falls back to pure-PyTorch implementations when the CUDA extension
is not compiled. The autograd Function handles both forward and
backward for the embedding lookup (sparse gradient accumulation).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch.autograd import Function

try:
    import engram_cuda

    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


# ── Autograd wrapper for multi-head N-gram lookup ───────────────────

class MultiHeadNgramLookupFn(Function):
    """Autograd-compatible fused hash + embedding gather.

    Forward:  compute hash IDs, gather embedding rows.
    Backward: sparse scatter-add gradients to embedding table.
    """

    @staticmethod
    def forward(
        ctx,
        compressed_ids: torch.Tensor,   # [B, T] int64
        embed_table: torch.Tensor,      # [total_vocab, D]
        multipliers: torch.Tensor,      # [max_n] int64
        head_primes: torch.Tensor,      # [total_heads] int64
        head_offsets: torch.Tensor,     # [total_heads] int64
        max_ngram_size: int,
        n_heads_per_ngram: int,
        pad_id: int,
    ) -> torch.Tensor:
        if CUDA_AVAILABLE and compressed_ids.is_cuda:
            output = engram_cuda.multi_head_ngram_lookup(
                compressed_ids, embed_table, multipliers,
                head_primes, head_offsets,
                max_ngram_size, n_heads_per_ngram, pad_id,
            )
        else:
            output = _pytorch_ngram_lookup(
                compressed_ids, embed_table, multipliers,
                head_primes, head_offsets,
                max_ngram_size, n_heads_per_ngram, pad_id,
            )

        # Save for backward
        n_orders = max_ngram_size - 1
        total_heads = n_orders * n_heads_per_ngram
        ctx.save_for_backward(embed_table)
        ctx.total_vocab = embed_table.size(0)

        # Recompute global indices for backward (cheaper than saving [B,T,H])
        ctx._compressed_ids = compressed_ids
        ctx._multipliers = multipliers
        ctx._head_primes = head_primes
        ctx._head_offsets = head_offsets
        ctx._max_ngram_size = max_ngram_size
        ctx._n_heads = n_heads_per_ngram
        ctx._pad_id = pad_id

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Recompute the global indices
        global_indices = _compute_global_indices(
            ctx._compressed_ids, ctx._multipliers,
            ctx._head_primes, ctx._head_offsets,
            ctx._max_ngram_size, ctx._n_heads, ctx._pad_id,
        )

        if CUDA_AVAILABLE and grad_output.is_cuda:
            grad_embed = engram_cuda.sparse_embed_backward(
                grad_output, global_indices, ctx.total_vocab
            )
        else:
            grad_embed = _pytorch_sparse_backward(
                grad_output, global_indices, ctx.total_vocab
            )

        return None, grad_embed, None, None, None, None, None, None


def _compute_global_indices(
    compressed_ids, multipliers, head_primes, head_offsets,
    max_ngram_size, n_heads, pad_id,
):
    """Recompute global embedding indices from hash parameters."""
    B, T = compressed_ids.shape
    n_orders = max_ngram_size - 1
    total_heads = n_orders * n_heads
    device = compressed_ids.device

    all_indices = []
    for n in range(2, max_ngram_size + 1):
        ngram_idx = n - 2
        mix = torch.zeros(B, T, dtype=torch.long, device=device)
        for k in range(n):
            pos_shift = k
            if pos_shift > 0:
                pad = torch.full(
                    (B, pos_shift), pad_id, dtype=torch.long, device=device
                )
                tokens = torch.cat(
                    [pad, compressed_ids[:, : T - pos_shift]], dim=1
                )
            else:
                tokens = compressed_ids
            mult = multipliers[k].item()
            mix = mix ^ (tokens * mult)

        for h in range(n_heads):
            head_idx = ngram_idx * n_heads + h
            prime = head_primes[head_idx].item()
            offset = head_offsets[head_idx].item()
            local_ids = mix % prime
            local_ids = torch.where(local_ids < 0, local_ids + prime, local_ids)
            global_ids = offset + local_ids
            all_indices.append(global_ids)

    return torch.stack(all_indices, dim=2)  # [B, T, total_heads]


def _pytorch_ngram_lookup(
    compressed_ids, embed_table, multipliers, head_primes, head_offsets,
    max_ngram_size, n_heads, pad_id,
):
    """Pure PyTorch fallback for multi-head N-gram lookup."""
    global_indices = _compute_global_indices(
        compressed_ids, multipliers, head_primes, head_offsets,
        max_ngram_size, n_heads, pad_id,
    )
    B, T, H = global_indices.shape
    D = embed_table.size(1)
    flat = global_indices.reshape(-1)
    embs = embed_table[flat]
    return embs.reshape(B, T, H, D)


def _pytorch_sparse_backward(grad_output, global_indices, total_vocab):
    """Pure PyTorch sparse gradient scatter-add."""
    D = grad_output.size(-1)
    flat_grad = grad_output.reshape(-1, D)
    flat_idx = global_indices.reshape(-1)
    grad_embed = torch.zeros(
        total_vocab, D, dtype=flat_grad.dtype, device=flat_grad.device
    )
    grad_embed.index_add_(0, flat_idx, flat_grad)
    return grad_embed


# ── Fused gating wrapper ────────────────────────────────────────────

def fused_gate(
    e_t: torch.Tensor,
    h_t: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
    norm_q_weight: torch.Tensor,
    norm_k_weight: torch.Tensor,
    inv_sqrt_d: float,
    norm_eps: float = 1e-5,
) -> torch.Tensor:
    """Fused gating operation with CUDA kernel or PyTorch fallback.

    Args:
        e_t: [B, T, E] engram embeddings
        h_t: [B, T, HC, D] hidden states
        W_K: [HC, D, E] per-branch key weights
        W_V: [D, E] shared value weights
        norm_q_weight: [HC, D] RMSNorm query scales
        norm_k_weight: [HC, D] RMSNorm key scales
    Returns:
        [B, T, HC, D] gated output
    """
    if CUDA_AVAILABLE and e_t.is_cuda:
        return engram_cuda.fused_engram_gate(
            e_t, h_t, W_K, W_V,
            norm_q_weight, norm_k_weight,
            inv_sqrt_d, norm_eps,
        )
    return _pytorch_fused_gate(
        e_t, h_t, W_K, W_V,
        norm_q_weight, norm_k_weight,
        inv_sqrt_d, norm_eps,
    )


def _pytorch_fused_gate(
    e_t, h_t, W_K, W_V,
    norm_q_weight, norm_k_weight,
    inv_sqrt_d, norm_eps,
):
    """Pure PyTorch fallback for fused gating."""
    B, T, HC, D = h_t.shape
    E = e_t.size(2)

    # V = W_V @ e_t: [B, T, D]
    v_t = torch.einsum("bte,de->btd", e_t, W_V)

    outputs = []
    for m in range(HC):
        h_m = h_t[:, :, m, :]  # [B, T, D]
        k_m = torch.einsum("bte,de->btd", e_t, W_K[m])  # [B, T, D]

        # RMSNorm
        q_rms = torch.rsqrt(h_m.pow(2).mean(-1, keepdim=True) + norm_eps)
        k_rms = torch.rsqrt(k_m.pow(2).mean(-1, keepdim=True) + norm_eps)
        q_norm = h_m * q_rms * norm_q_weight[m].unsqueeze(0).unsqueeze(0)
        k_norm = k_m * k_rms * norm_k_weight[m].unsqueeze(0).unsqueeze(0)

        dot = (q_norm * k_norm).sum(-1) * inv_sqrt_d  # [B, T]
        gate = dot.abs().clamp_min(1e-6).sqrt() * dot.sign()
        alpha = gate.sigmoid().unsqueeze(-1)  # [B, T, 1]

        outputs.append(alpha * v_t)

    return torch.stack(outputs, dim=2)  # [B, T, HC, D]

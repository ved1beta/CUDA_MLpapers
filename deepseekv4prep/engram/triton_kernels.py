"""
Triton kernel implementations for Engram operations.

These provide a more portable alternative to the raw CUDA kernels in csrc/.
Triton auto-tunes block sizes and handles memory coalescing automatically.

Kernels:
  1. fused_ngram_hash_lookup:  hash computation + embedding gather
  2. fused_gate_kernel:        K-proj + RMSNorm + gate + V-scaling
  3. sparse_embed_grad:        gradient scatter-add for embeddings
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════
# Kernel 1: Fused N-gram Hash + Embedding Lookup
# ═══════════════════════════════════════════════════════════════════

@triton.jit
def _ngram_hash_lookup_kernel(
    ids_ptr,          # [B, T] int64
    embed_ptr,        # [total_vocab, D]
    mults_ptr,        # [max_ngram_size] int64
    primes_ptr,       # [total_heads] int64
    offsets_ptr,      # [total_heads] int64
    out_ptr,          # [B, T, total_heads, D]
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    max_ngram_size: tl.constexpr,
    n_heads: tl.constexpr,
    pad_id: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Program ID: (batch, position, head_flat)
    pid_bt = tl.program_id(0)
    pid_h = tl.program_id(1)

    b = pid_bt // T
    t = pid_bt % T

    n_orders = max_ngram_size - 1
    total_heads = n_orders * n_heads

    if pid_h >= total_heads:
        return

    order_idx = pid_h // n_heads
    ngram_order = order_idx + 2

    # Load multipliers
    mult_0 = tl.load(mults_ptr + 0)
    mult_1 = tl.load(mults_ptr + 1)
    mult_2 = tl.load(mults_ptr + 2) if max_ngram_size >= 3 else tl.cast(0, tl.int64)

    # Compute XOR hash
    # token at position t-k
    tok0 = tl.load(ids_ptr + b * T + t) if t >= 0 else tl.cast(pad_id, tl.int64)
    tok1_pos = t - 1
    tok1 = tl.load(ids_ptr + b * T + tok1_pos) if tok1_pos >= 0 else tl.cast(pad_id, tl.int64)

    mix = tok0 * mult_0
    mix = mix ^ (tok1 * mult_1)

    if ngram_order >= 3:
        tok2_pos = t - 2
        tok2 = tl.load(ids_ptr + b * T + tok2_pos) if tok2_pos >= 0 else tl.cast(pad_id, tl.int64)
        mix = mix ^ (tok2 * mult_2)

    prime = tl.load(primes_ptr + pid_h)
    offset = tl.load(offsets_ptr + pid_h)
    local_id = mix % prime
    # Ensure positive
    local_id = tl.where(local_id < 0, local_id + prime, local_id)
    global_id = offset + local_id

    # Gather embedding row
    d_range = tl.arange(0, BLOCK_D)
    mask = d_range < D

    src_ptrs = embed_ptr + global_id * D + d_range
    emb_vals = tl.load(src_ptrs, mask=mask, other=0.0)

    out_base = ((b * T + t) * total_heads + pid_h) * D
    dst_ptrs = out_ptr + out_base + d_range
    tl.store(dst_ptrs, emb_vals, mask=mask)


def triton_ngram_hash_lookup(
    compressed_ids: torch.Tensor,   # [B, T] int64
    embed_table: torch.Tensor,      # [total_vocab, D]
    multipliers: torch.Tensor,      # [max_ngram_size] int64
    head_primes: torch.Tensor,      # [total_heads] int64
    head_offsets: torch.Tensor,     # [total_heads] int64
    max_ngram_size: int,
    n_heads_per_ngram: int,
    pad_id: int,
) -> torch.Tensor:
    """Triton implementation of fused hash + lookup."""
    B, T = compressed_ids.shape
    D = embed_table.size(1)
    n_orders = max_ngram_size - 1
    total_heads = n_orders * n_heads_per_ngram

    output = torch.empty(
        B, T, total_heads, D,
        dtype=embed_table.dtype, device=embed_table.device,
    )

    BLOCK_D = triton.next_power_of_2(D)

    grid = (B * T, total_heads)
    _ngram_hash_lookup_kernel[grid](
        compressed_ids, embed_table, multipliers,
        head_primes, head_offsets, output,
        B=B, T=T, D=D,
        max_ngram_size=max_ngram_size,
        n_heads=n_heads_per_ngram,
        pad_id=pad_id,
        BLOCK_D=BLOCK_D,
    )
    return output


# ═══════════════════════════════════════════════════════════════════
# Kernel 2: Fused Gating (K-proj + RMSNorm + gate + V-scaling)
# ═══════════════════════════════════════════════════════════════════

@triton.jit
def _fused_gate_kernel(
    e_ptr,              # [B, T, E]
    h_ptr,              # [B, T, HC, D]
    wk_ptr,             # [HC, D, E]
    wv_ptr,             # [D, E]
    nq_ptr,             # [HC, D]
    nk_ptr,             # [HC, D]
    out_ptr,            # [B, T, HC, D]
    inv_sqrt_d: tl.constexpr,
    norm_eps: tl.constexpr,
    B: tl.constexpr,
    T: tl.constexpr,
    HC: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,
):
    pid = tl.program_id(0)
    total = B * T * HC
    if pid >= total:
        return

    b = pid // (T * HC)
    rem = pid % (T * HC)
    t = rem // HC
    hc = rem % HC

    d_range = tl.arange(0, D)

    # Load e_t[b, t, :]
    e_base = (b * T + t) * E

    # Compute V = W_V @ e_t (accumulate over E dimension)
    v_val = tl.zeros([D], dtype=tl.float32)
    for e_start in range(0, E, BLOCK_E):
        e_range = e_start + tl.arange(0, BLOCK_E)
        e_mask = e_range < E
        e_vals = tl.load(e_ptr + e_base + e_range, mask=e_mask, other=0.0).to(tl.float32)

        # wv[d, e] for each d in d_range
        for d_idx in range(D):
            wv_val = tl.load(wv_ptr + d_idx * E + e_range, mask=e_mask, other=0.0).to(tl.float32)
            v_val = tl.where(d_range == d_idx, v_val + tl.sum(wv_val * e_vals), v_val)

    # Compute K = W_K[hc] @ e_t
    k_val = tl.zeros([D], dtype=tl.float32)
    for e_start in range(0, E, BLOCK_E):
        e_range = e_start + tl.arange(0, BLOCK_E)
        e_mask = e_range < E
        e_vals = tl.load(e_ptr + e_base + e_range, mask=e_mask, other=0.0).to(tl.float32)

        for d_idx in range(D):
            wk_val = tl.load(wk_ptr + (hc * D + d_idx) * E + e_range, mask=e_mask, other=0.0).to(tl.float32)
            k_val = tl.where(d_range == d_idx, k_val + tl.sum(wk_val * e_vals), k_val)

    # Load h_t[b, t, hc, :]
    h_base = ((b * T + t) * HC + hc) * D
    h_val = tl.load(h_ptr + h_base + d_range).to(tl.float32)

    # RMSNorm for q and k
    sq_h = tl.sum(h_val * h_val) / D
    sq_k = tl.sum(k_val * k_val) / D
    rms_q = 1.0 / tl.sqrt(sq_h + norm_eps)
    rms_k = 1.0 / tl.sqrt(sq_k + norm_eps)

    nq_val = tl.load(nq_ptr + hc * D + d_range).to(tl.float32)
    nk_val = tl.load(nk_ptr + hc * D + d_range).to(tl.float32)

    q_norm = h_val * rms_q * nq_val
    k_norm = k_val * rms_k * nk_val

    # Dot product -> sqrt-abs -> sigmoid
    dot = tl.sum(q_norm * k_norm) * inv_sqrt_d
    sign = tl.where(dot >= 0.0, 1.0, -1.0)
    abs_dot = tl.abs(dot)
    abs_dot = tl.maximum(abs_dot, 1e-6)
    gate_input = sign * tl.sqrt(abs_dot)
    alpha = tl.sigmoid(gate_input)

    # Output = alpha * V
    out_val = (alpha * v_val).to(out_ptr.dtype.element_ty)
    out_base = ((b * T + t) * HC + hc) * D
    tl.store(out_ptr + out_base + d_range, out_val)


def triton_fused_gate(
    e_t: torch.Tensor,
    h_t: torch.Tensor,
    W_K: torch.Tensor,
    W_V: torch.Tensor,
    norm_q_weight: torch.Tensor,
    norm_k_weight: torch.Tensor,
    inv_sqrt_d: float,
    norm_eps: float = 1e-5,
) -> torch.Tensor:
    """Triton implementation of fused gating."""
    B, T, HC, D = h_t.shape
    E = e_t.size(2)

    output = torch.empty_like(h_t)
    BLOCK_E = min(triton.next_power_of_2(E), 128)

    grid = (B * T * HC,)
    _fused_gate_kernel[grid](
        e_t, h_t, W_K, W_V,
        norm_q_weight, norm_k_weight, output,
        inv_sqrt_d=inv_sqrt_d,
        norm_eps=norm_eps,
        B=B, T=T, HC=HC, D=D, E=E,
        BLOCK_E=BLOCK_E,
    )
    return output


# ═══════════════════════════════════════════════════════════════════
# Kernel 3: Sparse Embedding Gradient Scatter-Add
# ═══════════════════════════════════════════════════════════════════

@triton.jit
def _sparse_embed_grad_kernel(
    grad_ptr,       # [N, D]
    idx_ptr,        # [N]
    out_ptr,        # [vocab, D]
    N: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= N:
        return

    row = tl.load(idx_ptr + pid)

    d_range = tl.arange(0, BLOCK_D)
    mask = d_range < D

    grad_vals = tl.load(grad_ptr + pid * D + d_range, mask=mask, other=0.0)
    tl.atomic_add(out_ptr + row * D + d_range, grad_vals, mask=mask)


def triton_sparse_embed_grad(
    grad_output: torch.Tensor,
    indices: torch.Tensor,
    total_vocab_size: int,
) -> torch.Tensor:
    """Triton implementation of sparse gradient scatter-add."""
    flat_grad = grad_output.reshape(-1, grad_output.size(-1)).contiguous()
    flat_idx = indices.reshape(-1).contiguous()
    N, D = flat_grad.shape

    grad_embed = torch.zeros(
        total_vocab_size, D,
        dtype=flat_grad.dtype, device=flat_grad.device,
    )

    if N == 0:
        return grad_embed

    BLOCK_D = triton.next_power_of_2(D)
    grid = (N,)
    _sparse_embed_grad_kernel[grid](
        flat_grad, flat_idx, grad_embed,
        N=N, D=D, BLOCK_D=BLOCK_D,
    )
    return grad_embed

#pragma once

#include <torch/extension.h>

// ── Kernel 1: Multi-Head N-gram Hash + Lookup ──────────────────────
// Fuses hash computation with embedding gather.
// One thread block per (batch, position); threads across heads.
torch::Tensor multi_head_ngram_lookup(
    torch::Tensor compressed_ids,   // [B, T] int64
    torch::Tensor embed_table,      // [total_vocab, D] bf16/fp16/fp32
    torch::Tensor multipliers,      // [max_ngram_size] int64
    torch::Tensor head_primes,      // [total_heads] int64
    torch::Tensor head_offsets,     // [total_heads] int64
    int64_t max_ngram_size,
    int64_t n_heads_per_ngram,
    int64_t pad_id
);

// ── Kernel 2: Fused Gating ─────────────────────────────────────────
// Fuses: K-projection, RMSNorm, dot product, sqrt-abs, sigmoid, V-scaling.
// Eliminates multiple HBM round-trips.
torch::Tensor fused_engram_gate(
    torch::Tensor e_t,              // [B, T, E] engram embeddings
    torch::Tensor h_t,              // [B, T, HC, D] hidden states
    torch::Tensor W_K,              // [HC, D, E] per-branch key projections
    torch::Tensor W_V,              // [D, E] shared value projection
    torch::Tensor norm_q_weight,    // [HC, D] RMSNorm query weights
    torch::Tensor norm_k_weight,    // [HC, D] RMSNorm key weights
    float inv_sqrt_d,
    float norm_eps
);

// ── Kernel 3: Sparse Embedding Gradient Accumulation ───────────────
// Backward pass: scatter-add gradients back to embedding rows.
// Uses atomicAdd for thread-safe accumulation.
torch::Tensor sparse_embed_backward(
    torch::Tensor grad_output,      // [B, T, total_heads, D]
    torch::Tensor indices,          // [B, T, total_heads] global indices
    int64_t total_vocab_size
);

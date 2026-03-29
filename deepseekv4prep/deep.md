# Engram & DeepSeek-V3.2: From Foundations to Distributed CUDA Implementation

---

## Table of Contents

1. [Foundational Concepts](#1-foundational-concepts)
2. [Engram Architecture — Deep Dive](#2-engram-architecture)
3. [DeepSeek-V3.2 / DSA Architecture](#3-deepseek-v32--dsa)
4. [Reference Python Implementation (Engram)](#4-reference-python-implementation)
5. [Distributed Training Design](#5-distributed-training)
6. [Inference Offloading & Prefetching](#6-inference-offloading)
7. [CUDA Kernels](#7-cuda-kernels)
8. [Putting It All Together — System Architecture](#8-system-architecture)

---

## 1. Foundational Concepts

### 1.1 The Two Tasks of Language Modeling

Language modeling does **two qualitatively different things**:

| Task | Nature | Example | Ideal Mechanism |
|------|--------|---------|-----------------|
| Compositional Reasoning | Dynamic, context-dependent | "Is this argument valid?" | Deep MLP / Attention |
| Knowledge Retrieval | Static, stereotyped | "Paris is in France" | Lookup table |

Standard Transformers have **only one mechanism** — computation — and are forced to simulate both. This wastes depth and attention capacity on tasks that are essentially O(1) lookups.

### 1.2 Sparsity in Neural Networks

**Dense model**: every parameter fires for every token. Expensive but simple.

**Mixture-of-Experts (MoE)**: *conditional computation* — for each token, a router picks top-k experts out of N total. You get a large model (N×expert_params) but only pay k-expert FLOPs. The key equation:

```
P_total  = P_dense + N_experts * P_per_expert
P_active = P_dense + k * P_per_expert       (k << N)
FLOPs    ∝ P_active   (not P_total)
```

The **sparsity ratio** = P_total / P_active. DeepSeek uses ~10x.

**Engram's insight**: MoE solves "activate less compute". But there's a second axis — **activate less *memory*** by replacing computation with lookup. Engram calls this *conditional memory*.

```
Sparsity Axis 1: Conditional Computation (MoE)    → select which FFN experts run
Sparsity Axis 2: Conditional Memory    (Engram)   → select which embedding rows to read
```

### 1.3 N-gram Models — The Ancestor

An N-gram model assigns probability to token sequences based on the previous N-1 tokens:

```
P(x_t | x_{t-1}, ..., x_1) ≈ P(x_t | x_{t-N+1}, ..., x_{t-1})
```

Key property: the *key* into this table is just the preceding N-1 tokens — completely **deterministic**, no routing network needed.

Classical N-gram tables are enormous (1-5 gram tables for Web text can be terabytes) but access is O(1) hash lookup.

### 1.4 Hash Embeddings — Bridging N-grams and Neural Networks

Instead of storing one float per N-gram count, store a **dense embedding vector** per N-gram. The address space is still too large to parameterize directly, so use hashing:

```
id = hash(ngram) % table_size
embedding = E[id]
```

Hash **collision** (two different N-grams → same id) is a known problem. Mitigation: **multi-head hashing** — use K independent hash functions. Each head produces an independent embedding; concatenate or sum them.

```
For head k:  id_k = hash_k(ngram) % M_k
e_k = E_k[id_k]
final_embedding = concat(e_1, e_2, ..., e_K)
```

Collision probability ≈ 1/M_k per head. With K=8 heads, catastrophic collisions on all heads simultaneously are extremely rare.

### 1.5 Tokenizer Compression

Standard subword tokenizers treat `Apple` and `apple` as different tokens (different IDs). For N-gram hashing, this is wasteful — they should map to the same key.

The solution is a **vocabulary projection** P: V → V' that collapses:
- Casing variants: `Apple`, `APPLE`, `apple` → same canonical ID
- Unicode normalization (NFKC): `ﬁ` → `fi`  
- Accent stripping: `café` → `cafe`

Result: ~23% reduction in effective vocabulary for 128k tokenizer. More N-gram slots map to actually-seen patterns rather than superficial spelling variants.

---

## 2. Engram Architecture

### 2.1 Overview

```
Input Hidden State H^(l) ∈ R^{T×d}
         │
    ┌────▼─────────────────────────────────────┐
    │              ENGRAM MODULE               │
    │                                          │
    │  Phase 1: RETRIEVAL                      │
    │  ┌─────────────────────────────────┐    │
    │  │ Tokenizer Compression           │    │
    │  │ input_ids → compressed_ids      │    │
    │  │                                 │    │
    │  │ Multi-Head Hashing              │    │
    │  │ for n in [2,3]:                 │    │
    │  │   for k in [1..K]:              │    │
    │  │     id = XOR_hash(ngram, k)     │    │
    │  │     e_nk = E_nk[id]             │    │
    │  │ e_t = concat(all e_nk)          │    │
    │  └─────────────────────────────────┘    │
    │                                          │
    │  Phase 2: FUSION                         │
    │  ┌─────────────────────────────────┐    │
    │  │ Context-Aware Gating            │    │
    │  │ k_t = W_K @ e_t                 │    │
    │  │ v_t = W_V @ e_t                 │    │
    │  │ α_t = σ(RMSNorm(h_t)·RMSNorm(k_t)/√d) │
    │  │ ṽ_t = α_t * v_t                 │    │
    │  │                                 │    │
    │  │ Short Depthwise Conv            │    │
    │  │ Y = SiLU(Conv1D(RMSNorm(Ṽ))) + Ṽ │  │
    │  └─────────────────────────────────┘    │
    └────────────────┬─────────────────────────┘
                     │ residual add
    H^(l) ← H^(l) + Y
```

### 2.2 Multi-Head Hashing — The Math

The hash function uses multiplicative XOR mixing:

```python
# For N-gram (x'_{t-n+1}, ..., x'_t) with multiplier vector M:
mix = x'_{t-n+1} * M[0]
mix = mix XOR (x'_{t-n+2} * M[1])
mix = mix XOR (x'_{t-n+3} * M[2])  # etc.
id  = mix % prime_table_size_k      # different prime per head
```

Why this works:
- XOR mixes bits non-linearly, giving good hash distribution
- Different multiplier M and prime per head → statistically independent hashes
- Deterministic → can be computed from token IDs alone (enables prefetching)

### 2.3 Context-Aware Gating — The Math

The retrieved embedding `e_t` is a *prior* — it doesn't know about surrounding context.
The gate decides how much to trust it:

```
k_t = W_K @ e_t                          # project to key space
v_t = W_V @ e_t                          # project to value space

α_t = σ( RMSNorm(h_t)ᵀ RMSNorm(k_t) / √d )   # scalar gate ∈ (0,1)

ṽ_t = α_t * v_t                          # modulated value
```

If the retrieved N-gram `e_t` *contradicts* the current hidden state `h_t`, the dot product will be small/negative → σ(.) near 0 → gate suppresses the memory.

This is analogous to cross-attention but with a scalar output gate rather than softmax over multiple keys.

### 2.4 Multi-Branch (mHC) Integration

DeepSeek uses Manifold-Constrained Hyper-Connections (mHC), which expands the residual stream into M=4 parallel branches. Engram adapts via parameter sharing:

- **Shared**: one embedding table E, one value projection W_V (expensive)
- **Per-branch**: M separate key projections {W_K^(m)}  

```python
# For branch m with hidden state h_t^(m):
α_t^(m) = σ( RMSNorm(h_t^(m))ᵀ RMSNorm(W_K^(m) @ e_t) / √d )
u_t^(m) = α_t^(m) * (W_V @ e_t)    # shared W_V, branch-specific gate
```

The K projections + V projection can be fused into a single FP8 matmul:

```python
# Fuse [W_V; W_K^1; W_K^2; W_K^3; W_K^4] into one matrix multiply
combined = e_t @ W_fused.T   # one matmul
v_t      = combined[:, :d]
k_t_list = [combined[:, d + m*d : d + (m+1)*d] for m in range(M)]
```

### 2.5 Short Depthwise Causal Convolution

After gating, a tiny causal conv expands receptive field:

```python
# kernel_size=4, dilation=max_ngram_order (e.g. 3)
# Effective receptive field: positions t, t-3, t-6, t-9
Y = SiLU(DepthwiseConv1D(RMSNorm(Ṽ), kernel=4, dilation=3)) + Ṽ
```

Why causal + dilated: stays auto-regressive (no future leakage) while capturing patterns spaced by N-gram order. The dilation of 3 means positions corresponding to N-gram boundaries are in view.

### 2.6 Sparsity Allocation Law

Define:
- `ρ` = fraction of inactive parameters given to MoE experts (0–1)
- `1-ρ` = fraction given to Engram embeddings

The paper finds a **U-shaped curve** in validation loss vs ρ, with the optimum at ρ≈0.75–0.80 (give 20-25% of the sparse budget to Engram).

Intuition:
- ρ=1.0 (pure MoE): no dedicated memory → backbone wastes early layers on static pattern reconstruction
- ρ=0.0 (pure Engram): no dynamic routing → can't reason about novel compositional inputs
- ρ=0.75: best of both — memory handles statics, MoE handles dynamics

---

## 3. DeepSeek-V3.2 / DSA

### 3.1 The Attention Bottleneck

Standard attention is O(L²) in both compute and memory. At L=128K tokens:
- 128K² attention scores per head = 16.4 billion values
- KV cache alone is enormous even with MLA compression

### 3.2 DeepSeek Sparse Attention (DSA)

DSA has two components:

**Lightning Indexer** — a cheap scoring function:
```
I_{t,s} = Σ_j  w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s)
```
- H_I heads (much fewer than main attention)
- Implemented in FP8 for throughput
- ReLU instead of softmax → sparse-friendly (many zero scores)
- Computes which past tokens are *relevant* to query token t

**Fine-grained Token Selection** — use the indexer to pick top-k keys:
```
S_t = {s : I_{t,s} ∈ Top-k(I_{t,:})}
u_t = Attention(h_t, {c_s}_{s ∈ S_t})
```
Only k=2048 tokens computed in full attention instead of L=128K.

Complexity: O(L² * cheap_indexer) + O(L * k * full_attention) → effectively O(L·k) for the expensive part.

### 3.3 Training DSA: Two Stages

**Stage 1 — Dense Warmup** (1000 steps, 2.1B tokens):
- Freeze main model; only train lightning indexer
- Loss: KL(main_attention_distribution || softmax(indexer_scores))
- Teaches the indexer to *mimic* where dense attention attends

**Stage 2 — Sparse Fine-tuning** (15000 steps, 943.7B tokens):
- Enable sparse selection; unfreeze all parameters
- Indexer loss only over selected set S_t
- Main model optimizes language modeling loss only
- Indexer gradients detached from main model

### 3.4 DSA + GRPO: Scaling RL

Key stabilization tricks for large-scale RL on MoE + DSA:

**Unbiased KL Estimator**: correct the standard K3 estimator with importance sampling ratio between current policy π_θ and sampling policy π_old. Prevents large unbounded gradients when π_θ << π_ref.

**Off-Policy Sequence Masking**: for negative-advantage sequences where KL(π_old || π_θ) > δ, mask out the GRPO loss entirely. Avoids training on stale samples that could destabilize optimization.

**Keep Routing**: during MoE RL training, save expert routing decisions made at inference time and *enforce identical routing* during the backward pass. Prevents different routing between forward/backward → no abrupt parameter subspace shifts.

**Keep Sampling Mask**: save the top-p/top-k truncation masks from sampling and apply to π_θ during training. Both policies share identical action subspaces → importance sampling is valid.

---

## 4. Reference Python Implementation

```python
"""
Full self-contained Engram implementation based on the paper + demo code analysis.
Includes: tokenizer compression, multi-head hashing, gating, short conv, mHC integration.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


# ─────────────────────────────────────────────────────────────────
# 1. Configuration
# ─────────────────────────────────────────────────────────────────

@dataclass
class EngramConfig:
    # vocab sizes per (ngram_order, head) - list of per-layer lists
    engram_vocab_size: List[int] = field(
        default_factory=lambda: [129280 * 5, 129280 * 5]
    )
    max_ngram_size: int = 3          # use 2-grams and 3-grams
    n_embed_per_ngram: int = 512     # total dim per ngram order
    n_head_per_ngram: int = 8        # hash heads per ngram order
    layer_ids: List[int] = field(default_factory=lambda: [1, 15])
    pad_id: int = 2
    seed: int = 42


@dataclass
class BackboneConfig:
    hidden_size: int = 2560
    hc_mult: int = 4       # number of hyper-connection branches


# ─────────────────────────────────────────────────────────────────
# 2. Tokenizer Compression
# ─────────────────────────────────────────────────────────────────

class VocabCompressor:
    """
    Surjective mapping P: V → V' that collapses semantically equivalent tokens.
    In production this is precomputed from the tokenizer's vocab.
    Here we approximate it with a random but fixed mapping for illustration.
    """
    def __init__(self, vocab_size: int, compression_ratio: float = 0.77, seed: int = 0):
        rng = np.random.default_rng(seed)
        target_size = int(vocab_size * compression_ratio)
        # Each original token maps to one of target_size canonical IDs
        self.mapping = torch.from_numpy(
            rng.integers(0, target_size, size=vocab_size, dtype=np.int64)
        )
        self.compressed_vocab_size = target_size

    def compress(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: [B, T] → compressed_ids: [B, T]"""
        return self.mapping[input_ids.cpu()].to(input_ids.device)


# ─────────────────────────────────────────────────────────────────
# 3. Multi-Head N-gram Hashing
# ─────────────────────────────────────────────────────────────────

def find_prime_greater_than(n: int) -> int:
    """Simple primality test to find next prime above n."""
    candidate = n + 1
    while True:
        if all(candidate % i != 0 for i in range(2, int(candidate**0.5) + 1)):
            return candidate
        candidate += 1


class NgramHashMapping:
    """
    For each (ngram_order n, head k), maintains:
      - a random multiplier vector M_{n,k} ∈ Z^n
      - a prime modulus P_{n,k}
    Computes: id = XOR_hash(compressed_ngram, M_{n,k}) % P_{n,k}
    """
    def __init__(self, cfg: EngramConfig, layer_id: int, compressed_vocab_size: int):
        rng = np.random.default_rng(cfg.seed + layer_id * 1000)
        self.max_n = cfg.max_ngram_size
        self.n_heads = cfg.n_head_per_ngram
        self.layer_id = layer_id
        self.pad_id = cfg.pad_id

        # Build multiplier matrices and prime moduli
        # Shape: {n: array of shape [n_heads, n]}
        self.multipliers = {}
        self.primes = {}
        for n in range(2, self.max_n + 1):
            mults = rng.integers(1, compressed_vocab_size, size=(self.n_heads, n))
            self.multipliers[n] = mults
            # Different prime per head
            base = cfg.engram_vocab_size[n - 2] // self.n_heads
            self.primes[n] = [find_prime_greater_than(base + k * 7) for k in range(self.n_heads)]

        self.compressed_vocab_size = compressed_vocab_size

    def hash_ngrams(self, compressed_ids: np.ndarray) -> dict:
        """
        compressed_ids: [B, T] numpy int64
        Returns dict: {n: array of shape [B, T, n_heads]} containing table indices
        """
        B, T = compressed_ids.shape
        results = {}
        for n in range(2, self.max_n + 1):
            mults = self.multipliers[n]   # [n_heads, n]
            primes = self.primes[n]       # list of n_heads ints
            head_ids = np.zeros((B, T, self.n_heads), dtype=np.int64)
            for k in range(self.n_heads):
                mix = np.zeros((B, T), dtype=np.int64)
                for i in range(n):
                    # Get token at position t-(n-1)+i, padded with pad_id
                    shift = n - 1 - i
                    if shift > 0:
                        padded = np.full((B, shift), self.pad_id, dtype=np.int64)
                        tokens = np.concatenate([padded, compressed_ids[:, :T - shift]], axis=1)
                    else:
                        tokens = compressed_ids
                    mix ^= (tokens * mults[k, i]).astype(np.int64)
                head_ids[:, :, k] = mix % primes[k]
            results[n] = head_ids
        return results


# ─────────────────────────────────────────────────────────────────
# 4. Multi-Head Embedding Table
# ─────────────────────────────────────────────────────────────────

class MultiHeadEmbedding(nn.Module):
    """
    Single nn.Embedding table serving multiple heads with offset addressing.
    More efficient than separate tables: one contiguous allocation, single gather.

    For head k with vocab_size V_k:
      global_id = local_id + offset[k]
      emb = E[global_id]  (single table of size sum(V_k))
    """
    def __init__(self, vocab_sizes: List[int], embed_dim: int):
        super().__init__()
        total_vocab = sum(vocab_sizes)
        self.embed = nn.Embedding(total_vocab, embed_dim)
        # Precompute offsets for each head
        offsets = [0]
        for v in vocab_sizes[:-1]:
            offsets.append(offsets[-1] + v)
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.long))
        self.vocab_sizes = vocab_sizes

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices: [B, T, n_heads]  (local indices per head)
        returns: [B, T, n_heads, embed_dim]
        """
        global_indices = indices + self.offsets.view(1, 1, -1)  # broadcast offsets
        return self.embed(global_indices)


# ─────────────────────────────────────────────────────────────────
# 5. Short Depthwise Causal Convolution
# ─────────────────────────────────────────────────────────────────

class ShortConv(nn.Module):
    """
    Dilated depthwise causal convolution for local context expansion.
    kernel_size=4, dilation=max_ngram_order (e.g. 3)
    Effective positions seen: t, t-3, t-6, t-9
    """
    def __init__(self, hidden_size: int, hc_mult: int,
                 kernel_size: int = 4, dilation: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation  # causal padding

        # Depthwise: groups=hidden_size*hc_mult
        channels = hidden_size * hc_mult
        self.conv = nn.Conv1d(
            channels, channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=channels,   # depthwise
            padding=self.padding,
            bias=True
        )
        self.norm = nn.RMSNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, hc_mult, hidden_size]
        returns: [B, T, hc_mult, hidden_size]
        """
        B, T, HC, D = x.shape
        # Reshape for conv: [B, HC*D, T]
        residual = x
        x = self.norm(x)  # norm over last dim
        x = x.permute(0, 2, 3, 1).reshape(B, HC * D, T)  # [B, HC*D, T]
        x = self.conv(x)
        x = x[:, :, :T]   # trim causal padding
        x = x.reshape(B, HC, D, T).permute(0, 3, 1, 2)  # [B, T, HC, D]
        x = F.silu(x)
        return x + residual


# ─────────────────────────────────────────────────────────────────
# 6. Core Engram Module
# ─────────────────────────────────────────────────────────────────

class EngramModule(nn.Module):
    """
    Full Engram module for a single layer position.
    
    Forward pipeline:
      1. Hash input_ids → table indices
      2. Embedding lookup → raw memory vectors
      3. Value projection (shared across branches)
      4. Key projection (per-branch for mHC)
      5. Context-aware gating
      6. Short depthwise conv
      7. Output projection → backbone hidden dim
    """

    def __init__(
        self,
        engram_cfg: EngramConfig,
        backbone_cfg: BackboneConfig,
        layer_id: int,
        compressed_vocab_size: int
    ):
        super().__init__()
        self.layer_id = layer_id
        self.max_n = engram_cfg.max_ngram_size
        self.n_heads = engram_cfg.n_head_per_ngram
        self.hc_mult = backbone_cfg.hc_mult
        self.hidden_size = backbone_cfg.hidden_size

        # Hash mapping (CPU, deterministic)
        self.hash_mapper = NgramHashMapping(engram_cfg, layer_id, compressed_vocab_size)

        # Head dimension for embedding
        head_dim = engram_cfg.n_embed_per_ngram // engram_cfg.n_head_per_ngram
        n_orders = engram_cfg.max_ngram_size - 1  # orders: 2, 3, ... max_n

        # Embedding tables per (order, head)
        # We create one MultiHeadEmbedding per ngram order
        self.embeddings = nn.ModuleList()
        for n in range(2, engram_cfg.max_ngram_size + 1):
            vocab_sizes = self.hash_mapper.primes[n]  # list of n_heads primes
            self.embeddings.append(MultiHeadEmbedding(vocab_sizes, head_dim))

        # After concatenating all ngram orders and heads:
        # engram_hidden_size = n_orders * n_heads * head_dim
        #                    = (max_n - 1) * n_heads * head_dim
        self.engram_hidden_size = n_orders * engram_cfg.n_head_per_ngram * head_dim

        D = backbone_cfg.hidden_size

        # Shared value projection: engram_hidden → D
        self.proj_v = nn.Linear(self.engram_hidden_size, D, bias=False)

        # Per-branch key projections: engram_hidden → D  (for gating)
        self.proj_k = nn.ModuleList([
            nn.Linear(self.engram_hidden_size, D, bias=False)
            for _ in range(self.hc_mult)
        ])

        # Per-branch output projections (after gating, D → D)
        self.proj_out = nn.ModuleList([
            nn.Linear(D, D, bias=False)
            for _ in range(self.hc_mult)
        ])

        # Norms for gating (query-side and key-side, per branch)
        self.norm_q = nn.ModuleList([nn.RMSNorm(D) for _ in range(self.hc_mult)])
        self.norm_k = nn.ModuleList([nn.RMSNorm(D) for _ in range(self.hc_mult)])

        # Short conv applied to fused output
        self.short_conv = ShortConv(D, self.hc_mult, kernel_size=4, dilation=self.max_n)

        self.scale = 1.0 / math.sqrt(D)

    def _retrieve_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, T]  (compressed)
        Returns: e_t [B, T, engram_hidden_size]
        """
        ids_np = input_ids.cpu().numpy().astype(np.int64)
        hash_ids = self.hash_mapper.hash_ngrams(ids_np)  # {n: [B, T, n_heads]}

        parts = []
        for i, n in enumerate(range(2, self.max_n + 1)):
            idx_tensor = torch.from_numpy(hash_ids[n]).to(input_ids.device)
            # [B, T, n_heads, head_dim]
            embs = self.embeddings[i](idx_tensor)
            B, T, H, D = embs.shape
            parts.append(embs.reshape(B, T, H * D))  # [B, T, n_heads * head_dim]

        return torch.cat(parts, dim=-1)  # [B, T, engram_hidden_size]

    def forward(
        self,
        hidden_states: torch.Tensor,           # [B, T, hc_mult, D]  (multi-branch)
        compressed_input_ids: torch.Tensor,    # [B, T]
    ) -> torch.Tensor:
        """
        Returns residual contribution: [B, T, hc_mult, D]
        """
        B, T, HC, D = hidden_states.shape

        # 1. Retrieve static embeddings
        e_t = self._retrieve_embeddings(compressed_input_ids)   # [B, T, E]
        e_t = e_t.to(hidden_states.dtype)

        # 2. Shared value projection
        v_t = self.proj_v(e_t)   # [B, T, D]

        # 3. Per-branch gating and output
        branch_outputs = []
        for m in range(self.hc_mult):
            h_m = hidden_states[:, :, m, :]  # [B, T, D]
            k_m = self.proj_k[m](e_t)         # [B, T, D]

            # Gating: scalar attention between hidden state and retrieved key
            q_norm = self.norm_q[m](h_m)      # [B, T, D]
            k_norm = self.norm_k[m](k_m)      # [B, T, D]

            # Scalar gate per position
            alpha = torch.sigmoid(
                (q_norm * k_norm).sum(dim=-1, keepdim=True) * self.scale
            )  # [B, T, 1]

            u_m = alpha * v_t   # [B, T, D]
            branch_outputs.append(u_m)

        # Stack branches: [B, T, HC, D]
        fused = torch.stack(branch_outputs, dim=2)

        # 4. Short conv (local context refinement)
        output = self.short_conv(fused)   # [B, T, HC, D]

        return output


# ─────────────────────────────────────────────────────────────────
# 7. Integration into Transformer Block
# ─────────────────────────────────────────────────────────────────

class TransformerBlockWithEngram(nn.Module):
    """
    Transformer block where Engram is inserted BEFORE attention + MoE.
    Engram's residual is added first, then attention, then MoE.
    """

    def __init__(
        self,
        backbone_cfg: BackboneConfig,
        engram_module: Optional[EngramModule] = None,
    ):
        super().__init__()
        self.engram = engram_module

        D = backbone_cfg.hidden_size
        HC = backbone_cfg.hc_mult

        # Mocked attention (replace with MLA in production)
        self.attn_norm = nn.RMSNorm(D)
        self.attn_proj = nn.Linear(D * HC, D * HC, bias=False)

        # Mocked MoE FFN (replace with DeepSeekMoE in production)
        self.ffn_norm = nn.RMSNorm(D)
        self.ffn = nn.Sequential(nn.Linear(D * HC, D * HC * 2), nn.SiLU(), nn.Linear(D * HC * 2, D * HC))

    def forward(
        self,
        hidden_states: torch.Tensor,           # [B, T, HC, D]
        compressed_input_ids: torch.Tensor,    # [B, T]
    ) -> torch.Tensor:
        B, T, HC, D = hidden_states.shape

        # Engram residual (before attention)
        if self.engram is not None:
            hidden_states = hidden_states + self.engram(hidden_states, compressed_input_ids)

        # Attention (simplified mock)
        flat = hidden_states.reshape(B, T, HC * D)
        flat = flat + self.attn_proj(self.attn_norm(flat.reshape(B * T, HC * D)).reshape(B, T, HC * D))

        # MoE FFN (simplified mock)
        flat = flat + self.ffn(self.ffn_norm(flat.reshape(B * T, HC * D)).reshape(B, T, HC * D))

        return flat.reshape(B, T, HC, D)
```

---

## 5. Distributed Training

### 5.1 The Challenge: Embedding Tables Don't Fit

A 27B-parameter Engram table at BF16 = **54 GB**. A single H100 has 80 GB HBM — and you need room for activations, optimizer states, and other parameters too.

For a 100B table: 200 GB — must be sharded.

### 5.2 Model Parallelism Strategy

```
┌─────────────────────────────────────────────────────────┐
│                    GPU CLUSTER (8×H100)                  │
│                                                         │
│  GPU 0: Embed shard [0, 12.5B)   + Transformer layers 0-3  │
│  GPU 1: Embed shard [12.5B, 25B) + Transformer layers 4-7  │
│  GPU 2: Embed shard [25B, 37.5B) + Transformer layers 8-11 │
│  ...                                                    │
│  GPU 7: Embed shard [87.5B, 100B) + Transformer layers 28-29│
└─────────────────────────────────────────────────────────┘
```

Each GPU holds:
- 1/N slice of the Engram embedding table
- The corresponding transformer layers (pipeline parallel)

### 5.3 All-to-All Communication Pattern

During training, each token needs embeddings that may live on any GPU:

```
Forward pass:
  Each GPU computes which embedding IDs it needs → sends requests to owners
  All-to-All scatter: GPU i sends "please give me rows [r1, r2, ...]" to GPU j
  All-to-All gather:  GPU j returns embedding rows to GPU i

Backward pass (gradient flow):
  Gradients flow back through gating → gradient for e_t
  All-to-All scatter: send gradient fragments back to owning GPUs
  Accumulate with optimizer (Adam for embeddings)
```

```python
import torch.distributed as dist

class DistributedEngramTable(nn.Module):
    """
    Embedding table sharded across world_size GPUs.
    Each rank owns rows [rank * shard_size, (rank+1) * shard_size).
    """
    def __init__(self, total_vocab: int, embed_dim: int):
        super().__init__()
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.total_vocab = total_vocab
        self.embed_dim = embed_dim

        # Each GPU owns a contiguous shard
        self.shard_size = math.ceil(total_vocab / self.world_size)
        local_vocab = min(self.shard_size, total_vocab - self.rank * self.shard_size)
        local_vocab = max(local_vocab, 0)

        self.local_embed = nn.Embedding(local_vocab, embed_dim)

    def forward(self, global_ids: torch.Tensor) -> torch.Tensor:
        """
        global_ids: [B, T, n_heads]
        Returns: [B, T, n_heads, embed_dim]
        """
        shape = global_ids.shape
        flat_ids = global_ids.reshape(-1)   # [N]
        N = flat_ids.shape[0]

        # Which rank owns each ID?
        owner_ranks = flat_ids // self.shard_size          # [N]
        local_ids   = flat_ids % self.shard_size            # [N]

        # --- Sparse All-to-All ---
        # 1. Sort by owner rank
        order = torch.argsort(owner_ranks)
        sorted_ids   = local_ids[order]
        sorted_ranks = owner_ranks[order]

        # 2. Count how many IDs go to each rank
        send_counts = torch.bincount(sorted_ranks, minlength=self.world_size).tolist()
        recv_counts = [0] * self.world_size
        dist.all_to_all_single(
            torch.tensor(recv_counts), torch.tensor(send_counts)
        )  # exchange count metadata

        # 3. Exchange the actual ID values
        recv_ids = torch.empty(sum(recv_counts), dtype=sorted_ids.dtype,
                               device=sorted_ids.device)
        dist.all_to_all_single(recv_ids, sorted_ids,
                               output_split_sizes=recv_counts,
                               input_split_sizes=send_counts)

        # 4. Local embedding lookup
        local_embs = self.local_embed(recv_ids)  # [sum(recv), D]

        # 5. Send embeddings back to requester
        send_back = torch.empty(N, self.embed_dim,
                                dtype=local_embs.dtype, device=local_embs.device)
        dist.all_to_all_single(send_back, local_embs,
                               output_split_sizes=send_counts,
                               input_split_sizes=recv_counts)

        # 6. Un-sort to original order
        inv_order = torch.argsort(order)
        result = send_back[inv_order]

        return result.reshape(*shape, self.embed_dim)
```

### 5.4 Optimizer Split

This is crucial for training stability and efficiency:

```python
# Backbone parameters (attention, FFN, norms) → Muon optimizer
backbone_params = [p for n, p in model.named_parameters() 
                   if 'local_embed' not in n]

# Embedding parameters → Adam with high LR and no weight decay
embed_params = [p for n, p in model.named_parameters() 
                if 'local_embed' in n]

backbone_opt = Muon(backbone_params, lr=4e-4, weight_decay=0.1)
embed_opt    = torch.optim.Adam(embed_params, 
                                 lr=4e-4 * 5,   # 5x higher LR
                                 weight_decay=0.0)
```

Why Adam for embeddings but Muon for backbone?
- Embeddings are sparse: only activated rows get gradients per step
- Adam handles sparse updates well (per-parameter adaptive LR)
- Muon (a momentum-based orthogonal optimizer) works better for dense dense weight matrices

### 5.5 Pipeline Parallelism + Engram Placement

```
Layer 0       Layer 1 (ENGRAM)  Layer 2      ...  Layer 15 (ENGRAM)  Layer 16  ...
[GPU 0-1]     [GPU 0-1]         [GPU 0-1]         [GPU 2-3]          [GPU 2-3]

Pipeline stages:
  Stage 0: Layers 0-7   (includes Engram at layer 1)
  Stage 1: Layers 8-15  (includes Engram at layer 15)
  ...
```

Engram is placed at layers 2 and 15 specifically so:
1. Each pipeline stage has exactly one Engram → clean stage boundaries
2. The earlier Engram has one round of attention before it (needed for good gating queries)

---

## 6. Inference Offloading & Prefetching

### 6.1 Why Offloading Works for Engram (but not MoE)

MoE routing depends on the **hidden state** — you don't know which experts to load until the forward pass runs. So you can't prefetch.

Engram routing depends only on **input_ids** — you can compute all hash IDs before any forward pass computation. This enables:

```
t=0:  Start computing input_ids hashes for Engram@layer2
t=1:  Launch async PCIe transfer from Host DRAM → GPU
t=1:  Begin Transformer Layer 0 computation (covers the PCIe latency)
t=2:  Transformer Layer 1 computation
t=2:  PCIe transfer completes (overlapped with layers 0-1)
t=3:  Engram@layer2 runs with embeddings already in GPU HBM
```

### 6.2 Zipfian Cache Hierarchy

N-gram frequencies follow Zipf's law: the top 1% of N-grams account for ~80% of accesses.

```
Tier 0: GPU HBM (80GB)        — top 1% most frequent N-grams (hot cache)
Tier 1: Host DRAM (1-2TB)     — next 10% (warm cache, PCIe prefetch)
Tier 2: NVMe SSD (tens of TB) — remaining 89% (cold, rarely accessed)
```

```python
class ZipfianCache:
    """
    Frequency-based embedding cache with LRU eviction per tier.
    """
    def __init__(self, gpu_capacity: int, dram_capacity: int,
                 embed_dim: int, dtype=torch.bfloat16):
        self.embed_dim = embed_dim
        self.dtype = dtype
        
        # GPU HBM tier: small, fast
        self.gpu_cache: dict[int, torch.Tensor] = {}
        self.gpu_capacity = gpu_capacity
        self.gpu_lru = []
        
        # DRAM tier: large, async prefetch
        self.dram_cache: dict[int, np.ndarray] = {}
        self.dram_capacity = dram_capacity
        
        # Access frequency counter
        self.freq: dict[int, int] = {}

    def prefetch_async(self, ids: list[int]) -> list[int]:
        """Launch async D2H transfers. Returns IDs not in GPU cache."""
        missing = [i for i in ids if i not in self.gpu_cache]
        dram_hits = [i for i in missing if i in self.dram_cache]
        # Transfer dram_hits → GPU asynchronously
        # (implemented via torch.cuda.Stream in production)
        return [i for i in missing if i not in self.dram_cache]  # cold misses

    def get(self, ids: torch.Tensor) -> torch.Tensor:
        flat = ids.reshape(-1).tolist()
        result = torch.zeros(len(flat), self.embed_dim, dtype=self.dtype)
        for i, gid in enumerate(flat):
            if gid in self.gpu_cache:
                result[i] = self.gpu_cache[gid]
                self.freq[gid] = self.freq.get(gid, 0) + 1
        return result.reshape(*ids.shape, self.embed_dim)
```

### 6.3 Prefetch Implementation with CUDA Streams

```python
class EngramInferenceEngine:
    def __init__(self, model, engram_table_path: str):
        self.model = model
        # Load table into pinned host memory for fast PCIe transfer
        self.host_table = np.load(engram_table_path, mmap_mode='r')
        self.stream = torch.cuda.Stream()  # dedicated transfer stream

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        
        # Step 1: Compute all hash IDs on CPU (deterministic, fast)
        compressed_ids = self.model.vocab_compressor.compress(input_ids)
        all_hash_ids = {}
        for layer_id in self.model.engram_layer_ids:
            mapper = self.model.hash_mappers[layer_id]
            all_hash_ids[layer_id] = mapper.hash_ngrams(
                compressed_ids.cpu().numpy()
            )
        
        # Step 2: Async transfer embeddings to GPU (on side stream)
        prefetched = {}
        with torch.cuda.stream(self.stream):
            for layer_id, hash_ids in all_hash_ids.items():
                flat_ids = np.unique(np.concatenate([
                    v.reshape(-1) for v in hash_ids.values()
                ]))
                # Gather rows from host table (pinned memory)
                rows = self.host_table[flat_ids]
                gpu_rows = torch.from_numpy(rows).cuda(non_blocking=True)
                prefetched[layer_id] = (flat_ids, gpu_rows)
        
        # Step 3: Run transformer layers 0 ... (engram_layer-1)
        # This computation overlaps with the PCIe transfer above
        hidden = self.model.embed(input_ids)
        for layer_idx in range(self.model.engram_layer_ids[0]):
            hidden = self.model.layers[layer_idx](hidden)
        
        # Step 4: Synchronize stream (ensure transfer complete)
        torch.cuda.current_stream().wait_stream(self.stream)
        
        # Step 5: Continue layers with Engram
        for layer_idx in range(self.model.engram_layer_ids[0], self.model.n_layers):
            if layer_idx in all_hash_ids:
                engram_out = self._lookup_from_prefetched(
                    all_hash_ids[layer_idx],
                    prefetched[layer_idx]
                )
                hidden = self.model.layers[layer_idx](hidden, engram_out)
            else:
                hidden = self.model.layers[layer_idx](hidden)
        
        return hidden

    def _lookup_from_prefetched(self, hash_ids, prefetched):
        flat_ids, gpu_rows = prefetched
        id_to_row = {gid: i for i, gid in enumerate(flat_ids)}
        # Index into the already-transferred GPU tensor
        return gpu_rows  # full scatter logic omitted for brevity
```

---

## 7. CUDA Kernels

### 7.1 Why Custom Kernels?

The standard `nn.Embedding` lookup is not optimized for:
1. **Multi-head hashing** (need to gather from multiple hash heads simultaneously)
2. **Sparse irregular access patterns** (N-gram embeddings have irregular access, bad for L2 cache)
3. **FP8 computation** for the gating matmuls
4. **Fused operations** (hash + lookup + gate in one kernel launch)

### 7.2 Kernel 1: Multi-Head Hash Lookup

```cuda
// multi_head_ngram_lookup.cu
// One thread per (batch, position, head)

#include <cuda_bf16.h>
#include <cuda_runtime.h>

__device__ inline int64_t xor_hash_ngram(
    const int64_t* compressed_ids,   // [B, T] flattened
    int           b, int t, int T,
    int           n,                 // ngram order (2 or 3)
    const int64_t* multipliers,      // [n] multipliers for this head
    int64_t        prime             // prime modulus for this head
) {
    int64_t mix = 0;
    for (int i = 0; i < n; i++) {
        int pos = t - (n - 1) + i;
        int64_t tok = (pos >= 0) ? compressed_ids[b * T + pos] : 2LL; // pad=2
        mix ^= tok * multipliers[i];
    }
    // Ensure positive modulo
    return ((mix % prime) + prime) % prime;
}

__global__ void multi_head_ngram_lookup_kernel(
    const int64_t* __restrict__ compressed_ids,    // [B, T]
    const __nv_bfloat16* __restrict__ embed_table, // [total_vocab, D]
    const int64_t* __restrict__ multipliers,       // [n_orders, n_heads, max_n]
    const int64_t* __restrict__ primes,            // [n_orders, n_heads]
    const int64_t* __restrict__ offsets,           // [n_orders, n_heads] cumulative offsets
    __nv_bfloat16* __restrict__ output,            // [B, T, n_orders*n_heads*D]
    int B, int T,
    int n_orders, int n_heads, int D,
    int max_ngram_size
) {
    // Grid: (B, T)   Block: (n_orders * n_heads)
    int b = blockIdx.x;
    int t = blockIdx.y;
    int head_flat = threadIdx.x;  // flat index over (order, head)

    if (b >= B || t >= T || head_flat >= n_orders * n_heads) return;

    int order_idx = head_flat / n_heads;  // 0 = bigram, 1 = trigram, ...
    int head_idx  = head_flat % n_heads;
    int n         = order_idx + 2;        // actual ngram order

    // Load multipliers for this head into registers
    int64_t mults[3];  // max 3-gram
    for (int i = 0; i < n; i++) {
        mults[i] = multipliers[(order_idx * n_heads + head_idx) * max_ngram_size + i];
    }
    int64_t prime  = primes[order_idx * n_heads + head_idx];
    int64_t offset = offsets[order_idx * n_heads + head_idx];

    // Compute hash
    int64_t local_id = xor_hash_ngram(compressed_ids, b, t, T, n, mults, prime);
    int64_t global_id = offset + local_id;

    // Load embedding row into output (coalesced if D is multiple of 16)
    // output layout: [B, T, n_orders*n_heads, D]
    int out_offset = ((b * T + t) * (n_orders * n_heads) + head_flat) * D;
    const __nv_bfloat16* src = embed_table + global_id * D;
    __nv_bfloat16* dst = output + out_offset;

    // Vectorized copy (128-bit loads)
    int chunks = D / 8;  // 8 bfloat16 = 128 bits
    const float4* src4 = reinterpret_cast<const float4*>(src);
    float4*       dst4 = reinterpret_cast<float4*>(dst);
    for (int c = 0; c < chunks; c++) {
        dst4[c] = src4[c];
    }
}
```

### 7.3 Kernel 2: Fused Gating (Gate Computation + Apply)

```cuda
// fused_engram_gate.cu
// Fuses: K-projection, gate score, sigmoid, gate application, V-scaling
// All in one kernel to avoid multiple HBM round-trips.

__global__ void fused_engram_gate_kernel(
    const __nv_bfloat16* __restrict__ e_t,         // [B, T, E]  memory embedding
    const __nv_bfloat16* __restrict__ h_t,         // [B, T, HC, D]  hidden states
    const __nv_bfloat16* __restrict__ W_K,         // [HC, D, E]  key projections
    const __nv_bfloat16* __restrict__ W_V,         // [D, E]  shared value projection
    const __nv_bfloat16* __restrict__ rms_scale_q, // [HC, D]  RMSNorm scale
    const __nv_bfloat16* __restrict__ rms_scale_k, // [HC, D]  RMSNorm scale
    __nv_bfloat16* __restrict__ output,            // [B, T, HC, D]
    float inv_sqrt_d,
    int B, int T, int HC, int D, int E
) {
    // One block per (B, T, HC)
    int b  = blockIdx.x;
    int t  = blockIdx.y;
    int hc = blockIdx.z;

    if (b >= B || t >= T || hc >= HC) return;

    extern __shared__ float smem[];
    // Layout: [v_proj: D] [k_proj: D] [rms_acc: 2] 

    int tid = threadIdx.x;  // 0..D-1

    // --- Step 1: Compute V = W_V @ e_t (shared across branches, but hc=0 computes) ---
    // (In practice computed once and passed in; simplified here)
    float v_val = 0.0f;
    const __nv_bfloat16* wv_row = W_V + tid * E;
    const __nv_bfloat16* e     = e_t + (b * T + t) * E;
    for (int i = 0; i < E; i += 4) {
        // Unrolled dot product
        v_val += __bfloat162float(wv_row[i])   * __bfloat162float(e[i]);
        v_val += __bfloat162float(wv_row[i+1]) * __bfloat162float(e[i+1]);
        v_val += __bfloat162float(wv_row[i+2]) * __bfloat162float(e[i+2]);
        v_val += __bfloat162float(wv_row[i+3]) * __bfloat162float(e[i+3]);
    }
    smem[tid] = v_val;  // cache V in shared mem

    __syncthreads();

    // --- Step 2: Compute K = W_K[hc] @ e_t ---
    float k_val = 0.0f;
    const __nv_bfloat16* wk_row = W_K + (hc * D + tid) * E;
    for (int i = 0; i < E; i += 4) {
        k_val += __bfloat162float(wk_row[i])   * __bfloat162float(e[i]);
        k_val += __bfloat162float(wk_row[i+1]) * __bfloat162float(e[i+1]);
        k_val += __bfloat162float(wk_row[i+2]) * __bfloat162float(e[i+2]);
        k_val += __bfloat162float(wk_row[i+3]) * __bfloat162float(e[i+3]);
    }

    // --- Step 3: RMSNorm on q and k ---
    float h_val = __bfloat162float(h_t[((b * T + t) * HC + hc) * D + tid]);

    // RMSNorm: compute sum of squares via warp reduction
    float sq_h = h_val * h_val;
    float sq_k = k_val * k_val;
    // Warp reduce
    for (int mask = 16; mask > 0; mask >>= 1) {
        sq_h += __shfl_xor_sync(0xffffffff, sq_h, mask);
        sq_k += __shfl_xor_sync(0xffffffff, sq_k, mask);
    }
    float rms_q = rsqrtf(sq_h / D + 1e-6f);
    float rms_k = rsqrtf(sq_k / D + 1e-6f);

    float scale_q = __bfloat162float(rms_scale_q[hc * D + tid]);
    float scale_k = __bfloat162float(rms_scale_k[hc * D + tid]);
    float q_norm  = h_val * rms_q * scale_q;
    float k_norm  = k_val * rms_k * scale_k;

    // --- Step 4: Scalar gate (dot product → sigmoid) ---
    float dot = q_norm * k_norm;
    // Warp reduce dot product
    for (int mask = 16; mask > 0; mask >>= 1) {
        dot += __shfl_xor_sync(0xffffffff, dot, mask);
    }
    float alpha = 1.0f / (1.0f + expf(-dot * inv_sqrt_d));  // sigmoid

    // --- Step 5: Apply gate and write output ---
    float out_val = alpha * smem[tid];  // α * V[tid]
    int out_idx = ((b * T + t) * HC + hc) * D + tid;
    output[out_idx] = __float2bfloat16(out_val);
}
```

### 7.4 Kernel 3: Fused RMS Norm (Inline with Attention for DSA)

```cuda
// For DSA's lightning indexer — ReLU dot product with FP8
// Using cutlass FP8 gemm primitives in production

__global__ void lightning_indexer_kernel(
    const __nv_fp8_e4m3* __restrict__ q_indexer,  // [B, T, H_I, d_I]
    const __nv_fp8_e4m3* __restrict__ k_indexer,  // [B, S, d_I]
    const float* __restrict__         w_indexer,  // [B, T, H_I]  scalar weights
    float* __restrict__                scores,     // [B, T, S]   index scores
    int B, int T, int S, int H_I, int d_I
) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int s = blockIdx.z;

    if (b >= B || t >= T || s >= S) return;

    float score = 0.0f;
    for (int h = 0; h < H_I; h++) {
        float w = w_indexer[(b * T + t) * H_I + h];
        // Dot product q[h] · k[s]
        float dot = 0.0f;
        for (int i = threadIdx.x; i < d_I; i += blockDim.x) {
            float q_val = __half2float(__nv_fp8_e4m3_to_half(
                q_indexer[((b * T + t) * H_I + h) * d_I + i]));
            float k_val = __half2float(__nv_fp8_e4m3_to_half(
                k_indexer[(b * S + s) * d_I + i]));
            dot += q_val * k_val;
        }
        // Warp reduce dot
        for (int mask = 16; mask > 0; mask >>= 1)
            dot += __shfl_xor_sync(0xffffffff, dot, mask);
        // ReLU + weighted sum
        score += w * fmaxf(dot, 0.0f);
    }
    scores[(b * T + t) * S + s] = score;
}
```

### 7.5 Python Bindings (torch.autograd)

```python
# engram_cuda_ops.py
import torch
from torch.autograd import Function

# Assumes CUDA extension compiled with:
# python setup.py build_ext --inplace

try:
    import engram_cuda  # custom CUDA extension
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

class MultiHeadNgramLookupFn(Function):
    @staticmethod
    def forward(ctx, compressed_ids, embed_table, multipliers, primes, offsets,
                B, T, n_orders, n_heads, D):
        if CUDA_AVAILABLE:
            output = engram_cuda.multi_head_lookup(
                compressed_ids, embed_table, multipliers, primes, offsets)
        else:
            # Fallback: pure PyTorch (slower)
            output = _ngram_lookup_pytorch(
                compressed_ids, embed_table, multipliers, primes, offsets,
                n_orders, n_heads, D)
        ctx.save_for_backward(embed_table, multipliers, primes, offsets)
        ctx.ids = compressed_ids
        ctx.shape_info = (B, T, n_orders, n_heads, D)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        embed_table, multipliers, primes, offsets = ctx.saved_tensors
        # Sparse gradient accumulation into embed_table
        if CUDA_AVAILABLE:
            grad_embed = engram_cuda.sparse_embed_grad(
                ctx.ids, grad_output, multipliers, primes, offsets,
                embed_table.shape[0])
        else:
            grad_embed = _sparse_grad_pytorch(
                ctx.ids, grad_output, embed_table.shape[0])
        return None, grad_embed, None, None, None, \
               None, None, None, None, None


def _ngram_lookup_pytorch(compressed_ids, embed_table, multipliers, primes,
                           offsets, n_orders, n_heads, D):
    """Pure PyTorch fallback for multi-head ngram lookup."""
    B, T = compressed_ids.shape[:2]
    parts = []
    for n_idx in range(n_orders):
        n = n_idx + 2
        for h_idx in range(n_heads):
            # Compute hash for this (order, head)
            mix = torch.zeros(B, T, dtype=torch.long, device=compressed_ids.device)
            for i in range(n):
                shift = n - 1 - i
                mult = multipliers[n_idx * n_heads * n + h_idx * n + i]
                if shift > 0:
                    pad = torch.full((B, shift), 2, dtype=torch.long,
                                     device=compressed_ids.device)
                    tokens = torch.cat([pad, compressed_ids[:, :T - shift]], dim=1)
                else:
                    tokens = compressed_ids
                mix = mix ^ (tokens * mult)

            prime  = primes[n_idx * n_heads + h_idx].item()
            offset = offsets[n_idx * n_heads + h_idx].item()
            local_ids = mix.abs() % prime
            global_ids = offset + local_ids.reshape(-1)
            embs = embed_table[global_ids].reshape(B, T, D)
            parts.append(embs)

    return torch.cat(parts, dim=-1)  # [B, T, n_orders * n_heads * D]
```

### 7.6 setup.py for Building the Extension

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='engram_cuda',
    ext_modules=[
        CUDAExtension(
            name='engram_cuda',
            sources=[
                'csrc/multi_head_ngram_lookup.cu',
                'csrc/fused_engram_gate.cu',
                'csrc/lightning_indexer.cu',
                'csrc/engram_bindings.cpp',
            ],
            extra_compile_args={
                'cxx': ['-O3', '-march=native'],
                'nvcc': [
                    '-O3',
                    '-arch=sm_90',           # H100
                    '--use_fast_math',
                    '-D_GLIBCXX_USE_CXX11_ABI=0',
                    '--expt-relaxed-constexpr',
                    '-DENABLE_FP8',
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

---

## 8. System Architecture

### 8.1 Complete Training Stack

```
┌────────────────────────────────────────────────────────────────────┐
│                     TRAINING CLUSTER (64× H100)                    │
│                                                                    │
│  Data Parallel (DP=8 replicas, each replica = 8 GPUs)             │
│  │                                                                 │
│  └── Tensor Parallel (TP=2) × Pipeline Parallel (PP=4)            │
│       │                                                            │
│       └── Per PP stage:                                            │
│            ┌──────────┬──────────────────────┐                    │
│            │ Backbone │     Engram Shards     │                    │
│            │ (TP=2)   │  (sharded across TP)  │                    │
│            └──────────┴──────────────────────┘                    │
│                                                                    │
│  Communication:                                                    │
│  - Backbone: NVLink (300 GB/s) for TP all-reduce                  │
│  - Engram: NVLink for all-to-all embedding gather                  │
│  - Pipeline: NVLink for activation tensors across stages           │
│  - Gradient sync: InfiniBand for DP all-reduce                    │
└────────────────────────────────────────────────────────────────────┘
```

### 8.2 Complete Inference Stack

```
┌────────────────────────────────────────────────────────────────────┐
│                     INFERENCE SERVER (8× H100)                     │
│                                                                    │
│  GPU HBM (8 × 80GB = 640GB total):                               │
│    - Model parameters (backbone + KV cache): ~400GB               │
│    - Hot Engram embeddings (top 1% N-grams):  ~100GB              │
│    - Activations + workspace:                 ~140GB              │
│                                                                    │
│  Host DRAM (2TB):                                                 │
│    - Warm Engram embeddings (top 10% N-grams): ~200GB             │
│    - Prefetch buffer + OS overhead:            ~1.8TB             │
│                                                                    │
│  PCIe Bus (64 GB/s bidirectional):                               │
│    - Async prefetch thread for next Engram layer                  │
│    - Overlapped with GPU computation                              │
│                                                                    │
│  NVMe SSD (tens of TB):                                           │
│    - Cold Engram embeddings (rare N-grams)                        │
│    - Accessed on cache miss (~1% of requests)                     │
└────────────────────────────────────────────────────────────────────┘
```

### 8.3 Implementation Roadmap

**Phase 1 — Understand (Week 1)**
- [ ] Run `engram_demo_v1.py` from the official repo
- [ ] Trace the data flow: tokenizer → hash → lookup → gate → conv
- [ ] Understand the sparsity allocation law (why U-shape)

**Phase 2 — Single-GPU Implementation (Week 2-3)**
- [ ] Implement `VocabCompressor` with real NFKC normalization
- [ ] Implement `NgramHashMapping` with XOR hashing
- [ ] Implement `EngramModule` end-to-end
- [ ] Integrate into a small GPT-2-size transformer
- [ ] Verify that ablating Engram degrades knowledge benchmarks

**Phase 3 — Multi-GPU Distributed (Week 4-6)**
- [ ] Implement `DistributedEngramTable` with all-to-all
- [ ] Add optimizer split (Adam for embeddings, Muon for backbone)
- [ ] Test on 2-4 GPUs with a toy model
- [ ] Profile embedding gather latency vs compute time

**Phase 4 — CUDA Kernels (Week 7-10)**
- [ ] Implement multi-head ngram lookup kernel
- [ ] Implement fused gating kernel
- [ ] Add backward pass with sparse gradient accumulation
- [ ] Benchmark vs PyTorch baseline

**Phase 5 — Inference Optimization (Week 11-12)**
- [ ] Implement prefetch-and-overlap with CUDA streams
- [ ] Add Zipfian LRU cache hierarchy
- [ ] Benchmark end-to-end throughput vs MoE baseline

---

## Quick Reference: Key Numbers

| Parameter | Value |
|-----------|-------|
| Optimal MoE/Engram split | 75-80% MoE / 20-25% Engram |
| Engram layer positions | 2 and 15 (out of 30) |
| N-gram orders | 2 and 3 |
| Hash heads per order | 8 |
| Tokenizer compression | 23% vocabulary reduction |
| Conv kernel | size=4, dilation=3, depthwise causal |
| 100B table offload overhead | <3% throughput loss |
| Multi-Query NIAH improvement | 84.2 → 97.0 |
| BBH improvement (reasoning) | +5.0 points |
| DSA top-k tokens | 2048 out of 128K |
| DSA warmup | 1000 steps / 2.1B tokens |
| DSA sparse training | 15000 steps / 943.7B tokens |
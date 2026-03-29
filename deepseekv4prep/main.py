"""
Engram: Conditional Memory via Scalable N-gram Lookup — End-to-End Demo

Demonstrates the full Engram pipeline integrated into a toy transformer:
  1. Tokenizer compression (NFKC + case folding)
  2. Multi-head N-gram hashing (multiplicative XOR)
  3. Embedding lookup from shared table
  4. Context-aware gating with sqrt-abs activation
  5. Short dilated depthwise causal convolution
  6. Residual integration with multi-branch (mHC) backbone

This runs a forward pass through a 30-layer transformer with Engram modules
at layers 1 and 15, matching the paper's recommended placement.
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from engram.config import EngramConfig, BackboneConfig
from engram.hashing import NgramHashMapping
from engram.module import EngramModule, TransformerBlockWithEngram


def build_model(
    engram_cfg: EngramConfig,
    backbone_cfg: BackboneConfig,
) -> tuple[nn.ModuleList, NgramHashMapping]:
    """Build a toy transformer with Engram modules at specified layers."""

    hash_mapping = NgramHashMapping(engram_cfg)

    layers = nn.ModuleList()
    for layer_id in range(backbone_cfg.num_layers):
        engram_mod = None
        if layer_id in engram_cfg.layer_ids:
            engram_mod = EngramModule(
                engram_cfg=engram_cfg,
                backbone_cfg=backbone_cfg,
                layer_id=layer_id,
                hash_mapping=hash_mapping,
            )
        layers.append(
            TransformerBlockWithEngram(
                backbone_cfg=backbone_cfg,
                layer_id=layer_id,
                engram_module=engram_mod,
            )
        )

    return layers, hash_mapping


def count_params(module: nn.Module) -> dict[str, int]:
    """Count total, embedding, and backbone parameters."""
    total = sum(p.numel() for p in module.parameters())
    embed = sum(
        p.numel() for n, p in module.named_parameters()
        if "multi_head_embedding" in n
    )
    return {"total": total, "embedding": embed, "backbone": total - embed}


def main():
    engram_cfg = EngramConfig()
    backbone_cfg = BackboneConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print("=" * 70)
    print("  Engram: Conditional Memory via Scalable N-gram Lookup")
    print("=" * 70)
    print(f"\nDevice: {device}  |  Dtype: {dtype}")
    print(f"Backbone: hidden={backbone_cfg.hidden_size}, "
          f"layers={backbone_cfg.num_layers}, hc_mult={backbone_cfg.hc_mult}")
    print(f"Engram: max_ngram={engram_cfg.max_ngram_size}, "
          f"heads={engram_cfg.n_head_per_ngram}, "
          f"embed_per_ngram={engram_cfg.n_embed_per_ngram}")
    print(f"Engram layers: {engram_cfg.layer_ids}")

    # ── Build model ─────────────────────────────────────────────────
    print("\nBuilding model...")
    t0 = time.time()
    layers, hash_mapping = build_model(engram_cfg, backbone_cfg)
    layers = layers.to(device=device, dtype=dtype)

    vocab_embed = nn.Embedding(backbone_cfg.vocab_size, backbone_cfg.hidden_size)
    vocab_embed = vocab_embed.to(device=device, dtype=dtype)
    lm_head = nn.Linear(backbone_cfg.hidden_size, backbone_cfg.vocab_size, bias=False)
    lm_head = lm_head.to(device=device, dtype=dtype)

    params = count_params(layers)
    print(f"  Layers built in {time.time() - t0:.1f}s")
    print(f"  Total params:     {params['total']:>12,}")
    print(f"  Embedding params: {params['embedding']:>12,}")
    print(f"  Backbone params:  {params['backbone']:>12,}")

    # ── Tokenize ────────────────────────────────────────────────────
    text = "Only Alexander the Great could tame the horse Bucephalus."
    print(f"\nInput: \"{text}\"")

    tokenizer = AutoTokenizer.from_pretrained(
        engram_cfg.tokenizer_name_or_path, trust_remote_code=True
    )
    input_ids = tokenizer(text, return_tensors="pt").input_ids  # [1, T]
    input_ids_np = input_ids.numpy().astype(np.int64)
    B, T = input_ids_np.shape

    print(f"Tokens: {tokenizer.convert_ids_to_tokens(input_ids[0].tolist())}")
    print(f"Shape: B={B}, T={T}")

    # ── Forward pass ────────────────────────────────────────────────
    print("\nRunning forward pass...")
    t0 = time.time()

    with torch.no_grad():
        # Vocab embedding -> expand to multi-branch (mHC)
        hidden = vocab_embed(input_ids.to(device))  # [B, T, D]
        hidden = hidden.unsqueeze(2).expand(
            -1, -1, backbone_cfg.hc_mult, -1
        ).contiguous()  # [B, T, HC, D]

        # Run through all transformer blocks
        for layer in layers:
            hidden = layer(hidden, input_ids_np)

        # Collapse mHC branches (take first branch for LM head)
        hidden_out = hidden[:, :, 0, :]  # [B, T, D]
        logits = lm_head(hidden_out)      # [B, T, V]

    elapsed = time.time() - t0
    print(f"  Forward complete in {elapsed:.3f}s")
    print(f"  Hidden shape:  {hidden.shape}")
    print(f"  Logits shape:  {logits.shape}")

    # ── Verify output ───────────────────────────────────────────────
    next_token_logits = logits[0, -1, :]
    top5_ids = torch.topk(next_token_logits, 5).indices.tolist()
    top5_tokens = [tokenizer.decode([tid]) for tid in top5_ids]
    print(f"\nTop-5 next token predictions: {top5_tokens}")

    # ── Benchmark hash computation ──────────────────────────────────
    print("\n--- Hash Computation Benchmark ---")
    n_iters = 100
    big_ids = np.random.randint(0, backbone_cfg.vocab_size, (4, 2048), dtype=np.int64)
    t0 = time.time()
    for _ in range(n_iters):
        hash_mapping.hash(big_ids)
    hash_time = (time.time() - t0) / n_iters * 1000
    print(f"  Hash time (B=4, T=2048): {hash_time:.2f} ms")

    print("\nDone.")


if __name__ == "__main__":
    main()

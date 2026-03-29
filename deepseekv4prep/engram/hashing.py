"""
Multi-head N-gram hash mapping with multiplicative-XOR mixing.

For each (ngram_order n, head k), maintains:
  - Random odd multiplier vector M_{n,k} in Z^n  (layer-seeded)
  - Prime modulus P_{n,k}  (unique per head across all layers)

Hash:  mix = x'[t] * M[0] XOR x'[t-1] * M[1] XOR ...
       id  = mix % P_{n,k}

The prime-per-head uniqueness prevents systematic collisions across heads.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sympy import isprime

from .config import EngramConfig
from .tokenizer import CompressedTokenizer


def _find_next_prime(start: int, seen: set[int]) -> int:
    candidate = start + 1
    while True:
        if isprime(candidate) and candidate not in seen:
            return candidate
        candidate += 1


class NgramHashMapping:
    """Deterministic N-gram hashing for a set of layers.

    Mirrors the official demo logic: one CompressedTokenizer shared,
    per-layer random odd multipliers, globally-unique primes per head.
    """

    def __init__(self, cfg: EngramConfig):
        self.cfg = cfg
        self.max_n = cfg.max_ngram_size
        self.n_heads = cfg.n_head_per_ngram
        self.layer_ids = cfg.layer_ids

        self.compressed_tokenizer = CompressedTokenizer(cfg.tokenizer_name_or_path)
        self.tokenizer_vocab_size = len(self.compressed_tokenizer)

        self.pad_id = cfg.pad_id
        if self.pad_id is not None:
            self.pad_id = int(
                self.compressed_tokenizer.lookup_table[self.pad_id]
            )

        self.layer_multipliers = self._build_multipliers()
        self.vocab_sizes_per_layer = self._build_vocab_sizes()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_multipliers(self) -> dict[int, np.ndarray]:
        """Generate per-layer random odd multiplier vectors of shape [max_n]."""
        max_long = np.iinfo(np.int64).max
        M_max = int(max_long // self.tokenizer_vocab_size)
        half_bound = max(1, M_max // 2)
        PRIME_1 = 10007

        layer_mults: dict[int, np.ndarray] = {}
        for layer_id in self.layer_ids:
            base_seed = int(self.cfg.seed + PRIME_1 * int(layer_id))
            rng = np.random.default_rng(base_seed)
            r = rng.integers(0, half_bound, size=(self.max_n,), dtype=np.int64)
            layer_mults[layer_id] = r * 2 + 1  # ensure odd
        return layer_mults

    def _build_vocab_sizes(self) -> dict[int, list[list[int]]]:
        """Assign globally-unique prime table sizes per (ngram_order, head).

        Returns: {layer_id: [[prime_h0, prime_h1, ...], ...]}
                  outer list indexed by (n-2), inner by head.
        """
        seen_primes: set[int] = set()
        result: dict[int, list[list[int]]] = {}

        for layer_id in self.layer_ids:
            per_layer: list[list[int]] = []
            for ngram_idx in range(self.max_n - 1):
                base_vocab = self.cfg.engram_vocab_size[ngram_idx]
                head_primes: list[int] = []
                search_start = base_vocab - 1
                for _ in range(self.n_heads):
                    p = _find_next_prime(search_start, seen_primes)
                    seen_primes.add(p)
                    head_primes.append(p)
                    search_start = p
                per_layer.append(head_primes)
            result[layer_id] = per_layer
        return result

    # ------------------------------------------------------------------
    # Core hashing
    # ------------------------------------------------------------------

    def _hash_single_layer(
        self, compressed_ids: np.ndarray, layer_id: int
    ) -> np.ndarray:
        """Compute all multi-head hashes for one layer.

        Args:
            compressed_ids: [B, T] int64
            layer_id: which layer's multipliers to use
        Returns:
            [B, T, total_heads] int64 where total_heads = (max_n-1)*n_heads
        """
        B, T = compressed_ids.shape
        mults = self.layer_multipliers[layer_id]  # [max_n]

        # Pre-build shifted views: shift_k gives tokens at position t-k
        def _shift(k: int) -> np.ndarray:
            if k == 0:
                return compressed_ids
            padded = np.pad(
                compressed_ids,
                ((0, 0), (k, 0)),
                mode="constant",
                constant_values=self.pad_id,
            )
            return padded[:, :T]

        shifts = [_shift(k) for k in range(self.max_n)]

        all_hashes: list[np.ndarray] = []
        for n in range(2, self.max_n + 1):
            ngram_idx = n - 2
            # Multiplicative XOR mixing
            mix = shifts[0] * mults[0]
            for k in range(1, n):
                mix = np.bitwise_xor(mix, shifts[k] * mults[k])

            head_primes = self.vocab_sizes_per_layer[layer_id][ngram_idx]
            for j in range(self.n_heads):
                mod = int(head_primes[j])
                all_hashes.append((mix % mod).astype(np.int64, copy=False))

        return np.stack(all_hashes, axis=2)  # [B, T, total_heads]

    def hash(
        self, input_ids: np.ndarray
    ) -> dict[int, np.ndarray]:
        """Full pipeline: compress then hash for all layers.

        Args:
            input_ids: [B, T] raw token IDs
        Returns:
            {layer_id: [B, T, total_heads]}
        """
        compressed = self.compressed_tokenizer.compress(input_ids)
        return {
            lid: self._hash_single_layer(compressed, lid)
            for lid in self.layer_ids
        }

    def flat_vocab_sizes(self, layer_id: int) -> list[int]:
        """Flatten per-(order, head) prime sizes for MultiHeadEmbedding construction."""
        return [
            p
            for order_primes in self.vocab_sizes_per_layer[layer_id]
            for p in order_primes
        ]

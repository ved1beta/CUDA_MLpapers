"""
Multi-head embedding table with offset addressing.

All heads share a single contiguous nn.Embedding allocation.
Head k with local index i maps to global index: offset[k] + i.
This is more memory-efficient than separate tables and enables
a single fused gather operation in CUDA.
"""

from typing import List

import torch
import torch.nn as nn


class MultiHeadEmbedding(nn.Module):
    """Single nn.Embedding serving multiple heads via offset addressing.

    Args:
        vocab_sizes: per-head vocabulary sizes (prime moduli).
        embed_dim:   dimension per head embedding.
    """

    def __init__(self, vocab_sizes: List[int], embed_dim: int):
        super().__init__()
        self.num_heads = len(vocab_sizes)
        self.embed_dim = embed_dim
        self.vocab_sizes = list(vocab_sizes)

        total_vocab = sum(vocab_sizes)
        self.embedding = nn.Embedding(
            num_embeddings=total_vocab, embedding_dim=embed_dim
        )

        offsets = [0]
        for v in vocab_sizes[:-1]:
            offsets.append(offsets[-1] + v)
        self.register_buffer(
            "offsets", torch.tensor(offsets, dtype=torch.long)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Gather embeddings for all heads simultaneously.

        Args:
            input_ids: [B, T, num_heads] local indices per head.
        Returns:
            [B, T, num_heads, embed_dim]
        """
        # Broadcast offsets: [1, 1, num_heads]
        shifted = input_ids + self.offsets.unsqueeze(0).unsqueeze(0)
        return self.embedding(shifted)

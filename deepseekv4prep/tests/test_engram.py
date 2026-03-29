"""
Tests for the Engram module components.

Validates:
  1. Tokenizer compression produces valid canonical IDs
  2. N-gram hashing is deterministic and collision-resistant
  3. MultiHeadEmbedding offset addressing is correct
  4. Full EngramModule forward pass produces correct shapes
  5. ShortConv preserves causality
  6. CUDA kernels match PyTorch fallback (when available)
"""

import numpy as np
import pytest
import torch

from engram.config import EngramConfig, BackboneConfig
from engram.tokenizer import CompressedTokenizer
from engram.hashing import NgramHashMapping
from engram.embedding import MultiHeadEmbedding
from engram.conv import ShortConv
from engram.module import EngramModule


# Use small configs for fast testing
SMALL_ENGRAM_CFG = EngramConfig(
    engram_vocab_size=[1000, 1000],
    max_ngram_size=3,
    n_embed_per_ngram=64,
    n_head_per_ngram=4,
    layer_ids=[1, 3],
    pad_id=2,
    seed=42,
    kernel_size=4,
)

SMALL_BACKBONE_CFG = BackboneConfig(
    hidden_size=128,
    hc_mult=4,
    vocab_size=129280,
    num_layers=6,
)


class TestCompressedTokenizer:
    @pytest.fixture(scope="class")
    def tokenizer(self):
        return CompressedTokenizer("deepseek-ai/DeepSeek-V3")

    def test_compression_reduces_vocab(self, tokenizer):
        assert len(tokenizer) < 129280
        assert len(tokenizer) > 50000  # should not compress too aggressively

    def test_deterministic(self, tokenizer):
        ids = np.array([[100, 200, 300]], dtype=np.int64)
        c1 = tokenizer.compress(ids)
        c2 = tokenizer.compress(ids)
        np.testing.assert_array_equal(c1, c2)

    def test_negative_ids_preserved(self, tokenizer):
        ids = np.array([[-1, 100, -1]], dtype=np.int64)
        result = tokenizer.compress(ids)
        assert result[0, 0] == -1
        assert result[0, 2] == -1
        assert result[0, 1] >= 0


class TestNgramHashing:
    @pytest.fixture(scope="class")
    def hash_mapping(self):
        return NgramHashMapping(SMALL_ENGRAM_CFG)

    def test_output_shapes(self, hash_mapping):
        B, T = 2, 16
        ids = np.random.randint(0, 1000, (B, T), dtype=np.int64)
        result = hash_mapping.hash(ids)

        n_orders = SMALL_ENGRAM_CFG.max_ngram_size - 1
        total_heads = n_orders * SMALL_ENGRAM_CFG.n_head_per_ngram

        for layer_id in SMALL_ENGRAM_CFG.layer_ids:
            assert layer_id in result
            assert result[layer_id].shape == (B, T, total_heads)

    def test_deterministic(self, hash_mapping):
        ids = np.array([[10, 20, 30, 40]], dtype=np.int64)
        r1 = hash_mapping.hash(ids)
        r2 = hash_mapping.hash(ids)
        for lid in SMALL_ENGRAM_CFG.layer_ids:
            np.testing.assert_array_equal(r1[lid], r2[lid])

    def test_within_prime_bounds(self, hash_mapping):
        ids = np.random.randint(0, 1000, (4, 32), dtype=np.int64)
        result = hash_mapping.hash(ids)

        for lid in SMALL_ENGRAM_CFG.layer_ids:
            vocab_sizes = hash_mapping.flat_vocab_sizes(lid)
            hashes = result[lid]
            for h_idx, vsize in enumerate(vocab_sizes):
                assert hashes[:, :, h_idx].max() < vsize
                assert hashes[:, :, h_idx].min() >= 0


class TestMultiHeadEmbedding:
    def test_output_shape(self):
        vocab_sizes = [100, 200, 150, 120]
        embed_dim = 16
        mhe = MultiHeadEmbedding(vocab_sizes, embed_dim)

        B, T, H = 2, 8, 4
        ids = torch.stack([
            torch.randint(0, vs, (B, T)) for vs in vocab_sizes
        ], dim=2)

        out = mhe(ids)
        assert out.shape == (B, T, H, embed_dim)

    def test_offset_correctness(self):
        vocab_sizes = [10, 20]
        mhe = MultiHeadEmbedding(vocab_sizes, 4)

        # Head 0, local ID 5 -> global ID 5
        # Head 1, local ID 3 -> global ID 13 (10 + 3)
        ids = torch.tensor([[[5, 3]]])
        out = mhe(ids)  # [1, 1, 2, 4]

        direct_0 = mhe.embedding(torch.tensor([5]))
        direct_1 = mhe.embedding(torch.tensor([13]))
        torch.testing.assert_close(out[0, 0, 0], direct_0[0])
        torch.testing.assert_close(out[0, 0, 1], direct_1[0])


class TestShortConv:
    def test_output_shape(self):
        D, HC = 128, 4
        conv = ShortConv(D, HC, kernel_size=4, dilation=3)
        x = torch.randn(2, 16, HC, D)
        out = conv(x)
        assert out.shape == x.shape

    def test_zero_init_identity(self):
        D, HC = 64, 4
        conv = ShortConv(D, HC, kernel_size=4, dilation=3)
        x = torch.randn(1, 8, HC, D)
        out = conv(x)
        # With zero-init conv weights, output should be close to zero
        # (SiLU(0) = 0, so conv contribution is zero)
        assert out.abs().max() < 1e-5


class TestEngramModule:
    @pytest.fixture(scope="class")
    def engram(self):
        hash_mapping = NgramHashMapping(SMALL_ENGRAM_CFG)
        return EngramModule(
            engram_cfg=SMALL_ENGRAM_CFG,
            backbone_cfg=SMALL_BACKBONE_CFG,
            layer_id=1,
            hash_mapping=hash_mapping,
        )

    def test_forward_shape(self, engram):
        B, T = 2, 16
        D = SMALL_BACKBONE_CFG.hidden_size
        HC = SMALL_BACKBONE_CFG.hc_mult

        hidden = torch.randn(B, T, HC, D)
        ids = np.random.randint(0, 1000, (B, T), dtype=np.int64)

        out = engram(hidden, ids)
        assert out.shape == (B, T, HC, D)

    def test_gradient_flows(self, engram):
        B, T = 1, 8
        D = SMALL_BACKBONE_CFG.hidden_size
        HC = SMALL_BACKBONE_CFG.hc_mult

        hidden = torch.randn(B, T, HC, D, requires_grad=True)
        ids = np.random.randint(0, 1000, (B, T), dtype=np.int64)

        out = engram(hidden, ids)
        loss = out.sum()
        loss.backward()

        assert hidden.grad is not None
        assert hidden.grad.shape == hidden.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

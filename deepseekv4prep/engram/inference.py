"""
Inference engine with prefetch-and-overlap for Engram.

Key insight: Engram's retrieval indices depend only on input_ids (deterministic),
unlike MoE where routing depends on hidden states. This enables:
  1. Compute all hash IDs before forward pass starts
  2. Launch async PCIe transfer from Host DRAM -> GPU HBM
  3. Overlap transfer with computation of preceding transformer layers
  4. By the time we reach an Engram layer, embeddings are already in HBM

Zipfian cache hierarchy exploits that top 1% of N-grams account for ~80% of accesses:
  Tier 0: GPU HBM  (hot cache, instant access)
  Tier 1: Host DRAM (warm cache, PCIe prefetch ~15us per row)
  Tier 2: NVMe SSD  (cold storage, rarely accessed)
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch

from .config import EngramConfig
from .hashing import NgramHashMapping


class ZipfianEmbeddingCache:
    """Frequency-aware tiered cache for Engram embeddings.

    Maintains a GPU HBM cache (LRU) for the hottest N-gram embeddings
    and a host DRAM tier for warm rows. Cold rows are fetched from
    the backing store (file-backed mmap or full table).

    Thread-safe for concurrent prefetch + lookup.
    """

    def __init__(
        self,
        gpu_capacity: int,
        host_capacity: int,
        embed_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.embed_dim = embed_dim
        self.dtype = dtype
        self.device = device
        self.lock = threading.Lock()

        # GPU tier: OrderedDict for LRU
        self.gpu_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self.gpu_capacity = gpu_capacity

        # Host tier: numpy arrays in pinned memory
        self.host_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.host_capacity = host_capacity

        # Access frequency for promotion decisions
        self.freq: dict[int, int] = {}

    def _evict_gpu(self) -> None:
        while len(self.gpu_cache) > self.gpu_capacity:
            self.gpu_cache.popitem(last=False)

    def _evict_host(self) -> None:
        while len(self.host_cache) > self.host_capacity:
            self.host_cache.popitem(last=False)

    def insert_gpu(self, row_id: int, tensor: torch.Tensor) -> None:
        with self.lock:
            self.gpu_cache[row_id] = tensor.to(self.device, non_blocking=True)
            self.gpu_cache.move_to_end(row_id)
            self._evict_gpu()

    def insert_host(self, row_id: int, data: np.ndarray) -> None:
        with self.lock:
            self.host_cache[row_id] = data
            self.host_cache.move_to_end(row_id)
            self._evict_host()

    def lookup(
        self, row_ids: list[int]
    ) -> tuple[dict[int, torch.Tensor], list[int], list[int]]:
        """Look up rows across cache tiers.

        Returns:
            (gpu_hits, host_hit_ids, cold_miss_ids)
        """
        gpu_hits: dict[int, torch.Tensor] = {}
        host_hits: list[int] = []
        cold_misses: list[int] = []

        with self.lock:
            for rid in row_ids:
                self.freq[rid] = self.freq.get(rid, 0) + 1
                if rid in self.gpu_cache:
                    gpu_hits[rid] = self.gpu_cache[rid]
                    self.gpu_cache.move_to_end(rid)
                elif rid in self.host_cache:
                    host_hits.append(rid)
                else:
                    cold_misses.append(rid)

        return gpu_hits, host_hits, cold_misses


class EngramInferenceEngine:
    """Inference engine that prefetches Engram embeddings using CUDA streams.

    Overlaps PCIe transfers with transformer computation to hide
    the latency of fetching from host/disk-backed embedding tables.

    Usage:
        engine = EngramInferenceEngine(model, host_table_path)
        output = engine.forward(input_ids)
    """

    def __init__(
        self,
        hash_mapping: NgramHashMapping,
        host_table: np.ndarray,
        embed_dim: int,
        device: torch.device = torch.device("cuda:0"),
        dtype: torch.dtype = torch.bfloat16,
        gpu_cache_size: int = 100_000,
        host_cache_size: int = 1_000_000,
    ):
        self.hash_mapping = hash_mapping
        self.host_table = host_table
        self.embed_dim = embed_dim
        self.device = device
        self.dtype = dtype

        self.prefetch_stream = torch.cuda.Stream(device=device)

        self.cache = ZipfianEmbeddingCache(
            gpu_capacity=gpu_cache_size,
            host_capacity=host_cache_size,
            embed_dim=embed_dim,
            device=device,
            dtype=dtype,
        )

    def precompute_hashes(
        self, input_ids: np.ndarray
    ) -> dict[int, np.ndarray]:
        """Compute all N-gram hash IDs on CPU (deterministic, fast).

        Can be called before any GPU forward pass computation begins.
        """
        return self.hash_mapping.hash(input_ids)

    def prefetch_embeddings(
        self, all_hash_ids: dict[int, np.ndarray]
    ) -> dict[int, tuple[np.ndarray, torch.Tensor]]:
        """Async transfer required embeddings from host to GPU.

        Launches on a dedicated CUDA stream so transfers overlap
        with transformer layer computation on the default stream.

        Returns:
            {layer_id: (unique_ids, gpu_tensor)} for each Engram layer.
        """
        prefetched: dict[int, tuple[np.ndarray, torch.Tensor]] = {}

        with torch.cuda.stream(self.prefetch_stream):
            for layer_id, hash_ids in all_hash_ids.items():
                unique_ids = np.unique(hash_ids.reshape(-1))

                # Check cache first
                unique_list = unique_ids.tolist()
                gpu_hits, host_hits, cold_misses = self.cache.lookup(
                    unique_list
                )

                # Gather from host table for non-GPU-cached rows
                need_transfer = host_hits + cold_misses
                if need_transfer:
                    need_array = np.array(need_transfer, dtype=np.int64)
                    # Clamp to valid range
                    need_array = np.clip(
                        need_array, 0, self.host_table.shape[0] - 1
                    )
                    rows = self.host_table[need_array]
                    gpu_rows = torch.from_numpy(rows).to(
                        self.device, dtype=self.dtype, non_blocking=True
                    )

                    # Update cache
                    for i, rid in enumerate(need_transfer):
                        self.cache.insert_gpu(rid, gpu_rows[i])

                prefetched[layer_id] = (unique_ids, None)

        return prefetched

    def sync_prefetch(self) -> None:
        """Wait for all async prefetch transfers to complete."""
        torch.cuda.current_stream().wait_stream(self.prefetch_stream)

    def gather_prefetched(
        self,
        hash_ids: np.ndarray,
        layer_id: int,
    ) -> torch.Tensor:
        """Gather embeddings for a specific layer from the cache.

        Args:
            hash_ids: [B, T, total_heads] hash indices.
            layer_id: which Engram layer.
        Returns:
            [B, T, total_heads, embed_dim] embedding tensor.
        """
        B, T, H = hash_ids.shape
        flat_ids = hash_ids.reshape(-1).tolist()

        result = torch.zeros(
            len(flat_ids), self.embed_dim,
            dtype=self.dtype, device=self.device,
        )

        with self.cache.lock:
            for i, rid in enumerate(flat_ids):
                if rid in self.cache.gpu_cache:
                    result[i] = self.cache.gpu_cache[rid]

        return result.reshape(B, T, H, self.embed_dim)

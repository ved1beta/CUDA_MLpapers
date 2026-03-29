"""
Distributed Engram embedding table with All-to-All communication.

Shards the massive N-gram embedding table across GPUs. Each rank owns
a contiguous slice: [rank * shard_size, (rank+1) * shard_size).

Communication pattern:
  Forward:  All-to-All scatter (send ID requests) + gather (receive embeddings)
  Backward: All-to-All scatter (send gradients) + local accumulate

Optimizer split: embeddings use Adam (sparse-friendly, higher LR),
backbone uses Muon/AdamW.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist


class DistributedEngramTable(nn.Module):
    """Embedding table sharded across world_size GPUs via All-to-All.

    Each rank holds rows [rank * shard_size, (rank+1) * shard_size).
    Lookup of arbitrary global IDs requires cross-GPU communication.

    Args:
        total_vocab:  total number of embedding rows across all shards.
        embed_dim:    embedding dimension per row.
        process_group: optional process group (defaults to WORLD).
    """

    def __init__(
        self,
        total_vocab: int,
        embed_dim: int,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        super().__init__()
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        self.rank = dist.get_rank(process_group)
        self.total_vocab = total_vocab
        self.embed_dim = embed_dim

        self.shard_size = math.ceil(total_vocab / self.world_size)
        local_vocab = min(
            self.shard_size,
            max(0, total_vocab - self.rank * self.shard_size),
        )

        self.local_embed = nn.Embedding(local_vocab, embed_dim)

    def forward(self, global_ids: torch.Tensor) -> torch.Tensor:
        """Distributed embedding lookup via sparse All-to-All.

        Args:
            global_ids: [*shape] int64 tensor of global embedding IDs.
        Returns:
            [*shape, embed_dim] embedding tensor.
        """
        orig_shape = global_ids.shape
        flat_ids = global_ids.reshape(-1)
        N = flat_ids.shape[0]

        # Determine which rank owns each ID
        owner_ranks = flat_ids.div(self.shard_size, rounding_mode="floor")
        local_ids = flat_ids % self.shard_size

        # Sort by owner rank for contiguous communication
        sort_order = torch.argsort(owner_ranks)
        sorted_local_ids = local_ids[sort_order]
        sorted_owners = owner_ranks[sort_order]

        # Count how many IDs go to each rank
        send_counts = torch.zeros(
            self.world_size, dtype=torch.long, device=flat_ids.device
        )
        send_counts.scatter_add_(
            0, sorted_owners, torch.ones_like(sorted_owners)
        )
        send_counts_list = send_counts.tolist()

        # Exchange counts
        recv_counts = torch.zeros_like(send_counts)
        dist.all_to_all_single(
            recv_counts, send_counts, group=self.process_group
        )
        recv_counts_list = recv_counts.tolist()

        total_recv = sum(recv_counts_list)

        # Exchange the actual ID values
        recv_ids = torch.empty(
            total_recv, dtype=sorted_local_ids.dtype,
            device=sorted_local_ids.device,
        )
        dist.all_to_all_single(
            recv_ids, sorted_local_ids,
            output_split_sizes=recv_counts_list,
            input_split_sizes=send_counts_list,
            group=self.process_group,
        )

        # Local embedding lookup on this rank's shard
        recv_ids_clamped = recv_ids.clamp(
            0, self.local_embed.num_embeddings - 1
        )
        local_embs = self.local_embed(recv_ids_clamped)  # [total_recv, D]

        # Send embeddings back to requesters
        send_back = torch.empty(
            N, self.embed_dim,
            dtype=local_embs.dtype, device=local_embs.device,
        )
        dist.all_to_all_single(
            send_back, local_embs,
            output_split_sizes=send_counts_list,
            input_split_sizes=recv_counts_list,
            group=self.process_group,
        )

        # Unsort to restore original ordering
        inv_order = torch.argsort(sort_order)
        result = send_back[inv_order]

        return result.reshape(*orig_shape, self.embed_dim)


def build_optimizer_groups(
    model: nn.Module,
    backbone_lr: float = 4e-4,
    embed_lr_scale: float = 5.0,
    backbone_weight_decay: float = 0.1,
) -> list[dict]:
    """Split parameters into backbone (Muon/AdamW) and embedding (Adam) groups.

    Embeddings get higher LR and no weight decay because:
    - They receive sparse gradients (only activated rows updated per step)
    - Adam handles sparse updates well via per-parameter adaptive LR
    - Weight decay on sparse embeddings causes unused rows to decay to zero

    Args:
        model: the full model with DistributedEngramTable modules.
        backbone_lr: learning rate for backbone parameters.
        embed_lr_scale: multiplier for embedding LR relative to backbone.
        backbone_weight_decay: weight decay for backbone params.
    Returns:
        list of param group dicts for torch.optim.
    """
    embed_params = []
    backbone_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "local_embed" in name or "multi_head_embedding" in name:
            embed_params.append(param)
        else:
            backbone_params.append(param)

    return [
        {
            "params": backbone_params,
            "lr": backbone_lr,
            "weight_decay": backbone_weight_decay,
            "name": "backbone",
        },
        {
            "params": embed_params,
            "lr": backbone_lr * embed_lr_scale,
            "weight_decay": 0.0,
            "name": "embeddings",
        },
    ]

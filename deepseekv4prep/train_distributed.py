"""
Distributed training script for Engram-augmented transformer.

Supports:
  - Data Parallel (DDP) across multiple GPUs
  - Sharded Engram embedding table via DistributedEngramTable
  - Split optimizer: Adam for embeddings (5x LR, no weight decay),
    AdamW for backbone (standard LR + weight decay)

Launch:
  torchrun --nproc_per_node=4 train_distributed.py
  torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
           --master_addr=<MASTER> --master_port=29500 train_distributed.py
"""

from __future__ import annotations

import os
import time
import math
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from engram.config import EngramConfig, BackboneConfig
from engram.hashing import NgramHashMapping
from engram.module import EngramModule, TransformerBlockWithEngram
from engram.distributed import DistributedEngramTable, build_optimizer_groups


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed Engram Training")
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=30)
    parser.add_argument("--hc-mult", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--embed-lr-scale", type=float, default=5.0)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


class EngramTransformer(nn.Module):
    """Full transformer with Engram modules and distributed embedding tables."""

    def __init__(
        self,
        engram_cfg: EngramConfig,
        backbone_cfg: BackboneConfig,
        hash_mapping: NgramHashMapping,
        use_distributed_embed: bool = False,
    ):
        super().__init__()
        self.backbone_cfg = backbone_cfg
        self.engram_cfg = engram_cfg
        self.hash_mapping = hash_mapping

        self.token_embed = nn.Embedding(
            backbone_cfg.vocab_size, backbone_cfg.hidden_size
        )

        self.layers = nn.ModuleList()
        for layer_id in range(backbone_cfg.num_layers):
            engram_mod = None
            if layer_id in engram_cfg.layer_ids:
                engram_mod = EngramModule(
                    engram_cfg=engram_cfg,
                    backbone_cfg=backbone_cfg,
                    layer_id=layer_id,
                    hash_mapping=hash_mapping,
                )
            self.layers.append(
                TransformerBlockWithEngram(
                    backbone_cfg=backbone_cfg,
                    layer_id=layer_id,
                    engram_module=engram_mod,
                )
            )

        self.lm_head = nn.Linear(
            backbone_cfg.hidden_size, backbone_cfg.vocab_size, bias=False
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        input_ids_np: np.ndarray,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        HC = self.backbone_cfg.hc_mult

        # Token embedding -> expand to multi-branch
        hidden = self.token_embed(input_ids)
        hidden = hidden.unsqueeze(2).expand(-1, -1, HC, -1).contiguous()

        for layer in self.layers:
            hidden = layer(hidden, input_ids_np)

        # Collapse branches (first branch for LM head)
        logits = self.lm_head(hidden[:, :, 0, :])
        return logits


class SyntheticDataset(torch.utils.data.Dataset):
    """Synthetic token dataset for testing distributed training."""

    def __init__(self, vocab_size: int, seq_len: int, num_samples: int = 10000):
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens = self.data[idx]
        return tokens[:-1], tokens[1:]  # input, target


def main():
    args = parse_args()

    # Initialize distributed
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    if rank == 0:
        print(f"Training with {world_size} GPUs")
        print(f"Config: hidden={args.hidden_size}, layers={args.num_layers}, "
              f"hc_mult={args.hc_mult}")

    # Configs
    engram_cfg = EngramConfig(
        engram_vocab_size=[1000, 1000],  # small for demo
        n_embed_per_ngram=128,
        n_head_per_ngram=4,
        layer_ids=[1, 15] if args.num_layers > 15 else [1],
    )
    backbone_cfg = BackboneConfig(
        hidden_size=args.hidden_size,
        hc_mult=args.hc_mult,
        num_layers=args.num_layers,
    )

    torch.manual_seed(args.seed + rank)

    # Build model
    hash_mapping = NgramHashMapping(engram_cfg)
    model = EngramTransformer(
        engram_cfg=engram_cfg,
        backbone_cfg=backbone_cfg,
        hash_mapping=hash_mapping,
    ).to(device=device, dtype=torch.bfloat16)

    model = DDP(model, device_ids=[local_rank])

    # Optimizer split: backbone vs embeddings
    param_groups = build_optimizer_groups(
        model,
        backbone_lr=args.lr,
        embed_lr_scale=args.embed_lr_scale,
        backbone_weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups)

    # Synthetic data
    dataset = SyntheticDataset(
        backbone_cfg.vocab_size, args.seq_len + 1, num_samples=args.steps * args.batch_size
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=2, pin_memory=True,
    )

    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    step = 0
    t0 = time.time()

    for epoch in range(100):
        sampler.set_epoch(epoch)
        for input_ids, targets in loader:
            if step >= args.steps:
                break

            input_ids = input_ids.to(device)
            targets = targets.to(device)
            input_ids_np = input_ids.cpu().numpy().astype(np.int64)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids, input_ids_np)
                loss = loss_fn(
                    logits.reshape(-1, backbone_cfg.vocab_size),
                    targets.reshape(-1),
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1

            if rank == 0 and step % args.log_interval == 0:
                elapsed = time.time() - t0
                tokens_per_sec = (
                    step * args.batch_size * args.seq_len * world_size / elapsed
                )
                print(
                    f"Step {step}/{args.steps} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Tokens/s: {tokens_per_sec:.0f} | "
                    f"Time: {elapsed:.1f}s"
                )

        if step >= args.steps:
            break

    if rank == 0:
        total_time = time.time() - t0
        print(f"\nTraining complete in {total_time:.1f}s")
        print(f"Average tokens/s: {args.steps * args.batch_size * args.seq_len * world_size / total_time:.0f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

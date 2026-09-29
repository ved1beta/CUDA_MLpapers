"""Render per-framework configs for one (seq_len, run_tag). Shared settings live in COMMON."""
import json
import sys
from pathlib import Path

import yaml

B = Path("/workspace/data/bench")
MODEL = "zai-org/GLM-4.7-Flash"
COMMON = dict(
    tokens_per_step=32768,
    steps=40,
    lr=1e-4,
    seed=42,
    lora_r=16,
    lora_alpha=32,
    targets=["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"],
    wandb_project="glm47-flash-lora-bench",
)


def axolotl(seq_len, tag, extra=None):
    c = COMMON
    cfg = {
        "base_model": MODEL,
        "plugins": ["axolotl.integrations.cut_cross_entropy.CutCrossEntropyPlugin"],
        "datasets": [{
            "path": "synthetic", "type": "_synthetic", "length": c["tokens_per_step"] * 50 // seq_len,
            "sequence_length": seq_len, "min_input_id": 100, "max_input_id": 32000, "seed": 42,
        }],
        "dataset_prepared_path": str(B / "prepared" / f"axolotl-{seq_len}"),
        "val_set_size": 0,
        "output_dir": str(B / "runs" / tag),
        "sequence_len": seq_len,
        "sample_packing": False,
        "pad_to_sequence_len": False,
        "adapter": "lora",
        "lora_r": c["lora_r"], "lora_alpha": c["lora_alpha"], "lora_dropout": 0.0,
        "lora_target_modules": c["targets"],
        "lora_mlp_kernel": False, "lora_qkv_kernel": False, "lora_o_kernel": False,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": c["tokens_per_step"] // seq_len,
        "max_steps": c["steps"],
        "optimizer": "adamw_torch_fused",
        "lr_scheduler": "constant",
        "learning_rate": c["lr"],
        "warmup_steps": 0,
        "bf16": True, "tf32": True,
        "gradient_checkpointing": True,
        "attn_implementation": "flash_attention_2",
        "logging_steps": 1,
        "save_strategy": "no",
        "seed": c["seed"],
        "wandb_project": c["wandb_project"],
        "wandb_name": tag,
    }
    if (extra or {}).pop("_real_data", False):
        cfg["datasets"] = [{"path": "tatsu-lab/alpaca", "type": "alpaca"}]
        cfg["dataset_prepared_path"] = str(B / "prepared" / f"axolotl-alpaca-{seq_len}")
        cfg["sample_packing"] = True
    cfg.update(extra or {})
    return cfg


def primerl(seq_len, tag, extra=None):
    c = COMMON
    ga = c["tokens_per_step"] // seq_len
    cfg = {
        "max_steps": c["steps"],
        "output_dir": str(B / "runs" / tag),
        "model": {
            "name": MODEL, "seq_len": seq_len, "attn": "flash_attention_2", "impl": "auto",
            "ac": {"mode": "full"}, "ac_offloading": "None", "optim_cpu_offload": False,
            "optimization_dtype": "bfloat16",  # fp32 (default) = 120GB master weights, cannot fit
            "lora": {"rank": c["lora_r"], "alpha": float(c["lora_alpha"]), "dropout": 0.0,
                     "target_modules": c["targets"]},
        },
        "data": {"type": "fake", "batch_size": ga, "micro_batch_size": 1, "seq_len": seq_len,
                 "input_ids": "random", "length": "fixed", "seed": c["seed"]},
        "optim": {"type": "adamw", "lr": c["lr"]},
        "scheduler": {"type": "constant"},
        "monitors": {"wandb": {"project": c["wandb_project"], "name": tag, "group": "primerl"}},
        "dashboard": False,
    }
    for k, v in (extra or {}).items():
        d = cfg
        *path, last = k.split(".")
        for p in path:
            d = d.setdefault(p, {})
        d[last] = v
    return cfg


def llamafactory(seq_len, tag, extra=None):
    c = COMMON
    cfg = {
        "model_name_or_path": MODEL,
        "trust_remote_code": True,
        "stage": "sft", "do_train": True, "finetuning_type": "lora",
        "lora_rank": c["lora_r"], "lora_alpha": c["lora_alpha"], "lora_dropout": 0.0,
        "lora_target": ",".join(c["targets"]),
        "tokenized_path": str(B / "datasets" / f"synthetic-{seq_len}"),
        # required by arg validation; ignored once tokenized_path exists
        "dataset": "identity", "dataset_dir": str(B / "llamafactory/src/data"), "template": "empty",
        "cutoff_len": seq_len,
        "output_dir": str(B / "runs" / tag),
        "overwrite_output_dir": True,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": c["tokens_per_step"] // seq_len,
        "max_steps": c["steps"],
        "learning_rate": c["lr"], "lr_scheduler_type": "constant", "warmup_steps": 0,
        "optim": "adamw_torch_fused",
        "bf16": True, "tf32": True,
        "flash_attn": "fa2",  # GC is on by default (disable_gradient_checkpointing=False)
        "logging_steps": 1, "save_strategy": "no",
        "seed": c["seed"],
        "report_to": "wandb", "run_name": tag,
    }
    cfg.update(extra or {})
    return cfg


def unsloth(seq_len, tag, extra=None):
    c = COMMON
    cfg = {
        "model": MODEL, "seq_len": seq_len, "tag": tag,
        "dataset": str(B / "datasets" / f"synthetic-{seq_len}"),
        "output_dir": str(B / "runs" / tag),
        "lora_r": c["lora_r"], "lora_alpha": c["lora_alpha"], "targets": c["targets"],
        "grad_accum": c["tokens_per_step"] // seq_len, "steps": c["steps"],
        "lr": c["lr"], "seed": c["seed"], "wandb_project": c["wandb_project"],
    }
    cfg.update(extra or {})
    return cfg


if __name__ == "__main__":
    fw, seq_len, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    extra = json.loads(sys.argv[4]) if len(sys.argv) > 4 else None
    cfg = globals()[fw](seq_len, tag, extra)
    out = B / "configs" / f"{tag}.{'toml' if fw == 'primerl' else 'yaml'}"
    out.parent.mkdir(parents=True, exist_ok=True)
    if fw == "primerl":
        import tomli_w
        out.write_bytes(tomli_w.dumps(cfg).encode())
    else:
        out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(out)

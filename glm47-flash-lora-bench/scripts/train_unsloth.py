"""Unsloth LoRA SFT on the pre-tokenized shared synthetic rows."""
import sys

import yaml

cfg = yaml.safe_load(open(sys.argv[1]))

import unsloth  # noqa: F401  must precede transformers/trl
from unsloth import FastLanguageModel
from datasets import load_from_disk
from transformers import DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=cfg["model"],
    max_seq_length=cfg["seq_len"],
    dtype=None,
    load_in_4bit=False,
    load_in_8bit=False,
    full_finetuning=False,
    attn_implementation=cfg.get("attn_implementation", "flash_attention_2"),
)
model = FastLanguageModel.get_peft_model(
    model,
    r=cfg["lora_r"],
    lora_alpha=cfg["lora_alpha"],
    lora_dropout=0,
    target_modules=cfg["targets"],
    bias="none",
    use_gradient_checkpointing=cfg.get("use_gradient_checkpointing", "unsloth"),
    random_state=cfg["seed"],
)
model.print_trainable_parameters()

ds = load_from_disk(cfg["dataset"])

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer, padding=False),
    args=SFTConfig(
        output_dir=cfg["output_dir"],
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=cfg["seq_len"],
        packing=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cfg["grad_accum"],
        max_steps=cfg["steps"],
        learning_rate=cfg["lr"],
        lr_scheduler_type="constant",
        warmup_steps=0,
        optim="adamw_torch_fused",
        bf16=True,
        tf32=True,
        logging_steps=1,
        save_strategy="no",
        seed=cfg["seed"],
        report_to="wandb",
        run_name=cfg["tag"],
        remove_unused_columns=False,
    ),
)
trainer.train()

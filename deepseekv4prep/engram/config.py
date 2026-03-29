from dataclasses import dataclass, field
from typing import List


@dataclass
class EngramConfig:
    tokenizer_name_or_path: str = "deepseek-ai/DeepSeek-V3"
    engram_vocab_size: List[int] = field(
        default_factory=lambda: [129280 * 5, 129280 * 5]
    )
    max_ngram_size: int = 3
    n_embed_per_ngram: int = 512
    n_head_per_ngram: int = 8
    layer_ids: List[int] = field(default_factory=lambda: [1, 15])
    pad_id: int = 2
    seed: int = 0
    kernel_size: int = 4


@dataclass
class BackboneConfig:
    hidden_size: int = 2560
    hc_mult: int = 4
    vocab_size: int = 129280
    num_layers: int = 30

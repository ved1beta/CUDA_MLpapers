"""Export axolotl `_synthetic` rows (identical RNG path) as HF datasets on disk."""
import sys
from axolotl.prompt_strategies._synthetic import SyntheticDatasetStrategy

TOKENS = 32768 * 50  # 50 optimizer steps worth
for seq_len in map(int, sys.argv[1:]):
    n = TOKENS // seq_len
    ds = SyntheticDatasetStrategy(sequence_length=seq_len, length=n, min_input_id=100,
                                  max_input_id=32000, seed=42).wrap_dataset(None)
    ds.save_to_disk(f"/workspace/data/bench/datasets/synthetic-{seq_len}")
    print(seq_len, ds)

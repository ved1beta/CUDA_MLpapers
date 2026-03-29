"""
Vocabulary compression via surjective mapping P: V -> V'.
Collapses semantically equivalent tokens (case, accents, unicode normalization)
to maximize N-gram hash density. Achieves ~23% reduction for 128k tokenizers.
"""

import numpy as np
from transformers import AutoTokenizer
from tokenizers import normalizers, Regex


class CompressedTokenizer:
    """Pre-computes a lookup table that maps raw token IDs to canonical IDs.

    Normalization pipeline: NFKC -> NFD -> StripAccents -> Lowercase -> whitespace collapse.
    Tokens containing replacement char (byte-fallback) are keyed by their raw piece instead.
    """

    def __init__(self, tokenizer_name_or_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path, trust_remote_code=True
        )

        SENTINEL = "\uE000"
        self.normalizer = normalizers.Sequence([
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), SENTINEL),
            normalizers.Strip(),
            normalizers.Replace(SENTINEL, " "),
        ])

        self.lookup_table, self.num_new_tokens = self._build_lookup_table()

    def __len__(self) -> int:
        return self.num_new_tokens

    def _build_lookup_table(self) -> tuple[np.ndarray, int]:
        old2new: dict[int, int] = {}
        key2new: dict[str, int] = {}
        new_tokens: list[str] = []

        vocab_size = len(self.tokenizer)
        for tid in range(vocab_size):
            text = self.tokenizer.decode([tid], skip_special_tokens=False)

            if "\ufffd" in text:
                key = self.tokenizer.convert_ids_to_tokens(tid)
            else:
                norm = self.normalizer.normalize_str(text)
                key = norm if norm else text

            nid = key2new.get(key)
            if nid is None:
                nid = len(new_tokens)
                key2new[key] = nid
                new_tokens.append(key)
            old2new[tid] = nid

        lookup = np.empty(vocab_size, dtype=np.int64)
        for tid in range(vocab_size):
            lookup[tid] = old2new[tid]

        return lookup, len(new_tokens)

    def compress(self, input_ids: np.ndarray) -> np.ndarray:
        """Map raw token IDs to canonical IDs via the precomputed lookup table.

        Args:
            input_ids: int64 array of any shape.
        Returns:
            Compressed IDs with same shape.
        """
        arr = np.asarray(input_ids, dtype=np.int64)
        pos_mask = arr >= 0
        out = arr.copy()
        out[pos_mask] = self.lookup_table[arr[pos_mask]]
        return out

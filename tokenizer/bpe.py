from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    vocab: dict[int, bytes]     # index -> bytes
    merges: dict[tuple[int, int], int]  # index1,index2 -> new_index

def bep_tokerizer(text, merge):
    indices = list(map(int, text.encode('utf-8')))
    merge: dict[tuple[int, int], int] = {}
    counts = defaultdict(int)
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}

    for i in range(merge):
        counts = defaultdict(int)
        for index1 , index2 in zip(indices, indices[1:]):
            pair_dict = (index1, index2)
            counts[pair_dict]  += 1

        max(counts.items(), key=lambda x: x[1])

        new_index = 256 + i
        vocab[new_index] = vocab[index1] + vocab[index2]
    
        merge[pair_dict] = new_index
    
        indices = merge(indices, pair_dict, new_index)

        return (vocab, merge)
        
def merge(indices, pair_dict, new_index):
    merged_indices = []
    i = 0
    while i < len(indices):
        if i < len(indices) - 1 and (indices[i], indices[i + 1]) == pair_dict:
            merged_indices.append(new_index)
            i += 2
        else:
            merged_indices.append(indices[i])
            i += 1
    return merged_indices

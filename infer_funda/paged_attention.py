import torch 
import 

def paged_atten(q , k_C , v_c , block_table , seq_leq , block_size):
    num_heads , head_dim = q.shape
    kv_heads= k_C.shape[2]

    output = []
    group_size = num_heads // block_size

    for q_head in range(num_heads):

        kv_head = q_head // group_size
        key , value = [] , []

        for token in range(seq_leq):
            
            logical_block = token // block_size
            offset = token % block_size

            physical_block = block_table[logical_block]

            k_chache = k_C[physical_block , kv_head , offset]
            v_chache = v_c[physical_block , kv_head , offset]

            key.append(k_chache)
            value.append(v_chache)

    K = torch.stack(key)
    V = torch.stack(value)

    score = K @ q[q_head].unsqueeze(-1) / torch.sqrt(torch.tensor(head_dim , dtype=torch.float32))
    probs = torch.softmax(score , dim=0)
    
    temp = (probs * V)

    output.append(temp)        

    ###########
def gather_paged_kv(
    cache,
    block_table,
    seq_len,
    block_size,
):
    num_blocks = (
        seq_len + block_size - 1
    ) // block_size

    blocks = block_table[
        :num_blocks
    ]

    kv = cache[blocks]

    # [logical_blocks,
    #  block_size,
    #  heads,
    #  dim]

    kv = kv.reshape(
        -1,
        cache.shape[2],
        cache.shape[3],
    )

    return kv[:seq_len]

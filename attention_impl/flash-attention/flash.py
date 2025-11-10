import torch 
import torch.nn.functional as F
import math

def naive_attention(q, k, v, mask=None, causal=False):
    """
    Naive attention for correctness checks.
    q: (Lq, d)
    k: (Lk, d)
    v: (Lk, d_v)
    mask: optional (Lq, Lk) with False for positions to mask out (or -inf for logits)
    causal: if True, applies causal mask so position i cannot see j>i
    Returns: (Lq, d_v)
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    logits = q @ k.transpose(-2, -1) * scale  # (Lq, Lk)
    if mask is not None:
        # mask should be bool: True -> keep, False -> mask out
        logits = logits.masked_fill(~mask, float("-inf"))
    if causal:
        Lq, Lk = q.shape[0], k.shape[0]
        # allow positions j <= i only
        causal_mask = torch.tril(torch.ones((Lq, Lk), dtype=torch.bool, device=q.device))
        logits = logits.masked_fill(~causal_mask, float("-inf"))
    attn = F.softmax(logits, dim=-1)
    return attn @ v

def flash_atten(Q, K , V , mask = None , causal = False , q_block = 128 , kv_block = 128):
    device = Q.device

    Lq , d = Q.shape
    Lk = K.shape[0]
    d_v = V.shape[-1] # dimension of value vectors
    scale = 1.0 / math.sqrt(d)

    out = torch.zeros((Lq, d_v), dtype=Q.dtype, device=device)
    all_m, all_l = [], []
    for i in range(0 , Lq , q_block):

        q_end = min(i + q_block , Lq)
        q_block_tensor = Q[i : q_end]

        m = torch.full((q_end - i,), float("-inf"), device=device, dtype=Q.dtype) # track max 
        l = torch.zeros((q_end - i,), device=device, dtype=Q.dtype) # track difference
        numerator = torch.zeros((q_end - i, d_v), device=device, dtype=Q.dtype) # track numerator

        for kv_start in range(0 , Lk , kv_block):
            kv_end = min(kv_start + kv_block , Lk)
            k_block_tensor = K[kv_start : kv_end]
            v_block_tensor = V[kv_start : kv_end]

            logts = torch.matmul(q_block_tensor , k_block_tensor.transpose(-2 , -1)) * scale  # (q_block , kv_block)

            if mask is not None:
                mask_block = mask[i:q_end , kv_start:kv_end]
                logts = logts.masked_fill(~mask_block , float("-inf"))
                # mask = tensor([
#    [ True, False, False, False],
#    [ True,  True, False, False],
#    [ True,  True,  True, False],
#    [ True,  True,  True,  True]
#])
            if causal:
                q_pos = torch.arange(i, q_end , device=device)[:, None] # [Bq , 1]
                k_pos = torch.arange(kv_start, kv_end , device=device)[None , :] # [1 , Bk]
                causal_mask = (k_pos <= q_pos)
                logts = logts.masked_fill(~causal_mask , float("-inf"))

            m_block = torch.max(logts, dim=-1).values  # max of Bq(q_block) values
            exp_logits = torch.exp(logts - m_block.unsqueeze(-1))
            sum_exp = torch.sum(exp_logits , dim = -1)

            num_block = exp_logits @ v_block_tensor # (Bq , d_v)

            new_m = torch.maximum(m , m_block)

            if torch.isfinite(new_m).all():
                    exp_old = torch.exp(m - new_m)
                    exp_blk = torch.exp(m_block - new_m)
            else:
                    exp_old = torch.exp(m - new_m)
                    exp_blk = torch.exp(m_block - new_m)

            l = l * exp_old + sum_exp * exp_blk # correction 
            numerator = numerator * exp_old.unsqueeze(-1) + num_block * exp_blk.unsqueeze(-1)

            m = new_m  # Update m for next iteration

        denom = l.clamp_min(1e-24).unsqueeze(-1)  # (Bq, 1) # avoid division by zero
        out[i:q_end] = numerator / denom

        all_m.append(m)
        all_l.append(l)

    m_all = torch.cat(all_m, dim=0)
    l_all = torch.cat(all_l, dim=0)

    return out, m_all, l_all

def flash_attention_backward(dO , Q , K , V , out , l_all, m_all, block_size=64):

    batch_size , num_heads , seq_len , head_dim = Q.shape
    scale = 1.0 / math.sqrt(head_dim)

    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)

    """ Flash Attention Backward Pass.
    
    Mathematical Derivation:
    - Attention: P = softmax(QK^T / sqrt(d)), O = PV
    - We need: dQ, dK, dV given dO
    
    Key formulas:
    1. dV = P^T @ dO
    2. dP = dO @ V^T
    3. dS = P ⊙ (dP - D), where D = rowsum(dP ⊙ O)
    4. dQ = dS @ K / sqrt(d)
    5. dK = dS^T @ Q / sqrt(d)
    
    Args:
        dO: Gradient of output (batch, num_heads, seq_len, head_dim)
        Q, K, V: Input matrices
        O: Forward output
        L: Row-wise sum of exp(S) from forward
        m: Row-wise max of S from forward
    
    Returns:
        dQ, dK, dV: Gradients
        """
    num_blocks = (seq_len + block_size - 1)//block_size

    D = (dO * out).sum(dim=-1)

    for i in range(num_blocks):
        q_start = i * block_size 
        q_end = min((i *block_size ), seq_len)

        Qi = Q[:, :, q_start:q_end, :]
        Oi = out[:, :, q_start:q_end, :]
        dOi = dO[:, :, q_start:q_end, :]
        Li = l_all[:, :, q_start:q_end]
        mi = m_all[:, :, q_start:q_end]
        Di = D[:, :, q_start:q_end]
        
        dQi = torch.zeros_like(Qi)

        for j in range(num_blocks):
            kv_start = j * block_size
            kv_end = min((j+1)* block_size , seq_len)

            Kj = K[:, :, kv_start:kv_end, :]
            Vj = V[:, :, kv_start:kv_end, :]
        
        Sij = torch.matmul(Qi , Kj.transpose(-2, -1))*scale

        Pij = torch.exp(Sij) - mi.unsqueeze(-1) / Li.unsqueeze9(-1)

        dVj = torch.matmul(Pij.transpose(-2, -1), dOi)  # (batch, heads, Bk, head_dim)
        dV[:, :, kv_start:kv_end, :] += dVj
        
        # Step 2: dP = dO @ V^T
        dPij = torch.matmul(dOi, Vj.transpose(-2, -1))  # (batch, heads, Bq, Bk)
        
        # Step 3: dS = P ⊙ (dP - D)
        # This comes from softmax derivative: ∂L/∂S = P ⊙ (∂L/∂P - D)
        # where D = Σ_k (∂L/∂P_k × P_k) is computed as rowsum(dO ⊙ O)
        dSij = Pij * (dPij - Di.unsqueeze(-1))  # (batch, heads, Bq, Bk)
        
        # Step 4: dQ += dS @ K / sqrt(d)
        dQi += torch.matmul(dSij, Kj) * scale  # (batch, heads, Bq, head_dim)
        
        # Step 5: dK += dS^T @ Q / sqrt(d)
        dKj = torch.matmul(dSij.transpose(-2, -1), Qi) * scale 
        dK[:, :, kv_start:kv_end, :] += dKj
        
        dQ[:, :, q_start:q_end, :] = dQi
    
    return dQ, dK, dV

if __name__ == "__main__":
    torch.manual_seed(42)
    
    seq_len, head_dim = 128, 64
    block_size = 32
    
    # Create input tensors (2D for flash_atten)
    Q = torch.randn(seq_len, head_dim, requires_grad=True)
    K = torch.randn(seq_len, head_dim, requires_grad=True)
    V = torch.randn(seq_len, head_dim, requires_grad=True)
    
    # Forward pass
    O_flash, L, m = flash_atten(Q, K, V, q_block=block_size, kv_block=block_size)
    
    # Compare with naive attention
    O_naive = naive_attention(Q, K, V)
    
    # Check correctness
    print("Output difference:", (O_flash - O_naive).abs().max().item())
    print("Flash Attention output shape:", O_flash.shape)
    print("Match:", torch.allclose(O_flash, O_naive, atol=1e-5))
    









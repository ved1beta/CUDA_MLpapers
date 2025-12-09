import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # It projects the input to the same shape as the attention output
        # (Batch, Seq_Len, Num_Heads, Head_Dim)
        self.gate_proj = nn.Linear(d_model, d_model) 

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_probs = F.softmax(scores, dim=-1)
        

        attn_output = torch.matmul(attn_probs, v) 
        

        gate_score = self.gate_proj(x) 
   
        gate_score = gate_score.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply Sigmoid to ensure values are between 0 and 1
        gate_score = torch.sigmoid(gate_score) 

        # Element-wise multiplication: Y' = Y * Gate
        gated_output = attn_output * gate_score
        
        gated_output = gated_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(gated_output)
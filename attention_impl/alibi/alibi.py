import torch
from torch import nn 
import torch.nn.functional as F

class ALiBiMultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.register_buffer("m", get_alibi_slope(self.num_heads))
        # other initilizations

    def forward(self, x):
        batch_size , seq_len , _ = x.shape

        key , query , value = self.kqv(x).chunk(3, dim=-1)

        query = query.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, -1).permute(0, 2, 3, 1)
        value = value.view(batch_size, seq_len, self.num_heads, -1).transpose(1, 2)
        
        bias = (self.m * get_relative_positions(seq_len)).unsqueez(0)

        score = torch.matmul(query , key)/ self.scale + bias

        if self.causal:
            score = score.masked_fill(
                self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf")
            )
        attention = F.softmax(score, dim=-1)
        return torch.matmul(attention, value)


def get_relative_positions(seq_len: int) -> torch.Tensor:
    """Create distance matrix"""
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    return x - y 

def get_alibi_slope(num_heads):
    """Compute slopes: 2^(-8i/n) for each head"""
    x = (2 ** 8) ** (1 / num_heads)
    return (
        torch.tensor([1 / x ** (i + 1) for i in range(num_heads)])
        .unsqueeze(-1)
        .unsqueeze(-1)
    )
import torch
import torch.nn as nn

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace

        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        # x shape: [Batch, Seq_Len, Dim]
        # Element-wise multiplication broadcasts gamma across Batch and Seq_Len
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class TransformerBlockWithLayerScale(nn.Module):
    def __init__(self, dim, num_heads, init_values=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.ls1 = LayerScale(dim, init_values=init_values) # <--- LayerScale 1

        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        self.ls2 = LayerScale(dim, init_values=init_values) # <--- LayerScale 2

    def forward(self, x):
        #  LS is applied to the output of Attention, BEFORE adding to residual
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + self.ls1(attn_out)

        # 2. FFN Block with Residual Scaling
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.ls2(ffn_out)
        
        return x
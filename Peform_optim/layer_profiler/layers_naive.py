import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (B, N, C)
        # compute rms over last dim
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.scale

class SwiGLU_MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        # single linear -> 2*hidden_dim for (u, v)
        self.fc1 = nn.Linear(dim, 2 * hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)

    def forward(self, x):
        # x: (B, N, C)
        uv = self.fc1(x)  # (B, N, 2*hidden)
        u, v = uv.chunk(2, dim=-1)
        return self.fc2(F.silu(u) * v)

class RMS_SwiGLU_Block(nn.Module):
    def __init__(self, dim, hidden_dim, eps=1e-8):
        super().__init__()
        self.rms = RMSNorm(dim, eps=eps)
        self.mlp = SwiGLU_MLP(dim, hidden_dim)

    def forward(self, x):
        x_norm = self.rms(x)
        out = self.mlp(x_norm)
        return x + out  # residual

B, N, C = 8, 1024, 1024   # batch, seq_len, channels (example)
hidden = 4 * C
model = RMS_SwiGLU_Block(C, hidden).cuda().half()
x = torch.randn(B, N, C, device='cuda', dtype=torch.half)

# warmup
for _ in range(5):
    y = model(x)
torch.cuda.synchronize()

# single timing
import time
start = time.perf_counter()
y = model(x)
torch.cuda.synchronize()
print("Time (ms):", (time.perf_counter()-start)*1000)

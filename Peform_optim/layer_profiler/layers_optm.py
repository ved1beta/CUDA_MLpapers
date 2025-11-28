import torch
from torch.autograd import Function
import torch.nn as nn
import torch.nn.functional as F

class FusedRMS_SwiGLU_Function(Function):
    @staticmethod
    def forward(ctx, x, scale, fc1_weight, fc1_bias, fc2_weight, fc2_bias, eps):
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + eps).sqrt()  # (B,N,1)
        x_norm = x / rms * scale  # broadcast scale (C,) -> (B,N,C)

        uv = F.linear(x_norm, fc1_weight, fc1_bias)  # (B,N,2*hidden)
        u, v = uv.chunk(2, dim=-1)
        y = F.silu(u) * v
        out = F.linear(y, fc2_weight, fc2_bias)
        res = x + out

        # Save for backward if implementing custom backward later
        ctx.save_for_backward(x, scale, rms, uv, fc1_weight, fc2_weight)
        ctx.eps = eps
        return res

    @staticmethod
    def backward(ctx, grad_output):
        # For now, rely on autograd by re-computing grads using saved tensors
        x, scale, rms, uv, fc1_weight, fc2_weight = ctx.saved_tensors
        # Recompute forward path (same as forward) but with requires_grad to compute grads
        x = x.detach().requires_grad_(True)
        scale = scale.detach().requires_grad_(True)
        fc1_weight = fc1_weight.detach().requires_grad_(True)
        fc1_bias = None
        fc2_weight = fc2_weight.detach().requires_grad_(True)
        fc2_bias = None

        # Recompute using same ops to let PyTorch compute gradient graph
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + ctx.eps).sqrt()
        x_norm = x / rms * scale
        uv = F.linear(x_norm, fc1_weight, fc1_bias)  # if bias used include it
        u, v = uv.chunk(2, dim=-1)
        y = F.silu(u) * v
        out = F.linear(y, fc2_weight, fc2_bias)
        res = x + out

        res.backward(grad_output)

        # extract grads
        gx = x.grad
        gscale = scale.grad
        gfc1_w = fc1_weight.grad
        gfc2_w = fc2_weight.grad

        # For simplicity return None for biases if not used above.
        return gx, gscale, gfc1_w, None, gfc2_w, None, None

class FusedRMS_SwiGLU_Block(nn.Module):
    def __init__(self, dim, hidden_dim, eps=1e-8):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.fc1 = nn.Linear(dim, 2*hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, dim, bias=True)
        self.eps = eps

    def forward(self, x):
        return FusedRMS_SwiGLU_Function.apply(
            x, self.scale, self.fc1.weight, self.fc1.bias,
            self.fc2.weight, self.fc2.bias, self.eps
        )
    
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


def time_module(mod, x, iters=50):
    # warmup
    for _ in range(5):
        _ = mod(x)
    torch.cuda.synchronize()
    import time
    s = time.perf_counter()
    for _ in range(iters):
        _ = mod(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - s) / iters * 1000  # ms per iteration

t_naive = time_module(RMS_SwiGLU_Block(C, hidden).cuda().half(), x)
t_fused = time_module(FusedRMS_SwiGLU_Block(C, hidden).cuda().half(), x)
print("Naive ms:", t_naive, "Fused ms:", t_fused)

import torch.profiler

def run_profiler(model, x, name):
    print(f"\n--- Profiling {name} ---")
    # Reset/detach input to avoid growing graph across runs
    x_input = x.detach().clone()
    
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"./log/{name}"),
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        for step in range(6):
            _ = model(x_input)
            torch.cuda.synchronize()
            prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

# Profile Naive
model_naive = RMS_SwiGLU_Block(C, hidden).cuda().half()
run_profiler(model_naive, x, "naive")

# Profile Fused
model_fused = FusedRMS_SwiGLU_Block(C, hidden).cuda().half()
run_profiler(model_fused, x, "fused")
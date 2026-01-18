import torch 
import torch.multiprocessing as mp
import os 
import torch.distributed as dist
import torch.nn as nn
import math
def torch_ddp(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    model = torch.nn.Linear(10, 10).to(rank)

    ddp_model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    loss = torch.nn.MSELoss()

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.001)

    out = ddp_model(torch.randn(20, 10).to(rank))
    target = torch.randn(20, 10).to(rank)

    l = loss(out, target)
    optimizer.step()
    print(f"Rank {rank}, Loss: {l.item()}")

def main():
    world_size = 1
    mp.spawn(torch_ddp,
        args=(world_size,),
        nprocs=world_size,
        join=True)
    
class MHADDP(nn.Module):
    """
    DP Characteristics:
    - Each rank owns the FULL model (replicated)
    - Each rank processes DIFFERENT data (data parallelism)
    - Gradients are averaged across ranks after backward
    
    This is different from Tensor Parallelism (TP):
    - TP: Model is split across ranks, same data, communication during forward/backward
    """
    
    def __init__(self, d_model, num_heads, rank, world_size, bucket_size_mb=25):
        super().__init__()

            
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        
        assert d_model % num_heads == 0 
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.rank = rank
        self.world_size = world_size

        self.bucket_size_mb = bucket_size_mb  
        self.bucket_bytes = bucket_size_mb * 1024 * 1024
        self.current_bucket = []
        self.current_size = 0
        self.pending_buckets = []

        self.W_q = nn.Linear(d_model, d_model) # Query transformation
        self.W_k = nn.Linear(d_model, d_model) # Key transformation
        self.W_v = nn.Linear(d_model, d_model) # Value transformation
        self.W_o = nn.Linear(d_model, d_model) # Output transformation

        for p in self.parameters():
            if p.requires_grad:
                p.register_hook(self.bucket_gradient(p))
    
        
        
    
    
    def scaled_dot_product_attention(self, Q, K, V , mask=None):
        atten = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            atten = atten.masked_fill(mask == 0, -1e9)
        atten = torch.nn.functional.softmax(atten, dim=-1)
        return torch.matmul(atten, V)

    def split_heads(self, x):
        return x.view(x.size(0), x.size(1), self.num_heads, self.d_k).transpose(1, 2)

    def merge_heads(self, x):
        # Input x: (batch, num_heads, seq_len, d_k)
        # After transpose(1, 2): (batch, seq_len, num_heads, d_k)
        # We want: (batch, seq_len, d_model) where d_model = num_heads * d_k
        batch_size, num_heads, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, x, mask=None):

        query = self.split_heads(self.W_q(x))
        key = self.split_heads(self.W_k(x))
        value = self.split_heads(self.W_v(x))

        atten = self.scaled_dot_product_attention(query, key, value, mask)

        output = self.merge_heads(atten)
        return self.W_o(output)


    
    def _broadcast_params(self):
        for param in self.parameters():
            dist.broadcast(param.data, src=0)
            
    def bucket_gradient(self, param):
        def hook(grad):
            grad_bytes = grad.numel() * grad.element_size()

            self.current_bucket.append(grad)
            self.current_size += grad_bytes

            if self.current_size >= self.bucket_bytes:
                self.flush_bucket()

            return grad
        return hook


    def flush_bucket(self):
        bucket = self.current_bucket
        self.current_bucket = []
        self.current_size = 0
        
        # Flatten into a single buffer
        flat_buffer = torch.cat([g.view(-1) for g in bucket])
        
        work = dist.all_reduce(flat_buffer, async_op=True)
        
        # Store buffer AND original shapes to unflatten later
        shapes = [g.shape for g in bucket]
        self.pending_buckets.append((flat_buffer, shapes, bucket, work))

    def finish_gradient_sync(self):
        if self.current_bucket:
            self.flush_bucket()
    
def train_mha_ddp(rank, world_size):

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    model = MHADDP(d_model=512, num_heads=8, rank=rank, world_size=world_size).to(device)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    num_epochs = 2
    batch_size = 32
    seq_len = 128
    
    for epoch in range(num_epochs):
        for batch_idx in range(10):

            x = torch.randn(batch_size, seq_len, 512).to(device)
            target = torch.randn(batch_size, seq_len, 512).to(device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, target)

            loss.backward()
 
            model.finish_gradient_sync()


            for flat_buffer, shapes, original_grads, work in model.pending_buckets:
                work.wait()
                flat_buffer /= world_size
                
                offset = 0
                for grad, shape in zip(original_grads, shapes):
                    numel = grad.numel()
                    grad.copy_(flat_buffer[offset:offset + numel].view(shape))
                    offset += numel
            model.pending_buckets = []


            optimizer.step()
            
            if batch_idx == 0 and rank == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    # Cleanup
    dist.destroy_process_group()
    print(f"Rank {rank} finished training")

def main_mha_ddp():
    """
    Main function to spawn multiple processes for DDP training.
    """
    # Set environment variables for process group initialization
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    

    world_size = 2 

    mp.spawn(
        train_mha_ddp,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":

    print("Running custom MHA DDP implementation...")
    main_mha_ddp()


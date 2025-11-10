#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 256
#define SHARED_MEM_PER_BLOCK (48 * 1024)

__global__ void flash_atten_fwd(
    const float* __restrict__ Q, 
    const float* __restrict__ K, 
    const float* __restrict__ V, 
    float* __restrict__ O, 
    float* __restrict__ L, 
    float* __restrict__ M,
    int B, int H, int N, int d,
    int Bc, int Br,
    float scale,
    bool causal
){
   int batch_idx = blockIdx.x;
   int head_idx = blockIdx.y;
   int query_block_idx = blockIdx.z;

   int local_q_idx = threadIdx.x;
   int global_q_idx = query_block_idx * Br + local_q_idx;

   if (global_q_idx >= N) return;

   extern __shared__ float smem[];

   float* Kj_shared = smem;                              // Bc x (d+1)
   float* Vj_shared = &smem[Bc * (d + 1)];              // Bc x (d+1)
   float* Sij_shared = &smem[2 * Bc * (d + 1)];         // Br x Bc

   int offset = (batch_idx * H + head_idx) * N * d;
   const float* Q_base = Q + offset;
   const float* K_base = K + offset;
   const float* V_base = V + offset;
   float* O_base = O + offset;
   
   float Qi_local[64];  // Max head_dim = 64 for RTX 3050
   float Oi[64];
    
   // Initialize
   float mi = -INFINITY;
   float li = 0.0f;

   for (int i = 0; i < d; i++) {
       Qi_local[i] = Q_base[global_q_idx * d + i];
       Oi[i] = 0.0f;
   }
    
   int num_kv_blocks = (N + Bc - 1) / Bc;

   for(int kv_block = 0; kv_block < num_kv_blocks; kv_block++){
       int kv_start = kv_block * Bc;
       int kv_end = min(kv_start + Bc, N);
       int kv_size = kv_end - kv_start;
        
       // Cooperative loading with coalesced access
       int total_elements = kv_size * d;
       for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
           int k_row = idx / d;
           int k_col = idx % d;
           if (k_row < kv_size) {
               // Padded storage to avoid bank conflicts
               Kj_shared[k_row * (d + 1) + k_col] = K_base[(kv_start + k_row) * d + k_col];
               Vj_shared[k_row * (d + 1) + k_col] = V_base[(kv_start + k_row) * d + k_col];
           }
       }
       __syncthreads();

       // Compute attention scores: S = Q @ K^T * scale
       for (int k = 0; k < kv_size; k++) {
           float score = 0.0f;
           for (int i = 0; i < d; i++) {
               score += Qi_local[i] * Kj_shared[k * (d + 1) + i];
           }
           score *= scale;
           
           // Causal mask
           if (causal && (kv_start + k) > global_q_idx) {
               score = -INFINITY;
           }
           
           Sij_shared[local_q_idx * Bc + k] = score;
       }

       float mij = -INFINITY;
       for (int k = 0; k < kv_size; k++) {
           mij = fmaxf(mij, Sij_shared[local_q_idx * Bc + k]);
       }

       float mi_new = fmaxf(mi, mij);
       float alpha = expf(mi - mi_new);
        
       // Compute exponentials and sum
       float lij = 0.0f;
       for (int k = 0; k < kv_size; k++) {
           float Pij = expf(Sij_shared[local_q_idx * Bc + k] - mi_new);
           Sij_shared[local_q_idx * Bc + k] = Pij;  // Store P for matmul
           lij += Pij;
       }
        
       li = alpha * li + lij;
        
       for (int i = 0; i < d; i++) {
           Oi[i] *= alpha;
           for (int k = 0; k < kv_size; k++) {
               Oi[i] += Sij_shared[local_q_idx * Bc + k] * Vj_shared[k * (d + 1) + i];
           }
       }
        
       mi = mi_new;
       __syncthreads();
   }
    
   // Final normalization
   for (int i = 0; i < d; i++) {
       O_base[global_q_idx * d + i] = Oi[i] / li;
   }
    
   // Write statistics
   int stat_offset = (batch_idx * H + head_idx) * N;
   L[stat_offset + global_q_idx] = li;
   M[stat_offset + global_q_idx] = mi;
}

__global__ void flash_attention_backward_kernel(
    const float* dO,     // (B, H, N, d) - gradient of output
    const float* Q,      // (B, H, N, d)
    const float* K,      // (B, H, N, d)
    const float* V,      // (B, H, N, d)
    const float* O,      // (B, H, N, d) - forward output
    const float* L,      // (B, H, N) - row sum from forward
    const float* M,      // (B, H, N) - row max from forward
    float* dQ,           // (B, H, N, d) - output gradient
    float* dK,           // (B, H, N, d) - output gradient
    float* dV,           // (B, H, N, d) - output gradient
    int B,
    int H,
    int N,
    int d,
    int Bc,
    int Br,
    float scale,
    bool causal
){
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int query_block_idx = blockIdx.z;
    
    int local_q_idx = threadIdx.x;
    int global_q_idx = query_block_idx * Br + local_q_idx;
    
    if (global_q_idx >= N) return;
    
    // Shared memory - REMOVED unused Qi_shared
    extern __shared__ float smem[];
    float* Kj_shared = smem;                                    // Bc x (d+1)
    float* Vj_shared = &smem[Bc * (d + 1)];                     // Bc x (d+1)
    float* Pij_shared = &smem[2 * Bc * (d + 1)];                // Br x Bc
    float* dPij_shared = &smem[2 * Bc * (d + 1) + Br * Bc];     // Br x Bc

    int offset = (batch_idx * H + head_idx) * N * d;
    const float* Q_base = Q + offset;
    const float* K_base = K + offset;
    const float* V_base = V + offset;
    const float* O_base = O + offset;
    const float* dO_base = dO + offset;
    float* dQ_base = dQ + offset;
    float* dK_base = dK + offset;
    float* dV_base = dV + offset;
    
    int stat_offset = (batch_idx * H + head_idx) * N;

    float Qi_local[64];
    float dOi_local[64];
    float Oi_local[64];
    
    for (int i = 0; i < d; i++) {
        Qi_local[i] = Q_base[global_q_idx * d + i];
        dOi_local[i] = dO_base[global_q_idx * d + i];
        Oi_local[i] = O_base[global_q_idx * d + i];
    }

    float li = L[stat_offset + global_q_idx];
    float mi = M[stat_offset + global_q_idx];

    // Compute D = rowsum(dO ⊙ O)
    float Di = 0.0f;
    for (int i = 0; i < d; i++) {
        Di += dOi_local[i] * Oi_local[i];
    }
    
    // Accumulator for dQ
    float dQi[64];
    for (int i = 0; i < d; i++) {
        dQi[i] = 0.0f;
    }

    int num_kv_blocks = (N + Bc - 1) / Bc;
    
    for (int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        int kv_start = kv_block * Bc;
        int kv_end = min(kv_start + Bc, N);
        int kv_size = kv_end - kv_start;
        
        // Load K and V blocks with padding (cooperative)
        int total_elements = kv_size * d;
        for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
            int k_row = idx / d;
            int k_col = idx % d;
            if (k_row < kv_size) {
                // Padded storage to match forward pass
                Kj_shared[k_row * (d + 1) + k_col] = K_base[(kv_start + k_row) * d + k_col];
                Vj_shared[k_row * (d + 1) + k_col] = V_base[(kv_start + k_row) * d + k_col];
            }
        }
        __syncthreads();

        // Recompute attention scores and probabilities
        for (int k = 0; k < kv_size; k++) {
            float score = 0.0f;
            for (int i = 0; i < d; i++) {
                score += Qi_local[i] * Kj_shared[k * (d + 1) + i];
            }
            score *= scale;
            
            // Causal mask (must match forward pass)
            int k_pos = kv_start + k;
            if (causal && k_pos > global_q_idx) {
                score = -INFINITY;
            }
            
            // Recompute P = exp(S - m) / l
            float Pij = expf(score - mi) / li;
            Pij_shared[local_q_idx * Bc + k] = Pij;
        }
        __syncthreads();

        // Compute dP = dO @ V^T
        for (int k = 0; k < kv_size; k++) {
            float dPij = 0.0f;
            for (int i = 0; i < d; i++) {
                dPij += dOi_local[i] * Vj_shared[k * (d + 1) + i];
            }
            dPij_shared[local_q_idx * Bc + k] = dPij;
        }
        __syncthreads();
        
        // Compute dS = P ⊙ (dP - D)
        float dSij[64];  // Assuming Bc <= 64
        for (int k = 0; k < kv_size; k++) {
            float Pij = Pij_shared[local_q_idx * Bc + k];
            float dPij = dPij_shared[local_q_idx * Bc + k];
            dSij[k] = Pij * (dPij - Di);
        }
        
        // Compute dQ += dS @ K * scale
        for (int i = 0; i < d; i++) {
            for (int k = 0; k < kv_size; k++) {
                dQi[i] += dSij[k] * Kj_shared[k * (d + 1) + i] * scale;
            }
        }
        
        // Compute dV += P^T @ dO (atomic add for thread cooperation)
        for (int k = 0; k < kv_size; k++) {
            float Pij = Pij_shared[local_q_idx * Bc + k];
            for (int i = 0; i < d; i++) {
                atomicAdd(&dV_base[(kv_start + k) * d + i], Pij * dOi_local[i]);
            }
        }
        
        // Compute dK += dS^T @ Q * scale (atomic add)
        for (int k = 0; k < kv_size; k++) {
            for (int i = 0; i < d; i++) {
                atomicAdd(&dK_base[(kv_start + k) * d + i], dSij[k] * Qi_local[i] * scale);
            }
        }
        
        __syncthreads();
    }
    
    // Write dQ
    for (int i = 0; i < d; i++) {
        dQ_base[global_q_idx * d + i] = dQi[i];
    }
}

// Host function to launch forward pass
void flash_attention_forward_cuda(
    const float* Q, const float* K, const float* V,
    float* O, float* L, float* M,
    int B, int H, int N, int d,
    int Bc, int Br,
    bool causal = false
) {
    float scale = 1.0f / sqrtf((float)d);
    
    dim3 grid(B, H, (N + Br - 1) / Br);
    dim3 block(Br);
    
    // Shared memory: Kj(Bc*(d+1)) + Vj(Bc*(d+1)) + Sij(Br*Bc)
    size_t smem_size = (2 * Bc * (d + 1) + Br * Bc) * sizeof(float);
    
    flash_atten_fwd<<<grid, block, smem_size>>>(
        Q, K, V, O, L, M, B, H, N, d, Bc, Br, scale, causal
    );
    
    cudaDeviceSynchronize();
}

// Host function to launch backward pass
void flash_attention_backward_cuda(
    const float* dO, const float* Q, const float* K, const float* V,
    const float* O, const float* L, const float* M,
    float* dQ, float* dK, float* dV,
    int B, int H, int N, int d,
    int Bc, int Br,
    bool causal = false
) {
    float scale = 1.0f / sqrtf((float)d);
    
    // Initialize gradients to zero
    size_t size = B * H * N * d * sizeof(float);
    cudaMemset(dQ, 0, size);
    cudaMemset(dK, 0, size);
    cudaMemset(dV, 0, size);
    
    dim3 grid(B, H, (N + Br - 1) / Br);
    dim3 block(Br);
    
    // Shared memory: Kj + Vj + Pij + dPij (removed unused Qi_shared)
    size_t smem_size = (2 * Bc * (d + 1) + 2 * Br * Bc) * sizeof(float);
    
    flash_attention_backward_kernel<<<grid, block, smem_size>>>(
        dO, Q, K, V, O, L, M, dQ, dK, dV, B, H, N, d, Bc, Br, scale, causal
    );
    
    cudaDeviceSynchronize();
}
void naive_attention_cpu(
    const float* Q, const float* K, const float* V,
    float* O, int B, int H, int N, int d, bool causal
) {
    float scale = 1.0f / sqrtf((float)d);
    
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            int offset = (b * H + h) * N * d;
            
            for (int i = 0; i < N; i++) {
                // Compute attention scores
                float scores[1024];  // Max sequence length
                float max_score = -INFINITY;
                
                for (int j = 0; j < N; j++) {
                    if (causal && j > i) {
                        scores[j] = -INFINITY;
                    } else {
                        float score = 0.0f;
                        for (int k = 0; k < d; k++) {
                            score += Q[offset + i * d + k] * K[offset + j * d + k];
                        }
                        scores[j] = score * scale;
                    }
                    max_score = fmaxf(max_score, scores[j]);
                }
                
                // Compute softmax
                float sum = 0.0f;
                for (int j = 0; j < N; j++) {
                    scores[j] = expf(scores[j] - max_score);
                    sum += scores[j];
                }
                
                // Compute output
                for (int k = 0; k < d; k++) {
                    float val = 0.0f;
                    for (int j = 0; j < N; j++) {
                        val += (scores[j] / sum) * V[offset + j * d + k];
                    }
                    O[offset + i * d + k] = val;
                }
            }
        }
    }
}


float compute_relative_error(const float* a, const float* b, int n) {
    float max_diff = 0.0f;
    float max_val = 0.0f;
    
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        max_diff = fmaxf(max_diff, diff);
        max_val = fmaxf(max_val, fabsf(a[i]));
    }
    
    return max_diff / (max_val + 1e-8f);
}


int main() {
    // Test multiple configurations
    struct Config {
        int B, H, N, d, Bc, Br;
        bool causal;
        const char* name;
    };
    
    Config configs[] = {
        {1, 2, 128, 64, 32, 32, false, "Tiny (B=1, H=2, N=128, d=64)"},
        {2, 8, 512, 64, 32, 32, false, "Small (B=2, H=8, N=512, d=64)"},
        {4, 8, 1024, 64, 32, 32, false, "Medium (B=4, H=8, N=1024, d=64)"},
        {2, 16, 2048, 64, 32, 32, false, "Large (B=2, H=16, N=2048, d=64)"},
        {2, 8, 1024, 64, 32, 32, true, "Causal (B=2, H=8, N=1024, d=64)"},
    };
    
    int num_configs = sizeof(configs) / sizeof(Config);
    
    printf("=== Flash Attention Benchmark ===\n\n");
    
    for (int cfg_idx = 0; cfg_idx < num_configs; cfg_idx++) {
        Config cfg = configs[cfg_idx];
        printf("Testing: %s\n", cfg.name);
        printf("----------------------------------------\n");
        
        int B = cfg.B, H = cfg.H, N = cfg.N, d = cfg.d;
        int Bc = cfg.Bc, Br = cfg.Br;
        bool causal = cfg.causal;
        
        size_t size = B * H * N * d * sizeof(float);
        size_t stat_size = B * H * N * sizeof(float);
        
        // Allocate device memory
        float *d_Q, *d_K, *d_V, *d_O, *d_L, *d_M;
        float *d_dO, *d_dQ, *d_dK, *d_dV;
        
        cudaMalloc(&d_Q, size);
        cudaMalloc(&d_K, size);
        cudaMalloc(&d_V, size);
        cudaMalloc(&d_O, size);
        cudaMalloc(&d_L, stat_size);
        cudaMalloc(&d_M, stat_size);
        cudaMalloc(&d_dO, size);
        cudaMalloc(&d_dQ, size);
        cudaMalloc(&d_dK, size);
        cudaMalloc(&d_dV, size);
        
        // Allocate host memory
        float *h_Q = (float*)malloc(size);
        float *h_K = (float*)malloc(size);
        float *h_V = (float*)malloc(size);
        float *h_dO = (float*)malloc(size);
        float *h_O_flash = (float*)malloc(size);
        float *h_O_naive = (float*)malloc(size);
        float *h_dQ = (float*)malloc(size);
        float *h_dK = (float*)malloc(size);
        float *h_dV = (float*)malloc(size);
        
        // Initialize with random data
        srand(42);
        for (int i = 0; i < B * H * N * d; i++) {
            h_Q[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // [-1, 1]
            h_K[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            h_V[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            h_dO[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
        
        // Copy to device
        cudaMemcpy(d_Q, h_Q, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_K, h_K, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_V, h_V, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_dO, h_dO, size, cudaMemcpyHostToDevice);
        
        // Warmup to ensure GPU clocks are up and caches are primed
        for (int i = 0; i < 3; i++) {
            flash_attention_forward_cuda(d_Q, d_K, d_V, d_O, d_L, d_M, 
                                        B, H, N, d, Bc, Br, causal);
        }
        cudaDeviceSynchronize();
        
        // Check for any errors after warmup
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error after warmup: %s\n", cudaGetErrorString(err));
        }
        
        // Benchmark forward pass
        int num_iterations = 10;  // Reduced for faster testing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        for (int i = 0; i < num_iterations; i++) {
            flash_attention_forward_cuda(d_Q, d_K, d_V, d_O, d_L, d_M, 
                                        B, H, N, d, Bc, Br, causal);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float forward_ms = 0;
        cudaEventElapsedTime(&forward_ms, start, stop);
        forward_ms /= num_iterations;
        
        // Benchmark backward pass
        cudaEventRecord(start);
        for (int i = 0; i < num_iterations; i++) {
            flash_attention_backward_cuda(d_dO, d_Q, d_K, d_V, d_O, d_L, d_M,
                                         d_dQ, d_dK, d_dV, B, H, N, d, Bc, Br, causal);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float backward_ms = 0;
        cudaEventElapsedTime(&backward_ms, start, stop);
        backward_ms /= num_iterations;
        
        // Copy results back
        cudaMemcpy(h_O_flash, d_O, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_dQ, d_dQ, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_dK, d_dK, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_dV, d_dV, size, cudaMemcpyDeviceToHost);
        
        // Verify correctness (only for small sequences to save time)
        if (N <= 512) {
            naive_attention_cpu(h_Q, h_K, h_V, h_O_naive, B, H, N, d, causal);
            float rel_error = compute_relative_error(h_O_flash, h_O_naive, B * H * N * d);
            printf("Relative Error (vs naive): %.2e\n", rel_error);
            
            if (rel_error < 1e-3) {
                printf("✓ Correctness check PASSED\n");
            } else {
                printf("✗ Correctness check FAILED\n");
            }
        }
        
        // Compute FLOPs
        // Forward: 2*N^2*d per attention head (QK^T and PV), N^2 for softmax
        // Backward: ~4x forward
        long long forward_flops = (long long)B * H * (4LL * N * N * d + 2LL * N * N);
        long long backward_flops = 4LL * forward_flops;
        
        float forward_tflops = (forward_flops / (forward_ms * 1e-3)) / 1e12;
        float backward_tflops = (backward_flops / (backward_ms * 1e-3)) / 1e12;
        
        printf("\nPerformance:\n");
        if (forward_ms < 0.001f) {
            printf("  Forward:  %.6f ms  (timing too small, may be unreliable)\n", forward_ms);
        } else {
            printf("  Forward:  %.3f ms  (%.3f TFLOPS)\n", forward_ms, forward_tflops);
        }
        printf("  Backward: %.3f ms  (%.3f TFLOPS)\n", backward_ms, backward_tflops);
        printf("  Total:    %.3f ms\n", forward_ms + backward_ms);
        
        // Memory bandwidth utilization
        size_t forward_bytes = B * H * N * d * sizeof(float) * 6; // Q,K,V read, O,L,M write
        float forward_bw_gb = (forward_bytes / (forward_ms * 1e-3)) / 1e9;
        printf("  Memory BW: %.2f GB/s\n", forward_bw_gb);
        
        printf("\n");
        
        // Clean up
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        free(h_Q); free(h_K); free(h_V); free(h_dO);
        free(h_O_flash); free(h_O_naive);
        free(h_dQ); free(h_dK); free(h_dV);
        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
        cudaFree(d_L); cudaFree(d_M);
        cudaFree(d_dO); cudaFree(d_dQ); cudaFree(d_dK); cudaFree(d_dV);
    }
    
    printf("passs\n");
    return 0;
}
/*
 * CUDA kernels for the Engram conditional memory module.
 *
 * Kernel 1: multi_head_ngram_lookup_kernel
 *   - Fused hash computation + embedding gather
 *   - One block per (batch, position), threads across (order, head)
 *   - Vectorized 128-bit loads for embedding rows
 *
 * Kernel 2: fused_engram_gate_kernel
 *   - Fuses K-projection, RMSNorm, dot product, sqrt-abs-sigmoid, V-scaling
 *   - One block per (batch, position, branch), threads across D
 *   - Uses shared memory for value cache and warp shuffles for reductions
 *
 * Kernel 3: sparse_embed_backward_kernel
 *   - Scatter-add gradients to embedding table rows
 *   - Uses atomicAdd for thread safety
 *
 * Compile target: sm_80+ (A100/H100). Uses bf16 natively.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

// ═══════════════════════════════════════════════════════════════════
// Kernel 1: Multi-Head N-gram Hash + Lookup
// ═══════════════════════════════════════════════════════════════════

__device__ __forceinline__ int64_t compute_ngram_hash(
    const int64_t* __restrict__ ids,
    int b, int t, int T,
    int ngram_order,
    const int64_t* __restrict__ mults,
    int64_t prime,
    int64_t pad_id
) {
    int64_t mix = 0;
    for (int k = 0; k < ngram_order; k++) {
        int pos = t - k;
        int64_t tok = (pos >= 0) ? ids[b * T + pos] : pad_id;
        mix ^= tok * mults[k];
    }
    int64_t result = mix % prime;
    return (result < 0) ? result + prime : result;
}

template <typename scalar_t>
__global__ void multi_head_ngram_lookup_kernel(
    const int64_t* __restrict__ compressed_ids,     // [B, T]
    const scalar_t* __restrict__ embed_table,       // [total_vocab, D]
    const int64_t* __restrict__ multipliers,        // [max_ngram_size]
    const int64_t* __restrict__ head_primes,        // [total_heads]
    const int64_t* __restrict__ head_offsets,       // [total_heads]
    scalar_t* __restrict__ output,                  // [B, T, total_heads, D]
    int B, int T, int D,
    int max_ngram_size,
    int n_heads_per_ngram,
    int64_t pad_id
) {
    int b = blockIdx.x;
    int t = blockIdx.y;
    int head_flat = threadIdx.x;

    int n_orders = max_ngram_size - 1;
    int total_heads = n_orders * n_heads_per_ngram;

    if (b >= B || t >= T || head_flat >= total_heads) return;

    int order_idx = head_flat / n_heads_per_ngram;
    int ngram_order = order_idx + 2;

    int64_t prime = head_primes[head_flat];
    int64_t offset = head_offsets[head_flat];

    int64_t local_id = compute_ngram_hash(
        compressed_ids, b, t, T,
        ngram_order, multipliers, prime, pad_id
    );
    int64_t global_id = offset + local_id;

    // Copy embedding row to output
    int out_base = ((b * T + t) * total_heads + head_flat) * D;
    int src_base = global_id * D;

    for (int d = 0; d < D; d++) {
        output[out_base + d] = embed_table[src_base + d];
    }
}

torch::Tensor multi_head_ngram_lookup(
    torch::Tensor compressed_ids,
    torch::Tensor embed_table,
    torch::Tensor multipliers,
    torch::Tensor head_primes,
    torch::Tensor head_offsets,
    int64_t max_ngram_size,
    int64_t n_heads_per_ngram,
    int64_t pad_id
) {
    const auto B = compressed_ids.size(0);
    const auto T = compressed_ids.size(1);
    const auto D = embed_table.size(1);
    const int n_orders = max_ngram_size - 1;
    const int total_heads = n_orders * n_heads_per_ngram;

    auto options = torch::TensorOptions()
        .dtype(embed_table.dtype())
        .device(embed_table.device());
    auto output = torch::empty({B, T, total_heads, D}, options);

    dim3 grid(B, T);
    dim3 block(total_heads);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        embed_table.scalar_type(), "multi_head_ngram_lookup", [&] {
            multi_head_ngram_lookup_kernel<scalar_t><<<grid, block,
                0, at::cuda::getCurrentCUDAStream()>>>(
                compressed_ids.data_ptr<int64_t>(),
                embed_table.data_ptr<scalar_t>(),
                multipliers.data_ptr<int64_t>(),
                head_primes.data_ptr<int64_t>(),
                head_offsets.data_ptr<int64_t>(),
                output.data_ptr<scalar_t>(),
                B, T, D,
                max_ngram_size,
                n_heads_per_ngram,
                pad_id
            );
        }
    );

    return output;
}


// ═══════════════════════════════════════════════════════════════════
// Kernel 2: Fused Gating
// ═══════════════════════════════════════════════════════════════════

// Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// Block-level reduction using shared memory
__device__ float block_reduce_sum(float val, float* smem, int tid, int block_size) {
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    val = warp_reduce_sum(val);

    if (lane_id == 0) {
        smem[warp_id] = val;
    }
    __syncthreads();

    int num_warps = (block_size + 31) / 32;
    val = (tid < num_warps) ? smem[tid] : 0.0f;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}

__global__ void fused_engram_gate_kernel(
    const float* __restrict__ e_t,              // [B, T, E]
    const float* __restrict__ h_t,              // [B, T, HC, D]
    const float* __restrict__ W_K,              // [HC, D, E]
    const float* __restrict__ W_V,              // [D, E]
    const float* __restrict__ norm_q_weight,    // [HC, D]
    const float* __restrict__ norm_k_weight,    // [HC, D]
    float* __restrict__ output,                 // [B, T, HC, D]
    float inv_sqrt_d,
    float norm_eps,
    int B, int T, int HC, int D, int E
) {
    int b  = blockIdx.x;
    int t  = blockIdx.y;
    int hc = blockIdx.z;

    if (b >= B || t >= T || hc >= HC) return;

    extern __shared__ float smem[];
    float* v_cache = smem;               // [D]
    float* reduce_buf = smem + D;        // [ceil(D/32)]

    int tid = threadIdx.x;

    // Step 1: Compute V = W_V @ e_t (shared across branches)
    float v_val = 0.0f;
    if (tid < D) {
        const float* wv_row = W_V + tid * E;
        const float* e = e_t + (b * T + t) * E;
        for (int i = 0; i < E; i++) {
            v_val += wv_row[i] * e[i];
        }
        v_cache[tid] = v_val;
    }
    __syncthreads();

    if (tid >= D) return;

    // Step 2: Compute K = W_K[hc] @ e_t
    float k_val = 0.0f;
    {
        const float* wk_row = W_K + (hc * D + tid) * E;
        const float* e = e_t + (b * T + t) * E;
        for (int i = 0; i < E; i++) {
            k_val += wk_row[i] * e[i];
        }
    }

    // Step 3: RMSNorm on query and key
    float h_val = h_t[((b * T + t) * HC + hc) * D + tid];

    float sq_h = h_val * h_val;
    float sq_k = k_val * k_val;

    float sum_sq_h = block_reduce_sum(sq_h, reduce_buf, tid, D);
    __syncthreads();
    float sum_sq_k = block_reduce_sum(sq_k, reduce_buf, tid, D);
    __syncthreads();

    float rms_q = rsqrtf(sum_sq_h / D + norm_eps);
    float rms_k = rsqrtf(sum_sq_k / D + norm_eps);

    float scale_q = norm_q_weight[hc * D + tid];
    float scale_k = norm_k_weight[hc * D + tid];
    float q_norm = h_val * rms_q * scale_q;
    float k_norm = k_val * rms_k * scale_k;

    // Step 4: Scalar gate (dot product -> sqrt-abs -> sigmoid)
    float dot = q_norm * k_norm;
    float dot_sum = block_reduce_sum(dot, reduce_buf, tid, D);
    __syncthreads();

    float scaled = dot_sum * inv_sqrt_d;
    float sign_val = (scaled >= 0.0f) ? 1.0f : -1.0f;
    float abs_val = fabsf(scaled);
    abs_val = fmaxf(abs_val, 1e-6f);
    float gate_input = sign_val * sqrtf(abs_val);
    float alpha = 1.0f / (1.0f + expf(-gate_input));

    // Step 5: output = alpha * V[tid]
    int out_idx = ((b * T + t) * HC + hc) * D + tid;
    output[out_idx] = alpha * v_cache[tid];
}

torch::Tensor fused_engram_gate(
    torch::Tensor e_t,
    torch::Tensor h_t,
    torch::Tensor W_K,
    torch::Tensor W_V,
    torch::Tensor norm_q_weight,
    torch::Tensor norm_k_weight,
    float inv_sqrt_d,
    float norm_eps
) {
    const auto B = h_t.size(0);
    const auto T = h_t.size(1);
    const auto HC = h_t.size(2);
    const auto D = h_t.size(3);
    const auto E = e_t.size(2);

    // Ensure float32 for kernel
    auto e_f = e_t.to(torch::kFloat32).contiguous();
    auto h_f = h_t.to(torch::kFloat32).contiguous();
    auto wk_f = W_K.to(torch::kFloat32).contiguous();
    auto wv_f = W_V.to(torch::kFloat32).contiguous();
    auto nq_f = norm_q_weight.to(torch::kFloat32).contiguous();
    auto nk_f = norm_k_weight.to(torch::kFloat32).contiguous();

    auto output = torch::empty({B, T, HC, D},
        torch::TensorOptions().dtype(torch::kFloat32).device(h_t.device()));

    dim3 grid(B, T, HC);
    int block_size = D;
    // shared: D floats for v_cache + ceil(D/32) for reductions
    int smem_bytes = (D + (D + 31) / 32) * sizeof(float);

    fused_engram_gate_kernel<<<grid, block_size, smem_bytes,
        at::cuda::getCurrentCUDAStream()>>>(
        e_f.data_ptr<float>(),
        h_f.data_ptr<float>(),
        wk_f.data_ptr<float>(),
        wv_f.data_ptr<float>(),
        nq_f.data_ptr<float>(),
        nk_f.data_ptr<float>(),
        output.data_ptr<float>(),
        inv_sqrt_d,
        norm_eps,
        B, T, HC, D, E
    );

    return output.to(h_t.dtype());
}


// ═══════════════════════════════════════════════════════════════════
// Kernel 3: Sparse Embedding Gradient Accumulation
// ═══════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void sparse_embed_backward_kernel(
    const scalar_t* __restrict__ grad_output,   // [N, D]
    const int64_t* __restrict__ indices,        // [N]
    scalar_t* __restrict__ grad_embed,          // [total_vocab, D]
    int N, int D
) {
    int idx = blockIdx.x;
    int d = threadIdx.x;

    if (idx >= N || d >= D) return;

    int64_t row = indices[idx];
    atomicAdd(
        reinterpret_cast<float*>(&grad_embed[row * D + d]),
        static_cast<float>(grad_output[idx * D + d])
    );
}

torch::Tensor sparse_embed_backward(
    torch::Tensor grad_output,
    torch::Tensor indices,
    int64_t total_vocab_size
) {
    auto flat_grad = grad_output.reshape({-1, grad_output.size(-1)}).contiguous();
    auto flat_idx = indices.reshape({-1}).contiguous();

    const auto N = flat_grad.size(0);
    const auto D = flat_grad.size(1);

    auto grad_embed = torch::zeros(
        {total_vocab_size, D},
        torch::TensorOptions().dtype(torch::kFloat32).device(grad_output.device())
    );

    if (N == 0) return grad_embed;

    dim3 grid(N);
    dim3 block(min((int64_t)1024, D));

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        grad_output.scalar_type(), "sparse_embed_backward", [&] {
            sparse_embed_backward_kernel<scalar_t><<<grid, block,
                0, at::cuda::getCurrentCUDAStream()>>>(
                flat_grad.data_ptr<scalar_t>(),
                flat_idx.data_ptr<int64_t>(),
                grad_embed.data_ptr<scalar_t>(),
                N, D
            );
        }
    );

    return grad_embed;
}

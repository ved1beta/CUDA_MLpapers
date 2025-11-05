#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <stdio.h>

__device__ float wrap_reduce_max(float val){
    for (int offset= 16; offset>0; offset/=2 ){
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset)); // calculate max of 32 values in 1 wrap
    }
    return val;
}

__device__ float wrap_reduce_sum(float val){
    for (int offset=16 ; offset > 0; offset /=2){
        val += __shfl_down_sync(0xffffffff, val, offset); // calculate sum of 32 values in 1 wrap by 32,16,8,4,2,1 offset values 
    }
    return val;
}

__global__ void safe_softmax_kernel(const float* x,float* out, int n){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    __shared__ float shared_max;
    __shared__ float shared_sum;

    float local_max = -FLT_MAX;
    if (tid < n){
        local_max = x[tid];
    }
    float wrap_max = wrap_reduce_max(local_max);

    if (threadIdx.x % 32 == 0){
        atomicMax((int*)&shared_max, __float_as_int(wrap_max));  //calculate the max value of all warps in the block one by one and store it in shared memory
    }
    __syncthreads();

    float exp_val = 0.0f;
    if (tid < n){
        exp_val = expf(x[tid] - shared_max); 
    
    float wrap_sum = wrap_reduce_sum(exp_val);

    if (threadIdx.x % 32 == 0){
        atomicAdd(&shared_sum, wrap_sum);  //calculate the sum value of all warps in the block one by one and store it in shared memory
    }
    __syncthreads();

    if (tid < n){
        out[tid] = exp_val / shared_sum;
    }

    if (tid < n) {
        out[tid] = exp_val / shared_sum;
        }
    }
}
__global__ void online_softmax_kernel(const float* x, float* out, int n){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float shared_max;  // Current maximum
    __shared__ float shared_sum;  // Current denominator

    if (threadIdx.x == 0) {
        shared_max = -FLT_MAX;
        shared_sum   = 0.0f;
    }
    __syncthreads();

    for (int i = 0 ; i< n ; i++ ){
        float xi = (i<n) ?  x[i] : -FLT_MAX;

        float m_old = shared_max;  // Previous maximum
        if (threadIdx.x == 0) {
            shared_max = fmaxf(m_old, xi);  // Update maximum
        }
        __syncthreads();

        float c = expf(m_old - shared_max);

        if (threadIdx.x == 0) {
            shared_sum = shared_sum * c + expf(xi - shared_max);
        }

        __syncthreads();

        if (tid < n) {
            out[tid] = expf(x[tid]-shared_max) / shared_sum;
        }
        
    }
}

__global__ void flash_attention_softmax_kernel(
    const float* scores,  // [batch, heads, seq_len, seq_len]
    float* out,
    int seq_len
) {
    int row = blockIdx.x;  // Which row of attention matrix
    
    __shared__ float m_shared;
    __shared__ float d_shared;
    
    if (threadIdx.x == 0) {
        m_shared = -FLT_MAX;
        d_shared = 0.0f;
    }
    __syncthreads();
    
    // Online softmax over row
    for (int col = threadIdx.x; col < seq_len; col += blockDim.x) {
        float score = scores[row * seq_len + col];
        
        // Atomic max update
        atomicMax((int*)&m_shared, __float_as_int(score));
        __syncthreads();
        
        // Correction and sum update
        float m_old = m_shared;
        float c = expf(m_old - m_shared);
        
        atomicAdd(&d_shared, expf(score - m_shared));
        __syncthreads();
    }
    
    // Write final results
    for (int col = threadIdx.x; col < seq_len; col += blockDim.x) {
        float score = scores[row * seq_len + col];
        out[row * seq_len + col] = expf(score - m_shared) / d_shared;
    }
}

// Host code
void launch_safe_softmax(const float* d_x, float* d_out, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    safe_softmax_kernel<<<blocks, threads>>>(d_x, d_out, n);
}

void launch_online_softmax(const float* d_x, float* d_out, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    online_softmax_kernel<<<blocks, threads>>>(d_x, d_out, n);
}

int main() {
    const int n = 8;
    float h_x[n] = {1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0};
    float h_out[n];

    float *d_x, *d_out;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    cudaMemcpy(d_x, h_x, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    launch_safe_softmax(d_x, d_out, n);

    // Copy result back
    cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    printf("Softmax output:\n");
    for (int i = 0; i < n; i++) {
        printf("%f ", h_out[i]);
    }
    printf("\n");

    cudaFree(d_x);
    cudaFree(d_out);
    return 0;
}

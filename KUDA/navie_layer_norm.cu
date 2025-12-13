


#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define BLOCK_SIZE 256 

__device__ float reduce_sum(float val){
    __shared__ float shared[BLOCK_SIZE];

    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();

    for(int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2){
        if(tid < stride){
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    return shared[0];
}

__global__ void layer_norm(float* out, const float* inp, const float* gamma, const float* beta, int N, float epsilon){
    int row_idx = blockIdx.x;  
    int tid = threadIdx.x;

    const float *row_inp = inp + row_idx * N;
    float *row_out = out + row_idx * N;

    // MEAN: Each thread sums its portion of the row
    float thread_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x){
        thread_sum += row_inp[i];
    }
    
    // Reduce across block
    float row_sum = reduce_sum(thread_sum);
    float mean = row_sum / N;

    // VARIANCE: Each thread computes squared differences for its portion
    float thread_sq_diff = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float diff = row_inp[i] - mean;
        thread_sq_diff += diff * diff;
    }
    
    // Reduce across block
    float row_sq_diff = reduce_sum(thread_sq_diff);
    float variance = row_sq_diff / N;

    // INVERSE STANDARD DEVIATION
    float inv_std_dev = rsqrtf(variance + epsilon);

    // NORMALIZE: Apply normalization with affine transform
    for (int i = tid; i < N; i += blockDim.x) {
        float val = row_inp[i];
        float norm = (val - mean) * inv_std_dev;
        row_out[i] = norm * gamma[i] + beta[i];
    }
}

void verify_layernorm(float* out, const float* inp, int batch, int N, float epsilon){
    std::cout << "\n=== Verification ===" << std::endl;
    
    for(int row = 0; row < std::min(batch, 2); row++){
        const float* row_inp = inp + row * N;
        const float* row_out = out + row * N;
        
        // Compute expected mean and variance
        float sum = 0.0f;
        for(int i = 0; i < N; i++){
            sum += row_inp[i];
        }
        float mean = sum / N;
        
        float var_sum = 0.0f;
        for(int i = 0; i < N; i++){
            float diff = row_inp[i] - mean;
            var_sum += diff * diff;
        }
        float variance = var_sum / N;
        float std_dev = sqrtf(variance + epsilon);
        
        // Check output statistics (should have mean≈0, std≈1 before gamma/beta)
        float out_mean = 0.0f;
        for(int i = 0; i < N; i++){
            out_mean += row_out[i];
        }
        out_mean /= N;
        
        std::cout << "Row " << row << ":" << std::endl;
        std::cout << "  Input  - mean: " << mean << ", std: " << std_dev << std::endl;
        std::cout << "  Output - mean: " << out_mean << std::endl;
        std::cout << "  First 5 output values: ";
        for(int i = 0; i < 5; i++){
            std::cout << row_out[i] << " ";
        }
        std::cout << std::endl;
    }
}

int main(){
    // Configuration
    int batch = 4;      // Number of rows
    int N = 512;        // Features per row
    float epsilon = 1e-5f;
    
    size_t input_bytes = batch * N * sizeof(float);
    size_t param_bytes = N * sizeof(float);
    
    std::cout << "Testing LayerNorm with:" << std::endl;
    std::cout << "  Batch size: " << batch << std::endl;
    std::cout << "  Features (N): " << N << std::endl;
    std::cout << "  Epsilon: " << epsilon << std::endl;
    
    // Allocate host memory
    float *h_inp = new float[batch * N];
    float *h_out = new float[batch * N];
    float *h_gamma = new float[N];
    float *h_beta = new float[N];
    
    // Initialize input with different values per row
    for(int row = 0; row < batch; row++){
        for(int i = 0; i < N; i++){
            h_inp[row * N + i] = (row + 1) * 10.0f + i * 0.1f;  // Different range per row
        }
    }
    
    // Initialize gamma (scale) to 1.0 and beta (shift) to 0.0 (identity transform)
    for(int i = 0; i < N; i++){
        h_gamma[i] = 1.0f;
        h_beta[i] = 0.0f;
    }
    
    // Allocate device memory
    float *d_inp, *d_out, *d_gamma, *d_beta;
    cudaMalloc(&d_inp, input_bytes);
    cudaMalloc(&d_out, input_bytes);
    cudaMalloc(&d_gamma, param_bytes);
    cudaMalloc(&d_beta, param_bytes);
    
    // Copy to device
    cudaMemcpy(d_inp, h_inp, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, param_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, param_bytes, cudaMemcpyHostToDevice);
    
    // Launch kernel: 1 block per row, BLOCK_SIZE threads per block
    std::cout << "\nLaunching kernel with " << batch << " blocks, " 
              << BLOCK_SIZE << " threads per block" << std::endl;
    
    layer_norm<<<batch, BLOCK_SIZE>>>(d_out, d_inp, d_gamma, d_beta, N, epsilon);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        std::cout << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(h_out, d_out, input_bytes, cudaMemcpyDeviceToHost);
    
    // Verify results
    verify_layernorm(h_out, h_inp, batch, N, epsilon);
    
    std::cout << "\n✓ LayerNorm completed successfully!" << std::endl;
    
    // Test with non-identity gamma and beta
    std::cout << "\n=== Testing with gamma=2.0, beta=0.5 ===" << std::endl;
    for(int i = 0; i < N; i++){
        h_gamma[i] = 2.0f;
        h_beta[i] = 0.5f;
    }
    cudaMemcpy(d_gamma, h_gamma, param_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, param_bytes, cudaMemcpyHostToDevice);
    
    layer_norm<<<batch, BLOCK_SIZE>>>(d_out, d_inp, d_gamma, d_beta, N, epsilon);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, input_bytes, cudaMemcpyDeviceToHost);
    
    std::cout << "Row 0, first 5 output values (should be ~2.5): ";
    for(int i = 0; i < 5; i++){
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    delete[] h_inp;
    delete[] h_out;
    delete[] h_gamma;
    delete[] h_beta;
    
    cudaFree(d_inp);
    cudaFree(d_out);
    cudaFree(d_gamma);
    cudaFree(d_beta);
    
    return 0;
}
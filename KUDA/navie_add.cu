

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

__global__ void vec_add(float *a, float *b, float *c, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        c[idx] = a[idx] + b[idx];
    }
}

bool verify_results(float *a, float *b, float *c, int n){
    for(int i = 0; i < n; i++){
        float expected = a[i] + b[i];
        if(fabs(c[i] - expected) > 1e-5){
            std::cout << "Mismatch at index " << i 
                      << ": expected " << expected 
                      << ", got " << c[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main(){
    int n = 1000000;
    size_t bytes = n * sizeof(float);
    
    // Allocate host memory
    float *h_a = new float[n];
    float *h_b = new float[n];
    float *h_c = new float[n];
    
    // Initialize input arrays
    for(int i = 0; i < n; i++){
        h_a[i] = i * 1.0f;
        h_b[i] = i * 2.0f;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    
    std::cout << "Launching kernel with " << blocks << " blocks and " 
              << threads << " threads per block" << std::endl;
    
    vec_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess){
        std::cout << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    // Verify results
    std::cout << "Verifying results..." << std::endl;
    if(verify_results(h_a, h_b, h_c, n)){
        std::cout << "✓ Test PASSED! All " << n << " elements are correct." << std::endl;
    } else {
        std::cout << "✗ Test FAILED!" << std::endl;
    }
    
    // Print sample results
    std::cout << "\nSample results (first 5 elements):" << std::endl;
    for(int i = 0; i < 5; i++){
        std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << std::endl;
    }
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    
    return 0;
}

    
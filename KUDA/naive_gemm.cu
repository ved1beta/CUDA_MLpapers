
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <iostream>
#include <chrono>

using namespace nvcuda;

__global__ void gemm_naive(float *A , float *B , float *C , int M , int N , int K){
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if(row < M && col < N){
        float sum = 0.0f;
        for(int k = 0; k < K; k++){
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

#define TILE_SIZE 16

__global__ void gemm_tiled(float *A , float *B , float *C , int M , int N , int K){
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    float sum = 0.0f;
    
    for(int t = 0 ; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++){

        int A_col = t * TILE_SIZE + threadIdx.y;
        int B_row = t * TILE_SIZE + threadIdx.x;


    if(row < M && A_col < K){
        A_tile[threadIdx.x][threadIdx.y] = A[row * K + A_col];
    }
    else{
        A_tile[threadIdx.x][threadIdx.y] = 0.0f;
    }
    
    if(col < N && B_row < K){
        B_tile[threadIdx.x][threadIdx.y] = B[B_row * N + col];
    }
    else{
        B_tile[threadIdx.x][threadIdx.y] = 0.0f;
    }
    __syncthreads();

    for(int n = 0; n < TILE_SIZE; n++){
        sum += A_tile[threadIdx.x][n] * B_tile[n][threadIdx.y];
    }

    __syncthreads();

    }
    
    if(row < M && col < N){
        C[row * N + col] = sum;
    }
}

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void tensor_gemm(half *A , half *B , float *C , int M , int N , int K){

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    int warpM = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
    int warpN = (threadIdx.y + blockIdx.y * blockDim.y);

    wmma::fill_fragment(c_frag, 0.0f);

    for(int k = 0; k < K; k += WMMA_K){
        int aRow = warpM * WMMA_M;
        int aCol = k;
        int bRow = k;
        int bCol = warpN * WMMA_N;

        if(aRow < M && aCol < K && bRow < K && bCol < N){
            // Load 16x16 tile from A and B into fragments
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow * N + bCol, N);
            
            // Perform 16x16x16 matrix multiplication using tensor cores
            // This is a SINGLE hardware instruction!
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
}

    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;
    if(cRow < M && cCol < N){
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

void init_matrix(float *mat, int rows, int cols){
    for(int i = 0; i < rows * cols; i++){
        mat[i] = (float)(rand() % 10) / 10.0f;
    }
}

void init_matrix_half(half *mat, int rows, int cols){
    for(int i = 0; i < rows * cols; i++){
        mat[i] = __float2half((float)(rand() % 10) / 10.0f);
    }
}

bool verify_result(float *C_test, float *C_ref, int M, int N, float threshold = 0.01f){
    for(int i = 0; i < M * N; i++){
        if(fabs(C_test[i] - C_ref[i]) > threshold){
            std::cout << "Mismatch at index " << i << ": " 
                      << C_test[i] << " vs " << C_ref[i] << std::endl;
            return false;
        }
    }
    return true;
}

void benchmark_kernel(const char* name, void (*kernel_func)(float*, float*, float*, int, int, int), 
                     float *d_A, float *d_B, float *d_C, int M, int N, int K, dim3 grid, dim3 block){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    kernel_func<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    
    // Benchmark
    cudaEventRecord(start);
    for(int i = 0; i < 10; i++){
        kernel_func<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 10.0f;  // Average
    
    // Calculate GFLOPS: 2*M*N*K operations
    float gflops = (2.0f * M * N * K) / (ms * 1e6);
    
    std::cout << name << ": " << ms << " ms, " << gflops << " GFLOPS" << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(){
    // Matrix dimensions
    int M = 1024;
    int N = 1024;
    int K = 1024;
    
    std::cout << "Matrix Multiplication: C[" << M << "," << N << "] = A[" << M << "," << K 
              << "] * B[" << K << "," << N << "]" << std::endl;
    std::cout << "Operations: " << 2.0 * M * N * K / 1e9 << " billion" << std::endl << std::endl;
    
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C_naive, *d_C_tiled;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C_naive, size_C);
    cudaMalloc(&d_C_tiled, size_C);
    
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    
    // ========== NAIVE GEMM ==========
    std::cout << "1. NAIVE GEMM" << std::endl;
    dim3 block_naive(16, 16);
    dim3 grid_naive((N + 15) / 16, (M + 15) / 16);
    
    gemm_naive<<<grid_naive, block_naive>>>(d_A, d_B, d_C_naive, M, N, K);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C_naive, size_C, cudaMemcpyDeviceToHost);
    std::cout << "Sample result C[0,0] = " << h_C[0] << std::endl;
    
    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    gemm_naive<<<grid_naive, block_naive>>>(d_A, d_B, d_C_naive, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_naive;
    cudaEventElapsedTime(&ms_naive, start, stop);
    float gflops_naive = (2.0f * M * N * K) / (ms_naive * 1e6);
    std::cout << "Time: " << ms_naive << " ms, " << gflops_naive << " GFLOPS" << std::endl << std::endl;
    
    // ========== TILED GEMM ==========
    std::cout << "2. TILED GEMM (Shared Memory)" << std::endl;
    dim3 block_tiled(TILE_SIZE, TILE_SIZE);
    dim3 grid_tiled((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    gemm_tiled<<<grid_tiled, block_tiled>>>(d_A, d_B, d_C_tiled, M, N, K);
    cudaDeviceSynchronize();
    
    float *h_C_tiled = new float[M * N];
    cudaMemcpy(h_C_tiled, d_C_tiled, size_C, cudaMemcpyDeviceToHost);
    
    if(verify_result(h_C_tiled, h_C, M, N)){
        std::cout << "✓ Results match naive implementation" << std::endl;
    }
    
    cudaEventRecord(start);
    gemm_tiled<<<grid_tiled, block_tiled>>>(d_A, d_B, d_C_tiled, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_tiled;
    cudaEventElapsedTime(&ms_tiled, start, stop);
    float gflops_tiled = (2.0f * M * N * K) / (ms_tiled * 1e6);
    std::cout << "Time: " << ms_tiled << " ms, " << gflops_tiled << " GFLOPS" << std::endl;
    std::cout << "Speedup vs naive: " << ms_naive / ms_tiled << "x" << std::endl << std::endl;
    
    // ========== COMPARISON ==========
    std::cout << "=== SUMMARY ===" << std::endl;
    std::cout << "Naive:  " << gflops_naive << " GFLOPS" << std::endl;
    std::cout << "Tiled:  " << gflops_tiled << " GFLOPS (" 
              << (gflops_tiled/gflops_naive) << "x faster)" << std::endl;
    std::cout << "\nNote: Tensor Core implementation requires FP16 and Volta+ GPU" << std::endl;
    std::cout << "      (typically 10-20x faster than tiled for large matrices)" << std::endl;
    
    // Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_tiled;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_naive);
    cudaFree(d_C_tiled);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>


__global__ void paged_atten(
    const half* q, 
    const half* k_cache,
    const half* v_cache,
    const half* block_table
    half* output,
    int seq_len,
    int num_kv_heads, 
    int num_q_heads,
    int head_dim,
){
    int q_head = blockIdx.x;


    float max_score = -INFINITY;
    float sum = 0.0f;

    float acc[MAX_HAED_DIM] = {0.0f}; 

    for (int token ; token < seq_len; token++){

        logical_block = token / BLOCK_SIZE;
        offset = token % BLOCK_SIZE;

        for (int token = 0;
         token < seq_len;
         token++) {

        int logical_block =
            token / BLOCK_SIZE;

        int offset =
            token % BLOCK_SIZE;

        int physical_block =
            block_table[logical_block];

        const half* K =
            get_k(
                k_cache,
                physical_block,
                offset,
            );

        const half* V =
            get_v(
                v_cache,
                physical_block,
                offset,
            );

        float score = dot(q, K);

    }





}


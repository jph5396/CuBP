#include "kernels.h"
#include <cuda_runtime.h>

__global__ void scale_kernel(float* data, float scalar, size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scalar;
    }
}

void cuda_scale(const float* h_data, float* h_result, float scalar, size_t n) {
    float* d_data = nullptr;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);

    constexpr int block_size = 256;
    int num_blocks = static_cast<int>((n + block_size - 1) / block_size);
    scale_kernel<<<num_blocks, block_size>>>(d_data, scalar, n);

    cudaMemcpy(h_result, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}
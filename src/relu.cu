#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include "error_check.h"  // Error checking macro from include/error_check.h
#include "relu.h"         // Header for ReLU declarations

// ------------------------------------------------------------------------
// CUDA Kernel Implementation for ReLU Activation
// ------------------------------------------------------------------------
__global__ void relu_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Apply ReLU: if input is negative, output is zero; otherwise pass input through
        float in_val = input[idx];
        output[idx] = (in_val > 0.0f) ? in_val : 0.0f;
    }
}

// ------------------------------------------------------------------------
// CPU Reference Implementation for ReLU
// ------------------------------------------------------------------------
void cpu_relu(const float *input, float *output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
    }
}

// ------------------------------------------------------------------------
// Standalone Test for ReLU Module
// ------------------------------------------------------------------------
#ifdef STANDALONE_TEST

int main() {
    const int size = 1024;  // Example vector size
    // Allocate host memory
    float *h_input = (float*)malloc(size * sizeof(float));
    float *h_output_cpu = (float*)malloc(size * sizeof(float));
    float *h_output_gpu = (float*)malloc(size * sizeof(float));
    
    // Initialize h_input with random values in range [-10, 10]
    for (int i = 0; i < size; i++) {
        h_input[i] = (float)(rand() % 200) / 10.0f - 10.0f;
    }
    
    // Compute CPU reference ReLU results
    cpu_relu(h_input, h_output_cpu, size);
    
    // Allocate device memory
    float *d_input, *d_output;
    CHECK_CUDA_ERR(cudaMalloc((void**)&d_input, size * sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc((void**)&d_output, size * sizeof(float)));
    
    // Copy input data from host to device
    CHECK_CUDA_ERR(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Configure kernel launch parameters (1D grid)
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    // Launch the ReLU kernel
    relu_kernel<<<gridSize, blockSize>>>(d_input, d_output, size);
    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    
    // Copy output from device to host
    CHECK_CUDA_ERR(cudaMemcpy(h_output_gpu, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Validate GPU results against the CPU reference
    int errors = 0;
    for (int i = 0; i < size; i++) {
        if (fabs(h_output_cpu[i] - h_output_gpu[i]) > 1e-5f) {
            errors++;
        }
    }
    
    if (errors == 0)
        printf("ReLU Test: Results match! (%d elements tested)\n", size);
    else
        printf("ReLU Test: FAILED with %d mismatches.\n", errors);
    
    // Clean up host and device memory
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);
    CHECK_CUDA_ERR(cudaFree(d_input));
    CHECK_CUDA_ERR(cudaFree(d_output));
    
    return 0;
}

#endif  // STANDALONE_TEST

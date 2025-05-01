#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include "error_check.h"    // Error checking macro from include/error_check.h
#include "batch_norm.h"     // Declarations for batch normalization functions

// ------------------------------------------------------------------------
// GPU Kernel Implementation for Batch Normalization
// ------------------------------------------------------------------------
// Computes for each element: 
//   y = gamma * ((x - mean) / sqrt(variance + epsilon)) + beta
__global__ void batch_norm_kernel(const float* input, float* output, int size,
                                  float mean, float variance,
                                  float gamma, float beta, float epsilon) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float normalized = (input[idx] - mean) / sqrtf(variance + epsilon);
        output[idx] = gamma * normalized + beta;
    }
}

// ------------------------------------------------------------------------
// CPU Reference Implementation for Batch Normalization
// ------------------------------------------------------------------------
void cpu_batch_norm(const float *input, float *output, int size,
                    float gamma, float beta, float epsilon) {
    // Calculate mean
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += input[i];
    }
    float mean = sum / size;
    
    // Calculate variance
    float var_sum = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = input[i] - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / size;
    
    // Normalize each element
    for (int i = 0; i < size; i++) {
        output[i] = gamma * ((input[i] - mean) / sqrt(variance + epsilon)) + beta;
    }
}

// ------------------------------------------------------------------------
// Optional Standalone Test
// ------------------------------------------------------------------------
#ifdef STANDALONE_TEST

int main() {
    const int size = 1024;  // Example vector size for testing
    float gamma = 1.0f, beta = 0.0f, epsilon = 1e-5f;
    
    // Allocate host memory
    float *h_input = (float*)malloc(size * sizeof(float));
    float *h_output_cpu = (float*)malloc(size * sizeof(float));
    float *h_output_gpu = (float*)malloc(size * sizeof(float));
    
    // Initialize input with random values (e.g., range 0 to 10)
    for (int i = 0; i < size; i++) {
        h_input[i] = (float)(rand() % 100) / 10.0f;
    }
    
    // Compute reference results on CPU
    cpu_batch_norm(h_input, h_output_cpu, size, gamma, beta, epsilon);
    
    // Allocate device memory
    float *d_input, *d_output;
    CHECK_CUDA_ERR(cudaMalloc((void**)&d_input, size * sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc((void**)&d_output, size * sizeof(float)));
    
    // Copy input from host to device
    CHECK_CUDA_ERR(cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Compute mean and variance on the host (to use in the kernel)
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += h_input[i];
    }
    float mean = sum / size;
    
    float var_sum = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = h_input[i] - mean;
        var_sum += diff * diff;
    }
    float variance = var_sum / size;
    
    // Set up the kernel launch configuration (1D grid)
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    // Launch the batch normalization kernel on the GPU
    batch_norm_kernel<<<gridSize, blockSize>>>(d_input, d_output, size, mean, variance, gamma, beta, epsilon);
    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    
    // Copy the results back from device to host
    CHECK_CUDA_ERR(cudaMemcpy(h_output_gpu, d_output, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Validate the results
    int errors = 0;
    for (int i = 0; i < size; i++) {
        if (fabs(h_output_cpu[i] - h_output_gpu[i]) > 1e-5f)
            errors++;
    }
    
    if (errors == 0) {
        printf("Batch Norm Test: Results match!\n");
    } else {
        printf("Batch Norm Test: %d mismatches found!\n", errors);
    }
    
    // Clean up host and device memory
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);
    CHECK_CUDA_ERR(cudaFree(d_input));
    CHECK_CUDA_ERR(cudaFree(d_output));
    
    return 0;
}

#endif  // STANDALONE_TEST

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include "error_check.h"  // Error checking macro from include/error_check.h
#include "pooling.h"      // Header that declares max_pooling_kernel and cpu_max_pooling

// ------------------------------------------------------------------------
// CUDA Kernel Implementation for Max Pooling
// ------------------------------------------------------------------------
// This kernel performs max pooling on the input matrix.
// It assumes that the input dimensions are evenly divisible by the stride.
__global__ void max_pooling_kernel(const float* input, float* output,
                                   int inputWidth, int inputHeight,
                                   int poolSize, int stride) {
    // Compute the output indices for this thread
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Calculate output dimensions (assumes perfect division)
    int outputWidth = inputWidth / stride;
    int outputHeight = inputHeight / stride;
    
    if (out_x < outputWidth && out_y < outputHeight) {
        float max_val = -FLT_MAX;
        // Loop over the pooling window
        for (int dy = 0; dy < poolSize; dy++) {
            for (int dx = 0; dx < poolSize; dx++) {
                int in_x = out_x * stride + dx;
                int in_y = out_y * stride + dy;
                float val = input[in_y * inputWidth + in_x];
                if (val > max_val) {
                    max_val = val;
                }
            }
        }
        output[out_y * outputWidth + out_x] = max_val;
    }
}

// ------------------------------------------------------------------------
// CPU Reference Implementation for Max Pooling
// ------------------------------------------------------------------------
void cpu_max_pooling(const float *input, float *output,
                     int inputWidth, int inputHeight,
                     int poolSize, int stride) {
    int outputWidth = inputWidth / stride;
    int outputHeight = inputHeight / stride;
    for (int y = 0; y < outputHeight; y++) {
        for (int x = 0; x < outputWidth; x++) {
            float max_val = -FLT_MAX;
            for (int dy = 0; dy < poolSize; dy++) {
                for (int dx = 0; dx < poolSize; dx++) {
                    int in_x = x * stride + dx;
                    int in_y = y * stride + dy;
                    float val = input[in_y * inputWidth + in_x];
                    if (val > max_val) {
                        max_val = val;
                    }
                }
            }
            output[y * outputWidth + x] = max_val;
        }
    }
}

// ------------------------------------------------------------------------
// Optional Standalone Test for Pooling Module
// ------------------------------------------------------------------------
#ifdef STANDALONE_TEST

int main() {
    // Define an input with dimensions that are evenly divisible by the pooling stride.
    // For instance, let's use an 8x8 matrix for testing with a 2x2 pooling window and stride 2.
    const int inputWidth = 8;
    const int inputHeight = 8;
    const int poolSize = 2;
    const int stride = 2;
    const int outputWidth = inputWidth / stride;     // 8/2 = 4
    const int outputHeight = inputHeight / stride;   // 8/2 = 4
    const int inputSize = inputWidth * inputHeight;
    const int outputSize = outputWidth * outputHeight;
    
    // Allocate host memory
    float *h_input = (float*)malloc(inputSize * sizeof(float));
    float *h_output_cpu = (float*)malloc(outputSize * sizeof(float));
    float *h_output_gpu = (float*)malloc(outputSize * sizeof(float));
    
    // Initialize the input with random values in the range [0, 100]
    for (int i = 0; i < inputSize; i++) {
        h_input[i] = (float)(rand() % 101);
    }
    
    // Compute the expected pooling result using the CPU reference implementation
    cpu_max_pooling(h_input, h_output_cpu, inputWidth, inputHeight, poolSize, stride);
    
    // Allocate device memory
    float *d_input, *d_output;
    CHECK_CUDA_ERR(cudaMalloc((void**)&d_input, inputSize * sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc((void**)&d_output, outputSize * sizeof(float)));
    
    // Copy input data from host to device
    CHECK_CUDA_ERR(cudaMemcpy(d_input, h_input, inputSize * sizeof(float), cudaMemcpyHostToDevice));
    
    // Setup kernel launch configuration for 2D grid
    dim3 blockDim(16, 16);
    dim3 gridDim((outputWidth + blockDim.x - 1) / blockDim.x,
                 (outputHeight + blockDim.y - 1) / blockDim.y);
    
    // Launch the max pooling kernel
    max_pooling_kernel<<<gridDim, blockDim>>>(d_input, d_output,
                                              inputWidth, inputHeight,
                                              poolSize, stride);
    CHECK_CUDA_ERR(cudaGetLastError());
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    
    // Copy the kernel's output from device to host
    CHECK_CUDA_ERR(cudaMemcpy(h_output_gpu, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Validate GPU results against the CPU reference
    int errors = 0;
    for (int i = 0; i < outputSize; i++) {
        if (fabs(h_output_cpu[i] - h_output_gpu[i]) > 1e-5f)
            errors++;
    }
    
    if (errors == 0)
        printf("Pooling Test: Results match! (%d elements tested)\n", outputSize);
    else
        printf("Pooling Test: FAILED with %d mismatches.\n", errors);
    
    // Clean up host and device memory
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);
    CHECK_CUDA_ERR(cudaFree(d_input));
    CHECK_CUDA_ERR(cudaFree(d_output));
    
    return 0;
}

#endif  // STANDALONE_TEST

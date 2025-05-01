// File: src/cuda_convolution.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include "error_check.h"      // Error checking functions from include/error_check.h
#include "convolution.h"      // Contains the declaration for naive_convolution and cpu_convolution

// ------------------------
// Function Definitions
// ------------------------

// The actual kernel and CPU function definitions remain here.
// They should match the declarations in convolution.h.

// Naive CUDA convolution kernel
__global__ void naive_convolution(const float* input, const float* kernel, float* output,
                                    int inputWidth, int inputHeight, 
                                    int kernelWidth, int kernelHeight,
                                    int outputWidth, int outputHeight) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (out_x < outputWidth && out_y < outputHeight) {
        float sum = 0.0f;
        for (int ky = 0; ky < kernelHeight; ky++) {
            for (int kx = 0; kx < kernelWidth; kx++) {
                int in_x = out_x + kx;
                int in_y = out_y + ky;
                if (in_x < inputWidth && in_y < inputHeight) {
                    sum += input[in_y * inputWidth + in_x] * kernel[ky * kernelWidth + kx];
                }
            }
        }
        output[out_y * outputWidth + out_x] = sum;
    }
}

// CPU implementation for validation
void cpu_convolution(const float *input, const float *kernel, float *output,
                     int inputWidth, int inputHeight, int kernelWidth, int kernelHeight,
                     int outputWidth, int outputHeight) {
    for (int y = 0; y < outputHeight; y++) {
        for (int x = 0; x < outputWidth; x++) {
            float sum = 0.0f;
            for (int ky = 0; ky < kernelHeight; ky++) {
                for (int kx = 0; kx < kernelWidth; kx++) {
                    int in_x = x + kx;
                    int in_y = y + ky;
                    if (in_x < inputWidth && in_y < inputHeight)
                        sum += input[in_y * inputWidth + in_x] * kernel[ky * kernelWidth + kx];
                }
            }
            output[y * outputWidth + x] = sum;
        }
    }
}

// ------------------------
// Standalone Test Main Function
// ------------------------
#ifdef STANDALONE_TEST
int main() {
    // Define dimensions
    const int inputWidth = 128;
    const int inputHeight = 128;
    const int kernelWidth = 3;
    const int kernelHeight = 3;
    // Calculate output dimensions (assuming stride=1, no padding)
    const int outputWidth = inputWidth - kernelWidth + 1;
    const int outputHeight = inputHeight - kernelHeight + 1;
    
    // Compute sizes in bytes
    size_t inputSize = inputWidth * inputHeight * sizeof(float);
    size_t kernelSize = kernelWidth * kernelHeight * sizeof(float);
    size_t outputSize = outputWidth * outputHeight * sizeof(float);
    
    // Allocate host memory
    float *h_input      = (float*)malloc(inputSize);
    float *h_kernel     = (float*)malloc(kernelSize);
    float *h_output_cpu = (float*)malloc(outputSize);
    float *h_output_gpu = (float*)malloc(outputSize);
    
    // Initialize input data (random values between 0 and 9)
    for (int i = 0; i < inputWidth * inputHeight; i++)
        h_input[i] = (float)(rand() % 10);
    
    // Initialize kernel (simple edge-detection filter)
    float exampleKernel[9] = { 1, 0, -1, 1, 0, -1, 1, 0, -1 };
    for (int i = 0; i < kernelWidth * kernelHeight; i++)
        h_kernel[i] = exampleKernel[i];
    
    // Run CPU convolution for baseline reference
    cpu_convolution(h_input, h_kernel, h_output_cpu,
                    inputWidth, inputHeight,
                    kernelWidth, kernelHeight,
                    outputWidth, outputHeight);
    
    // Allocate device memory
    float *d_input, *d_kernel, *d_output;
    CHECK_CUDA_ERR(cudaMalloc((void**)&d_input, inputSize));
    CHECK_CUDA_ERR(cudaMalloc((void**)&d_kernel, kernelSize));
    CHECK_CUDA_ERR(cudaMalloc((void**)&d_output, outputSize));
    
    // Copy data from host to device
    CHECK_CUDA_ERR(cudaMemcpy(d_input, h_input, inputSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_kernel, h_kernel, kernelSize, cudaMemcpyHostToDevice));
    
    // Define block and grid dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((outputWidth + blockDim.x - 1) / blockDim.x,
                 (outputHeight + blockDim.y - 1) / blockDim.y);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERR(cudaEventCreate(&start));
    CHECK_CUDA_ERR(cudaEventCreate(&stop));
    
    // Warm-up iterations (not timed)
    for (int i = 0; i < 5; i++) {
        naive_convolution<<<gridDim, blockDim>>>(d_input, d_kernel, d_output,
                                                   inputWidth, inputHeight,
                                                   kernelWidth, kernelHeight,
                                                   outputWidth, outputHeight);
        CHECK_CUDA_ERR(cudaDeviceSynchronize());
    }
    
    // Timing iterations
    const int iterations = 100;
    float totalTime = 0.0f;
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA_ERR(cudaEventRecord(start, 0));
        naive_convolution<<<gridDim, blockDim>>>(d_input, d_kernel, d_output,
                                                   inputWidth, inputHeight,
                                                   kernelWidth, kernelHeight,
                                                   outputWidth, outputHeight);
        CHECK_CUDA_ERR(cudaGetLastError());
        CHECK_CUDA_ERR(cudaEventRecord(stop, 0));
        CHECK_CUDA_ERR(cudaEventSynchronize(stop));
    
        float iterTime;
        CHECK_CUDA_ERR(cudaEventElapsedTime(&iterTime, start, stop));
        totalTime += iterTime;
    }
    
    float averageTime = totalTime / iterations;
    printf("Average kernel execution time over %d iterations: %f ms\n", iterations, averageTime);
    
    // Copy result back to host
    CHECK_CUDA_ERR(cudaMemcpy(h_output_gpu, d_output, outputSize, cudaMemcpyDeviceToHost));
    
    // Validate the GPU results against the CPU results
    int errors = 0;
    for (int i = 0; i < outputWidth * outputHeight; i++) {
        if (fabs(h_output_cpu[i] - h_output_gpu[i]) > 1e-5)
            errors++;
    }
    
    if (errors == 0) {
        printf("Results match! Average Kernel execution time: %f ms\n", averageTime);
    } else {
        printf("There were %d mismatches between CPU and GPU results.\n", errors);
    }
    
    // Clean up memory and CUDA events
    CHECK_CUDA_ERR(cudaFree(d_input));
    CHECK_CUDA_ERR(cudaFree(d_kernel));
    CHECK_CUDA_ERR(cudaFree(d_output));
    free(h_input);
    free(h_kernel);
    free(h_output_cpu);
    free(h_output_gpu);
    CHECK_CUDA_ERR(cudaEventDestroy(start));
    CHECK_CUDA_ERR(cudaEventDestroy(stop));
    
    return 0;
}
#endif  // STANDALONE_TEST

#ifndef RELU_H
#define RELU_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA kernel for ReLU activation
// Computes output[i] = max(0, input[i]) for each element.
__global__ void relu_kernel(const float* input, float* output, int size);

// CPU reference implementation for ReLU
void cpu_relu(const float *input, float *output, int size);

#ifdef __cplusplus
}
#endif

#endif // RELU_H

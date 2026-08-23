#ifndef POOLING_H
#define POOLING_H

#include <cuda_runtime.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA kernel for max pooling (for example, 2x2 pooling with a given stride)
// Assumes that input dimensions are evenly divisible by the stride.
__global__ void max_pooling_kernel(const float* input, float* output,
                                   int inputWidth, int inputHeight,
                                   int poolSize, int stride);

// CPU reference implementation for max pooling
void cpu_max_pooling(const float *input, float *output,
                     int inputWidth, int inputHeight,
                     int poolSize, int stride);

#ifdef __cplusplus
}
#endif

#endif // POOLING_H

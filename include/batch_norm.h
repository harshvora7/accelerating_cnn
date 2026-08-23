#ifndef BATCH_NORM_H
#define BATCH_NORM_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA kernel for batch normalization
// Computes y = gamma * ((x - mean) / sqrt(variance + epsilon)) + beta
__global__ void batch_norm_kernel(const float* input, float* output, int size,
                                    float mean, float variance,
                                    float gamma, float beta, float epsilon);

// CPU reference implementation for batch normalization
void cpu_batch_norm(const float *input, float *output, int size,
                    float gamma, float beta, float epsilon);

#ifdef __cplusplus
}
#endif

#endif // BATCH_NORM_H

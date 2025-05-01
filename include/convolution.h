#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA kernel for naive convolution
__global__ void naive_convolution(const float* input, const float* kernel, float* output,
                                    int inputWidth, int inputHeight,
                                    int kernelWidth, int kernelHeight,
                                    int outputWidth, int outputHeight);

// CPU reference implementation for convolution
void cpu_convolution(const float *input, const float *kernel, float *output,
                     int inputWidth, int inputHeight,
                     int kernelWidth, int kernelHeight,
                     int outputWidth, int outputHeight);

#ifdef __cplusplus
}
#endif

#endif // CONVOLUTION_H

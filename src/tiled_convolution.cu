// File: src/tiled_convolution.cu
//
// Shared-memory tiled convolution with the filter held in constant memory.
// Hand-optimized counterpart to naive_convolution (cuda_convolution.cu): same math
// (valid cross-correlation, stride 1, no padding), but each input element is loaded
// from global memory once per block into shared memory and reused by every output
// pixel in the tile, instead of being re-read from global memory for every tap.

#include <stdio.h>
#include <cuda_runtime.h>
#include "error_check.h"
#include "tiled_convolution.h"

// Filter in constant memory: broadcast to all threads and cached on-chip.
__constant__ float c_kernel[MAX_KERNEL_DIM * MAX_KERNEL_DIM];

// Shared tile: one output tile (TILE_DIM x TILE_DIM) plus the halo the filter reaches
// into. Sized for the largest supported filter so the bound is compile-time constant;
// at runtime only the top-left (TILE_DIM + kW - 1) x (TILE_DIM + kH - 1) region is used.
#define SMEM_DIM (TILE_DIM + MAX_KERNEL_DIM - 1)

// Host helper: upload the filter to constant memory once before launching.
extern "C" void set_tiled_conv_kernel(const float* h_kernel,
                                      int kernelWidth, int kernelHeight) {
    CHECK_CUDA_ERR(cudaMemcpyToSymbol(c_kernel, h_kernel,
                                      kernelWidth * kernelHeight * sizeof(float)));
}

extern "C" __global__ void tiled_convolution(const float* input, float* output,
                                             int inputWidth, int inputHeight,
                                             int kernelWidth, int kernelHeight,
                                             int outputWidth, int outputHeight) {
    __shared__ float tile[SMEM_DIM][SMEM_DIM];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // For valid (no-pad, stride-1) convolution, output (ox,oy) reads input (ox+kx, oy+ky),
    // so the input tile origin equals the output tile origin.
    const int baseX = blockIdx.x * TILE_DIM;
    const int baseY = blockIdx.y * TILE_DIM;

    // Input region needed for this output tile (tile + halo).
    const int inTileW = TILE_DIM + kernelWidth  - 1;
    const int inTileH = TILE_DIM + kernelHeight - 1;

    // Cooperatively load the input tile into shared memory. The block has fewer threads
    // than the region it must load, so each thread strides to cover the halo.
    for (int ly = ty; ly < inTileH; ly += TILE_DIM) {
        for (int lx = tx; lx < inTileW; lx += TILE_DIM) {
            const int gx = baseX + lx;
            const int gy = baseY + ly;
            float v = 0.0f;
            if (gx < inputWidth && gy < inputHeight) {
                v = input[gy * inputWidth + gx];
            }
            tile[ly][lx] = v;
        }
    }

    __syncthreads();

    // Each thread computes one output pixel from shared memory + constant-memory taps.
    const int outX = baseX + tx;
    const int outY = baseY + ty;
    if (outX < outputWidth && outY < outputHeight) {
        float sum = 0.0f;
        for (int ky = 0; ky < kernelHeight; ++ky) {
            for (int kx = 0; kx < kernelWidth; ++kx) {
                sum += tile[ty + ky][tx + kx] * c_kernel[ky * kernelWidth + kx];
            }
        }
        output[outY * outputWidth + outX] = sum;
    }
}

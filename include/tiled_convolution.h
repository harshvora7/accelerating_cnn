#ifndef TILED_CONVOLUTION_H
#define TILED_CONVOLUTION_H

#include <cuda_runtime.h>

// Tile size == CUDA block dimension for the tiled convolution.
// The kernel MUST be launched with a (TILE_DIM x TILE_DIM) thread block.
#define TILE_DIM 16

// Largest filter (per side) the tiled kernel supports. Bounds both the
// shared-memory tile and the constant-memory filter buffer at compile time.
// 15 leaves headroom for the filter-size sweep in the next phase.
#define MAX_KERNEL_DIM 31

#ifdef __cplusplus
extern "C" {
#endif

// Copies the convolution filter into GPU constant memory.
// Call once on the host BEFORE launching tiled_convolution.
void set_tiled_conv_kernel(const float* h_kernel, int kernelWidth, int kernelHeight);

// Shared-memory tiled convolution: valid cross-correlation, stride 1, no padding.
// The filter is read from constant memory (see set_tiled_conv_kernel), so unlike
// naive_convolution there is NO kernel pointer argument.
// Launch with block = (TILE_DIM, TILE_DIM),
//            grid  = (ceil(outW/TILE_DIM), ceil(outH/TILE_DIM)).
__global__ void tiled_convolution(const float* input, float* output,
                                  int inputWidth, int inputHeight,
                                  int kernelWidth, int kernelHeight,
                                  int outputWidth, int outputHeight);

#ifdef __cplusplus
}
#endif

#endif // TILED_CONVOLUTION_H

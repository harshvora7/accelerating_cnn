#ifndef CUDNN_POOLING_H
#define CUDNN_POOLING_H

#include <cudnn.h>
#include <cuda_fp16.h> 
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque context for cuDNN pooling layer.
 */
typedef struct {
    cudnnTensorDescriptor_t    inDesc;    // descriptor for input tensor
    cudnnTensorDescriptor_t    outDesc;   // descriptor for output tensor
    cudnnPoolingDescriptor_t   poolDesc;  // descriptor for pooling operation
} CudnnPoolCtx;

/**
 * One-time setup: Create descriptors for a 2D pooling layer.
 *
 * @param N          batch size
 * @param C          number of channels
 * @param H          input height
 * @param W          input width
 * @param windowH    pooling window height
 * @param windowW    pooling window width
 * @param padH       vertical padding
 * @param padW       horizontal padding
 * @param strideH    vertical stride
 * @param strideW    horizontal stride
 * @returns          allocated CudnnPoolCtx*, destroy with cudnn_pooling_destroy()
 */
CudnnPoolCtx* cudnn_pooling_create(
    int N, int C, int H, int W,
    int windowH, int windowW,
    int padH, int padW,
    int strideH, int strideW);

/**
 * Timed forward pass: run max-pooling (NCHW layout).
 *
 * @param handle   cuDNN handle
 * @param ctx      context from cudnn_pooling_create()
 * @param d_input  device pointer to input tensor (N×C×H×W)
 * @param d_output device pointer to output tensor (N×C×H_out×W_out)
 */
void cudnn_pooling_forward(
    cudnnHandle_t handle,
    CudnnPoolCtx* ctx,
    const float* d_input,
    float* d_output);

/**
 * Cleanup: destroy descriptors and free context.
 */
void cudnn_pooling_destroy(CudnnPoolCtx* ctx);

typedef struct {
    cudnnTensorDescriptor_t    inDesc;
    cudnnTensorDescriptor_t    outDesc;
    cudnnPoolingDescriptor_t   poolDesc;
} CudnnPoolCtxFp16;

CudnnPoolCtxFp16* cudnn_pooling_create_fp16(
    int N, int C, int H, int W,
    int windowH, int windowW,
    int padH, int padW,
    int strideH, int strideW);

void cudnn_pooling_forward_fp16(cudnnHandle_t handle, CudnnPoolCtxFp16* ctx, const __half* d_input, __half* d_output);
void cudnn_pooling_destroy_fp16(CudnnPoolCtxFp16* ctx);
    
#ifdef __cplusplus
}
#endif

#endif // CUDNN_POOLING_H

#ifndef CUDNN_RELU_H
#define CUDNN_RELU_H

#include <cudnn.h>
#include <cuda_fp16.h>  // for __half

#ifdef __cplusplus
extern "C" {
#endif

// -------------------------
// FP32 ReLU API (existing)
// -------------------------
typedef struct {
    cudnnTensorDescriptor_t     desc;     // NCHW float tensor
    cudnnActivationDescriptor_t actDesc;  // ReLU activation
} CudnnReluCtx;

/**
 * One-time setup: creates tensor and activation descriptors.
 *
 * @param N  batch size
 * @param C  channels
 * @param H  height
 * @param W  width
 * @returns  allocated context, destroy with cudnn_relu_destroy()
 */
CudnnReluCtx* cudnn_relu_create(int N, int C, int H, int W);

/**
 * Timed forward pass (FP32): runs ReLU on d_input → d_output
 *
 * @param handle   cuDNN handle
 * @param ctx      context from cudnn_relu_create()
 * @param d_input  device input pointer (float*, N×C×H×W)
 * @param d_output device output pointer (float*, N×C×H×W)
 */
void cudnn_relu_forward(
    cudnnHandle_t    handle,
    CudnnReluCtx*    ctx,
    const float*     d_input,
    float*           d_output);

/**
 * Cleanup descriptors and free context (FP32).
 */
void cudnn_relu_destroy(CudnnReluCtx* ctx);

// -------------------------
// FP16 ReLU API (new)
// -------------------------
typedef struct {
    cudnnTensorDescriptor_t     desc;     // NCHW half tensor
    cudnnActivationDescriptor_t actDesc;  // ReLU activation
} CudnnReluCtxFp16;

/**
 * One-time setup: creates tensor and activation descriptors for half-precision.
 *
 * @param N  batch size
 * @param C  channels
 * @param H  height
 * @param W  width
 * @returns  allocated context, destroy with cudnn_relu_destroy_fp16()
 */
CudnnReluCtxFp16* cudnn_relu_create_fp16(int N, int C, int H, int W);

/**
 * Timed forward pass (FP16): runs ReLU on d_input_fp16 → d_output_fp16.
 *
 * @param handle        cuDNN handle
 * @param ctx           context from cudnn_relu_create_fp16()
 * @param d_input_fp16  device input pointer (__half*, N×C×H×W)
 * @param d_output_fp16 device output pointer (__half*, N×C×H×W)
 */
void cudnn_relu_forward_fp16(
    cudnnHandle_t      handle,
    CudnnReluCtxFp16*  ctx,
    const __half*      d_input_fp16,
    __half*            d_output_fp16);

/**
 * Cleanup descriptors and free context (FP16).
 */
void cudnn_relu_destroy_fp16(CudnnReluCtxFp16* ctx);

#ifdef __cplusplus
}
#endif

#endif // CUDNN_RELU_H

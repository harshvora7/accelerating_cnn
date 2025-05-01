// File: src/cudnn_relu.cu

#include "cudnn_relu.h"
#include "error_check.h"    // for CHECK_CUDA_ERR / CHECK_CUDNN
#include <cuda_fp16.h>      // for __half
#include <cudnn.h>
#include <stdio.h>
#include <stdlib.h>

// cuDNN error‐checking macro
#define CHECK_CUDNN(call)                                                     \
    do {                                                                      \
        cudnnStatus_t status = (call);                                        \
        if (status != CUDNN_STATUS_SUCCESS) {                                 \
            fprintf(stderr,                                                   \
                "cuDNN error at %s:%d: %s\n",                                 \
                __FILE__, __LINE__, cudnnGetErrorString(status));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)


// -------------------------
// FP32 ReLU (existing)
// -------------------------

CudnnReluCtx* cudnn_relu_create(int N, int C, int H, int W) {
    CudnnReluCtx* ctx = (CudnnReluCtx*)malloc(sizeof(*ctx));
    if (!ctx) {
        fprintf(stderr, "Failed to allocate CudnnReluCtx\n");
        exit(EXIT_FAILURE);
    }
    // Tensor descriptor (float)
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&ctx->desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        ctx->desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        N, C, H, W));
    // Activation descriptor (ReLU)
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&ctx->actDesc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(
        ctx->actDesc,
        CUDNN_ACTIVATION_RELU,
        CUDNN_PROPAGATE_NAN,
        /*reluCeiling=*/0.0f));
    return ctx;
}

void cudnn_relu_forward(
    cudnnHandle_t handle,
    CudnnReluCtx* ctx,
    const float*  d_input,
    float*        d_output)
{
    const float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnActivationForward(
        handle,
        ctx->actDesc,
        &alpha,
        ctx->desc, d_input,
        &beta,
        ctx->desc, d_output));
}

void cudnn_relu_destroy(CudnnReluCtx* ctx) {
    if (!ctx) return;
    cudnnDestroyActivationDescriptor(ctx->actDesc);
    cudnnDestroyTensorDescriptor(ctx->desc);
    free(ctx);
}


// -------------------------
// FP16 ReLU (new)
// -------------------------

CudnnReluCtxFp16* cudnn_relu_create_fp16(int N, int C, int H, int W) {
    CudnnReluCtxFp16* ctx = (CudnnReluCtxFp16*)malloc(sizeof(*ctx));
    if (!ctx) {
        fprintf(stderr, "Failed to allocate CudnnReluCtxFp16\n");
        exit(EXIT_FAILURE);
    }
    // Tensor descriptor (half)
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&ctx->desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        ctx->desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_HALF,
        N, C, H, W));
    // Activation descriptor (ReLU)
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&ctx->actDesc));
    CHECK_CUDNN(cudnnSetActivationDescriptor(
        ctx->actDesc,
        CUDNN_ACTIVATION_RELU,
        CUDNN_PROPAGATE_NAN,
        /*reluCeiling=*/0.0f));
    return ctx;
}

void cudnn_relu_forward_fp16(
    cudnnHandle_t       handle,
    CudnnReluCtxFp16*   ctx,
    const __half*       d_input_fp16,
    __half*             d_output_fp16)
{
    // cuDNN expects alpha/beta pointers that match tensor data type
    __half alpha = __float2half(1.0f);
    __half beta  = __float2half(0.0f);
    CHECK_CUDNN(cudnnActivationForward(
        handle,
        ctx->actDesc,
        &alpha,
        ctx->desc, d_input_fp16,
        &beta,
        ctx->desc, d_output_fp16));
}

void cudnn_relu_destroy_fp16(CudnnReluCtxFp16* ctx) {
    if (!ctx) return;
    cudnnDestroyActivationDescriptor(ctx->actDesc);
    cudnnDestroyTensorDescriptor(ctx->desc);
    free(ctx);
}

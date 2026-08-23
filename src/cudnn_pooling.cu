// File: src/cudnn_pooling.cu

#include "cudnn_pooling.h"
#include "error_check.h"   // for CHECK_CUDNN
#include <stdlib.h>
#include <stdio.h>

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
// FP32 Pooling (existing)
// -------------------------

CudnnPoolCtx* cudnn_pooling_create(
    int N, int C, int H, int W,
    int windowH, int windowW,
    int padH, int padW,
    int strideH, int strideW)
{
    CudnnPoolCtx* ctx = (CudnnPoolCtx*)malloc(sizeof(*ctx));
    if (!ctx) {
        fprintf(stderr, "Failed to alloc CudnnPoolCtx\n");
        exit(EXIT_FAILURE);
    }
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&ctx->inDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        ctx->inDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));

    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&ctx->poolDesc));
    CHECK_CUDNN(cudnnSetPooling2dDescriptor(
        ctx->poolDesc,
        CUDNN_POOLING_MAX,
        CUDNN_PROPAGATE_NAN,
        windowH, windowW,
        padH, padW,
        strideH, strideW));

    int n2, c2, h2, w2;
    CHECK_CUDNN(cudnnGetPooling2dForwardOutputDim(
        ctx->poolDesc, ctx->inDesc, &n2, &c2, &h2, &w2));

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&ctx->outDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        ctx->outDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n2, c2, h2, w2));

    return ctx;
}

void cudnn_pooling_forward(
    cudnnHandle_t handle,
    CudnnPoolCtx* ctx,
    const float*  d_input,
    float*        d_output)
{
    const float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnPoolingForward(
        handle,
        ctx->poolDesc,
        &alpha,
        ctx->inDesc, d_input,
        &beta,
        ctx->outDesc, d_output));
}

void cudnn_pooling_destroy(CudnnPoolCtx* ctx)
{
    if (!ctx) return;
    cudnnDestroyPoolingDescriptor(ctx->poolDesc);
    cudnnDestroyTensorDescriptor(ctx->inDesc);
    cudnnDestroyTensorDescriptor(ctx->outDesc);
    free(ctx);
}


// -------------------------
// FP16 Pooling (new)
// -------------------------

CudnnPoolCtxFp16* cudnn_pooling_create_fp16(
    int N, int C, int H, int W,
    int windowH, int windowW,
    int padH, int padW,
    int strideH, int strideW)
{
    CudnnPoolCtxFp16* ctx = (CudnnPoolCtxFp16*)malloc(sizeof(*ctx));
    if (!ctx) {
        fprintf(stderr, "Failed to alloc CudnnPoolCtxFp16\n");
        exit(EXIT_FAILURE);
    }
    // input descriptor (half)
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&ctx->inDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        ctx->inDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, N, C, H, W));

    // pooling descriptor
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&ctx->poolDesc));
    CHECK_CUDNN(cudnnSetPooling2dDescriptor(
        ctx->poolDesc,
        CUDNN_POOLING_MAX,
        CUDNN_PROPAGATE_NAN,
        windowH, windowW,
        padH, padW,
        strideH, strideW));

    // compute output dims
    int n2, c2, h2, w2;
    CHECK_CUDNN(cudnnGetPooling2dForwardOutputDim(
        ctx->poolDesc, ctx->inDesc, &n2, &c2, &h2, &w2));

    // output descriptor (half)
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&ctx->outDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        ctx->outDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, n2, c2, h2, w2));

    return ctx;
}

void cudnn_pooling_forward_fp16(
    cudnnHandle_t       handle,
    CudnnPoolCtxFp16*   ctx,
    const __half*       d_input,
    __half*             d_output)
{
    float alpha = 1.0f;   // cuDNN wants FP32 alpha/beta even for FP16 tensors
    float beta  = 0.0f;
    CHECK_CUDNN(cudnnPoolingForward(
        handle,
        ctx->poolDesc,
        &alpha,
        ctx->inDesc, d_input,
        &beta,
        ctx->outDesc, d_output));
}

void cudnn_pooling_destroy_fp16(CudnnPoolCtxFp16* ctx)
{
    if (!ctx) return;
    cudnnDestroyPoolingDescriptor(ctx->poolDesc);
    cudnnDestroyTensorDescriptor(ctx->inDesc);
    cudnnDestroyTensorDescriptor(ctx->outDesc);
    free(ctx);
}

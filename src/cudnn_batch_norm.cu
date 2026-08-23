// File: src/cudnn_batch_norm.cu

#include "cudnn_batch_norm.h"
#include "error_check.h"      // for CHECK_CUDA_ERR
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cuda_fp16.h>        // for __half
#include <stdio.h>
#include <stdlib.h>

// cuDNN error‐checking macro
#define CHECK_CUDNN(call)                                                   \
    do {                                                                    \
        cudnnStatus_t status = (call);                                      \
        if (status != CUDNN_STATUS_SUCCESS) {                               \
            fprintf(stderr,                                                 \
                "cuDNN error at %s:%d: %s\n",                               \
                __FILE__, __LINE__, cudnnGetErrorString(status));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// -------------------------
// FP32 BatchNorm (existing)
// -------------------------

CudnnBnCtx* cudnn_batch_norm_create(
    cudnnHandle_t handle,
    int N, int C, int H, int W,
    double      epsilon,
    const float* host_mean,
    const float* host_var,
    float        host_gamma,
    float        host_beta)
{
    CudnnBnCtx* ctx = (CudnnBnCtx*)malloc(sizeof(*ctx));
    if (!ctx) { fprintf(stderr,"OOM @ cudnn_batch_norm_create\n"); exit(EXIT_FAILURE); }
    ctx->epsilon = epsilon;

    // Create descriptors
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&ctx->xDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&ctx->paramDesc));

    // xDesc: N × C × H × W  (float)
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        ctx->xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, N, C, H, W));
    // paramDesc: 1 × C × 1 × 1  (float)
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        ctx->paramDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C, 1, 1));

    size_t paramSize = C * sizeof(float);
    // Allocate parameter buffers
    CHECK_CUDA_ERR(cudaMalloc(&ctx->d_scale, paramSize));
    CHECK_CUDA_ERR(cudaMalloc(&ctx->d_bias,  paramSize));
    CHECK_CUDA_ERR(cudaMalloc(&ctx->d_mean,  paramSize));
    CHECK_CUDA_ERR(cudaMalloc(&ctx->d_var,   paramSize));

    // Fill scale/bias arrays on host
    float* tmp = (float*)malloc(paramSize);
    for (int i = 0; i < C; ++i) tmp[i] = host_gamma;
    CHECK_CUDA_ERR(cudaMemcpy(ctx->d_scale, tmp, paramSize, cudaMemcpyHostToDevice));
    for (int i = 0; i < C; ++i) tmp[i] = host_beta;
    CHECK_CUDA_ERR(cudaMemcpy(ctx->d_bias,  tmp, paramSize, cudaMemcpyHostToDevice));
    free(tmp);

    // Copy running mean/var
    CHECK_CUDA_ERR(cudaMemcpy(ctx->d_mean, host_mean, paramSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(ctx->d_var,  host_var,  paramSize, cudaMemcpyHostToDevice));

    return ctx;
}

void cudnn_batch_norm_forward(
    cudnnHandle_t handle,
    CudnnBnCtx*   ctx,
    const float*  d_input,
    float*        d_output)
{
    const float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnBatchNormalizationForwardInference(
        handle,
        CUDNN_BATCHNORM_SPATIAL,
        &alpha, &beta,
        ctx->xDesc,    d_input,
        ctx->xDesc,    d_output,
        ctx->paramDesc,
        ctx->d_scale,
        ctx->d_bias,
        ctx->d_mean,
        ctx->d_var,
        ctx->epsilon));
}

void cudnn_batch_norm_destroy(CudnnBnCtx* ctx)
{
    cudaFree(ctx->d_scale);
    cudaFree(ctx->d_bias);
    cudaFree(ctx->d_mean);
    cudaFree(ctx->d_var);
    cudnnDestroyTensorDescriptor(ctx->xDesc);
    cudnnDestroyTensorDescriptor(ctx->paramDesc);
    free(ctx);
}

// -------------------------
// FP16 BatchNorm (fixed)
// -------------------------

CudnnBnCtxFp16* cudnn_batch_norm_create_fp16(
    cudnnHandle_t handle,
    int N, int C, int H, int W,
    double        epsilon,
    const float*  host_mean,
    const float*  host_var,
    float         host_gamma,
    float         host_beta)
{
    CudnnBnCtxFp16* ctx = (CudnnBnCtxFp16*)malloc(sizeof(*ctx));
    if (!ctx) { fprintf(stderr,"OOM @ cudnn_batch_norm_create_fp16\n"); exit(EXIT_FAILURE); }
    ctx->epsilon = epsilon;

    // Create activation descriptor  (half)
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&ctx->xDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        ctx->xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, N, C, H, W));

    // Create parameter descriptor (float!)
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&ctx->paramDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        ctx->paramDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C, 1, 1));

    size_t paramSize = C * sizeof(float);
    // Allocate float parameter buffers
    CHECK_CUDA_ERR(cudaMalloc(&ctx->d_scale, paramSize));
    CHECK_CUDA_ERR(cudaMalloc(&ctx->d_bias,  paramSize));
    CHECK_CUDA_ERR(cudaMalloc(&ctx->d_mean,  paramSize));
    CHECK_CUDA_ERR(cudaMalloc(&ctx->d_var,   paramSize));

    // Fill and copy scale/bias on host (float)
    float* tmp = (float*)malloc(paramSize);
    for (int i = 0; i < C; ++i) tmp[i] = host_gamma;
    CHECK_CUDA_ERR(cudaMemcpy(ctx->d_scale, tmp, paramSize, cudaMemcpyHostToDevice));
    for (int i = 0; i < C; ++i) tmp[i] = host_beta;
    CHECK_CUDA_ERR(cudaMemcpy(ctx->d_bias,  tmp, paramSize, cudaMemcpyHostToDevice));
    free(tmp);

    // Copy running mean/var (float)
    CHECK_CUDA_ERR(cudaMemcpy(ctx->d_mean, host_mean, paramSize, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(ctx->d_var,  host_var,  paramSize, cudaMemcpyHostToDevice));

    return ctx;
}

void cudnn_batch_norm_forward_fp16(
    cudnnHandle_t   handle,
    CudnnBnCtxFp16* ctx,
    const __half*   d_input_fp16,
    __half*         d_output_fp16)
{
    // Use float alpha/beta even though activations are half
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnBatchNormalizationForwardInference(
        handle,
        CUDNN_BATCHNORM_SPATIAL,
        &alpha, &beta,
        ctx->xDesc,        d_input_fp16,    // half activations
        ctx->xDesc,        d_output_fp16,   // half outputs
        ctx->paramDesc,    // float descriptor
        ctx->d_scale,      // float scale
        ctx->d_bias,       // float bias
        ctx->d_mean,       // float running mean
        ctx->d_var,        // float running var
        ctx->epsilon));
}

void cudnn_batch_norm_destroy_fp16(CudnnBnCtxFp16* ctx)
{
    cudaFree(ctx->d_scale);
    cudaFree(ctx->d_bias);
    cudaFree(ctx->d_mean);
    cudaFree(ctx->d_var);
    cudnnDestroyTensorDescriptor(ctx->xDesc);
    cudnnDestroyTensorDescriptor(ctx->paramDesc);
    free(ctx);
}

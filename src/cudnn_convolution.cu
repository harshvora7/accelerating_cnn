// File: src/cudnn_convolution.cu

#include "cudnn_convolution.h"
#include "error_check.h"
#include <cudnn.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

// cuDNN error‐checking macro
#define CHECK_CUDNN(call)                                                    \
    do {                                                                     \
        cudnnStatus_t status = (call);                                       \
        if (status != CUDNN_STATUS_SUCCESS) {                                \
            fprintf(stderr,                                                  \
                "cuDNN error at %s:%d: %s\n",                                \
                __FILE__, __LINE__, cudnnGetErrorString(status));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// --------------------------------------------------
// FP32 API (existing)
// --------------------------------------------------

cudnnHandle_t cudnn_init() {
    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));
    return handle;
}

void cudnn_destroy(cudnnHandle_t handle) {
    CHECK_CUDNN(cudnnDestroy(handle));
}

void cudnn_convolution_setup(
    cudnnHandle_t handle,
    const int inputDims[4],
    const int filterDims[4],
    const int outputDims[4],
    int padH, int padW,
    int strideH, int strideW,
    cudnnTensorDescriptor_t* inDesc,
    cudnnFilterDescriptor_t* filtDesc,
    cudnnConvolutionDescriptor_t* convDesc,
    cudnnTensorDescriptor_t* outDesc,
    cudnnConvolutionFwdAlgo_t* algo,
    void** workspace,
    size_t* workspaceSize)
{
    // Create descriptors
    CHECK_CUDNN(cudnnCreateTensorDescriptor(inDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(outDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(filtDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(convDesc));

    // Set to FP32
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        *inDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        inputDims[0], inputDims[1], inputDims[2], inputDims[3]));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        *outDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        outputDims[0], outputDims[1], outputDims[2], outputDims[3]));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        *filtDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        filterDims[0], filterDims[1], filterDims[2], filterDims[3]));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        *convDesc, padH, padW, strideH, strideW,
        /*dilationH=*/1, /*dilationW=*/1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

    // Pick best algorithm
    const int maxAlgos = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    int returnedAlgoCount = 0;
    cudnnConvolutionFwdAlgoPerf_t perfResults[maxAlgos];
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        handle, *inDesc, *filtDesc, *convDesc, *outDesc,
        maxAlgos, &returnedAlgoCount, perfResults));
    *algo = perfResults[0].algo;

    // Allocate workspace
    *workspaceSize = perfResults[0].memory;
    if (*workspaceSize > 0) {
        CHECK_CUDA_ERR(cudaMalloc(workspace, *workspaceSize));
    }
}

void cudnn_convolution_forward(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t inDesc,
    cudnnFilterDescriptor_t filtDesc,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnTensorDescriptor_t outDesc,
    cudnnConvolutionFwdAlgo_t algo,
    void* workspace,
    size_t workspaceSize,
    const float* input,
    const float* filter,
    float* output,
    float alpha,
    float beta)
{
    CHECK_CUDNN(cudnnConvolutionForward(
        handle,
        &alpha, inDesc, input,
                filtDesc, filter,
                convDesc, algo, workspace, workspaceSize,
        &beta,  outDesc, output));
}

// --------------------------------------------------
// FP16 API (new)
// --------------------------------------------------

void cudnn_convolution_setup_fp16(
    cudnnHandle_t handle,
    const int inputDims[4],
    const int filterDims[4],
    const int outputDims[4],
    int padH, int padW,
    int strideH, int strideW,
    cudnnTensorDescriptor_t* inDesc,
    cudnnFilterDescriptor_t* filtDesc,
    cudnnConvolutionDescriptor_t* convDesc,
    cudnnTensorDescriptor_t* outDesc,
    cudnnConvolutionFwdAlgo_t* algo,
    void** workspace,
    size_t* workspaceSize)
{
    // Create descriptors
    CHECK_CUDNN(cudnnCreateTensorDescriptor(inDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(outDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(filtDesc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(convDesc));

    // Set to FP16
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        *inDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
        inputDims[0], inputDims[1], inputDims[2], inputDims[3]));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(
        *outDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF,
        outputDims[0], outputDims[1], outputDims[2], outputDims[3]));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(
        *filtDesc, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW,
        filterDims[0], filterDims[1], filterDims[2], filterDims[3]));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(
        *convDesc, padH, padW, strideH, strideW,
        /*dilationH=*/1, /*dilationW=*/1,
        CUDNN_CROSS_CORRELATION, CUDNN_DATA_HALF));

    // Pick best algorithm
    const int maxAlgos = CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    int returnedAlgoCount = 0;
    cudnnConvolutionFwdAlgoPerf_t perfResults[maxAlgos];
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(
        handle, *inDesc, *filtDesc, *convDesc, *outDesc,
        maxAlgos, &returnedAlgoCount, perfResults));
    *algo = perfResults[0].algo;

    // Allocate workspace
    *workspaceSize = perfResults[0].memory;
    if (*workspaceSize > 0) {
        CHECK_CUDA_ERR(cudaMalloc(workspace, *workspaceSize));
    }
}

void cudnn_convolution_forward_fp16(
    cudnnHandle_t handle,
    cudnnTensorDescriptor_t inDesc,
    cudnnFilterDescriptor_t filtDesc,
    cudnnConvolutionDescriptor_t convDesc,
    cudnnTensorDescriptor_t outDesc,
    cudnnConvolutionFwdAlgo_t algo,
    void* workspace,
    size_t workspaceSize,
    const __half* input,
    const __half* filter,
    __half* output,
    float alpha_f,
    float beta_f)
{
    // convert float scalars to half
    __half alpha = __float2half(alpha_f);
    __half beta  = __float2half(beta_f);

    CHECK_CUDNN(cudnnConvolutionForward(
        handle,
        &alpha, inDesc, input,
                filtDesc, filter,
                convDesc, algo, workspace, workspaceSize,
        &beta,  outDesc, output));
}

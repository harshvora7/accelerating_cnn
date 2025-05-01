#ifndef CUDNN_CONVOLUTION_H
#define CUDNN_CONVOLUTION_H

#include <cuda_runtime.h>
#include <cudnn.h>
#include <cuda_fp16.h>    // for __half

#ifdef __cplusplus
extern "C" {
#endif

// ----------------------
// FP32 API (existing)
// ----------------------

// Initialize cuDNN context (call once at program start)
cudnnHandle_t cudnn_init();

// Clean up cuDNN context (call at program end)
void cudnn_destroy(cudnnHandle_t handle);

// One‐time setup for an FP32 convolution shape:
//   allocates/configures descriptors, picks algorithm, allocates workspace.
void cudnn_convolution_setup(
    cudnnHandle_t handle,
    const int inputDims[4],        // {N, C, H, W}
    const int filterDims[4],       // {K, C, R, S}
    const int outputDims[4],       // {N, K, H_out, W_out}
    int padH,
    int padW,
    int strideH,
    int strideW,
    cudnnTensorDescriptor_t* inDesc,
    cudnnFilterDescriptor_t* filtDesc,
    cudnnConvolutionDescriptor_t* convDesc,
    cudnnTensorDescriptor_t* outDesc,
    cudnnConvolutionFwdAlgo_t* algo,
    void** workspace,
    size_t* workspaceSize);

// Timed FP32 forward pass (after setup)
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
    float beta);

// ----------------------
// FP16 API (new)
// ----------------------

// One‐time setup for an FP16 convolution shape:
//   allocates/configures half‐precision descriptors, picks algorithm, allocates workspace.
void cudnn_convolution_setup_fp16(
    cudnnHandle_t handle,
    const int inputDims[4],        // {N, C, H, W}
    const int filterDims[4],       // {K, C, R, S}
    const int outputDims[4],       // {N, K, H_out, W_out}
    int padH,
    int padW,
    int strideH,
    int strideW,
    cudnnTensorDescriptor_t* inDesc,
    cudnnFilterDescriptor_t* filtDesc,
    cudnnConvolutionDescriptor_t* convDesc,
    cudnnTensorDescriptor_t* outDesc,
    cudnnConvolutionFwdAlgo_t* algo,
    void** workspace,
    size_t* workspaceSize);

// Timed FP16 forward pass (after fp16 setup)
//   Uses __half pointers for input, filter, output
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
    float alpha,   // passed as float, converted internally
    float beta);

#ifdef __cplusplus
}
#endif

#endif // CUDNN_CONVOLUTION_H

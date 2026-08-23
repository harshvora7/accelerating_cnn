#ifndef CUDNN_BATCH_NORM_H
#define CUDNN_BATCH_NORM_H

#include <cudnn.h>
#include <cuda_fp16.h>   // for __half

#ifdef __cplusplus
extern "C" {
#endif

// -------------------------------
// FP32 BatchNorm API (existing)
// -------------------------------

// Opaque context for FP32 cuDNN Batch Normalization inference
typedef struct {
    cudnnTensorDescriptor_t xDesc;       // descriptor for input/output
    cudnnTensorDescriptor_t paramDesc;   // descriptor for scale/bias/mean/var
    float*                 d_scale;      // device scale (gamma)
    float*                 d_bias;       // device bias  (beta)
    float*                 d_mean;       // device estimated mean
    float*                 d_var;        // device estimated variance
    double                 epsilon;      // epsilon for numerical stability
} CudnnBnCtx;

// One‐time setup (allocates descriptors + scale/bias/mean/var buffers).
// host_mean & host_var are length-C arrays, host_gamma/beta can be scalars or arrays.
CudnnBnCtx* cudnn_batch_norm_create(
    cudnnHandle_t handle,
    int N, int C, int H, int W,
    double     epsilon,
    const float* host_mean,
    const float* host_var,
    float        host_gamma,
    float        host_beta);

// Timed forward‐inference pass (FP32). Reads from d_input, writes to d_output.
void cudnn_batch_norm_forward(
    cudnnHandle_t handle,
    CudnnBnCtx*   ctx,
    const float*  d_input,
    float*        d_output);

// Cleanup descriptors and device allocations (FP32).
void cudnn_batch_norm_destroy(CudnnBnCtx* ctx);

// -------------------------------
// FP16 BatchNorm API (new)
// -------------------------------

// Opaque context for FP16 cuDNN Batch Normalization inference
typedef struct {
    cudnnTensorDescriptor_t xDesc;       // descriptor for input/output
    cudnnTensorDescriptor_t paramDesc;   // descriptor for scale/bias/mean/var
    __half*                d_scale;      // device scale (gamma) in half
    __half*                d_bias;       // device bias  (beta)  in half
    __half*                d_mean;       // device estimated mean in half
    __half*                d_var;        // device estimated variance in half
    double                 epsilon;      // same epsilon
} CudnnBnCtxFp16;

// One‐time setup (allocates descriptors + half‐precision buffers).
// host_mean & host_var are float arrays (length C), host_gamma/beta are floats.
CudnnBnCtxFp16* cudnn_batch_norm_create_fp16(
    cudnnHandle_t handle,
    int N, int C, int H, int W,
    double        epsilon,
    const float*  host_mean,
    const float*  host_var,
    float         host_gamma,
    float         host_beta);

// Timed forward‐inference pass (FP16). Reads from d_input_fp16, writes to d_output_fp16.
void cudnn_batch_norm_forward_fp16(
    cudnnHandle_t    handle,
    CudnnBnCtxFp16*  ctx,
    const __half*    d_input_fp16,
    __half*          d_output_fp16);

// Cleanup descriptors and device allocations (FP16).
void cudnn_batch_norm_destroy_fp16(CudnnBnCtxFp16* ctx);

#ifdef __cplusplus
}
#endif

#endif // CUDNN_BATCH_NORM_H

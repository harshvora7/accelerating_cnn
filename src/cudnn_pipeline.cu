// File: src/cudnn_pipeline.cu

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cuda_fp16.h>

#include "error_check.h"           // CHECK_CUDA_ERR
#include "convolution.h"           // cpu_convolution()
#include "batch_norm.h"            // cpu_batch_norm()
#include "pooling.h"               // cpu_max_pooling()
#include "cudnn_convolution.h"     // cudnn_init(), setup(), forward(), destroy()
#include "cudnn_batch_norm.h"      // cudnn_batch_norm_create(), forward(), destroy()
#include "cudnn_relu.h"            // cudnn_relu_create(), forward(), destroy()
#include "cudnn_pooling.h"         // cudnn_pooling_create(), forward(), destroy()

int main() {
    // --- Layer dims ---
    const int N = 1, C = 1;
    const int H = 512, W = 512;
    const int R = 3, S = 3;              // 3×3 kernel
    const int H_out = H - R + 1, W_out = W - S + 1;
    const int poolSize = 2, poolStride = 2;
    const int H_pool = (H_out - poolSize) / poolStride + 1;
    const int W_pool = (W_out - poolSize) / poolStride + 1;

    // BN params
    const double epsilon = 1e-5;
    const float gamma = 1.0f, beta = 0.0f;

    // --- Host buffers (FP32) ---
    size_t inBytes     = N*C*H*W          * sizeof(float);
    size_t outBytes    = N*C*H_out*W_out  * sizeof(float);
    size_t kernBytes   = C*R*S            * sizeof(float);
    size_t poolBytes   = N*C*H_pool*W_pool * sizeof(float);

    float *h_in   = (float*)malloc(inBytes);
    float *h_kern = (float*)malloc(kernBytes);
    float *h_conv = (float*)malloc(outBytes);
    float *h_bn   = (float*)malloc(outBytes);
    float *h_relu = (float*)malloc(outBytes);
    float *h_pool = (float*)malloc(poolBytes);

    // --- Host buffers (FP16) ---
    __half *h_in16   = (__half*)malloc(N*C*H*W          * sizeof(__half));
    __half *h_kern16 = (__half*)malloc(C*R*S            * sizeof(__half));

    // Initialize FP32 input + kernel, then convert to FP16
    for(int i = 0; i < N*C*H*W; ++i) {
        h_in[i] = (float)(rand() % 10);
        h_in16[i] = __float2half(h_in[i]);
    }
    float ek[9] = { 1,0,-1, 1,0,-1, 1,0,-1 };
    for(int i = 0; i < C*R*S; ++i) {
        h_kern[i] = ek[i];
        h_kern16[i] = __float2half(ek[i]);
    }

    // --- CPU reference: Conv → BN → ReLU → Pool (FP32) ---
    cpu_convolution(h_in, h_kern, h_conv, W, H, S, R, W_out, H_out);
    int M = N*C*H_out*W_out;
    double sum=0, vsum=0;
    for(int i=0;i<M;i++) sum += h_conv[i];
    double mean = sum / M;
    for(int i=0;i<M;i++){
        double d = h_conv[i] - mean; vsum += d*d;
    }
    double var = vsum / M;
    for(int i=0;i<M;i++){
        h_bn[i] = gamma * ((h_conv[i]-mean)/sqrt(var+epsilon)) + beta;
        h_relu[i] = h_bn[i] > 0.0f ? h_bn[i] : 0.0f;
    }
    cpu_max_pooling(h_relu, h_pool, W_out, H_out, poolSize, poolStride);

    // --- Device buffers (FP32) ---
    float  *d_in, *d_kern, *d_conv, *d_bn, *d_relu, *d_pool;
    CHECK_CUDA_ERR(cudaMalloc(&d_in,   inBytes));
    CHECK_CUDA_ERR(cudaMalloc(&d_kern, kernBytes));
    CHECK_CUDA_ERR(cudaMalloc(&d_conv, outBytes));
    CHECK_CUDA_ERR(cudaMalloc(&d_bn,   outBytes));
    CHECK_CUDA_ERR(cudaMalloc(&d_relu, outBytes));
    CHECK_CUDA_ERR(cudaMalloc(&d_pool, poolBytes));
    CHECK_CUDA_ERR(cudaMemcpy(d_in,   h_in,   inBytes,   cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_kern, h_kern, kernBytes, cudaMemcpyHostToDevice));

    // --- Device buffers (FP16) ---
    __half *d_in16, *d_kern16, *d_conv16, *d_bn16, *d_relu16, *d_pool16;
    CHECK_CUDA_ERR(cudaMalloc(&d_in16,   N*C*H*W          * sizeof(__half)));
    CHECK_CUDA_ERR(cudaMalloc(&d_kern16, C*R*S            * sizeof(__half)));
    CHECK_CUDA_ERR(cudaMalloc(&d_conv16, N*C*H_out*W_out  * sizeof(__half)));
    CHECK_CUDA_ERR(cudaMalloc(&d_bn16,   N*C*H_out*W_out  * sizeof(__half)));
    CHECK_CUDA_ERR(cudaMalloc(&d_relu16, N*C*H_out*W_out  * sizeof(__half)));
    CHECK_CUDA_ERR(cudaMalloc(&d_pool16, N*C*H_pool*W_pool * sizeof(__half)));
    CHECK_CUDA_ERR(cudaMemcpy(d_in16,   h_in16,   N*C*H*W          * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_kern16, h_kern16, C*R*S            * sizeof(__half), cudaMemcpyHostToDevice));

    // --- cuDNN setup ---
    cudnnHandle_t cudnn = cudnn_init();

    // (1) Convolution setup FP32 + FP16
    int inDims[4]   = { N, C, H,    W    };
    int filtDims[4] = { C, C, R,    S    };
    int outDims[4]  = { N, C, H_out, W_out };
    cudnnTensorDescriptor_t      inDesc_f32, outDesc_f32;
    cudnnFilterDescriptor_t      filtDesc_f32;
    cudnnConvolutionDescriptor_t convDesc_f32;
    cudnnConvolutionFwdAlgo_t    algo_f32;
    void* workspace_f32;  size_t wsSize_f32;

    cudnn_convolution_setup(
        cudnn, inDims, filtDims, outDims,
        0,0,1,1,
        &inDesc_f32, &filtDesc_f32, &convDesc_f32, &outDesc_f32,
        &algo_f32, &workspace_f32, &wsSize_f32
    );

    // for FP16, identical dims but half data type
    cudnnTensorDescriptor_t      inDesc_f16, outDesc_f16;
    cudnnFilterDescriptor_t      filtDesc_f16;
    cudnnConvolutionDescriptor_t convDesc_f16;
    cudnnConvolutionFwdAlgo_t    algo_f16;
    void* workspace_f16;  size_t wsSize_f16;

    cudnn_convolution_setup_fp16(
        cudnn, inDims, filtDims, outDims,
        0,0,1,1,
        &inDesc_f16, &filtDesc_f16, &convDesc_f16, &outDesc_f16,
        &algo_f16, &workspace_f16, &wsSize_f16
    );

    // (2) BatchNorm setup FP32 + FP16
    float host_mean_f = (float)mean, host_var_f = (float)var;
    CudnnBnCtx*    bnCtx32 = cudnn_batch_norm_create(
        cudnn, N, C, H_out, W_out,
        epsilon, &host_mean_f, &host_var_f,
        gamma, beta
    );
    CudnnBnCtxFp16* bnCtx16 = cudnn_batch_norm_create_fp16(
        cudnn, N, C, H_out, W_out,
        epsilon, &host_mean_f, &host_var_f,
        gamma, beta
    );

    // (3) ReLU setup FP32 + FP16
    CudnnReluCtx* reluCtx32 = cudnn_relu_create(N, C, H_out, W_out);
    CudnnReluCtxFp16* reluCtx16 = cudnn_relu_create_fp16(N, C, H_out, W_out);

    // (4) Pooling setup FP32 + FP16
    CudnnPoolCtx* poolCtx32 = cudnn_pooling_create(
        N, C, H_out, W_out,
        poolSize, poolSize,
        0, 0,
        poolStride, poolStride
    );
    CudnnPoolCtxFp16* poolCtx16 = cudnn_pooling_create_fp16(
        N, C, H_out, W_out,
        poolSize, poolSize,
        0, 0,
        poolStride, poolStride
    );

    // --- Timing events ---
    cudaEvent_t s0,e0, s0_16,e0_16;
    cudaEvent_t s1,e1, s1_16,e1_16;
    cudaEvent_t s2,e2, s2_16,e2_16;
    cudaEvent_t s3,e3, s3_16,e3_16;
    CHECK_CUDA_ERR(cudaEventCreate(&s0));     CHECK_CUDA_ERR(cudaEventCreate(&e0));
    CHECK_CUDA_ERR(cudaEventCreate(&s0_16));  CHECK_CUDA_ERR(cudaEventCreate(&e0_16));
    CHECK_CUDA_ERR(cudaEventCreate(&s1));     CHECK_CUDA_ERR(cudaEventCreate(&e1));
    CHECK_CUDA_ERR(cudaEventCreate(&s1_16));  CHECK_CUDA_ERR(cudaEventCreate(&e1_16));
    CHECK_CUDA_ERR(cudaEventCreate(&s2));     CHECK_CUDA_ERR(cudaEventCreate(&e2));
    CHECK_CUDA_ERR(cudaEventCreate(&s2_16));  CHECK_CUDA_ERR(cudaEventCreate(&e2_16));
    CHECK_CUDA_ERR(cudaEventCreate(&s3));     CHECK_CUDA_ERR(cudaEventCreate(&e3));
    CHECK_CUDA_ERR(cudaEventCreate(&s3_16));  CHECK_CUDA_ERR(cudaEventCreate(&e3_16));

    // Warm-up
    for(int i=0; i<5; ++i) {
        cudnn_convolution_forward     (cudnn, inDesc_f32, filtDesc_f32, convDesc_f32, outDesc_f32, algo_f32, workspace_f32, wsSize_f32, d_in,   d_kern, d_conv, 1.0f,0.0f);
        cudnn_convolution_forward_fp16(cudnn, inDesc_f16, filtDesc_f16, convDesc_f16, outDesc_f16, algo_f16, workspace_f16, wsSize_f16, d_in16, d_kern16, d_conv16, __float2half(1.0f), __float2half(0.0f));
        cudnn_batch_norm_forward      (cudnn, bnCtx32, d_conv,  d_bn);
        cudnn_batch_norm_forward_fp16 (cudnn, bnCtx16, d_conv16, d_bn16);
        cudnn_relu_forward            (cudnn, reluCtx32, d_bn,   d_relu);
        cudnn_relu_forward_fp16       (cudnn, reluCtx16, d_bn16, d_relu16);
        cudnn_pooling_forward         (cudnn, poolCtx32, d_relu, d_pool);
        cudnn_pooling_forward_fp16    (cudnn, poolCtx16, d_relu16, d_pool16);
    }

    const int ITERS = 50;
    float total_conv32=0, total_conv16=0;
    float total_bn32=0,   total_bn16=0;
    float total_relu32=0, total_relu16=0;
    float total_pool32=0, total_pool16=0;

    // Timed loop
    for(int i = 0; i < ITERS; ++i) {
      // FP32 conv
      CHECK_CUDA_ERR(cudaEventRecord(s0,0));
      cudnn_convolution_forward(cudnn, inDesc_f32, filtDesc_f32, convDesc_f32, outDesc_f32,
                                algo_f32, workspace_f32, wsSize_f32,
                                d_in, d_kern, d_conv, 1.0f, 0.0f);
      CHECK_CUDA_ERR(cudaEventRecord(e0,0));
      CHECK_CUDA_ERR(cudaEventSynchronize(e0));
      { float t; CHECK_CUDA_ERR(cudaEventElapsedTime(&t,s0,e0)); total_conv32 += t; }

      // FP16 conv
      CHECK_CUDA_ERR(cudaEventRecord(s0_16,0));
      cudnn_convolution_forward_fp16(cudnn, inDesc_f16, filtDesc_f16, convDesc_f16, outDesc_f16,
                                     algo_f16, workspace_f16, wsSize_f16,
                                     d_in16, d_kern16, d_conv16,
                                     __float2half(1.0f), __float2half(0.0f));
      CHECK_CUDA_ERR(cudaEventRecord(e0_16,0));
      CHECK_CUDA_ERR(cudaEventSynchronize(e0_16));
      { float t; CHECK_CUDA_ERR(cudaEventElapsedTime(&t,s0_16,e0_16)); total_conv16 += t; }

      // FP32 BN
      CHECK_CUDA_ERR(cudaEventRecord(s1,0));
      cudnn_batch_norm_forward(cudnn, bnCtx32, d_conv, d_bn);
      CHECK_CUDA_ERR(cudaEventRecord(e1,0));
      CHECK_CUDA_ERR(cudaEventSynchronize(e1));
      { float t; CHECK_CUDA_ERR(cudaEventElapsedTime(&t,s1,e1)); total_bn32 += t; }

      // FP16 BN
      CHECK_CUDA_ERR(cudaEventRecord(s1_16,0));
      cudnn_batch_norm_forward_fp16(cudnn, bnCtx16, d_conv16, d_bn16);
      CHECK_CUDA_ERR(cudaEventRecord(e1_16,0));
      CHECK_CUDA_ERR(cudaEventSynchronize(e1_16));
      { float t; CHECK_CUDA_ERR(cudaEventElapsedTime(&t,s1_16,e1_16)); total_bn16 += t; }

      // FP32 ReLU
      CHECK_CUDA_ERR(cudaEventRecord(s2,0));
      cudnn_relu_forward(cudnn, reluCtx32, d_bn, d_relu);
      CHECK_CUDA_ERR(cudaEventRecord(e2,0));
      CHECK_CUDA_ERR(cudaEventSynchronize(e2));
      { float t; CHECK_CUDA_ERR(cudaEventElapsedTime(&t,s2,e2)); total_relu32 += t; }

      // FP16 ReLU
      CHECK_CUDA_ERR(cudaEventRecord(s2_16,0));
      cudnn_relu_forward_fp16(cudnn, reluCtx16, d_bn16, d_relu16);
      CHECK_CUDA_ERR(cudaEventRecord(e2_16,0));
      CHECK_CUDA_ERR(cudaEventSynchronize(e2_16));
      { float t; CHECK_CUDA_ERR(cudaEventElapsedTime(&t,s2_16,e2_16)); total_relu16 += t; }

      // FP32 Pool
      CHECK_CUDA_ERR(cudaEventRecord(s3,0));
      cudnn_pooling_forward(cudnn, poolCtx32, d_relu, d_pool);
      CHECK_CUDA_ERR(cudaEventRecord(e3,0));
      CHECK_CUDA_ERR(cudaEventSynchronize(e3));
      { float t; CHECK_CUDA_ERR(cudaEventElapsedTime(&t,s3,e3)); total_pool32 += t; }

      // FP16 Pool
      CHECK_CUDA_ERR(cudaEventRecord(s3_16,0));
      cudnn_pooling_forward_fp16(cudnn, poolCtx16, d_relu16, d_pool16);
      CHECK_CUDA_ERR(cudaEventRecord(e3_16,0));
      CHECK_CUDA_ERR(cudaEventSynchronize(e3_16));
      { float t; CHECK_CUDA_ERR(cudaEventElapsedTime(&t,s3_16,e3_16)); total_pool16 += t; }
    }

    // Report averages
    printf("  FP32  Conv avg: %f ms\n", total_conv32/ITERS);
    printf("  FP16  Conv avg: %f ms\n", total_conv16/ITERS);
    printf("  FP32  BN   avg: %f ms\n", total_bn32  /ITERS);
    printf("  FP16  BN   avg: %f ms\n", total_bn16  /ITERS);
    printf("  FP32  ReLU avg: %f ms\n", total_relu32/ITERS);
    printf("  FP16  ReLU avg: %f ms\n", total_relu16/ITERS);
    printf("  FP32  Pool avg: %f ms\n", total_pool32/ITERS);
    printf("  FP16  Pool avg: %f ms\n", total_pool16/ITERS);

    // ---- FP16 correctness: FP16 pipeline output vs FP32 CPU reference ----
    int poolCount = N*C*H_pool*W_pool;
    __half* h_pool16 = (__half*)malloc(poolCount * sizeof(__half));
    CHECK_CUDA_ERR(cudaMemcpy(h_pool16, d_pool16, poolCount * sizeof(__half),
                              cudaMemcpyDeviceToHost));
    int   fp16_errors = 0;
    float fp16_max_abs = 0.0f, fp16_max_rel = 0.0f;
    for (int i = 0; i < poolCount; i++) {
        float ref = h_pool[i];                  // FP32 CPU reference
        float got = __half2float(h_pool16[i]);  // FP16 GPU result -> float
        float abs_diff = fabsf(ref - got);
        float rel_diff = abs_diff / (fabsf(ref) + 1e-6f);
        if (abs_diff > fp16_max_abs) fp16_max_abs = abs_diff;
        if (rel_diff > fp16_max_rel) fp16_max_rel = rel_diff;
        if (rel_diff > 2e-2f) fp16_errors++;    // FP16 ~3 decimal digits -> looser tol than FP32
    }
    printf("\n  FP16 vs FP32-CPU -> Max abs: %e | Max rel: %e\n", fp16_max_abs, fp16_max_rel);
    printf(fp16_errors == 0 ? "  FP16 Correctness: PASS (within 2e-2 relative)\n"
                            : "  FP16 Correctness: %d values exceed 2e-2 rel tol\n", fp16_errors);
    free(h_pool16);
    // Destroy pooling contexts
    cudnn_pooling_destroy(poolCtx32);
    cudnn_pooling_destroy_fp16(poolCtx16);

    // Destroy ReLU contexts
    cudnn_relu_destroy(reluCtx32);
    cudnn_relu_destroy_fp16(reluCtx16);

    // Destroy batchnorm contexts
    cudnn_batch_norm_destroy(bnCtx32);
    cudnn_batch_norm_destroy_fp16(bnCtx16);

    // Free conv workspaces & destroy conv descriptors
    cudaFree(workspace_f32);
    cudnnDestroyTensorDescriptor(inDesc_f32);
    cudnnDestroyFilterDescriptor(filtDesc_f32);
    cudnnDestroyConvolutionDescriptor(convDesc_f32);
    cudnnDestroyTensorDescriptor(outDesc_f32);

    cudaFree(workspace_f16);
    cudnnDestroyTensorDescriptor(inDesc_f16);
    cudnnDestroyFilterDescriptor(filtDesc_f16);
    cudnnDestroyConvolutionDescriptor(convDesc_f16);
    cudnnDestroyTensorDescriptor(outDesc_f16);

    // Destroy cuDNN handle
    cudnn_destroy(cudnn);

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_kern);
    cudaFree(d_conv);
    cudaFree(d_bn);
    cudaFree(d_relu);
    cudaFree(d_pool);

    cudaFree(d_in16);
    cudaFree(d_kern16);
    cudaFree(d_conv16);
    cudaFree(d_bn16);
    cudaFree(d_relu16);
    cudaFree(d_pool16);

    // Free host memory
    free(h_in);    free(h_kern);
    free(h_conv);  free(h_bn);
    free(h_relu);  free(h_pool);

    free(h_in16);   free(h_kern16);

    // Destroy all CUDA events
    cudaEventDestroy(s0);      cudaEventDestroy(e0);
    cudaEventDestroy(s0_16);   cudaEventDestroy(e0_16);
    cudaEventDestroy(s1);      cudaEventDestroy(e1);
    cudaEventDestroy(s1_16);   cudaEventDestroy(e1_16);
    cudaEventDestroy(s2);      cudaEventDestroy(e2);
    cudaEventDestroy(s2_16);   cudaEventDestroy(e2_16);
    cudaEventDestroy(s3);      cudaEventDestroy(e3);
    cudaEventDestroy(s3_16);   cudaEventDestroy(e3_16);

    return 0;
}

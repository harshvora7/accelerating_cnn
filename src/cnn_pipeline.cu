// File: src/cnn_pipeline.cu

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <cudnn.h>

#include "error_check.h"        // CHECK_CUDA_ERR
#include "convolution.h"        // naive_convolution(), cpu_convolution()
#include "batch_norm.h"         // batch_norm_kernel(), cpu_batch_norm()
#include "relu.h"               // relu_kernel(), cpu_relu()
#include "pooling.h"            // max_pooling_kernel(), cpu_max_pooling()
#include "cudnn_convolution.h"  // cudnn_init(), cudnn_convolution_forward(), etc.
#include "tiled_convolution.h"   // tiled_convolution(), set_tiled_conv_kernel()

enum ImplMode { MODE_CUSTOM, MODE_CUDNN, MODE_TILED };

ImplMode parseMode(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--mode=", 7) == 0) {
            const char* m = argv[i] + 7;
            if (strcmp(m, "custom") == 0) return MODE_CUSTOM;
            if (strcmp(m, "cudnn")  == 0) return MODE_CUDNN;
            if (strcmp(m, "tiled")  == 0) return MODE_TILED;
        }
    }
    return MODE_CUSTOM;
}

// Parse --ksize=N (odd, 3..15); default 3.
int parseKsize(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (strncmp(argv[i], "--ksize=", 8) == 0) {
            int k = atoi(argv[i] + 8);
            if (k >= 3 && k <= 31 && (k % 2 == 1)) return k;
        }
    }
    return 3;
}

int main(int argc, char** argv) {
    ImplMode mode = parseMode(argc, argv);
    int ksize = parseKsize(argc, argv);
    if (mode == MODE_CUDNN) {
        printf("Running with cuDNN convolution\n");
    } else if (mode == MODE_TILED) {
        printf("Running with custom TILED CUDA convolution\n");
    } else {
        printf("Running with custom CUDA convolution\n");
    }

    // descriptors for cuDNN
    cudnnTensorDescriptor_t    inDesc, outDesc;
    cudnnFilterDescriptor_t    filtDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t  algo;
    void*                      workspace     = nullptr;
    size_t                     workspaceSize = 0;

    // -------------------------------------
    // Define dimensions for each layer
    // -------------------------------------
    const int inputWidth  = 512;
    const int inputHeight = 512;
    const int convKernelWidth  = ksize;
    const int convKernelHeight = ksize;
    const int convOutWidth  = inputWidth  - convKernelWidth  + 1;
    const int convOutHeight = inputHeight - convKernelHeight + 1;

    const int poolSize   = 2;
    const int poolStride = 2;
    const int poolOutWidth  = convOutWidth  / poolStride;
    const int poolOutHeight = convOutHeight / poolStride;

    const float gamma   = 1.0f;
    const float beta    = 0.0f;
    const float epsilon = 1e-5f;

    // Number of elements for conv output
    int convOutElements = convOutWidth * convOutHeight;

    // -------------------------------------
    // Allocate Host Memory
    // -------------------------------------
    size_t inputSizeBytes  = inputWidth  * inputHeight * sizeof(float);
    size_t kernelSizeBytes = convKernelWidth * convKernelHeight * sizeof(float);
    size_t convOutBytes    = convOutWidth  * convOutHeight * sizeof(float);
    size_t poolOutBytes    = poolOutWidth  * poolOutHeight * sizeof(float);

    float *h_input      = (float*)malloc(inputSizeBytes);
    float *h_conv_kernel= (float*)malloc(kernelSizeBytes);

    float *h_conv_cpu   = (float*)malloc(convOutBytes);
    float *h_bn_cpu     = (float*)malloc(convOutBytes);
    float *h_relu_cpu   = (float*)malloc(convOutBytes);
    float *h_pool_cpu   = (float*)malloc(poolOutBytes);

    float *h_pool_gpu   = (float*)malloc(poolOutBytes);

    // -------------------------------------
    // Initialize Input and Kernel Data
    // -------------------------------------
    for (int i = 0; i < inputWidth*inputHeight; i++) {
        h_input[i] = (float)(rand() % 10);
    }
    // Normalized box-blur filter, sized convKernelWidth x convKernelHeight.
    // Well-defined at any kernel size; small output magnitudes keep the
    // FP32 GPU-vs-CPU validation tight across the filter-size sweep.
    {
        float w = 1.0f / (float)(convKernelWidth * convKernelHeight);
        for (int i = 0; i < convKernelWidth*convKernelHeight; i++) {
            h_conv_kernel[i] = w;
        }
    }

    // Upload filter to constant memory for the tiled kernel (MODE_TILED)
    if (mode == MODE_TILED) {
        set_tiled_conv_kernel(h_conv_kernel, convKernelWidth, convKernelHeight);
    }

    // -------------------------------------
    // Build CPU Reference Pipeline
    // -------------------------------------
    cpu_convolution(
      h_input, h_conv_kernel, h_conv_cpu,
      inputWidth, inputHeight,
      convKernelWidth, convKernelHeight,
      convOutWidth, convOutHeight);

    // compute mean & variance
    float sum = 0.0f;
    for (int i = 0; i < convOutElements; i++) sum += h_conv_cpu[i];
    float mean = sum / convOutElements;

    float var_sum = 0.0f;
    for (int i = 0; i < convOutElements; i++) {
        float d = h_conv_cpu[i] - mean;
        var_sum += d*d;
    }
    float variance = var_sum / convOutElements;

    // CPU BatchNorm
    for (int i = 0; i < convOutElements; i++) {
        h_bn_cpu[i] = gamma * ((h_conv_cpu[i] - mean) / sqrt(variance + epsilon)) + beta;
    }
    // CPU ReLU
    cpu_relu(h_bn_cpu, h_relu_cpu, convOutElements);
    // CPU Pooling
    cpu_max_pooling(
      h_relu_cpu, h_pool_cpu,
      convOutWidth, convOutHeight,
      poolSize, poolStride);

    // -------------------------------------
    // Allocate Device Memory
    // -------------------------------------
    float *d_input, *d_conv_kernel;
    float *d_conv, *d_bn, *d_relu, *d_pool;
    CHECK_CUDA_ERR(cudaMalloc(&d_input,       inputSizeBytes));
    CHECK_CUDA_ERR(cudaMalloc(&d_conv_kernel, kernelSizeBytes));
    CHECK_CUDA_ERR(cudaMalloc(&d_conv,        convOutBytes));
    CHECK_CUDA_ERR(cudaMalloc(&d_bn,          convOutBytes));
    CHECK_CUDA_ERR(cudaMalloc(&d_relu,        convOutBytes));
    CHECK_CUDA_ERR(cudaMalloc(&d_pool,        poolOutBytes));

    CHECK_CUDA_ERR(cudaMemcpy(d_input,       h_input,       inputSizeBytes,  cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_conv_kernel, h_conv_kernel, kernelSizeBytes, cudaMemcpyHostToDevice));

    // -------------------------------------
    // cuDNN Setup (only for convolution)
    // -------------------------------------
    cudnnHandle_t cudnnHandle = nullptr;
    if (mode == MODE_CUDNN) {
        cudnnHandle = cudnn_init();
        int inDims[4]   = {1,1,inputHeight, inputWidth};
        int filtDims[4] = {1,1,convKernelHeight,convKernelWidth};
        int outDims[4]  = {1,1,convOutHeight,  convOutWidth};

        cudnn_convolution_setup(
          cudnnHandle,
          inDims, filtDims, outDims,
          0,0, 1,1,
          &inDesc, &filtDesc, &convDesc, &outDesc,
          &algo, &workspace, &workspaceSize
        );
    }

    // -------------------------------------
    // Create CUDA events for timing
    // -------------------------------------
    cudaEvent_t conv_start,  conv_stop;
    cudaEvent_t bn_start,    bn_stop;
    cudaEvent_t relu_start,  relu_stop;
    cudaEvent_t pool_start,  pool_stop;
    CHECK_CUDA_ERR(cudaEventCreate(&conv_start));
    CHECK_CUDA_ERR(cudaEventCreate(&conv_stop));
    CHECK_CUDA_ERR(cudaEventCreate(&bn_start));
    CHECK_CUDA_ERR(cudaEventCreate(&bn_stop));
    CHECK_CUDA_ERR(cudaEventCreate(&relu_start));
    CHECK_CUDA_ERR(cudaEventCreate(&relu_stop));
    CHECK_CUDA_ERR(cudaEventCreate(&pool_start));
    CHECK_CUDA_ERR(cudaEventCreate(&pool_stop));

    // -------------------------------------
    // Warm‐up & Timed iterations
    // -------------------------------------
    const int WARMUPS = 5;
    const int ITERS   = 50;
    float totalConv = 0.0f, totalBN = 0.0f, totalReLU = 0.0f, totalPool = 0.0f;

    // Warm‐up (uncaptured)
    for (int i = 0; i < WARMUPS; i++) {
        if (mode == MODE_CUSTOM) {
            dim3 b2d(16,16), g2d((convOutWidth+15)/16,(convOutHeight+15)/16);
            naive_convolution<<<g2d,b2d>>>(d_input, d_conv_kernel, d_conv,
                                           inputWidth, inputHeight,
                                           convKernelWidth, convKernelHeight,
                                           convOutWidth, convOutHeight);
            CHECK_CUDA_ERR(cudaDeviceSynchronize());
        } else if (mode == MODE_TILED) {
            dim3 b2d(TILE_DIM,TILE_DIM), g2d((convOutWidth+TILE_DIM-1)/TILE_DIM,(convOutHeight+TILE_DIM-1)/TILE_DIM);
            tiled_convolution<<<g2d,b2d>>>(d_input, d_conv,
                                           inputWidth, inputHeight,
                                           convKernelWidth, convKernelHeight,
                                           convOutWidth, convOutHeight);
            CHECK_CUDA_ERR(cudaDeviceSynchronize());
        } else {
            cudnn_convolution_forward(
              cudnnHandle,
              inDesc,filtDesc,convDesc,outDesc,
              algo, workspace, workspaceSize,
              d_input, d_conv_kernel, d_conv,
              1.0f, 0.0f
            );
        }
        // BN, ReLU, Pool (always custom)
        batch_norm_kernel<<<(convOutElements+255)/256,256>>>(d_conv, d_bn,
                                                    convOutElements,
                                                    mean, variance, gamma, beta, epsilon);
        CHECK_CUDA_ERR(cudaDeviceSynchronize());
        relu_kernel<<<(convOutElements+255)/256,256>>>(d_bn, d_relu, convOutElements);
        CHECK_CUDA_ERR(cudaDeviceSynchronize());
        {
          dim3 b2d(16,16), g2d((convOutWidth+15)/16,(convOutHeight+15)/16);
          max_pooling_kernel<<<g2d,b2d>>>(d_relu, d_pool,
                                          convOutWidth, convOutHeight,
                                          poolSize, poolStride);
          CHECK_CUDA_ERR(cudaDeviceSynchronize());
        }
    }

    // Timed loop
    for (int i = 0; i < ITERS; i++) {
        // (1) Convolution
        CHECK_CUDA_ERR(cudaEventRecord(conv_start,0));
        if (mode == MODE_CUSTOM) {
            dim3 b2d(16,16), g2d((convOutWidth+15)/16,(convOutHeight+15)/16);
            naive_convolution<<<g2d,b2d>>>(d_input, d_conv_kernel, d_conv,
                                           inputWidth, inputHeight,
                                           convKernelWidth, convKernelHeight,
                                           convOutWidth, convOutHeight);
        } else if (mode == MODE_TILED) {
            dim3 b2d(TILE_DIM,TILE_DIM), g2d((convOutWidth+TILE_DIM-1)/TILE_DIM,(convOutHeight+TILE_DIM-1)/TILE_DIM);
            tiled_convolution<<<g2d,b2d>>>(d_input, d_conv,
                                           inputWidth, inputHeight,
                                           convKernelWidth, convKernelHeight,
                                           convOutWidth, convOutHeight);
        } else {
            cudnn_convolution_forward(
              cudnnHandle,
              inDesc,filtDesc,convDesc,outDesc,
              algo, workspace, workspaceSize,
              d_input, d_conv_kernel, d_conv,
              1.0f, 0.0f
            );
        }
        CHECK_CUDA_ERR(cudaEventRecord(conv_stop,0));
        CHECK_CUDA_ERR(cudaEventSynchronize(conv_stop));
        { float t; CHECK_CUDA_ERR(cudaEventElapsedTime(&t,conv_start,conv_stop)); totalConv += t; }

        // (2) Batch Normalization
        CHECK_CUDA_ERR(cudaEventRecord(bn_start,0));
        batch_norm_kernel<<<(convOutElements+255)/256,256>>>(d_conv, d_bn,
                                                    convOutElements,
                                                    mean, variance, gamma, beta, epsilon);
        CHECK_CUDA_ERR(cudaEventRecord(bn_stop,0));
        CHECK_CUDA_ERR(cudaEventSynchronize(bn_stop));
        { float t; CHECK_CUDA_ERR(cudaEventElapsedTime(&t,bn_start,bn_stop)); totalBN += t; }

        // (3) ReLU
        CHECK_CUDA_ERR(cudaEventRecord(relu_start,0));
        relu_kernel<<<(convOutElements+255)/256,256>>>(d_bn, d_relu, convOutElements);
        CHECK_CUDA_ERR(cudaEventRecord(relu_stop,0));
        CHECK_CUDA_ERR(cudaEventSynchronize(relu_stop));
        { float t; CHECK_CUDA_ERR(cudaEventElapsedTime(&t,relu_start,relu_stop)); totalReLU += t; }

        // (4) Pooling
        CHECK_CUDA_ERR(cudaEventRecord(pool_start,0));
        {
          dim3 b2d(16,16), g2d((convOutWidth+15)/16,(convOutHeight+15)/16);
          max_pooling_kernel<<<g2d,b2d>>>(d_relu, d_pool,
                                          convOutWidth, convOutHeight,
                                          poolSize, poolStride);
        }
        CHECK_CUDA_ERR(cudaEventRecord(pool_stop,0));
        CHECK_CUDA_ERR(cudaEventSynchronize(pool_stop));
        { float t; CHECK_CUDA_ERR(cudaEventElapsedTime(&t,pool_start,pool_stop)); totalPool += t; }
    }

    // -------------------------------------
    // Print averaged times
    // -------------------------------------
    printf("Avg Convolution time    : %f ms\n", totalConv / ITERS);
    printf("Avg BatchNorm time      : %f ms\n", totalBN   / ITERS);
    printf("Avg ReLU time           : %f ms\n", totalReLU / ITERS);
    printf("Avg Pooling time        : %f ms\n", totalPool / ITERS);

    // -------------------------------------
    // Validate GPU pipeline result against CPU reference
    // -------------------------------------
    CHECK_CUDA_ERR(cudaMemcpy(h_pool_gpu, d_pool, poolOutBytes, cudaMemcpyDeviceToHost));
    int errors = 0;
    float max_abs = 0.0f, max_rel = 0.0f;
    for (int i = 0; i < poolOutWidth*poolOutHeight; i++) {
        float a = h_pool_cpu[i], b = h_pool_gpu[i];
        float abs_diff = fabsf(a - b);
        float rel_diff = abs_diff / (fabsf(a) + 1e-6f);
        if (abs_diff > max_abs) max_abs = abs_diff;
        if (rel_diff > max_rel) max_rel = rel_diff;
        if (rel_diff > 1e-3f) errors++;   // relative tolerance: robust to FMA rounding + magnitude
    }
    printf("Max abs diff: %e | Max rel diff: %e\n", max_abs, max_rel);
    if (errors == 0) {
        printf("Integrated Pipeline Test: Results match!\n");
    } else {
        printf("Integrated Pipeline Test: %d mismatches detected!\n", errors);
    }

    // -------------------------------------
    // Clean up host and device memory, CUDA events
    // -------------------------------------
    if (mode == MODE_CUDNN) {
        if (workspace) cudaFree(workspace);
        cudnnDestroyTensorDescriptor(inDesc);
        cudnnDestroyTensorDescriptor(outDesc);
        cudnnDestroyFilterDescriptor(filtDesc);
        cudnnDestroyConvolutionDescriptor(convDesc);
        cudnn_destroy(cudnnHandle);
    }

    cudaFree(d_input);
    cudaFree(d_conv_kernel);
    cudaFree(d_conv);
    cudaFree(d_bn);
    cudaFree(d_relu);
    cudaFree(d_pool);

    CHECK_CUDA_ERR(cudaEventDestroy(conv_start));
    CHECK_CUDA_ERR(cudaEventDestroy(conv_stop));
    CHECK_CUDA_ERR(cudaEventDestroy(bn_start));
    CHECK_CUDA_ERR(cudaEventDestroy(bn_stop));
    CHECK_CUDA_ERR(cudaEventDestroy(relu_start));
    CHECK_CUDA_ERR(cudaEventDestroy(relu_stop));
    CHECK_CUDA_ERR(cudaEventDestroy(pool_start));
    CHECK_CUDA_ERR(cudaEventDestroy(pool_stop));

    free(h_input);
    free(h_conv_kernel);
    free(h_conv_cpu);
    free(h_bn_cpu);
    free(h_relu_cpu);
    free(h_pool_cpu);
    free(h_pool_gpu);

    return 0;
}

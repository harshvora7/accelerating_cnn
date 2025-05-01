// File: src/tensorrt_pipeline.cu

#include <iostream>
#include <vector>
#include <chrono>

#include "NvInfer.h"
#include "cuda_runtime_api.h"

#include "error_check.h"              // CHECK_CUDA_ERR
#include "convolution.h"              // cpu_convolution()
#include "batch_norm.h"               // cpu_batch_norm()
#include "relu.h"                     // cpu_relu()
#include "pooling.h"                  // cpu_max_pooling()

#include "tensorrt_convolution.h"     // tensorrt_addConvolution()
#include "tensorrt_batch_norm.h"      // tensorrt_addBatchNorm()
#include "tensorrt_relu.h"            // tensorrt_addReLU()
#include "tensorrt_pooling.h"         // tensorrt_addPooling()

using namespace nvinfer1;

// Simple logger for TensorRT info/warning/errors
class TRTLogger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kINFO) {
            std::cerr << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

int main() {
    // --- Layer dimensions ---
    const int N = 1, C = 1;
    const int H = 512, W = 512;
    const int R = 3, S = 3;                    // conv kernel
    const int H_out = H - R + 1, W_out = W - S + 1;
    const int poolSize = 2, poolStride = 2;
    const int H_pool = (H_out - poolSize) / poolStride + 1;
    const int W_pool = (W_out - poolSize) / poolStride + 1;

    // --- CPU reference pipeline ---
    std::vector<float> h_input(N*C*H*W), h_kernel(C*R*S);
    std::vector<float> h_conv(N*C*H_out*W_out),
                       h_bn(N*C*H_out*W_out),
                       h_relu(N*C*H_out*W_out),
                       h_pool(N*C*H_pool*W_pool);

    // Initialize host input and kernel
    for (int i = 0; i < N*C*H*W; ++i) {
        h_input[i] = static_cast<float>(rand() % 10);
    }
    float ek[9] = {1,0,-1, 1,0,-1, 1,0,-1};
    for (int i = 0; i < C*R*S; ++i) {
        h_kernel[i] = ek[i];
    }

    // CPU conv
    cpu_convolution(h_input.data(), h_kernel.data(), h_conv.data(),
                    W, H, S, R, W_out, H_out);
    // CPU batch norm
    int M = N*C*H_out*W_out;
    double sum = 0, vsum = 0;
    for (int i = 0; i < M; ++i) sum += h_conv[i];
    double mean = sum / M;
    for (int i = 0; i < M; ++i) {
        double d = h_conv[i] - mean;
        vsum += d*d;
    }
    double var = vsum / M;
    const float gamma = 1.0f, beta = 0.0f, eps = 1e-5f;
    for (int i = 0; i < M; ++i) {
        h_bn[i]   = gamma * ((h_conv[i] - mean) / sqrt(var + eps)) + beta;
        h_relu[i] = h_bn[i] > 0 ? h_bn[i] : 0.0f;
    }
    cpu_max_pooling(h_relu.data(), h_pool.data(),
                    W_out, H_out, poolSize, poolStride);

    // --- Create TensorRT builder & network ---
    IBuilder* builder = createInferBuilder(gLogger);
    INetworkDefinition* network = builder->createNetworkV2(0);

    // Input tensor
    ITensor* input = network->addInput("input", DataType::kFLOAT, Dims4{N, C, H, W});
    assert(input);

    // Build layers
    ITensor* conv = tensorrt_addConvolution(network, input,
                        /*outChannels=*/C, R, S,
                        /*strideH=*/1, /*strideW=*/1,
                        /*padH=*/0, /*padW=*/0,
                        /*kernelWeights=*/h_kernel.data());
    assert(conv);

    // BatchNorm parameters
    std::vector<float> meanArr(C, static_cast<float>(mean)),
                        varArr(C, static_cast<float>(var)),
                        gammaArr(C, gamma),
                        betaArr(C, beta);
    ITensor* bn = tensorrt_addBatchNorm(network, conv,
                        meanArr.data(), varArr.data(),
                        gammaArr.data(), betaArr.data(),
                        eps);
    assert(bn);

    ITensor* relu = tensorrt_addReLU(network, bn);
    assert(relu);

    ITensor* pool = tensorrt_addPooling(network, relu,
                        poolSize, poolSize,
                        poolStride, poolStride,
                        /*padH=*/0, /*padW=*/0);
    assert(pool);

    pool->setName("output");
    network->markOutput(*pool);

    // --- Build engine ---
    IBuilderConfig* config = builder->createBuilderConfig();
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    IExecutionContext* ctx = engine->createExecutionContext();

    // Cleanup network & builder
    network->destroy();
    builder->destroy();
    config->destroy();

    // --- Allocate GPU buffers ---
    void* d_in = nullptr;
    void* d_out = nullptr;
    size_t inBytes  = h_input.size() * sizeof(float);
    size_t outBytes = h_pool.size()  * sizeof(float);
    CHECK_CUDA_ERR(cudaMalloc(&d_in,  inBytes));
    CHECK_CUDA_ERR(cudaMalloc(&d_out, outBytes));
    CHECK_CUDA_ERR(cudaMemcpy(d_in, h_input.data(), inBytes, cudaMemcpyHostToDevice));

    // --- Run inference ---
    const int ITERS = 50;
    // Warm-up
    for (int i = 0; i < 5; ++i) {
        void* bindings[] = { d_in, d_out };
        ctx->enqueueV2(bindings, 0, nullptr);
    }

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERR(cudaEventCreate(&start));
    CHECK_CUDA_ERR(cudaEventCreate(&stop));
    float totalMs = 0;
    for (int i = 0; i < ITERS; ++i) {
        void* bindings[] = { d_in, d_out };
        CHECK_CUDA_ERR(cudaEventRecord(start));
        ctx->enqueueV2(bindings, 0, nullptr);
        CHECK_CUDA_ERR(cudaEventRecord(stop));
        CHECK_CUDA_ERR(cudaEventSynchronize(stop));
        float ms;
        CHECK_CUDA_ERR(cudaEventElapsedTime(&ms, start, stop));
        totalMs += ms;
    }
    std::cout << "TensorRT Conv→BN→ReLU→Pool avg: "
              << (totalMs / ITERS) << " ms\n";

    // --- Copy back and validate ---
    std::vector<float> h_out(h_pool.size());
    CHECK_CUDA_ERR(cudaMemcpy(h_out.data(), d_out, outBytes, cudaMemcpyDeviceToHost));
    int mismatches = 0;
    for (size_t i = 0; i < h_out.size(); ++i) {
        if (fabs(h_out[i] - h_pool[i]) > 1e-4f) ++mismatches;
    }
    std::cout << "Validation: " << (mismatches ? "FAIL" : "PASS")
              << " (" << mismatches << " mismatches)\n";

    // --- Cleanup ---
    cudaFree(d_in);
    cudaFree(d_out);
    ctx->destroy();
    engine->destroy();

    CHECK_CUDA_ERR(cudaEventDestroy(start));
    CHECK_CUDA_ERR(cudaEventDestroy(stop));

    return 0;
}

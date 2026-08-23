// File: src/tensorrt_batch_norm.cu

#include "tensorrt_batch_norm.h"
#include <NvInfer.h>
#include <cassert>
#include <cstring>

// Add a Batch Normalization (Scale + Shift) layer to a TensorRT network.
// This wraps TensorRT's IScaleLayer with channel-wise weights.
extern "C" nvinfer1::ITensor* tensorrt_addBatchNorm(
    nvinfer1::INetworkDefinition* network,
    nvinfer1::ITensor*            input,
    int                            channels,
    nvinfer1::Weights              scaleWeights,
    nvinfer1::Weights              shiftWeights,
    nvinfer1::Weights              powerWeights)
{
    assert(network && "network must not be null");
    assert(input   && "input tensor must not be null");
    assert(channels > 0 && "channels must be positive");

    // Create the IScaleLayer: applies Y = (X * scale) + shift, with optional power
    nvinfer1::IScaleLayer* scaleLayer = network->addScale(
        *input,
        nvinfer1::ScaleMode::kCHANNEL,
        shiftWeights,   // Bias term (β)
        scaleWeights,   // Scale term (γ)
        powerWeights    // Power term (usually all 1s)
    );
    if (!scaleLayer) {
        // In practice, handle this error more gracefully
        fprintf(stderr, "TensorRT addScale (BatchNorm) returned null\n");
        return nullptr;
    }
    scaleLayer->setName("BatchNorm_ScaleShift");

    // The output tensor of the scale layer is the normalized tensor
    return scaleLayer->getOutput(0);
}

// File: src/tensorrt_convolution.cu

#include "tensorrt_convolution.h"
#include <NvInfer.h>
#include <cassert>

using namespace nvinfer1;

extern "C" ITensor* tensorrt_addConvolution(
    INetworkDefinition* network,
    ITensor*            input,
    int                 outChannels,
    DimsHW              kernelSize,
    Weights             kernelWeights,
    Weights             biasWeights,
    DimsHW              padding,
    DimsHW              stride)
{
    // Sanity checks
    assert(network && "network must not be null");
    assert(input   && "input tensor must not be null");

    // Create the convolution layer
    // addConvolutionNd takes Dims (kernel size) as its 3rd arg; DimsHW implicitly convertible
    IConvolutionLayer* conv = network->addConvolutionNd(
        *input,
        outChannels,
        kernelSize,
        kernelWeights,
        biasWeights);
    assert(conv && "failed to create convolution layer");

    // Set convolution parameters
    conv->setPadding(padding);
    conv->setStride(stride);

    // Return the output tensor of the layer
    return conv->getOutput(0);
}

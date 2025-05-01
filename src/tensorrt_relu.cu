// File: src/tensorrt_relu.cu

#include "tensorrt_relu.h"
#include <iostream>

// Adds a ReLU activation layer to the TensorRT network.
// Returns the output tensor of the ReLU layer, or nullptr on failure.
nvinfer1::ITensor* tensorrt_addReLU(
    nvinfer1::INetworkDefinition* network,
    nvinfer1::ITensor*            input)
{
    // Create a ReLU activation layer
    auto relu = network->addActivation(*input, nvinfer1::ActivationType::kRELU);
    if (!relu) {
        std::cerr << "TensorRT error: failed to create ReLU layer\n";
        return nullptr;
    }
    // Return the output tensor of the ReLU layer
    return relu->getOutput(0);
}

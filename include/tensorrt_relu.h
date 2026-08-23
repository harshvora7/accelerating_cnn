// File: src/tensorrt_relu.h

#ifndef TENSORRT_RELU_H
#define TENSORRT_RELU_H

#include <NvInfer.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Add a ReLU activation layer to a TensorRT network.
 *
 * @param network  Pointer to the INetworkDefinition
 * @param input    Input ITensor to apply ReLU to
 * @returns        The ITensor output of the ReLU layer, or nullptr on failure
 */
nvinfer1::ITensor* tensorrt_addReLU(
    nvinfer1::INetworkDefinition* network,
    nvinfer1::ITensor*            input);

#ifdef __cplusplus
}
#endif

#endif // TENSORRT_RELU_H

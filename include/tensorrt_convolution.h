// File: include/tensorrt_convolution.h
#ifndef TENSORRT_CONVOLUTION_H
#define TENSORRT_CONVOLUTION_H

#include <NvInfer.h>
using namespace nvinfer1;

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Add a 2D convolution layer to a TensorRT network.
 *
 * @param network       The TensorRT network definition to which to add the layer.
 * @param input         The input tensor.
 * @param outChannels   Number of output feature maps (K).
 * @param kernelSize    Convolution kernel size (R, S).
 * @param kernelWeights Weights struct containing the convolution kernel values.
 * @param biasWeights   Weights struct containing the bias values (can be empty).
 * @param padding       Amount of zero‐padding (padH, padW).
 * @param stride        Convolution stride (strideH, strideW).
 * @returns             The tensor corresponding to the convolution output.
 */
ITensor* tensorrt_addConvolution(
    INetworkDefinition* network,
    ITensor*            input,
    int                 outChannels,
    DimsHW              kernelSize,
    Weights             kernelWeights,
    Weights             biasWeights,
    DimsHW              padding,
    DimsHW              stride);

#ifdef __cplusplus
}
#endif

#endif // TENSORRT_CONVOLUTION_H

// File: src/tensorrt_pooling.h

#ifndef TENSORRT_POOLING_H
#define TENSORRT_POOLING_H

#include "NvInfer.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Adds a 2D max-pooling layer to the TensorRT network.
 *
 * @param network   Pointer to the TensorRT INetworkDefinition
 * @param input     Pointer to the input ITensor (NCHW layout)
 * @param windowH   Height of the pooling window
 * @param windowW   Width of the pooling window
 * @param strideH   Vertical stride
 * @param strideW   Horizontal stride
 * @param padH      Vertical padding
 * @param padW      Horizontal padding
 * @return          Pointer to the output ITensor of the pooling layer, or nullptr on failure
 */
nvinfer1::ITensor* tensorrt_addPooling(
    nvinfer1::INetworkDefinition* network,
    nvinfer1::ITensor*            input,
    int                           windowH,
    int                           windowW,
    int                           strideH,
    int                           strideW,
    int                           padH,
    int                           padW);

#ifdef __cplusplus
}
#endif

#endif // TENSORRT_POOLING_H

// File: include/tensorrt_batch_norm.h

#ifndef TENSORRT_BATCH_NORM_H
#define TENSORRT_BATCH_NORM_H

#include <NvInfer.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Add a Batch Normalization (Scale + Shift) layer to a TensorRT network.
 *
 * @param network      Pointer to an existing TensorRT INetworkDefinition.
 * @param input        Input tensor to be normalized (NCHW layout).
 * @param channels     Number of channels in the input tensor.
 * @param scaleWeights Weights for the scale (γ) term; length = channels.
 * @param shiftWeights Weights for the shift (β) term; length = channels.
 * @param powerWeights Weights for the power term (usually all 1s); length = channels.
 * @return             The output ITensor of the added IScaleLayer.
 */
nvinfer1::ITensor* tensorrt_addBatchNorm(
    nvinfer1::INetworkDefinition* network,
    nvinfer1::ITensor*            input,
    int                            channels,
    nvinfer1::Weights              scaleWeights,
    nvinfer1::Weights              shiftWeights,
    nvinfer1::Weights              powerWeights
);

#ifdef __cplusplus
}
#endif

#endif // TENSORRT_BATCH_NORM_H

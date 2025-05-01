// File: src/tensorrt_pooling.cu

#include "tensorrt_pooling.h"
#include "NvInfer.h"
#include <cassert>

extern "C" nvinfer1::ITensor* tensorrt_addPooling(
    nvinfer1::INetworkDefinition* network,
    nvinfer1::ITensor*            input,
    int                           windowH,
    int                           windowW,
    int                           strideH,
    int                           strideW,
    int                           padH,
    int                           padW)
{
    assert(network && input);

    // Create a max-pooling layer
    auto pool = network->addPoolingNd(
        *input,
        nvinfer1::PoolingType::kMAX,
        nvinfer1::DimsHW{windowH, windowW}
    );
    if (!pool) {
        // Failed to create pooling layer
        return nullptr;
    }

    // Set stride and padding
    pool->setStrideNd(nvinfer1::DimsHW{strideH, strideW});
    pool->setPaddingNd(nvinfer1::DimsHW{padH, padW});

    // Return the output tensor of the pooling layer
    return pool->getOutput(0);
}

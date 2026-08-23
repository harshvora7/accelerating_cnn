// Standalone demo: run the tiled convolution kernel on a real image (Sobel edge detection).
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "error_check.h"
#include "tiled_convolution.h"

// Combine Sobel-X and Sobel-Y gradients into an edge-magnitude image (clamped to 0..255).
__global__ void sobel_magnitude(const float* gx, const float* gy, unsigned char* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float m = sqrtf(gx[i]*gx[i] + gy[i]*gy[i]);
        out[i] = (unsigned char)(m > 255.0f ? 255.0f : m);
    }
}

int main(int argc, char** argv) {
    const char* inpath  = argc > 1 ? argv[1] : "input.jpg";
    const char* outpath = argc > 2 ? argv[2] : "edges.png";

    int w, h, ch;
    unsigned char* img = stbi_load(inpath, &w, &h, &ch, 1);  // force grayscale
    if (!img) { printf("Could not load %s\n", inpath); return 1; }
    printf("Loaded %s (%dx%d)\n", inpath, w, h);

    int n = w*h, outW = w-2, outH = h-2, outN = outW*outH;   // valid 3x3 conv
    float* h_gray = (float*)malloc(n*sizeof(float));
    for (int i = 0; i < n; i++) h_gray[i] = (float)img[i];

    float *d_gray, *d_gx, *d_gy; unsigned char* d_out;
    CHECK_CUDA_ERR(cudaMalloc(&d_gray, n*sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc(&d_gx, outN*sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc(&d_gy, outN*sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc(&d_out, outN));
    CHECK_CUDA_ERR(cudaMemcpy(d_gray, h_gray, n*sizeof(float), cudaMemcpyHostToDevice));

    float sobel_x[9] = {-1,0,1, -2,0,2, -1,0,1};
    float sobel_y[9] = {-1,-2,-1, 0,0,0, 1,2,1};
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((outW+TILE_DIM-1)/TILE_DIM, (outH+TILE_DIM-1)/TILE_DIM);

    set_tiled_conv_kernel(sobel_x, 3, 3);
    tiled_convolution<<<grid, block>>>(d_gray, d_gx, w, h, 3, 3, outW, outH);
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    set_tiled_conv_kernel(sobel_y, 3, 3);
    tiled_convolution<<<grid, block>>>(d_gray, d_gy, w, h, 3, 3, outW, outH);
    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    int threads = 256, blocks = (outN+threads-1)/threads;
    sobel_magnitude<<<blocks, threads>>>(d_gx, d_gy, d_out, outN);
    CHECK_CUDA_ERR(cudaDeviceSynchronize());

    unsigned char* h_out = (unsigned char*)malloc(outN);
    CHECK_CUDA_ERR(cudaMemcpy(h_out, d_out, outN, cudaMemcpyDeviceToHost));
    stbi_write_png(outpath, outW, outH, 1, h_out, outW);
    stbi_write_png("input_gray.png", w, h, 1, img, w);   // grayscale input, for a clean before/after
    printf("Wrote %s (%dx%d) and input_gray.png\n", outpath, outW, outH);

    free(img); free(h_gray); free(h_out);
    cudaFree(d_gray); cudaFree(d_gx); cudaFree(d_gy); cudaFree(d_out);
    return 0;
}

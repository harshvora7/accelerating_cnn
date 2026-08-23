// Multi-channel convolution benchmark.
//   Compares custom naive + custom tiled against cuDNN (FP32), all validated
//   against a CPU reference. Foundation for the channel-count sweep: at C=1 the
//   custom kernels win; as channels grow cuDNN's implicit-GEMM pulls ahead.
//
// Usage: ./mc_conv_bench --cin=16 --cout=16 --hw=256
// Layout (NCHW, N=1): in [Cin,H,W]  filter [Cout,Cin,R,S]  out [Cout,Hout,Wout]
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include "error_check.h"
#include "cudnn_convolution.h"

// ---- custom multi-channel naive convolution (valid, stride 1, no padding) ----
__global__ void mc_naive_conv(const float* in, const float* filt, float* out,
                              int Cin, int Cout, int H, int W, int R, int S,
                              int Hout, int Wout) {
    int x  = blockIdx.x * blockDim.x + threadIdx.x;
    int y  = blockIdx.y * blockDim.y + threadIdx.y;
    int co = blockIdx.z;                       // one grid-z slice per output channel
    if (x >= Wout || y >= Hout || co >= Cout) return;
    float sum = 0.0f;
    for (int ci = 0; ci < Cin; ++ci) {
        const float* inC   = in   + (size_t)ci * H * W;
        const float* filtC = filt + ((size_t)co * Cin + ci) * R * S;
        for (int ky = 0; ky < R; ++ky)
            for (int kx = 0; kx < S; ++kx)
                sum += inC[(y + ky) * W + (x + kx)] * filtC[ky * S + kx];
    }
    out[((size_t)co * Hout + y) * Wout + x] = sum;
}

#define MC_TILE 16
// ---- custom multi-channel TILED convolution ----
// Per block, load each input channel's tile (+halo) into shared memory once and
// reuse it across the R*S window; accumulate over all Cin. One grid-z slice per
// output channel. Filter passed via global memory (constant memory can't hold
// Cout*Cin*R*S once channels grow -- a good illustration of that 64KB limit).
__global__ void mc_tiled_conv(const float* in, const float* filt, float* out,
                              int Cin, int Cout, int H, int W, int R, int S,
                              int Hout, int Wout) {
    extern __shared__ float tile[];            // (MC_TILE+R-1) x (MC_TILE+S-1)
    const int tw = MC_TILE + S - 1;
    const int th = MC_TILE + R - 1;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int baseX = blockIdx.x * MC_TILE, baseY = blockIdx.y * MC_TILE;
    const int co = blockIdx.z;
    const int outX = baseX + tx, outY = baseY + ty;

    float sum = 0.0f;
    for (int ci = 0; ci < Cin; ++ci) {
        const float* inC   = in   + (size_t)ci * H * W;
        const float* filtC = filt + ((size_t)co * Cin + ci) * R * S;
        for (int ly = ty; ly < th; ly += MC_TILE)
            for (int lx = tx; lx < tw; lx += MC_TILE) {
                int gx = baseX + lx, gy = baseY + ly;
                tile[ly*tw + lx] = (gx < W && gy < H) ? inC[gy*W + gx] : 0.0f;
            }
        __syncthreads();
        if (outX < Wout && outY < Hout)
            for (int ky = 0; ky < R; ++ky)
                for (int kx = 0; kx < S; ++kx)
                    sum += tile[(ty+ky)*tw + (tx+kx)] * filtC[ky*S + kx];
        __syncthreads();                        // protect tile before next channel overwrites it
    }
    if (outX < Wout && outY < Hout)
        out[((size_t)co*Hout + outY)*Wout + outX] = sum;
}

static void cpu_mc_conv(const float* in, const float* filt, float* out,
                        int Cin, int Cout, int H, int W, int R, int S,
                        int Hout, int Wout) {
    for (int co = 0; co < Cout; ++co)
      for (int y = 0; y < Hout; ++y)
        for (int x = 0; x < Wout; ++x) {
            float sum = 0.0f;
            for (int ci = 0; ci < Cin; ++ci)
              for (int ky = 0; ky < R; ++ky)
                for (int kx = 0; kx < S; ++kx)
                    sum += in[((size_t)ci*H + (y+ky))*W + (x+kx)]
                         * filt[(((size_t)co*Cin+ci)*R+ky)*S+kx];
            out[((size_t)co*Hout + y)*Wout + x] = sum;
        }
}

static void check(float* d_out, float* h_gpu, const float* h_cpu,
                  size_t outN, const char* name) {
    CHECK_CUDA_ERR(cudaMemcpy(h_gpu, d_out, outN*sizeof(float), cudaMemcpyDeviceToHost));
    // Relative L2-norm error over the whole tensor: robust to outputs that cancel
    // toward zero (per-element relative error explodes there). max_abs shows the truth.
    double max_abs = 0.0, num = 0.0, den = 0.0;
    for (size_t i = 0; i < outN; ++i) {
        double d = (double)h_cpu[i] - (double)h_gpu[i];
        if (fabs(d) > max_abs) max_abs = fabs(d);
        num += d*d; den += (double)h_cpu[i]*(double)h_cpu[i];
    }
    double rel_l2 = sqrt(num) / (sqrt(den) + 1e-12);
    printf("  %-6s max_abs=%.3e  rel_L2=%.3e  %s\n",
           name, max_abs, rel_l2, rel_l2 < 1e-3 ? "PASS" : "MISMATCH");
}

static int argi(int argc, char** argv, const char* key, int defv) {
    for (int i = 1; i < argc; ++i)
        if (strncmp(argv[i], key, strlen(key)) == 0)
            return atoi(argv[i] + strlen(key));
    return defv;
}

int main(int argc, char** argv) {
    const int N = 1;
    int Cin  = argi(argc, argv, "--cin=",  16);
    int Cout = argi(argc, argv, "--cout=", 16);
    int H    = argi(argc, argv, "--hw=",  256), W = H;
    const int R = 3, S = 3;
    int Hout = H - R + 1, Wout = W - S + 1;

    size_t inN = (size_t)Cin*H*W, filtN = (size_t)Cout*Cin*R*S, outN = (size_t)Cout*Hout*Wout;
    float *h_in=(float*)malloc(inN*sizeof(float)), *h_filt=(float*)malloc(filtN*sizeof(float));
    float *h_cpu=(float*)malloc(outN*sizeof(float)), *h_gpu=(float*)malloc(outN*sizeof(float));
    srand(0);
    for (size_t i=0;i<inN;i++)   h_in[i]   = (float)(rand()%10);
    for (size_t i=0;i<filtN;i++) h_filt[i] = ((rand()%2001)/1000.0f) - 1.0f;   // [-1,1]

    cpu_mc_conv(h_in, h_filt, h_cpu, Cin, Cout, H, W, R, S, Hout, Wout);

    float *d_in,*d_filt,*d_naive,*d_cudnn;
    CHECK_CUDA_ERR(cudaMalloc(&d_in,    inN*sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc(&d_filt,  filtN*sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc(&d_naive, outN*sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc(&d_cudnn, outN*sizeof(float)));
    CHECK_CUDA_ERR(cudaMemcpy(d_in,   h_in,   inN*sizeof(float),   cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_filt, h_filt, filtN*sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16,16,1), grid((Wout+15)/16, (Hout+15)/16, Cout);
    const int WARM=5, IT=50;
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);

    // ---- custom naive ----
    for (int i=0;i<WARM;i++) mc_naive_conv<<<grid,block>>>(d_in,d_filt,d_naive,Cin,Cout,H,W,R,S,Hout,Wout);
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    cudaEventRecord(s);
    for (int i=0;i<IT;i++)   mc_naive_conv<<<grid,block>>>(d_in,d_filt,d_naive,Cin,Cout,H,W,R,S,Hout,Wout);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms_naive=0; cudaEventElapsedTime(&ms_naive,s,e); ms_naive/=IT;

    // ---- custom tiled ----
    float *d_tiled; CHECK_CUDA_ERR(cudaMalloc(&d_tiled, outN*sizeof(float)));
    {
        int tw = MC_TILE + S - 1, th = MC_TILE + R - 1;
        size_t shmem = (size_t)tw*th*sizeof(float);
        for (int i=0;i<WARM;i++) mc_tiled_conv<<<grid,block,shmem>>>(d_in,d_filt,d_tiled,Cin,Cout,H,W,R,S,Hout,Wout);
        CHECK_CUDA_ERR(cudaDeviceSynchronize());
        cudaEventRecord(s);
        for (int i=0;i<IT;i++)   mc_tiled_conv<<<grid,block,shmem>>>(d_in,d_filt,d_tiled,Cin,Cout,H,W,R,S,Hout,Wout);
        cudaEventRecord(e); cudaEventSynchronize(e);
    }
    float ms_tiled=0; cudaEventElapsedTime(&ms_tiled,s,e); ms_tiled/=IT;

    // ---- cuDNN FP32 (multi-channel via dims; reuses existing setup/forward) ----
    int inDims[4]={N,Cin,H,W}, filtDims[4]={Cout,Cin,R,S}, outDims[4]={N,Cout,Hout,Wout};
    cudnnHandle_t cudnn = cudnn_init();
    cudnnTensorDescriptor_t inDesc,outDesc; cudnnFilterDescriptor_t filtDesc;
    cudnnConvolutionDescriptor_t convDesc; cudnnConvolutionFwdAlgo_t algo;
    void* ws=nullptr; size_t wsSize=0;
    cudnn_convolution_setup(cudnn, inDims, filtDims, outDims, 0,0,1,1,
        &inDesc,&filtDesc,&convDesc,&outDesc,&algo,&ws,&wsSize);
    for (int i=0;i<WARM;i++) cudnn_convolution_forward(cudnn,inDesc,filtDesc,convDesc,outDesc,algo,ws,wsSize,d_in,d_filt,d_cudnn,1.0f,0.0f);
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    cudaEventRecord(s);
    for (int i=0;i<IT;i++)   cudnn_convolution_forward(cudnn,inDesc,filtDesc,convDesc,outDesc,algo,ws,wsSize,d_in,d_filt,d_cudnn,1.0f,0.0f);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms_cudnn=0; cudaEventElapsedTime(&ms_cudnn,s,e); ms_cudnn/=IT;

    printf("Multi-channel conv  Cin=%d Cout=%d  %dx%d  3x3\n", Cin, Cout, H, W);
    printf("  naive : %8.4f ms\n", ms_naive);
    printf("  tiled : %8.4f ms   (naive/tiled = %.2fx)\n", ms_tiled, ms_naive/ms_tiled);
    printf("  cuDNN : %8.4f ms   (naive/cuDNN = %.2fx, tiled/cuDNN = %.2fx)\n",
           ms_cudnn, ms_naive/ms_cudnn, ms_tiled/ms_cudnn);
    check(d_naive, h_gpu, h_cpu, outN, "naive");
    check(d_tiled, h_gpu, h_cpu, outN, "tiled");
    check(d_cudnn, h_gpu, h_cpu, outN, "cuDNN");
    return 0;
}

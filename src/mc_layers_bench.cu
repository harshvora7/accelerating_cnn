// Multi-channel BatchNorm / ReLU / MaxPool benchmark.
//   Custom multi-channel kernels validated against a CPU reference (L2 metric).
//   BatchNorm uses PER-CHANNEL statistics (each channel its own mean/var/gamma/beta
//   -- the interesting part); ReLU is elementwise; MaxPool is per-channel 2x2/stride2.
//   These kernels feed the fused pipeline in Phase 7.
//
// Usage: ./mc_layers_bench --c=64 --hw=256      (H,W assumed even for 2x2 pooling)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include "error_check.h"

// ---------------- kernels ----------------
__global__ void mc_bn(const float* in, float* out,
                      const float* mean, const float* var,
                      const float* gamma, const float* beta,
                      int C, int HW, float eps) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= C*HW) return;
    int c = idx / HW;                          // which channel this element belongs to
    float inv = rsqrtf(var[c] + eps);
    out[idx] = gamma[c] * (in[idx] - mean[c]) * inv + beta[c];
}

__global__ void mc_relu(const float* in, float* out, int total) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < total) out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

__global__ void mc_maxpool(const float* in, float* out,
                           int C, int H, int W, int Hout, int Wout) {
    int ox = blockIdx.x*blockDim.x + threadIdx.x;
    int oy = blockIdx.y*blockDim.y + threadIdx.y;
    int c  = blockIdx.z;
    if (ox >= Wout || oy >= Hout || c >= C) return;
    const float* inC = in + (size_t)c*H*W;
    int iy = 2*oy, ix = 2*ox;
    float m = inC[iy*W + ix];
    m = fmaxf(m, inC[iy*W + ix + 1]);
    m = fmaxf(m, inC[(iy+1)*W + ix]);
    m = fmaxf(m, inC[(iy+1)*W + ix + 1]);
    out[((size_t)c*Hout + oy)*Wout + ox] = m;
}

// ---------------- CPU references ----------------
static void cpu_bn(const float* in, float* out, const float* mean, const float* var,
                   const float* gamma, const float* beta, int C, int HW, float eps) {
    for (int c=0;c<C;c++) {
        float inv = 1.0f/sqrtf(var[c]+eps);
        for (int i=0;i<HW;i++) { int idx=c*HW+i; out[idx]=gamma[c]*(in[idx]-mean[c])*inv+beta[c]; }
    }
}
static void cpu_relu(const float* in, float* out, int total) {
    for (int i=0;i<total;i++) out[i]=in[i]>0.0f?in[i]:0.0f;
}
static void cpu_maxpool(const float* in, float* out, int C, int H, int W, int Hout, int Wout) {
    for (int c=0;c<C;c++)
      for (int oy=0;oy<Hout;oy++)
        for (int ox=0;ox<Wout;ox++) {
            const float* inC=in+(size_t)c*H*W; int iy=2*oy, ix=2*ox;
            float m=inC[iy*W+ix];
            m=fmaxf(m,inC[iy*W+ix+1]); m=fmaxf(m,inC[(iy+1)*W+ix]); m=fmaxf(m,inC[(iy+1)*W+ix+1]);
            out[((size_t)c*Hout+oy)*Wout+ox]=m;
        }
}

// ---------------- L2 check ----------------
static void check(float* d_out, float* h_gpu, const float* h_cpu, size_t n, const char* name) {
    CHECK_CUDA_ERR(cudaMemcpy(h_gpu, d_out, n*sizeof(float), cudaMemcpyDeviceToHost));
    double max_abs=0, num=0, den=0;
    for (size_t i=0;i<n;i++) {
        double d=(double)h_cpu[i]-(double)h_gpu[i];
        if (fabs(d)>max_abs) max_abs=fabs(d);
        num+=d*d; den+=(double)h_cpu[i]*(double)h_cpu[i];
    }
    double rel=sqrt(num)/(sqrt(den)+1e-12);
    printf("  %-8s max_abs=%.3e  rel_L2=%.3e  %s\n", name, max_abs, rel, rel<1e-3?"PASS":"MISMATCH");
}

static int argi(int argc, char** argv, const char* key, int defv) {
    for (int i=1;i<argc;i++) if (strncmp(argv[i],key,strlen(key))==0) return atoi(argv[i]+strlen(key));
    return defv;
}

int main(int argc, char** argv) {
    int C = argi(argc,argv,"--c=",64);
    int H = argi(argc,argv,"--hw=",256), W=H;
    int HW=H*W, total=C*HW, Hout=H/2, Wout=W/2, poolN=C*Hout*Wout;
    float eps=1e-5f;

    float *h_in =(float*)malloc(total*sizeof(float));
    float *h_cpu=(float*)malloc(total*sizeof(float)), *h_gpu=(float*)malloc(total*sizeof(float));
    float *h_cpu_pool=(float*)malloc(poolN*sizeof(float)), *h_gpu_pool=(float*)malloc(poolN*sizeof(float));
    float *mean=(float*)malloc(C*sizeof(float)), *var=(float*)malloc(C*sizeof(float));
    float *gamma=(float*)malloc(C*sizeof(float)), *beta=(float*)malloc(C*sizeof(float));

    srand(0);
    for (int i=0;i<total;i++) h_in[i]=((rand()%2001)/100.0f)-10.0f;   // [-10,10]: exercises ReLU
    for (int c=0;c<C;c++) {                                           // per-channel stats + params
        double sum=0; for (int i=0;i<HW;i++) sum+=h_in[c*HW+i];
        double mu=sum/HW, vs=0;
        for (int i=0;i<HW;i++){ double d=h_in[c*HW+i]-mu; vs+=d*d; }
        mean[c]=(float)mu; var[c]=(float)(vs/HW);
        gamma[c]=0.5f+((rand()%1000)/1000.0f);    // ~[0.5,1.5]
        beta[c]=((rand()%2001)/1000.0f)-1.0f;      // ~[-1,1]
    }

    float *d_in,*d_out,*d_pool,*d_mean,*d_var,*d_gamma,*d_beta;
    CHECK_CUDA_ERR(cudaMalloc(&d_in,total*sizeof(float)));  CHECK_CUDA_ERR(cudaMalloc(&d_out,total*sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc(&d_pool,poolN*sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc(&d_mean,C*sizeof(float)));    CHECK_CUDA_ERR(cudaMalloc(&d_var,C*sizeof(float)));
    CHECK_CUDA_ERR(cudaMalloc(&d_gamma,C*sizeof(float)));   CHECK_CUDA_ERR(cudaMalloc(&d_beta,C*sizeof(float)));
    CHECK_CUDA_ERR(cudaMemcpy(d_in,h_in,total*sizeof(float),cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_mean,mean,C*sizeof(float),cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_var,var,C*sizeof(float),cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_gamma,gamma,C*sizeof(float),cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_beta,beta,C*sizeof(float),cudaMemcpyHostToDevice));

    const int WARM=5, IT=50, TPB=256;
    int blocks=(total+TPB-1)/TPB;
    dim3 pblock(16,16,1), pgrid((Wout+15)/16,(Hout+15)/16,C);
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    float ms;
    printf("Multi-channel layers  C=%d  %dx%d\n", C, H, W);

    for(int i=0;i<WARM;i++) mc_bn<<<blocks,TPB>>>(d_in,d_out,d_mean,d_var,d_gamma,d_beta,C,HW,eps);
    CHECK_CUDA_ERR(cudaDeviceSynchronize()); cudaEventRecord(s);
    for(int i=0;i<IT;i++)   mc_bn<<<blocks,TPB>>>(d_in,d_out,d_mean,d_var,d_gamma,d_beta,C,HW,eps);
    cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e); ms/=IT;
    cpu_bn(h_in,h_cpu,mean,var,gamma,beta,C,HW,eps);
    printf("  BatchNorm: %.4f ms\n", ms); check(d_out,h_gpu,h_cpu,total,"BN");

    for(int i=0;i<WARM;i++) mc_relu<<<blocks,TPB>>>(d_in,d_out,total);
    CHECK_CUDA_ERR(cudaDeviceSynchronize()); cudaEventRecord(s);
    for(int i=0;i<IT;i++)   mc_relu<<<blocks,TPB>>>(d_in,d_out,total);
    cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e); ms/=IT;
    cpu_relu(h_in,h_cpu,total);
    printf("  ReLU     : %.4f ms\n", ms); check(d_out,h_gpu,h_cpu,total,"ReLU");

    for(int i=0;i<WARM;i++) mc_maxpool<<<pgrid,pblock>>>(d_in,d_pool,C,H,W,Hout,Wout);
    CHECK_CUDA_ERR(cudaDeviceSynchronize()); cudaEventRecord(s);
    for(int i=0;i<IT;i++)   mc_maxpool<<<pgrid,pblock>>>(d_in,d_pool,C,H,W,Hout,Wout);
    cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e); ms/=IT;
    cpu_maxpool(h_in,h_cpu_pool,C,H,W,Hout,Wout);
    printf("  MaxPool  : %.4f ms\n", ms); check(d_pool,h_gpu_pool,h_cpu_pool,poolN,"MaxPool");
    return 0;
}

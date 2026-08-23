// Kernel fusion benchmark: Conv -> BatchNorm -> ReLU.
//   Fusion collapses the chain into one kernel: each thread computes a conv output,
//   applies BN + ReLU while the value is still in a register, and writes once --
//   eliminating the global-memory round-trips between layers. TensorRT's core trick.
//
//   Two comparisons, both L2-validated vs a CPU reference (ref = relu(bn(conv(in)))):
//     (1) epilogue : BN + ReLU as 2 kernels        vs  fused BN+ReLU  (1 kernel)
//     (2) full     : conv + BN + ReLU as 3 kernels  vs  fused conv+BN+ReLU (1 kernel)
//   BN statistics are per OUTPUT channel, computed from the conv output.
//
// Usage: ./fuse_bench --cin=64 --cout=64 --hw=256
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include "error_check.h"

__global__ void mc_naive_conv(const float* in, const float* filt, float* out,
                              int Cin, int Cout, int H, int W, int R, int S, int Hout, int Wout) {
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y, co=blockIdx.z;
    if (x>=Wout||y>=Hout||co>=Cout) return;
    float sum=0.0f;
    for (int ci=0;ci<Cin;++ci){
        const float* inC=in+(size_t)ci*H*W; const float* fC=filt+((size_t)co*Cin+ci)*R*S;
        for (int ky=0;ky<R;++ky) for (int kx=0;kx<S;++kx) sum+=inC[(y+ky)*W+(x+kx)]*fC[ky*S+kx];
    }
    out[((size_t)co*Hout+y)*Wout+x]=sum;
}
__global__ void mc_bn(const float* in, float* out, const float* mean, const float* var,
                      const float* gamma, const float* beta, int C, int HW, float eps){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=C*HW)return; int c=i/HW;
    out[i]=gamma[c]*(in[i]-mean[c])*rsqrtf(var[c]+eps)+beta[c];
}
__global__ void mc_relu(const float* in, float* out, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=in[i]>0.0f?in[i]:0.0f;
}
// fused elementwise epilogue: BN then ReLU in one pass
__global__ void fused_bn_relu(const float* in, float* out, const float* mean, const float* var,
                              const float* gamma, const float* beta, int C, int HW, float eps){
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=C*HW)return; int c=i/HW;
    float v=gamma[c]*(in[i]-mean[c])*rsqrtf(var[c]+eps)+beta[c];
    out[i]=v>0.0f?v:0.0f;
}
// fully fused conv+BN+ReLU (BN stats indexed by OUTPUT channel co)
__global__ void fused_conv_bn_relu(const float* in, const float* filt, float* out,
                                   const float* mean, const float* var,
                                   const float* gamma, const float* beta, float eps,
                                   int Cin, int Cout, int H, int W, int R, int S, int Hout, int Wout){
    int x=blockIdx.x*blockDim.x+threadIdx.x, y=blockIdx.y*blockDim.y+threadIdx.y, co=blockIdx.z;
    if (x>=Wout||y>=Hout||co>=Cout) return;
    float sum=0.0f;
    for (int ci=0;ci<Cin;++ci){
        const float* inC=in+(size_t)ci*H*W; const float* fC=filt+((size_t)co*Cin+ci)*R*S;
        for (int ky=0;ky<R;++ky) for (int kx=0;kx<S;++kx) sum+=inC[(y+ky)*W+(x+kx)]*fC[ky*S+kx];
    }
    float v=gamma[co]*(sum-mean[co])*rsqrtf(var[co]+eps)+beta[co];   // BN + ReLU in-register
    out[((size_t)co*Hout+y)*Wout+x]=v>0.0f?v:0.0f;
}

static void cpu_conv(const float* in,const float* filt,float* out,
                     int Cin,int Cout,int H,int W,int R,int S,int Hout,int Wout){
    for(int co=0;co<Cout;co++)for(int y=0;y<Hout;y++)for(int x=0;x<Wout;x++){
        float sum=0.0f;
        for(int ci=0;ci<Cin;ci++)for(int ky=0;ky<R;ky++)for(int kx=0;kx<S;kx++)
            sum+=in[((size_t)ci*H+(y+ky))*W+(x+kx)]*filt[(((size_t)co*Cin+ci)*R+ky)*S+kx];
        out[((size_t)co*Hout+y)*Wout+x]=sum;
    }
}
static void check(float* d,float* hg,const float* hc,size_t n,const char* name){
    CHECK_CUDA_ERR(cudaMemcpy(hg,d,n*sizeof(float),cudaMemcpyDeviceToHost));
    double ma=0,num=0,den=0;
    for(size_t i=0;i<n;i++){double e=(double)hc[i]-(double)hg[i]; if(fabs(e)>ma)ma=fabs(e); num+=e*e; den+=(double)hc[i]*hc[i];}
    double rel=sqrt(num)/(sqrt(den)+1e-12);
    printf("  %-16s max_abs=%.3e rel_L2=%.3e %s\n",name,ma,rel,rel<1e-3?"PASS":"MISMATCH");
}
static int argi(int c,char** v,const char* k,int d){for(int i=1;i<c;i++)if(strncmp(v[i],k,strlen(k))==0)return atoi(v[i]+strlen(k));return d;}

int main(int argc,char** argv){
    int Cin=argi(argc,argv,"--cin=",64), Cout=argi(argc,argv,"--cout=",64);
    int H=argi(argc,argv,"--hw=",256), W=H; const int R=3,S=3; float eps=1e-5f;
    int Hout=H-R+1, Wout=W-S+1, HWout=Hout*Wout;
    size_t inN=(size_t)Cin*H*W, filtN=(size_t)Cout*Cin*R*S, outN=(size_t)Cout*Hout*Wout;

    float *h_in=(float*)malloc(inN*4), *h_filt=(float*)malloc(filtN*4);
    float *h_conv=(float*)malloc(outN*4), *h_ref=(float*)malloc(outN*4), *h_gpu=(float*)malloc(outN*4);
    float *mean=(float*)malloc(Cout*4), *var=(float*)malloc(Cout*4), *gamma=(float*)malloc(Cout*4), *beta=(float*)malloc(Cout*4);
    srand(0);
    for(size_t i=0;i<inN;i++)   h_in[i]=((rand()%2001)/100.0f)-10.0f;
    for(size_t i=0;i<filtN;i++) h_filt[i]=((rand()%2001)/1000.0f)-1.0f;

    cpu_conv(h_in,h_filt,h_conv,Cin,Cout,H,W,R,S,Hout,Wout);
    for(int co=0;co<Cout;co++){                         // per-output-channel stats from conv output
        double sum=0; for(int i=0;i<HWout;i++) sum+=h_conv[(size_t)co*HWout+i];
        double mu=sum/HWout, vs=0;
        for(int i=0;i<HWout;i++){double d=h_conv[(size_t)co*HWout+i]-mu; vs+=d*d;}
        mean[co]=(float)mu; var[co]=(float)(vs/HWout);
        gamma[co]=0.5f+((rand()%1000)/1000.0f); beta[co]=((rand()%2001)/1000.0f)-1.0f;
    }
    for(int co=0;co<Cout;co++){                         // ref = relu(bn(conv))
        float inv=1.0f/sqrtf(var[co]+eps);
        for(int i=0;i<HWout;i++){ size_t idx=(size_t)co*HWout+i;
            float v=gamma[co]*(h_conv[idx]-mean[co])*inv+beta[co]; h_ref[idx]=v>0.0f?v:0.0f; }
    }

    float *d_in,*d_filt,*d_conv,*d_tmp,*d_out,*d_mean,*d_var,*d_gamma,*d_beta;
    CHECK_CUDA_ERR(cudaMalloc(&d_in,inN*4));   CHECK_CUDA_ERR(cudaMalloc(&d_filt,filtN*4));
    CHECK_CUDA_ERR(cudaMalloc(&d_conv,outN*4));CHECK_CUDA_ERR(cudaMalloc(&d_tmp,outN*4));CHECK_CUDA_ERR(cudaMalloc(&d_out,outN*4));
    CHECK_CUDA_ERR(cudaMalloc(&d_mean,Cout*4));CHECK_CUDA_ERR(cudaMalloc(&d_var,Cout*4));
    CHECK_CUDA_ERR(cudaMalloc(&d_gamma,Cout*4));CHECK_CUDA_ERR(cudaMalloc(&d_beta,Cout*4));
    CHECK_CUDA_ERR(cudaMemcpy(d_in,h_in,inN*4,cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_filt,h_filt,filtN*4,cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_mean,mean,Cout*4,cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_var,var,Cout*4,cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_gamma,gamma,Cout*4,cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_beta,beta,Cout*4,cudaMemcpyHostToDevice));

    const int WARM=5, IT=50, TPB=256;
    int eblocks=(int)((outN+TPB-1)/TPB);
    dim3 cblock(16,16,1), cgrid((Wout+15)/16,(Hout+15)/16,Cout);
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e); float ms;

    mc_naive_conv<<<cgrid,cblock>>>(d_in,d_filt,d_conv,Cin,Cout,H,W,R,S,Hout,Wout); // conv once for the epilogue demo
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    printf("Fusion benchmark  Cin=%d Cout=%d  %dx%d  3x3\n",Cin,Cout,H,W);

    // (1) epilogue: BN+ReLU unfused vs fused
    for(int i=0;i<WARM;i++){ mc_bn<<<eblocks,TPB>>>(d_conv,d_tmp,d_mean,d_var,d_gamma,d_beta,Cout,HWout,eps); mc_relu<<<eblocks,TPB>>>(d_tmp,d_out,(int)outN); }
    CHECK_CUDA_ERR(cudaDeviceSynchronize()); cudaEventRecord(s);
    for(int i=0;i<IT;i++){ mc_bn<<<eblocks,TPB>>>(d_conv,d_tmp,d_mean,d_var,d_gamma,d_beta,Cout,HWout,eps); mc_relu<<<eblocks,TPB>>>(d_tmp,d_out,(int)outN); }
    cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e); float t_epi_unf=ms/IT;
    check(d_out,h_gpu,h_ref,outN,"epilogue unfused");
    for(int i=0;i<WARM;i++) fused_bn_relu<<<eblocks,TPB>>>(d_conv,d_out,d_mean,d_var,d_gamma,d_beta,Cout,HWout,eps);
    CHECK_CUDA_ERR(cudaDeviceSynchronize()); cudaEventRecord(s);
    for(int i=0;i<IT;i++) fused_bn_relu<<<eblocks,TPB>>>(d_conv,d_out,d_mean,d_var,d_gamma,d_beta,Cout,HWout,eps);
    cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e); float t_epi_fus=ms/IT;
    check(d_out,h_gpu,h_ref,outN,"epilogue fused");
    printf("  epilogue  unfused %.4f ms   fused %.4f ms   -> %.2fx\n", t_epi_unf, t_epi_fus, t_epi_unf/t_epi_fus);

    // (2) full: conv+BN+ReLU unfused vs fused
    for(int i=0;i<WARM;i++){ mc_naive_conv<<<cgrid,cblock>>>(d_in,d_filt,d_conv,Cin,Cout,H,W,R,S,Hout,Wout); mc_bn<<<eblocks,TPB>>>(d_conv,d_tmp,d_mean,d_var,d_gamma,d_beta,Cout,HWout,eps); mc_relu<<<eblocks,TPB>>>(d_tmp,d_out,(int)outN); }
    CHECK_CUDA_ERR(cudaDeviceSynchronize()); cudaEventRecord(s);
    for(int i=0;i<IT;i++){ mc_naive_conv<<<cgrid,cblock>>>(d_in,d_filt,d_conv,Cin,Cout,H,W,R,S,Hout,Wout); mc_bn<<<eblocks,TPB>>>(d_conv,d_tmp,d_mean,d_var,d_gamma,d_beta,Cout,HWout,eps); mc_relu<<<eblocks,TPB>>>(d_tmp,d_out,(int)outN); }
    cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e); float t_full_unf=ms/IT;
    check(d_out,h_gpu,h_ref,outN,"full unfused");
    for(int i=0;i<WARM;i++) fused_conv_bn_relu<<<cgrid,cblock>>>(d_in,d_filt,d_out,d_mean,d_var,d_gamma,d_beta,eps,Cin,Cout,H,W,R,S,Hout,Wout);
    CHECK_CUDA_ERR(cudaDeviceSynchronize()); cudaEventRecord(s);
    for(int i=0;i<IT;i++) fused_conv_bn_relu<<<cgrid,cblock>>>(d_in,d_filt,d_out,d_mean,d_var,d_gamma,d_beta,eps,Cin,Cout,H,W,R,S,Hout,Wout);
    cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e); float t_full_fus=ms/IT;
    check(d_out,h_gpu,h_ref,outN,"full fused");
    printf("  full      unfused %.4f ms   fused %.4f ms   -> %.2fx\n", t_full_unf, t_full_fus, t_full_unf/t_full_fus);
    printf("  (fusion removes the BN+ReLU round-trip ~= %.4f ms; its share grows as the conv gets faster)\n", t_epi_unf);
    return 0;
}

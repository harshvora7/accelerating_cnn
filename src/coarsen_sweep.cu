// Coarsening-factor sweep: times the output-channel-coarsened conv for COARSEN in
// {1,2,4,8} via a templated kernel (one binary), each L2-validated vs a CPU reference.
// Produces the "coarsening factor vs speedup" bonus chart.  Usage: ./coarsen_sweep --c=64 --hw=256
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>
#include "error_check.h"
#define MC_TILE 16

template<int COARSEN>
__global__ void coarsened_conv(const float* in, const float* filt, float* out,
                               int Cin, int Cout, int H, int W, int R, int S, int Hout, int Wout) {
    extern __shared__ float tile[];
    const int tw = MC_TILE + S - 1, th = MC_TILE + R - 1;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int baseX = blockIdx.x*MC_TILE, baseY = blockIdx.y*MC_TILE;
    const int co_base = blockIdx.z*COARSEN;
    const int outX = baseX+tx, outY = baseY+ty;
    float acc[COARSEN];
    #pragma unroll
    for (int j=0;j<COARSEN;++j) acc[j]=0.0f;
    for (int ci=0; ci<Cin; ++ci) {
        const float* inC = in + (size_t)ci*H*W;
        for (int ly=ty; ly<th; ly+=MC_TILE)
            for (int lx=tx; lx<tw; lx+=MC_TILE) {
                int gx=baseX+lx, gy=baseY+ly;
                tile[ly*tw+lx] = (gx<W&&gy<H)? inC[gy*W+gx] : 0.0f;
            }
        __syncthreads();
        if (outX<Wout && outY<Hout) {
            #pragma unroll
            for (int j=0;j<COARSEN;++j) {
                const float* fC = filt + ((size_t)(co_base+j)*Cin+ci)*R*S;
                float sum=0.0f;
                for (int ky=0;ky<R;++ky) for (int kx=0;kx<S;++kx)
                    sum += tile[(ty+ky)*tw+(tx+kx)]*fC[ky*S+kx];
                acc[j]+=sum;
            }
        }
        __syncthreads();
    }
    if (outX<Wout && outY<Hout) {
        #pragma unroll
        for (int j=0;j<COARSEN;++j)
            out[((size_t)(co_base+j)*Hout+outY)*Wout+outX]=acc[j];
    }
}

static void cpu_conv(const float* in,const float* filt,float* out,int Cin,int Cout,int H,int W,int R,int S,int Hout,int Wout){
    for(int co=0;co<Cout;co++)for(int y=0;y<Hout;y++)for(int x=0;x<Wout;x++){
        float sum=0.0f;
        for(int ci=0;ci<Cin;ci++)for(int ky=0;ky<R;ky++)for(int kx=0;kx<S;kx++)
            sum+=in[((size_t)ci*H+(y+ky))*W+(x+kx)]*filt[(((size_t)co*Cin+ci)*R+ky)*S+kx];
        out[((size_t)co*Hout+y)*Wout+x]=sum;
    }
}
static double rel_l2(const float* a,const float* b,size_t n){
    double num=0,den=0; for(size_t i=0;i<n;i++){double d=(double)a[i]-b[i]; num+=d*d; den+=(double)a[i]*a[i];}
    return sqrt(num)/(sqrt(den)+1e-12);
}
static int argi(int c,char**v,const char*k,int d){for(int i=1;i<c;i++)if(strncmp(v[i],k,strlen(k))==0)return atoi(v[i]+strlen(k));return d;}

template<int COARSEN>
static float time_coarsen(float* d_in,float* d_filt,float* d_out,int Cin,int Cout,int H,int W,int R,int S,int Hout,int Wout,
                          cudaEvent_t s,cudaEvent_t e,int WARM,int IT){
    dim3 block(MC_TILE,MC_TILE,1);
    dim3 grid((Wout+MC_TILE-1)/MC_TILE,(Hout+MC_TILE-1)/MC_TILE,Cout/COARSEN);
    size_t shmem=(size_t)(MC_TILE+S-1)*(MC_TILE+R-1)*sizeof(float);
    for(int i=0;i<WARM;i++) coarsened_conv<COARSEN><<<grid,block,shmem>>>(d_in,d_filt,d_out,Cin,Cout,H,W,R,S,Hout,Wout);
    CHECK_CUDA_ERR(cudaDeviceSynchronize());
    cudaEventRecord(s);
    for(int i=0;i<IT;i++) coarsened_conv<COARSEN><<<grid,block,shmem>>>(d_in,d_filt,d_out,Cin,Cout,H,W,R,S,Hout,Wout);
    cudaEventRecord(e); cudaEventSynchronize(e);
    float ms=0; cudaEventElapsedTime(&ms,s,e); return ms/IT;
}

int main(int argc,char**argv){
    int C=argi(argc,argv,"--c=",64); int Cin=C,Cout=C;
    int H=argi(argc,argv,"--hw=",256),W=H; const int R=3,S=3;
    int Hout=H-R+1,Wout=W-S+1;
    size_t inN=(size_t)Cin*H*W, filtN=(size_t)Cout*Cin*R*S, outN=(size_t)Cout*Hout*Wout;
    float *h_in=(float*)malloc(inN*4),*h_filt=(float*)malloc(filtN*4),*h_cpu=(float*)malloc(outN*4),*h_gpu=(float*)malloc(outN*4);
    srand(0);
    for(size_t i=0;i<inN;i++)h_in[i]=(float)(rand()%10);
    for(size_t i=0;i<filtN;i++)h_filt[i]=((rand()%2001)/1000.0f)-1.0f;
    cpu_conv(h_in,h_filt,h_cpu,Cin,Cout,H,W,R,S,Hout,Wout);
    float *d_in,*d_filt,*d_out;
    CHECK_CUDA_ERR(cudaMalloc(&d_in,inN*4));CHECK_CUDA_ERR(cudaMalloc(&d_filt,filtN*4));CHECK_CUDA_ERR(cudaMalloc(&d_out,outN*4));
    CHECK_CUDA_ERR(cudaMemcpy(d_in,h_in,inN*4,cudaMemcpyHostToDevice));
    CHECK_CUDA_ERR(cudaMemcpy(d_filt,h_filt,filtN*4,cudaMemcpyHostToDevice));
    cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
    const int WARM=5,IT=50;
    printf("Coarsen sweep  C=%d  %dx%d\n",C,H,W);
    #define RUN(K) if(Cout%(K)==0){ \
        float ms=time_coarsen<K>(d_in,d_filt,d_out,Cin,Cout,H,W,R,S,Hout,Wout,s,e,WARM,IT); \
        CHECK_CUDA_ERR(cudaMemcpy(h_gpu,d_out,outN*4,cudaMemcpyDeviceToHost)); \
        double r=rel_l2(h_cpu,h_gpu,outN); \
        printf("  COARSEN=%d : %8.4f ms   rel_L2=%.2e %s\n",K,ms,r,r<1e-3?"PASS":"MISMATCH"); }
    RUN(1) RUN(2) RUN(4) RUN(8)
    return 0;
}

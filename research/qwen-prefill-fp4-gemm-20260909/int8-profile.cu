// Same integer dots and FP32 scale-fold body, different tile instantiations.
#include "../../crates/memra-engine/cu/mmq_nvfp4_w4a8.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"line %d: %s\n",__LINE__,cudaGetErrorString(e));exit(2);} }while(0)
int main(int argc,char** argv){
    void* test;CK(cudaMalloc(&test,4));CK(cudaMemset(test,0,4));CK(cudaDeviceSynchronize());CK(cudaFree(test));
    const int shape=argc>1?atoi(argv[1]):0;
    {
        int m=1024,k=shape?17408:5120,n=shape?5120:17408;
        std::mt19937 rng(20260909);std::normal_distribution<float> normal(0,1);
        std::vector<float>x(size_t(m)*k);for(float&v:x)v=normal(rng);
        const size_t blocks=size_t(n)*k/64;
        std::vector<uint8_t>w(blocks*36);
        for(size_t i=0;i<blocks;++i){
            for(int j=0;j<32;++j)w[i*32+j]=uint8_t(rng());
            for(int j=0;j<4;++j)w[blocks*32+i*4+j]=uint8_t(40+rng()%25);
        }
        void *dw,*scratch;float *dx,*dy;
        CK(cudaMalloc(&dw,w.size()));CK(cudaMalloc(&dx,x.size()*4));CK(cudaMalloc(&dy,size_t(m)*n*4));
        CK(cudaMalloc(&scratch,memra_mmq_nvfp4_w4a8_act_bytes(k,m)));
        CK(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice));CK(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));
        int rc=memra_mmq_nvfp4_w4a8(dw,dx,dy,k,n,m,scratch,nullptr,0.37f,1);
        if(rc){fprintf(stderr,"baseline rc=%d\n",rc);return 3;}
        CK(cudaDeviceSynchronize());
        CK(cudaFree(dw));CK(cudaFree(dx));CK(cudaFree(dy));CK(cudaFree(scratch));
    }
}

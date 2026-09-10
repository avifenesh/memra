// Same integer dots and FP32 scale-fold body, different tile instantiations.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"line %d: %s\n",__LINE__,cudaGetErrorString(e));exit(2);} }while(0)
extern "C" size_t base_bytes(int,int);
using Gemm=int(*)(const void*,const float*,float*,int,int,int,void*,void*,float,int);
extern "C" int base_gemm(const void*,const float*,float*,int,int,int,void*,void*,float,int);
extern "C" int unpack_gemm(const void*,const float*,float*,int,int,int,void*,void*,float,int);
extern "C" int fold_gemm(const void*,const float*,float*,int,int,int,void*,void*,float,int);
extern "C" int act_gemm(const void*,const float*,float*,int,int,int,void*,void*,float,int);
Gemm funcs[]={base_gemm,unpack_gemm,fold_gemm,act_gemm};
int main(){
    void* test;CK(cudaMalloc(&test,4));CK(cudaMemset(test,0,4));CK(cudaDeviceSynchronize());CK(cudaFree(test));
    for(int shape=0;shape<2;++shape){
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
        CK(cudaMalloc(&scratch,base_bytes(k,m)));
        CK(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice));CK(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));
        auto run=[&](int arm){
            int rc=funcs[arm](dw,dx,dy,k,n,m,scratch,nullptr,0.37f,1);
            if(rc){fprintf(stderr,"arm %d rc=%d\n",arm,rc);exit(3);}
        };
        std::vector<float>ref(size_t(m)*n),out(ref.size());
        run(0);CK(cudaMemcpy(ref.data(),dy,ref.size()*4,cudaMemcpyDeviceToHost));
        for(int arm=0;arm<4;++arm){
            run(arm);CK(cudaMemcpy(out.data(),dy,out.size()*4,cudaMemcpyDeviceToHost));
            size_t mismatch=0,bad=0;double ma=0,err2=0,ref2=0;
            for(size_t i=0;i<ref.size();++i){
                mismatch+=memcmp(&ref[i],&out[i],4)!=0;
                bad+=!std::isfinite(ref[i])||!std::isfinite(out[i]);
                double d=double(ref[i])-out[i];ma=std::max(ma,fabs(d));err2+=d*d;ref2+=double(ref[i])*ref[i];
            }
            printf("{\"kind\":\"identity\",\"m\":%d,\"k\":%d,\"n\":%d,\"arm\":%d,\"elements\":%zu,\"bit_mismatch\":%zu,\"nonfinite\":%zu,\"max_abs\":%.9g,\"relative_l2\":%.9g}\n",m,k,n,arm,ref.size(),mismatch,bad,ma,sqrt(err2/ref2));
            if(bad)return 4;
        }
        for(int arm=0;arm<4;++arm) for(int rep=0;rep<3;++rep)run(arm);
        CK(cudaDeviceSynchronize());
        cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));
        for(int round=0;round<6;++round)for(int pos=0;pos<4;++pos){
            int arm=round%2?3-pos:pos;
            CK(cudaEventRecord(a));for(int rep=0;rep<20;++rep)run(arm);CK(cudaEventRecord(b));CK(cudaEventSynchronize(b));
            float ms;CK(cudaEventElapsedTime(&ms,a,b));ms/=20;
            printf("{\"kind\":\"timing\",\"m\":%d,\"k\":%d,\"n\":%d,\"arm\":%d,\"round\":%d,\"calls\":20,\"ms\":%.9g,\"input_tok_s\":%.9g,\"tops\":%.9g}\n",m,k,n,arm,round,ms,1024000./ms,2.*m*k*n/(ms*1e9));fflush(stdout);
        }
        CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));CK(cudaFree(dw));CK(cudaFree(dx));CK(cudaFree(dy));CK(cudaFree(scratch));
    }
}

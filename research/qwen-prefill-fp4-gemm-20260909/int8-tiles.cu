// Same integer dots and FP32 scale-fold body, different tile instantiations.
#include "../../crates/memra-engine/cu/mmq_nvfp4_w4a8.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"line %d: %s\n",__LINE__,cudaGetErrorString(e));exit(2);} }while(0)
template<int X,int Y,int Pipe>
void launch_tile(const void* w,const float* x,float* y,int k,int n,int m,void* scratch) {
    const int padded=GGML_PAD(k,MATRIX_ROW_PADDING);
    const dim3 qb(m,(padded+4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ-1)/(4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ));
    quantize_mmq_q8_1_d4_kernel<<<qb,CUDA_QUANTIZE_BLOCK_SIZE_MMQ>>>(x,scratch,k,k,padded,m);
    const size_t smem=mmq_nvfp4_w4a8_nbytes_shared(Pipe,X,Y);
    auto kernel=mul_mat_q_nvfp4_w4a8<X,Y,Pipe,false,true>;
    CK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    kernel<<<dim3(n/Y,m/X),dim3(32,Y/16),smem>>>(
        (const char*)w,(const char*)w+size_t(n)*(k/64)*32,(const int*)scratch,y,n,m,k/64,m,n,k/64,0.37f);
    CK(cudaGetLastError());
}
template<int X,int Y,int Pipe>
void resource(int arm) {
    auto kernel=mul_mat_q_nvfp4_w4a8<X,Y,Pipe,false,true>;
    cudaFuncAttributes a;CK(cudaFuncGetAttributes(&a,kernel));
    const size_t smem=mmq_nvfp4_w4a8_nbytes_shared(Pipe,X,Y);
    CK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    int blocks;CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,kernel,32*(Y/16),smem));
    printf("{\"kind\":\"resources\",\"arm\":%d,\"tile_m\":%d,\"tile_n\":%d,\"pipeline\":%d,\"regs\":%d,\"local_bytes\":%zu,\"smem\":%zu,\"max_ctas_per_sm\":%d}\n",arm,X,Y,Pipe,a.numRegs,a.localSizeBytes,smem,blocks);
}
int main(){
    void* test;CK(cudaMalloc(&test,4));CK(cudaMemset(test,0,4));CK(cudaDeviceSynchronize());CK(cudaFree(test));
    resource<128,128,1>(0);resource<64,64,1>(1);resource<32,64,1>(2);resource<64,32,1>(3);
    resource<128,64,1>(4);resource<64,128,1>(5);resource<64,64,0>(6);
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
        CK(cudaMalloc(&scratch,memra_mmq_nvfp4_w4a8_act_bytes(k,m)));
        CK(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice));CK(cudaMemcpy(dx,x.data(),x.size()*4,cudaMemcpyHostToDevice));
        auto run=[&](int arm){
            switch(arm){
            case 0: {int rc=memra_mmq_nvfp4_w4a8(dw,dx,dy,k,n,m,scratch,nullptr,0.37f,1);if(rc){fprintf(stderr,"baseline rc=%d\n",rc);exit(3);}break;}
            case 1:launch_tile<64,64,1>(dw,dx,dy,k,n,m,scratch);break;
            case 2:launch_tile<32,64,1>(dw,dx,dy,k,n,m,scratch);break;
            case 3:launch_tile<64,32,1>(dw,dx,dy,k,n,m,scratch);break;
            case 4:launch_tile<128,64,1>(dw,dx,dy,k,n,m,scratch);break;
            case 5:launch_tile<64,128,1>(dw,dx,dy,k,n,m,scratch);break;
            case 6:launch_tile<64,64,0>(dw,dx,dy,k,n,m,scratch);break;
            }
        };
        std::vector<float>ref(size_t(m)*n),out(ref.size());
        run(0);CK(cudaMemcpy(ref.data(),dy,ref.size()*4,cudaMemcpyDeviceToHost));
        for(int arm=0;arm<7;++arm){
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
        for(int arm=0;arm<7;++arm) for(int rep=0;rep<3;++rep)run(arm);
        CK(cudaDeviceSynchronize());
        cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));
        for(int round=0;round<6;++round)for(int pos=0;pos<7;++pos){
            int arm=round%2?6-pos:pos;
            CK(cudaEventRecord(a));for(int rep=0;rep<20;++rep)run(arm);CK(cudaEventRecord(b));CK(cudaEventSynchronize(b));
            float ms;CK(cudaEventElapsedTime(&ms,a,b));ms/=20;
            printf("{\"kind\":\"timing\",\"m\":%d,\"k\":%d,\"n\":%d,\"arm\":%d,\"round\":%d,\"calls\":20,\"ms\":%.9g,\"input_tok_s\":%.9g,\"tops\":%.9g}\n",m,k,n,arm,round,ms,1024000./ms,2.*m*k*n/(ms*1e9));fflush(stdout);
        }
        CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));CK(cudaFree(dw));CK(cudaFree(dx));CK(cudaFree(dy));CK(cudaFree(scratch));
    }
}

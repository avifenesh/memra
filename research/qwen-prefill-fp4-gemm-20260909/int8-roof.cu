// Register-only INT8 instruction roof. No loads/stores inside the measured loop.
#include <cuda_runtime.h>
#include <initializer_list>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){fprintf(stderr,"line %d: %s\n",__LINE__,cudaGetErrorString(e));exit(2);} }while(0)
template<int K,int A>
__global__ void roof(int* out,unsigned long long* cycles,int iters) {
    // Nonzero small integers; no signed accumulation overflow at the tested iteration count.
    const uint32_t a=0x01010101u,b=0x01010101u;
    int d[A][4]={};
    __syncthreads();
    unsigned long long start=clock64();
    for(int it=0;it<iters;++it) {
        #pragma unroll
        for(int j=0;j<A;++j) {
            if constexpr(K==16) asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3};" : "+r"(d[j][0]),"+r"(d[j][1]),"+r"(d[j][2]),"+r"(d[j][3]) : "r"(a),"r"(a),"r"(b));
            else asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};" : "+r"(d[j][0]),"+r"(d[j][1]),"+r"(d[j][2]),"+r"(d[j][3]) : "r"(a),"r"(a),"r"(a),"r"(a),"r"(b),"r"(b));
        }
    }
    unsigned long long stop=clock64();
    int sum=0;
    #pragma unroll
    for(int j=0;j<A;++j) for(int k=0;k<4;++k) sum+=d[j][k];
    out[blockIdx.x*blockDim.x+threadIdx.x]=sum;
    if(threadIdx.x==0) cycles[blockIdx.x]=stop-start;
}
template<int K,int A>
void measure(int grid,int* out,unsigned long long* cycles) {
    constexpr int iters=65536,threads=256,calls=20;
    cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));
    for(int i=0;i<3;++i) roof<K,A><<<grid,threads>>>(out,cycles,iters);
    CK(cudaDeviceSynchronize());
    for(int r=0;r<5;++r) {
        CK(cudaEventRecord(start));
        for(int call=0;call<calls;++call) roof<K,A><<<grid,threads>>>(out,cycles,iters);
        CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));
        float ms;CK(cudaEventElapsedTime(&ms,start,end));
        unsigned long long ticks;int result;
        CK(cudaMemcpy(&ticks,cycles,8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(&result,out,4,cudaMemcpyDeviceToHost));
        if(result!=A*4*K*iters) {fprintf(stderr,"integer control mismatch %d\n",result);exit(3);}
        const double ops=double(grid)*(threads/32)*iters*A*2*16*8*K;
        printf("{\"k\":%d,\"accumulators\":%d,\"grid\":%d,\"threads\":256,\"iters\":%d,\"calls\":20,\"rep\":%d,\"ms\":%.9g,\"tops\":%.9g,\"cta0_cycles_per_warp_mma\":%.9g,\"integer_control\":%d}\n",K,A,grid,iters,r,ms,calls*ops/(ms*1e9),double(ticks)/(iters*A),result);fflush(stdout);
    }
    CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));
}
int main(){
    cudaDeviceProp prop;CK(cudaGetDeviceProperties(&prop,0));int* out;unsigned long long* cycles;
    CK(cudaMalloc(&out,prop.multiProcessorCount*4*256*4));CK(cudaMalloc(&cycles,prop.multiProcessorCount*4*8));
    CK(cudaMemset(out,0,4));CK(cudaDeviceSynchronize());
    printf("{\"device\":\"%s\",\"sms\":%d}\n",prop.name,prop.multiProcessorCount);
    for(int grid:{prop.multiProcessorCount,prop.multiProcessorCount*2,prop.multiProcessorCount*4}) {
        measure<16,1>(grid,out,cycles);measure<16,2>(grid,out,cycles);measure<16,4>(grid,out,cycles);measure<16,8>(grid,out,cycles);
        measure<32,4>(grid,out,cycles);
    }
    CK(cudaFree(out));CK(cudaFree(cycles));
}

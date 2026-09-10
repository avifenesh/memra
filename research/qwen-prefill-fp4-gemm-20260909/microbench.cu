// Research-only direct ABI benchmark; not linked into the engine.
#include "../../crates/memra-engine/cu/mmq_fp4.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" size_t memra_mmq_nvfp4_w4a8_act_bytes(int, int);
extern "C" int memra_mmq_nvfp4_w4a8(const void*, const float*, float*, int, int, int, void*, void*, float, int);
#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(2); } } while(0)

// Row-local quantization. Each thread owns four adjacent values; a four-lane
// group chooses one per-16 scale. SiLU/mul is optional and has no intermediate
// global activation tensor on the fused arm. E2M1 conversion is native RN.
template<bool Fused>
__device__ float input_value(const float* a, const float* b, size_t i) {
    const float v=a[i];
    if constexpr(Fused) return (v/(1.f+expf(-v)))*b[i];
    return v;
}
template<bool Fused>
__global__ void fp4gemm_quant(const float* a, const float* b, block_fp4_mmq* q,
                             float* scales, int k, int m) {
    const int row=blockIdx.x, lane=threadIdx.x%32, warp=threadIdx.x/32;
    const size_t base=size_t(row)*k;
    float mx=0;
    for(int i=threadIdx.x;i<k;i+=blockDim.x) mx=fmaxf(mx,fabsf(input_value<Fused>(a,b,base+i)));
    mx=mmq_warp_reduce_max(mx);
    __shared__ float wm[8];
    if(lane==0) wm[warp]=mx;
    __syncthreads();
    if(warp==0) {
        mx=lane<8?wm[lane]:0;
        mx=mmq_warp_reduce_max(mx);
        if(lane==0) { wm[0]=mx/(6.f*448.f); scales[row]=wm[0]; }
    }
    __syncthreads();
    const float inv=wm[0]>0?1.f/wm[0]:0;
    const int padded=GGML_PAD(k,512);
    for(int i=threadIdx.x*4;i<padded;i+=blockDim.x*4) {
        float v[4]; float sm=0;
        #pragma unroll
        for(int j=0;j<4;++j) { v[j]=i+j<k?input_value<Fused>(a,b,base+i+j)*inv:0; sm=fmaxf(sm,fabsf(v[j])); }
        sm=fmaxf(sm,__shfl_xor_sync(0xffffffff,sm,1,4));
        sm=fmaxf(sm,__shfl_xor_sync(0xffffffff,sm,2,4));
        const __nv_fp8_e4m3 sf(sm/6.f);
        const float ds=float(sf), qi=ds>0?1.f/ds:0;
        const __nv_fp4x4_e2m1 packed(make_float4(v[0]*qi,v[1]*qi,v[2]*qi,v[3]*qi));
        const unsigned bits=packed.__x;
        const unsigned pair=__shfl_xor_sync(0xffffffff,bits,2,4);
        const int sub=(i%256)/16;
        block_fp4_mmq* out=q+size_t(i/256)*m+row;
        if((threadIdx.x&3)==0) reinterpret_cast<uint8_t*>(out->d4)[sub]=sf.__x;
        if((threadIdx.x&3)<2) {
            uint32_t bytes=0;
            #pragma unroll
            for(int j=0;j<4;++j) bytes|=(((bits>>(4*j))&15)|(((pair>>(4*j))&15)<<4))<<(8*j);
            reinterpret_cast<uint32_t*>(out->qs)[2*sub+(threadIdx.x&1)]=bytes;
        }
    }
}
__global__ void silu_input(const float* a,const float* b,float* x,size_t len) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<len) x[i]=input_value<true>(a,b,i);
}
static int candidate(const void* w,const float* a,const float* b,float* y,int k,int n,int m,void* scratch,bool fused) {
    float* scales=mmq_nvfp4_scale_ptr(scratch,k,m);
    if(fused) fp4gemm_quant<true><<<m,256>>>(a,b,(block_fp4_mmq*)scratch,scales,k,m);
    else fp4gemm_quant<false><<<m,256>>>(a,b,(block_fp4_mmq*)scratch,scales,k,m);
    CK(cudaGetLastError());
    const size_t smem=mmq_nvfp4_nbytes_shared();
    CK(cudaFuncSetAttribute(mul_mat_q_nvfp4<128,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    mul_mat_q_nvfp4<128,false><<<dim3(n/128,m/128),dim3(32,8),smem>>>(
        (const char*)w,(const int*)scratch,y,scales,n,m,k/64,m,n,k/64,1.f);
    CK(cudaGetLastError()); return 0;
}
int main() {
    void* allocation=nullptr; CK(cudaMalloc(&allocation,4)); CK(cudaMemset(allocation,0,4)); CK(cudaDeviceSynchronize()); CK(cudaFree(allocation));
    cudaDeviceProp p{}; CK(cudaGetDeviceProperties(&p,0));
    printf("{\"device\":\"%s\",\"sms\":%d,\"seed\":20260909}\n",p.name,p.multiProcessorCount);
    for(int shape=0;shape<2;++shape) for(int fused=0;fused<2;++fused) {
        const int m=1024,k=shape?17408:5120,n=shape?5120:17408;
        std::mt19937 rng(20260909); std::normal_distribution<float> normal(0,1);
        std::vector<float> a(size_t(m)*k),b(a.size());
        for(size_t i=0;i<a.size();++i) { a[i]=normal(rng); b[i]=normal(rng); }
        std::vector<uint8_t> w(size_t(n)*k/64*36);
        for(size_t i=0;i<w.size();i+=36) {
            for(int j=0;j<4;++j) w[i+j]=uint8_t(40+(rng()%25));
            for(int j=4;j<36;++j) w[i+j]=uint8_t(rng());
        }
        std::vector<uint8_t> rp(w.size());
        const size_t blocks=w.size()/36;
        for(size_t ib=0;ib<blocks;++ib) {
            memcpy(rp.data()+ib*32,w.data()+ib*36+4,32);
            memcpy(rp.data()+blocks*32+ib*4,w.data()+ib*36,4);
        }
        float *da,*db,*dx,*dy; void *dw,*dwrp,*scratch;
        CK(cudaMalloc(&da,a.size()*4)); CK(cudaMalloc(&db,b.size()*4)); CK(cudaMalloc(&dx,a.size()*4));
        CK(cudaMalloc(&dy,size_t(m)*n*4)); CK(cudaMalloc(&dw,w.size())); CK(cudaMalloc(&dwrp,rp.size()));
        CK(cudaMalloc(&scratch,std::max(memra_mmq_nvfp4_act_bytes(k,m),memra_mmq_nvfp4_w4a8_act_bytes(k,m))));
        CK(cudaMemcpy(da,a.data(),a.size()*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(db,b.data(),b.size()*4,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dw,w.data(),w.size(),cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dwrp,rp.data(),rp.size(),cudaMemcpyHostToDevice));
        auto run=[&](int arm) {
            int rc;
            if(arm<2 && fused) silu_input<<<(a.size()+255)/256,256>>>(da,db,dx,a.size());
            const float* input=fused?dx:da;
            if(arm==0) rc=memra_mmq_nvfp4_w4a8(dwrp,input,dy,k,n,m,scratch,nullptr,1.f,1);
            else if(arm==1) rc=memra_mmq_nvfp4_ex2(dw,input,dy,k,n,m,scratch,nullptr,1.f,1,0);
            else rc=candidate(dw,da,db,dy,k,n,m,scratch,fused);
            if(rc) { fprintf(stderr,"arm %d rc=%d\n",arm,rc); exit(3); }
        };
        std::vector<float> reference(size_t(m)*n),out(reference.size());
        run(0); CK(cudaMemcpy(reference.data(),dy,reference.size()*4,cudaMemcpyDeviceToHost));
        double refmax=0,ref2=0; for(float v:reference) { if(!std::isfinite(v)) return 4; refmax=std::max(refmax,double(fabs(v)));ref2+=double(v)*v; }
        for(int arm=0;arm<3;++arm) {
            run(arm); CK(cudaMemcpy(out.data(),dy,out.size()*4,cudaMemcpyDeviceToHost));
            double ma=0,err2=0;size_t bad=0;
            for(size_t i=0;i<out.size();++i) { if(!std::isfinite(out[i])) ++bad; double d=double(out[i])-reference[i]; ma=std::max(ma,fabs(d));err2+=d*d; }
            printf("{\"kind\":\"deviation\",\"m\":%d,\"k\":%d,\"n\":%d,\"silu\":%d,\"arm\":%d,\"max_abs\":%.9g,\"max_normalized\":%.9g,\"relative_l2\":%.9g,\"nonfinite\":%zu}\n",m,k,n,fused,arm,ma,ma/refmax,sqrt(err2/ref2),bad);
            if(bad) return 4;
        }
        cudaEvent_t start,end; CK(cudaEventCreate(&start)); CK(cudaEventCreate(&end));
        for(int j=0;j<9;++j) run(j%3); CK(cudaDeviceSynchronize());
        for(int round=0;round<6;++round) for(int pos=0;pos<3;++pos) {
            const int arm=round%2?2-pos:pos;
            CK(cudaEventRecord(start)); for(int rep=0;rep<10;++rep) run(arm); CK(cudaEventRecord(end)); CK(cudaEventSynchronize(end));
            float ms; CK(cudaEventElapsedTime(&ms,start,end)); ms/=10;
            printf("{\"kind\":\"timing\",\"m\":%d,\"k\":%d,\"n\":%d,\"silu\":%d,\"arm\":%d,\"round\":%d,\"ms\":%.9g,\"input_tok_s\":%.9g,\"tops\":%.9g}\n",m,k,n,fused,arm,round,ms,m*1000./ms,2.*m*k*n/(ms*1e9)); fflush(stdout);
        }
        CK(cudaEventDestroy(start)); CK(cudaEventDestroy(end));
        CK(cudaFree(da)); CK(cudaFree(db)); CK(cudaFree(dx)); CK(cudaFree(dy)); CK(cudaFree(dw)); CK(cudaFree(dwrp)); CK(cudaFree(scratch));
    }
}

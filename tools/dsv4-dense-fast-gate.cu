// Shape microbench for the current exact-tail kernels. No ncu/nsys required.
// nvcc -t 2 -std=c++17 -O3 -fmad=false -Xcompiler=-ffp-contract=off
//   -arch=sm_120a -lineinfo -Xptxas=-v tools/dsv4-dense-fast-gate.cu
//   -lcublasLt -lcublas -ldl -o <empty-owned-target>/component
#include "../crates/memra-engine/cu/dsv4_gpu.cu"
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static void ck(cudaError_t rc) {
    if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
}
struct Shape { int kind, n, k; }; // 0 FP8, 1 F32 dots, 2 BF16 dots
static const Shape shapes[] = {
    {0,512,4096},{0,1024,4096},{0,2048,4096},{0,4096,2048},
    {0,4096,4096},{0,8192,1024},{0,16384,1024},
    {1,4,16384},{1,24,16384},{1,256,4096},{1,512,4096},
    {1,1024,4096},{2,129280,4096}
};
struct Buffer {
    void* p{};
    explicit Buffer(size_t bytes) { ck(cudaMalloc(&p,bytes)); }
    ~Buffer() { cudaFree(p); }
};
static void bench(int rank, Shape sh, cudaStream_t stream, void* flush) {
    int n=sh.n,k=sh.k,sc_cols=(k+127)/128;
    size_t wb=size_t(n)*k*(sh.kind==0?1:sh.kind==1?4:2);
    size_t xb=size_t(k)*(sh.kind==0?2:4),sb=sh.kind==0?4*size_t((n+127)/128)*sc_cols:0;
    Buffer w(wb),x(xb),sc(sb?sb:4),y(4*size_t(n));
    // Finite deterministic operands. Baseline diagnostics only, not the real-input gate.
    ck(cudaMemsetAsync(w.p,0x38,wb,stream));
    ck(cudaMemsetAsync(x.p,0x3f,xb,stream));
    std::vector<float> scales(sb/4,1.0f);
    if(sb) ck(cudaMemcpyAsync(sc.p,scales.data(),sb,cudaMemcpyHostToDevice,stream));
    auto launch=[&] {
        int rc=sh.kind==0?memra_dsv4_dense_exact_tail_fp8(w.p,(float*)sc.p,sc_cols,x.p,(float*)y.p,1,n,k,0,0,stream)
            :memra_dsv4_dense_exact_tail_dots((float*)x.p,w.p,sh.kind==2,(float*)y.p,1,n,k,stream);
        if(rc) throw std::runtime_error("kernel rc="+std::to_string(rc));
    };
    const void* fn=sh.kind==0?(const void*)dsv4_dense_exact_tail_fp8_kernel<1,false>
        :(const void*)dsv4_dense_exact_tail_dots_kernel<1>;
    cudaFuncAttributes attr{}; ck(cudaFuncGetAttributes(&attr,fn));
    int blocks=0; ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,fn,128,0));
    cudaDeviceProp prop{}; ck(cudaGetDeviceProperties(&prop,rank));
    printf("RESOURCE rank=%d kind=%d n=%d k=%d registers=%d shared=%zu local=%zu blocks_per_sm=%d resident_warps=%d sm_count=%d grid_ctas=%d occupancy_limit_pct=%.6f\n",
        rank,sh.kind,n,k,attr.numRegs,attr.sharedSizeBytes,attr.localSizeBytes,blocks,blocks*4,prop.multiProcessorCount,n,
        100.0*blocks*128/prop.maxThreadsPerMultiProcessor);
    ck(cudaStreamSynchronize(stream));
    cudaGraph_t graph{}; cudaGraphExec_t exec{};
    ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)); launch();
    ck(cudaStreamEndCapture(stream,&graph)); ck(cudaGraphInstantiate(&exec,graph,0));
    for(int i=0;i<32;++i) ck(cudaGraphLaunch(exec,stream));
    ck(cudaStreamSynchronize(stream));
    cudaEvent_t a{},b{}; ck(cudaEventCreate(&a)); ck(cudaEventCreate(&b));
    for(int cold=0;cold<2;++cold) {
        for(int rep=0;rep<50;++rep) {
            if(cold) ck(cudaMemsetAsync(flush,rep,256ull<<20,stream));
            ck(cudaEventRecord(a,stream)); ck(cudaGraphLaunch(exec,stream));
            ck(cudaEventRecord(b,stream)); ck(cudaEventSynchronize(b));
            float ms; ck(cudaEventElapsedTime(&ms,a,b));
            printf("TIME rank=%d kind=%d n=%d k=%d cold=%d rep=%d us=%.6f bytes=%zu tensor_gbs=%.6f\n",
                rank,sh.kind,n,k,cold,rep,ms*1000.0,wb+xb+sb+4*size_t(n),(wb+xb+sb+4*size_t(n))/(ms*1.e6));
        }
    }
    ck(cudaEventDestroy(a));ck(cudaEventDestroy(b));
    ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(graph));fflush(stdout);
}
int main(int argc,char** argv) {
    try {
        bool reverse=argc==2 && std::string(argv[1])=="--reverse";
        if(argc>2 || (argc==2&&!reverse)) throw std::runtime_error("usage: component [--reverse]");
        int count=0;ck(cudaGetDeviceCount(&count));if(count!=2)throw std::runtime_error("two GPUs required");
        for(int r=0;r<2;++r) {
            int rank=reverse?1-r:r;ck(cudaSetDevice(rank));cudaStream_t stream{};
            ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
            { Buffer flush(256ull<<20);
              for(int s=0;s<13;++s)bench(rank,shapes[reverse?12-s:s],stream,flush.p); }
            ck(cudaStreamDestroy(stream));
        }
        puts("PASS baseline_shapes=26 timing_rows=2600 occupancy=static_limit instructions=disassembly");return 0;
    } catch(const std::exception& e) {fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}

// Captured real-operand dense batch component. Build/run only on the owned dev pair.
// nvcc -t 2 -std=c++17 -O3 -fmad=false -Xcompiler=-ffp-contract=off
//   -arch=sm_120a -lineinfo -Xptxas=-v tools/dsv4-dense-batch-gate.cu
//   -lcublasLt -lcublas -ldl -o <owned-target>/component
#include "../crates/memra-engine/cu/dsv4_gpu.cu"
#include <array>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static void insist(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
static void ck(cudaError_t rc) { if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc)); }
static void api(int rc) { if (rc) throw std::runtime_error("launcher rc=" + std::to_string(rc)); }
struct Stream {
    cudaStream_t value{};
    Stream() { ck(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamSynchronize(value); cudaStreamDestroy(value); }
};
template<class T> struct Guard {
    static constexpr size_t pad = 64 / sizeof(T);
    T* raw{};
    size_t n;
    cudaStream_t stream;
    std::vector<T> original;
    Guard(size_t count, cudaStream_t s): n(count), stream(s), original(n + 2*pad) {
        memset(original.data(), 0xa5, original.size()*sizeof(T));
        ck(cudaMalloc(&raw, original.size()*sizeof(T)));
        reset();
    }
    ~Guard() { cudaStreamSynchronize(stream); cudaFree(raw); }
    T* data() { return raw + pad; }
    void upload(const std::vector<T>& values) {
        insist(values.size() == n, "input count");
        memcpy(original.data()+pad, values.data(), n*sizeof(T)); reset();
    }
    void reset() { ck(cudaMemcpyAsync(raw, original.data(), original.size()*sizeof(T), cudaMemcpyHostToDevice, stream)); ck(cudaStreamSynchronize(stream)); }
    std::vector<T> read() {
        std::vector<T> v(original.size());
        ck(cudaMemcpyAsync(v.data(), raw, v.size()*sizeof(T), cudaMemcpyDeviceToHost, stream));
        ck(cudaStreamSynchronize(stream)); return v;
    }
    void immutable() { auto v=read(); insist(!memcmp(v.data(),original.data(),v.size()*sizeof(T)),"input/weight mutated"); }
    std::vector<T> output(size_t written) {
        auto v=read(); insist(written<=n,"output size");
        insist(!memcmp(v.data(),original.data(),pad*sizeof(T)),"prefix guard");
        insist(!memcmp(v.data()+pad+written,original.data()+pad+written,(n-written+pad)*sizeof(T)),"output tail/suffix guard");
        return std::vector<T>(v.begin()+pad,v.begin()+pad+written);
    }
};
#include <filesystem>
#include <fstream>
#include <algorithm>
struct Graph {
    cudaGraph_t graph{};
    cudaGraphExec_t exec{};
    cudaStream_t stream;
    Graph(cudaStream_t s): stream(s) {}
    ~Graph() { cudaStreamSynchronize(stream); if(exec)cudaGraphExecDestroy(exec); if(graph)cudaGraphDestroy(graph); }
    template<class F> void capture(F launch, bool batch) {
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        launch();
        ck(cudaStreamEndCapture(stream,&graph));
        ck(cudaGraphInstantiate(&exec,graph,0));
        size_t n=0; ck(cudaGraphGetNodes(graph,nullptr,&n));
        insist(n == (batch ? 1u : 2u), "launch census");
        std::vector<cudaGraphNode_t> nodes(n); ck(cudaGraphGetNodes(graph,nodes.data(),&n));
        for(auto node:nodes) {
            cudaGraphNodeType type{}; ck(cudaGraphNodeGetType(node,&type));
            insist(type==cudaGraphNodeTypeKernel,"kernel nodes only");
            cudaKernelNodeParams p{}; ck(cudaGraphKernelNodeGetParams(node,&p));
            const char* name=nullptr; ck(cudaFuncGetName(&name,p.func));
            insist(strstr(name,batch ? "dsv4_dense_batch_" : "dsv4_dense_exact_tail_")!=nullptr,"actual graph function");
            insist(p.blockDim.x==128 && p.blockDim.y==1 && p.blockDim.z==1,"128-leaf geometry");
        }
    }
    void launch() { ck(cudaGraphLaunch(exec,stream)); }
};
static std::vector<unsigned char> read_bytes(std::ifstream& f,size_t n) {
    std::vector<unsigned char> out(n);
    insist(bool(f.read((char*)out.data(),n)),"short operand tape"); return out;
}
static void equal(const std::vector<float>& a,const std::vector<float>& b) {
    insist(a.size()==b.size(),"output size");
    for(size_t i=0;i<a.size();++i) {
        insist(std::isfinite(a[i])&&std::isfinite(b[i]),"finite output");
        if(memcmp(&a[i],&b[i],4)) throw std::runtime_error("raw bit mismatch row="+std::to_string(i));
    }
}
__global__ void dense_batch_gate_flush(uint32_t* p,size_t n) {
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x)
        ((volatile uint32_t*)p)[i]=(uint32_t)i;
}
static double timed(Graph& graph,uint32_t* flush,bool cold) {
    cudaEvent_t start{},end{}; ck(cudaEventCreate(&start)); ck(cudaEventCreate(&end));
    double sum=0; const int reps=cold?8:32;
    for(int i=0;i<reps;++i) {
        if(cold) dense_batch_gate_flush<<<4096,256,0,graph.stream>>>(flush,256u*1024*1024/4);
        ck(cudaGetLastError());
        ck(cudaEventRecord(start,graph.stream)); graph.launch(); ck(cudaEventRecord(end,graph.stream));
        ck(cudaEventSynchronize(end)); float ms=0; ck(cudaEventElapsedTime(&ms,start,end)); sum+=ms;
    }
    ck(cudaEventDestroy(start)); ck(cudaEventDestroy(end)); return sum*1000/reps;
}
static void run_file(const std::filesystem::path& path,bool check_only) {
    std::ifstream f(path,std::ios::binary); uint32_t h[8]{};
    insist(bool(f.read((char*)h,sizeof(h))),"header");
    insist(h[0]==0x44534231 && h[1]<=1 && h[7]<2,"tape version/kind/device");
    const bool fp8=h[1]==0; const int k=h[2],na=h[3],nb=h[4];
    insist(k>0&&k%8==0&&k<=32768&&na>0&&nb>0&&na<=32768&&nb<=32768,"bounded shape");
    ck(cudaSetDevice(h[7])); Stream stream; const auto s=stream.value;
    auto hx=read_bytes(f,(size_t)k*(fp8?2:4));
    auto wa=read_bytes(f,(size_t)na*k*(fp8?1:(h[5]?2:4)));
    auto sa=fp8?read_bytes(f,(size_t)((na+127)/128)*h[5]*4):std::vector<unsigned char>();
    auto wb=read_bytes(f,(size_t)nb*k*(fp8?1:(h[6]?2:4)));
    auto sb=fp8?read_bytes(f,(size_t)((nb+127)/128)*h[6]*4):std::vector<unsigned char>();
    insist(f.peek()==EOF,"trailing tape bytes");
    Guard<unsigned char> x(hx.size(),s),a(wa.size(),s),b(wb.size(),s),asc(sa.size(),s),bsc(sb.size(),s);
    x.upload(hx);a.upload(wa);b.upload(wb);asc.upload(sa);bsc.upload(sb);
    Guard<float> ya(na,s),yb(nb,s),za(na,s),zb(nb,s);
    Dsv4DenseBatchFp8 pf[2]={{a.data(),(float*)asc.data(),za.data(),na,(int)h[5]},
        {b.data(),(float*)bsc.data(),zb.data(),nb,(int)h[6]}};
    Dsv4DenseBatchDots pd[2]={{a.data(),za.data(),na,(int)h[5]}, {b.data(),zb.data(),nb,(int)h[6]}};
    auto control=[&] {
        if(fp8) {
            api(memra_dsv4_dense_exact_tail_fp8(a.data(),(float*)asc.data(),h[5],x.data(),ya.data(),1,na,k,0,0,s));
            api(memra_dsv4_dense_exact_tail_fp8(b.data(),(float*)bsc.data(),h[6],x.data(),yb.data(),1,nb,k,0,0,s));
        } else {
            api(memra_dsv4_dense_exact_tail_dots((float*)x.data(),a.data(),h[5],ya.data(),1,na,k,s));
            api(memra_dsv4_dense_exact_tail_dots((float*)x.data(),b.data(),h[6],yb.data(),1,nb,k,s));
        }
    };
    auto batch=[&] {
        if(fp8) api(memra_dsv4_dense_batch_fp8(x.data(),k,pf,s));
        else api(memra_dsv4_dense_batch_dots((float*)x.data(),k,pd,s));
    };
    control();batch();ck(cudaStreamSynchronize(s));equal(ya.output(na),za.output(na));equal(yb.output(nb),zb.output(nb));
    Graph off(s),on(s);off.capture(control,false);on.capture(batch,true);
    for(int iteration=0;iteration<2;++iteration) {
        ya.reset();yb.reset();za.reset();zb.reset();off.launch();on.launch();ck(cudaStreamSynchronize(s));
        equal(ya.output(na),za.output(na));equal(yb.output(nb),zb.output(nb));
    }
    x.immutable();a.immutable();b.immutable();asc.immutable();bsc.immutable();
    uint64_t before[2],after[2]; api(memra_dsv4_dense_batch_counts_for_gate(before,before+1));
    if(fp8) {
        auto bad=pf[1];pf[1].y=pf[0].y;
        insist(memra_dsv4_dense_batch_fp8(x.data(),k,pf,s)==40075,"overlap refusal");pf[1]=bad;
        insist(memra_dsv4_dense_batch_fp8(x.data(),k-1,pf,s)==40075,"K refusal");
    } else {
        auto bad=pd[1];pd[1].y=pd[0].y;
        insist(memra_dsv4_dense_batch_dots((float*)x.data(),k,pd,s)==40075,"overlap refusal");pd[1]=bad;
        insist(memra_dsv4_dense_batch_dots((float*)x.data(),k-1,pd,s)==40075,"K refusal");
    }
    api(memra_dsv4_dense_batch_counts_for_gate(after,after+1));
    insist(before[0]==after[0]&&before[1]==after[1],"refusals enqueue nothing");
    double warm_off=0,warm_on=0,cold_off=0,cold_on=0;
    if(!check_only) {
        uint32_t* flush{};ck(cudaMalloc(&flush,256u*1024*1024));
        off.launch();on.launch();ck(cudaStreamSynchronize(s));
        warm_off=timed(off,flush,false);warm_on=timed(on,flush,false);
        cold_off=timed(off,flush,true);cold_on=timed(on,flush,true);ck(cudaFree(flush));
    }
    printf("COMPONENT file=%s device=%u kind=%s k=%d na=%d nb=%d raw_bits=equal guards=pass immutable=pass graph=pass refusals=pass launches_before=2 launches_after=1 timing=%s warm_off_us=%.6f warm_on_us=%.6f cold_off_us=%.6f cold_on_us=%.6f\n",
        path.filename().c_str(),h[7],fp8?"fp8":"dots",k,na,nb,check_only?"skipped":"events",warm_off,warm_on,cold_off,cold_on);fflush(stdout);
}
int main(int argc,char** argv) {
    try {
        insist(argc==2||(argc==3&&!strcmp(argv[2],"--check-only")),"usage: component tape-dir [--check-only]");
        std::vector<std::filesystem::path> paths;
        for(auto& p:std::filesystem::directory_iterator(argv[1])) if(p.path().extension()==".bin")paths.push_back(p.path());
        std::sort(paths.begin(),paths.end());insist(paths.size()==252,"252 captured layer-pairs required");
        for(auto& p:paths)run_file(p,argc==3);
        printf("PASS component captured_pairs=%zu raw_bits=equal\n",paths.size());return 0;
    } catch(const std::exception& e) {fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}

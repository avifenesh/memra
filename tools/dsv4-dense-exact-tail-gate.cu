// Standalone actual-kernel component. Build/run only on the owned dev pair.
// nvcc -t 2 -std=c++17 -O3 -fmad=false -Xcompiler=-ffp-contract=off
//   -arch=sm_120a -lineinfo -Xptxas=-v tools/dsv4-dense-exact-tail-gate.cu
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
struct Graph {
    cudaGraph_t graph{}; cudaGraphExec_t exec{}; cudaStream_t stream;
    explicit Graph(cudaStream_t s):stream(s){}
    ~Graph(){ cudaStreamSynchronize(stream); if(exec)cudaGraphExecDestroy(exec); if(graph)cudaGraphDestroy(graph); }
    template<class F> void capture(F&& launch, bool candidate) {
        ck(cudaStreamSynchronize(stream)); ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        try { launch(); } catch (...) { cudaStreamEndCapture(stream,&graph); throw; }
        ck(cudaStreamEndCapture(stream,&graph)); ck(cudaGraphInstantiate(&exec,graph,0));
        size_t count=0; ck(cudaGraphGetNodes(graph,nullptr,&count)); insist(count==1,"one kernel per component graph");
        cudaGraphNode_t node{}; ck(cudaGraphGetNodes(graph,&node,&count));
        cudaGraphNodeType type{}; ck(cudaGraphNodeGetType(node,&type)); insist(type==cudaGraphNodeTypeKernel,"kernel graph");
        cudaKernelNodeParams p{}; ck(cudaGraphKernelNodeGetParams(node,&p));
        const char* name=nullptr; ck(cudaFuncGetName(&name,p.func));
        insist((strstr(name,"dsv4_dense_exact_tail_")!=nullptr)==candidate,"actual candidate node identity");
    }
    void launch(){ ck(cudaGraphLaunch(exec,stream)); }
};
static std::array<uint64_t,2> counts() { std::array<uint64_t,2> c{}; api(memra_dsv4_dense_exact_tail_counts_for_gate(&c[0],&c[1]));return c; }
static uint16_t bf(float x) { uint32_t b; memcpy(&b,&x,4); return uint16_t(b>>16); }
static std::vector<float> floats(size_t n,int salt) {
    std::vector<float> v(n); for(size_t i=0;i<n;++i)v[i]=float(int((i*13+salt*7)%97)-48)/64.0f;return v;
}
static void equal(const std::vector<float>& a,const std::vector<float>& b) {
    insist(a.size()==b.size(),"output lengths");
    for(size_t i=0;i<a.size();++i){insist(std::isfinite(a[i])&&std::isfinite(b[i]),"finite output");
        if(memcmp(&a[i],&b[i],4))throw std::runtime_error("output bits differ row="+std::to_string(i));}
}
static void fp8_case(int n,int k,int pattern) {
    Stream s; int sc_cols=(k+127)/128;
    Guard<uint8_t> w(size_t(n)*k,s.value); Guard<float> sc(size_t((n+127)/128)*sc_cols,s.value);
    Guard<uint16_t> x(k,s.value); Guard<float> a(n+7,s.value),b(n+7,s.value);
    std::vector<uint8_t> weights(size_t(n)*k);
    for(size_t i=0;i<weights.size();++i) weights[i]=pattern?0x38:uint8_t((i*37+19)%256);
    w.upload(weights); auto scales=floats(sc.n,1);
    for(size_t i=0;i<scales.size();++i)scales[i]=pattern?1.0f:ldexpf(1.0f,int(i%9)-4);
    sc.upload(scales);
    auto input=[&](int generation){auto f=floats(k,generation);std::vector<uint16_t> v(k);
        for(int i=0;i<k;++i){
            if(pattern==1){int chunk=i/1024;int element=i%8;f[i]=element?1.0f:(chunk%2?-0x1p30f:0x1p30f);}
            if(pattern==2)f[i]=(i%2)?-0.0f:0.0f;
            v[i]=bf(f[i]);
        }return v;};
    x.upload(input(1));
    auto launch=[&](float* y){api(memra_dsv4_gemv_fp8_m(w.data(),sc.data(),sc_cols,x.data(),y,1,n,k,0,0,s.value));};
    api(memra_dsv4_dense_exact_tail_set_for_gate(0)); auto c0=counts();launch(a.data());ck(cudaStreamSynchronize(s.value));insist(counts()==c0,"OFF enqueued candidate");
    api(memra_dsv4_dense_exact_tail_set_for_gate(1)); launch(b.data());ck(cudaStreamSynchronize(s.value));auto c1=counts();insist(c1[0]==c0[0]+1&&c1[1]==c0[1],"FP8 candidate enqueue");
    equal(a.output(n),b.output(n));
    Graph ga(s.value),gb(s.value);
    api(memra_dsv4_dense_exact_tail_set_for_gate(0));ga.capture([&]{launch(a.data());},false);
    api(memra_dsv4_dense_exact_tail_set_for_gate(1));gb.capture([&]{launch(b.data());},true);
    auto captured=counts();
    for(int generation:{3,7}){
        x.upload(input(generation));a.reset();b.reset();
        // Reverse the current host selector to prove retained graph selection is immutable.
        api(memra_dsv4_dense_exact_tail_set_for_gate(1));ga.launch();
        api(memra_dsv4_dense_exact_tail_set_for_gate(0));gb.launch();ck(cudaStreamSynchronize(s.value));
        equal(a.output(n),b.output(n));
        auto prior=b.output(n);gb.launch();ck(cudaStreamSynchronize(s.value));equal(prior,b.output(n));
    }
    insist(counts()==captured,"host counter must not masquerade as graph replay count");
    w.immutable();sc.immutable();x.immutable();
    printf("PASS fp8 n=%d k=%d pattern=%d bits=1 guards=1 immutable_inputs=1 changing_input_graphs=2 fixed_node_selection=1\n",n,k,pattern);fflush(stdout);
}
static void dots_case(int n,int k,int w_bf16,int pattern){
    Stream s; Guard<float> x(k,s.value),a(n+7,s.value),b(n+7,s.value);
    Guard<uint8_t> w(size_t(n)*k*(w_bf16?2:4),s.value);
    std::vector<uint8_t> weights(w.n);
    for(size_t i=0;i<size_t(n)*k;++i){float f=pattern?1.0f:float(int((i*17+5)%101)-50)/64.0f;
        if(w_bf16){auto h=bf(f);memcpy(weights.data()+2*i,&h,2);}else memcpy(weights.data()+4*i,&f,4);}
    w.upload(weights);
    auto input=[&](int generation){auto f=floats(k,generation);
        if(pattern)for(int i=0;i<k;++i){
            if(pattern==1)f[i]=(i%8)?1.0f:((i/1024)%2?-0x1p30f:0x1p30f);
            else f[i]=(i%2)?-0.0f:0.0f;
        }return f;};
    x.upload(input(1));auto launch=[&](float* y){api(memra_dsv4_dots_f32acc_mrow(x.data(),w.data(),w_bf16,y,1,k,n,s.value));};
    api(memra_dsv4_dense_exact_tail_set_for_gate(0));auto c0=counts();launch(a.data());ck(cudaStreamSynchronize(s.value));insist(counts()==c0,"OFF dots candidate");
    api(memra_dsv4_dense_exact_tail_set_for_gate(1));launch(b.data());ck(cudaStreamSynchronize(s.value));auto c1=counts();insist(c1[1]==c0[1]+1&&c1[0]==c0[0],"dots candidate enqueue");equal(a.output(n),b.output(n));
    Graph ga(s.value),gb(s.value);api(memra_dsv4_dense_exact_tail_set_for_gate(0));ga.capture([&]{launch(a.data());},false);
    api(memra_dsv4_dense_exact_tail_set_for_gate(1));gb.capture([&]{launch(b.data());},true);auto captured=counts();
    for(int generation:{3,7}){x.upload(input(generation));a.reset();b.reset();api(memra_dsv4_dense_exact_tail_set_for_gate(1));ga.launch();api(memra_dsv4_dense_exact_tail_set_for_gate(0));gb.launch();ck(cudaStreamSynchronize(s.value));equal(a.output(n),b.output(n));auto prior=b.output(n);gb.launch();ck(cudaStreamSynchronize(s.value));equal(prior,b.output(n));}
    insist(counts()==captured,"graph count domain");w.immutable();x.immutable();
    printf("PASS dots n=%d k=%d bf16=%d pattern=%d bits=1 guards=1 immutable_inputs=1 changing_input_graphs=2 fixed_node_selection=1\n",n,k,w_bf16,pattern);fflush(stdout);
}
// A tree-only tape distinguishes the canonical halving association from warp-first
// sums, independently of the common product-loop body. Expected values use volatile
// float nodes, preserving each rounding boundary on the host too.
__global__ void dense_tail_tree_probe(const float* in,float* out){
    __shared__ float p[128];p[threadIdx.x]=in[threadIdx.x];__syncthreads();
    if(threadIdx.x<32){float v=dsv4_dense_exact_tail_reduce(p);if(!threadIdx.x)*out=v;}
}
static void tree_case(){
    Stream s;Guard<float> p(128,s.value),y(3,s.value);std::vector<float> v(128,0.0f);
    v[0]=0x1p30f;v[32]=-0x1p30f;v[64]=1.0f;v[96]=1.0f;
    volatile float tree[128];for(int i=0;i<128;++i)tree[i]=v[i];
    for(int off=64;off;off>>=1)for(int i=0;i<off;++i)tree[i]=tree[i]+tree[i+off];
    float expected=tree[0];insist(expected==0.0f,"canonical tree cancellation witness");
    p.upload(v);dense_tail_tree_probe<<<1,128,0,s.value>>>(p.data(),y.data());ck(cudaGetLastError());auto got=y.output(1);insist(!memcmp(&got[0],&expected,4),"tree association witness");p.immutable();
    printf("PASS canonical128_tree_cancellation expected=0 warp_first_would_be=2\n");
}
static void admission(){
    Stream s;Guard<uint8_t>w(129*256*4,s.value);Guard<float>x(33*256,s.value),sc(258,s.value),a(33*129+7,s.value),b(33*129+7,s.value);
    w.upload(std::vector<uint8_t>(w.n,0));x.upload(std::vector<float>(x.n,0));sc.upload(std::vector<float>(sc.n,1));
    api(memra_dsv4_dense_exact_tail_set_for_gate(0));auto c=counts();
    insist(memra_dsv4_dense_exact_tail_set_for_gate(2)==40075,"selector refuses typo");
    insist(!dsv4_dense_exact_tail_fp8_admits(w.data()+1,sc.data(),2,x.data(),a.data(),1,129,256),"FP8 alignment");
    insist(!dsv4_dense_exact_tail_dots_admits(x.data()+1,w.data(),0,a.data(),1,129,256),"dots alignment");
    insist(memra_dsv4_dense_exact_tail_fp8(w.data(),sc.data(),1,x.data(),a.data(),1,129,256,0,0,s.value)==40075,"scale bounds");
    insist(memra_dsv4_dense_exact_tail_fp8(w.data(),sc.data(),2,x.data(),a.data(),2,129,256,0,0,s.value)==40075,"M>1 raw refusal");
    insist(memra_dsv4_dense_exact_tail_dots(x.data(),w.data(),2,a.data(),1,129,256,s.value)==40075,"weight type");
    insist(memra_dsv4_dense_exact_tail_dots(x.data(),w.data(),0,a.data(),1,129,255,s.value)==40075,"K alignment");
    a.immutable();insist(counts()==c,"refusal enqueued");
    for(int m:{2,33}){
        api(memra_dsv4_dense_exact_tail_set_for_gate(0));api(memra_dsv4_gemv_fp8_m(w.data(),sc.data(),2,x.data(),a.data(),m,129,256,0,0,s.value));
        api(memra_dsv4_dense_exact_tail_set_for_gate(1));api(memra_dsv4_gemv_fp8_m(w.data(),sc.data(),2,x.data(),b.data(),m,129,256,0,0,s.value));ck(cudaStreamSynchronize(s.value));equal(a.output(m*129),b.output(m*129));a.reset();b.reset();
        api(memra_dsv4_dense_exact_tail_set_for_gate(0));api(memra_dsv4_dots_f32acc_mrow(x.data(),w.data(),0,a.data(),m,256,129,s.value));
        api(memra_dsv4_dense_exact_tail_set_for_gate(1));api(memra_dsv4_dots_f32acc_mrow(x.data(),w.data(),0,b.data(),m,256,129,s.value));ck(cudaStreamSynchronize(s.value));equal(a.output(m*129),b.output(m*129));a.reset();b.reset();
    }
    api(memra_dsv4_dense_exact_tail_set_for_gate(0));
    api(memra_dsv4_gemv_fp8_grouped_m1(w.data(),sc.data(),2,x.data(),a.data(),2,128,256,256,128,s.value));
    api(memra_dsv4_dense_exact_tail_set_for_gate(1));
    api(memra_dsv4_gemv_fp8_grouped_m1(w.data(),sc.data(),2,x.data(),b.data(),2,128,256,256,128,s.value));ck(cudaStreamSynchronize(s.value));equal(a.output(256),b.output(256));
    insist(counts()==c,"M>1/recursive/grouped used candidate");w.immutable();x.immutable();sc.immutable();api(memra_dsv4_dense_exact_tail_set_for_gate(0));
    printf("PASS admission raw_refusals=1 M2_M33_grouped_control=1 candidate_enqueues=0\n");
}
int main(int argc, char** argv) try {
    // This gate measures exact-tail, not the newer dense-fast implementation.
    api(memra_dsv4_dense_fast_set_for_gate(0));
    insist(memra_dsv4_dense_fast_enabled_for_gate()==0,"dense-fast control override");
    if(argc==2 && !strcmp(argv[1],"--check-controls")) {
        puts("PASS exact_tail_control dense_fast=0 cpu_policy_only=true"); return 0;
    }
    int n=0;ck(cudaGetDeviceCount(&n));insist(n==2,"requires exact visible pair");
    for(int rank=0;rank<2;++rank){ck(cudaSetDevice(rank));tree_case();admission();
        for(auto shape:std::vector<std::pair<int,int>>{{1024,4096},{16384,1024},{512,4096},{4096,4096},{2048,4096},{4096,2048},{8192,1024}})fp8_case(shape.first,shape.second,0);
        for(auto shape:std::vector<std::pair<int,int>>{{24,16384},{1024,4096},{512,4096},{256,4096},{4,16384},{129280,4096}})dots_case(shape.first,shape.second,shape.first==129280,0);
        for(int k:{8,120,128,136,1016,1024,1032,2040,2048,2056,4096,8192}){
            int rows=129;fp8_case(rows,k,0);dots_case(rows,k,0,0);dots_case(rows,k,1,0);
        }
        for(int rows:{1,63,64,65,127,128}){fp8_case(rows,256,0);dots_case(rows,256,0,0);dots_case(rows,256,1,0);}
        for(int pattern:{1,2}){fp8_case(129,4096,pattern);dots_case(129,4096,0,pattern);dots_case(129,4096,1,pattern);}
        printf("PASS rank=%d full_shapes_edges_graphs_admission=1\n",rank);fflush(stdout);
    }
    printf("COMPLETE dense_exact_tail component_only=true no_model_performance_claim=true\n");return 0;
}catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}

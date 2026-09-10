// Standalone dense-fast component gate. Existing gate supplies guarded buffers
// and the canonical cancellation witness. Its main is never invoked here.
// Build with nice10 nvcc -t 2 -std=c++17 -O3 -fmad=false
// -Xcompiler=-ffp-contract=off -arch=sm_120a -lineinfo -Xptxas=-v
// tools/dsv4-dense-fast-gate.cu -lcublasLt -lcublas -ldl -o <empty-target>/component
#define main dense_tail_legacy_main
#include "dsv4-dense-exact-tail-gate.cu"
#undef main
#include <algorithm>
#include <filesystem>
#include <fstream>

struct FastGraph {
    cudaGraph_t graph{}; cudaGraphExec_t exec{}; cudaStream_t stream;
    template<class F> FastGraph(cudaStream_t s,F launch,bool on):stream(s) {
        ck(cudaStreamSynchronize(stream));
        ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal)); launch();
        ck(cudaStreamEndCapture(stream,&graph)); ck(cudaGraphInstantiate(&exec,graph,0));
        size_t count=0;ck(cudaGraphGetNodes(graph,nullptr,&count));insist(count==1,"single component kernel");
        cudaGraphNode_t node{};ck(cudaGraphGetNodes(graph,&node,&count));
        cudaKernelNodeParams params{};ck(cudaGraphKernelNodeGetParams(node,&params));
        const char* name=nullptr;ck(cudaFuncGetName(&name,params.func));
        insist((strstr(name,"dsv4_dense_fast_")!=nullptr)==on,"actual fast function selection");
        insist((strstr(name,"dsv4_dense_exact_tail_")!=nullptr)==!on,"actual OFF function selection");
        cudaFuncAttributes attr{};ck(cudaFuncGetAttributes(&attr,params.func));
        int blocks=0,threads=params.blockDim.x;ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,params.func,threads,0));
        printf("RESOURCE on=%d registers=%d shared=%zu local=%zu blocks_per_sm=%d resident_warps=%d grid=%d threads=%d symbol=%s\n",
               on,attr.numRegs,attr.sharedSizeBytes,attr.localSizeBytes,blocks,blocks*threads/32,params.gridDim.x,threads,name);
    }
    ~FastGraph(){cudaStreamSynchronize(stream);cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);}
    void launch(){ck(cudaGraphLaunch(exec,stream));}
};
struct Case {int rank,kind,n,k,sc_cols;std::vector<uint8_t> w,x;std::vector<float> sc;};
static Case read_case(const std::filesystem::path& path) {
    std::ifstream f(path,std::ios::binary);insist(bool(f),"operand file open");
    char magic[8];f.read(magic,8);insist(!memcmp(magic,"DENSEF01",8),"operand format");
    int h[5];f.read((char*)h,sizeof h);Case c{h[0],h[1],h[2],h[3],h[4],{},{},{}};
    insist(c.rank>=0&&c.rank<2&&c.kind>=0&&c.kind<3&&c.n>0&&c.k>0&&c.k%8==0,"operand header");
    c.w.resize(size_t(c.n)*c.k*(c.kind==0?1:c.kind==1?4:2));
    c.x.resize(size_t(c.k)*(c.kind==0?2:4));
    if(c.kind==0)c.sc.resize(size_t((c.n+127)/128)*c.sc_cols);
    f.read((char*)c.w.data(),c.w.size());f.read((char*)c.x.data(),c.x.size());
    f.read((char*)c.sc.data(),c.sc.size()*4);insist(bool(f)&&f.peek()==EOF,"exact operand file size");return c;
}
static Case synthetic(int rank,int kind,int n,int k) {
    Case c{rank,kind,n,k,(k+127)/128,{},{},{}};
    c.w.resize(size_t(n)*k*(kind==0?1:kind==1?4:2));c.x.resize(size_t(k)*(kind==0?2:4));
    for(size_t i=0;i<size_t(n)*k;++i){
        if(kind==0)c.w[i]=uint8_t((i*37+19)%256);
        else {float v=float(int((i*17+5)%101)-50)/64.0f;
            if(kind==1)memcpy(c.w.data()+i*4,&v,4);else{uint16_t b=bf(v);memcpy(c.w.data()+i*2,&b,2);}}
    }
    for(int i=0;i<k;++i){float v=float((i*13)%97-48)/64.0f;
        if(kind==0){uint16_t b=bf(v);memcpy(c.x.data()+i*2,&b,2);}else memcpy(c.x.data()+i*4,&v,4);}
    if(kind==0)c.sc.assign(size_t((n+127)/128)*c.sc_cols,1.0f);return c;
}
static void run_case(Case& c, bool timing, bool real) {
    ck(cudaSetDevice(c.rank));Stream s;
    Guard<uint8_t> w(c.w.size(),s.value),x(c.x.size(),s.value);
    Guard<float> sc(std::max(size_t(1),c.sc.size()),s.value),a(c.n+7,s.value),b(c.n+7,s.value);
    w.upload(c.w);x.upload(c.x);sc.upload(c.sc.empty()?std::vector<float>{1.0f}:c.sc);
    auto launch=[&](float* y){api(c.kind==0?
        memra_dsv4_dense_exact_tail_fp8(w.data(),sc.data(),c.sc_cols,x.data(),y,1,c.n,c.k,0,0,s.value):
        memra_dsv4_dense_exact_tail_dots((float*)x.data(),w.data(),c.kind==2,y,1,c.n,c.k,s.value));};
    api(memra_dsv4_dense_fast_set_for_gate(0));launch(a.data());ck(cudaStreamSynchronize(s.value));
    api(memra_dsv4_dense_fast_set_for_gate(1));launch(b.data());ck(cudaStreamSynchronize(s.value));equal(a.output(c.n),b.output(c.n));
    api(memra_dsv4_dense_fast_set_for_gate(0));FastGraph ga(s.value,[&]{launch(a.data());},false);
    api(memra_dsv4_dense_fast_set_for_gate(1));FastGraph gb(s.value,[&]{launch(b.data());},true);
    uint64_t fc=0,dc=0;api(memra_dsv4_dense_fast_counts_for_gate(&fc,&dc));
    for(int gen=0;gen<2;++gen){
        // Retained graphs must ignore a later selector change and use current operands.
        if(gen){auto changed=c.x;const size_t item=c.kind==0?2:4;
            for(size_t i=0;i<changed.size();i+=item)changed[i+item-1]^=0x80;x.upload(changed);}
        a.reset();b.reset();api(memra_dsv4_dense_fast_set_for_gate(1));ga.launch();
        api(memra_dsv4_dense_fast_set_for_gate(0));gb.launch();ck(cudaStreamSynchronize(s.value));equal(a.output(c.n),b.output(c.n));
    }
    x.upload(c.x);
    if(timing){
        void* flush=nullptr;ck(cudaMalloc(&flush,256ull<<20));cudaEvent_t start{},end{};ck(cudaEventCreate(&start));ck(cudaEventCreate(&end));
        for(int i=0;i<32;++i){ga.launch();gb.launch();}ck(cudaStreamSynchronize(s.value));
        for(int cold=0;cold<2;++cold)for(int rep=0;rep<50;++rep)for(int slot=0;slot<2;++slot){
            int on=(rep%2)^slot;
            if(cold)ck(cudaMemsetAsync(flush,rep,256ull<<20,s.value));
            ck(cudaEventRecord(start,s.value));if(on)gb.launch();else ga.launch();
            ck(cudaEventRecord(end,s.value));ck(cudaEventSynchronize(end));float ms=0;ck(cudaEventElapsedTime(&ms,start,end));
            size_t bytes=c.w.size()+c.x.size()+c.sc.size()*4+size_t(c.n)*4;
            printf("TIME rank=%d kind=%d n=%d k=%d real=%d on=%d cold=%d rep=%d us=%.6f bytes=%zu tensor_gbs=%.6f\n",
                   c.rank,c.kind,c.n,c.k,real,on,cold,rep,ms*1000.0,bytes,bytes/(ms*1.e6));
        }
        ck(cudaFree(flush));ck(cudaEventDestroy(start));ck(cudaEventDestroy(end));
    }
    uint64_t f2=0,d2=0;api(memra_dsv4_dense_fast_counts_for_gate(&f2,&d2));insist(fc==f2&&dc==d2,"replays are not host enqueues");
    w.immutable();x.immutable();sc.immutable();equal(a.output(c.n),b.output(c.n));
    printf("PASS rank=%d kind=%d n=%d k=%d real=%d bits=1 guards=1 immutable=1 retained_graphs=1\n",c.rank,c.kind,c.n,c.k,real);fflush(stdout);
}
int main(int argc,char** argv){try{
    api(memra_dsv4_hc_dot_split_set_for_gate(0));
    insist(memra_dsv4_hc_dot_split_slices_for_gate()==0,"HC split control override");
    bool reverse=false,timing=true;std::string dir;
    for(int i=1;i<argc;++i){std::string a=argv[i];if(a=="--reverse")reverse=true;else if(a=="--check-only")timing=false;else if(dir.empty())dir=a;else throw std::runtime_error("usage: component [operand-dir] [--reverse] [--check-only]");}
    int devices=0;ck(cudaGetDeviceCount(&devices));insist(devices==2,"two GPUs");
    if(!dir.empty()){
        std::vector<std::filesystem::path> files;for(auto& e:std::filesystem::directory_iterator(dir))if(e.path().extension()==".bin")files.push_back(e.path());
        std::sort(files.begin(),files.end());if(reverse)std::reverse(files.begin(),files.end());insist(files.size()==24,"24 actual rank/shape cases");
        for(auto& path:files){auto c=read_case(path);run_case(c,timing,true);}
    }else{
        const int shapes[][3]={{0,512,4096},{0,1024,4096},{0,2048,4096},{0,4096,2048},{0,4096,4096},{0,8192,1024},{0,16384,1024},
            {1,4,16384},{1,24,16384},{1,256,4096},{1,512,4096},{1,1024,4096},{2,129280,4096}};
        for(int rank=0;rank<2;++rank)for(int i=0;i<13;++i){auto sh=shapes[reverse?12-i:i];auto c=synthetic(rank,sh[0],sh[1],sh[2]);run_case(c,timing,false);}
    }
    // Tail and K boundaries cover partial two-row tiles and unroll remainders.
    for(int rank=0;rank<2;++rank){ck(cudaSetDevice(rank));tree_case();
        for(int kind=0;kind<3;++kind)for(int k:{8,1016,1024,1032,2048,4096}){auto c=synthetic(rank,kind,3,k);run_case(c,false,false);}}
    puts("PASS dense_fast_components");return 0;
}catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}

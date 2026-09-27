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
// memra #710: two matrices over the same x rows in one pair launch give each matrix the bits of
// its own launch, and the capture holds one pair kernel. kind 0 = FP8, 2 = BF16 dots.
static void pair_case(int rank,int kind,int na,int nb,int k,int m){
    ck(cudaSetDevice(rank));Stream s;
    Case a=synthetic(rank,kind,na,k),b=synthetic(rank,kind,nb,k);
    const size_t item=kind==0?1:2;  // b: a's rows reversed, element by element
    {auto w=b.w;const size_t count=w.size()/item;
        for(size_t i=0;i<count;++i)memcpy(b.w.data()+i*item,w.data()+(count-1-i)*item,item);}
    const size_t xitem=kind==0?2:4;std::vector<uint8_t> xs(size_t(m)*k*xitem);
    for(int r=0;r<m;++r)for(int i=0;i<k;++i)memcpy(xs.data()+(size_t(r)*k+i)*xitem,a.x.data()+size_t((i+7*r)%k)*xitem,xitem);
    Guard<uint8_t> wa(a.w.size(),s.value),wb(b.w.size(),s.value),x(xs.size(),s.value);
    Guard<float> sa(std::max(size_t(1),a.sc.size()),s.value),sb(std::max(size_t(1),b.sc.size()),s.value);
    Guard<float> ya1(size_t(m)*na,s.value),yb1(size_t(m)*nb,s.value),ya2(size_t(m)*na,s.value),yb2(size_t(m)*nb,s.value);
    wa.upload(a.w);wb.upload(b.w);x.upload(xs);
    sa.upload(a.sc.empty()?std::vector<float>{1.0f}:a.sc);sb.upload(b.sc.empty()?std::vector<float>{1.0f}:b.sc);
    api(memra_dsv4_dense_fast_set_for_gate(1));
    if(kind==0){
        api(memra_dsv4_gemv_fp8_m(wa.data(),sa.data(),a.sc_cols,x.data(),ya1.data(),m,na,k,0,0,s.value));
        api(memra_dsv4_gemv_fp8_m(wb.data(),sb.data(),b.sc_cols,x.data(),yb1.data(),m,nb,k,0,0,s.value));
    }else{
        api(memra_dsv4_dots_f32acc_mrow((float*)x.data(),wa.data(),1,ya1.data(),m,k,na,s.value));
        api(memra_dsv4_dots_f32acc_mrow((float*)x.data(),wb.data(),1,yb1.data(),m,k,nb,s.value));
    }
    auto pair=[&]{api(kind==0?
        memra_dsv4_gemv_fp8_m_pair(wa.data(),sa.data(),a.sc_cols,ya2.data(),na,0,wb.data(),sb.data(),b.sc_cols,yb2.data(),nb,0,x.data(),m,k,0,s.value):
        memra_dsv4_dots_f32acc_mrow_pair((float*)x.data(),wa.data(),ya2.data(),na,wb.data(),yb2.data(),nb,1,m,k,s.value));};
    pair();ck(cudaStreamSynchronize(s.value));
    equal(ya1.output(size_t(m)*na),ya2.output(size_t(m)*na));equal(yb1.output(size_t(m)*nb),yb2.output(size_t(m)*nb));
    cudaGraph_t g{};ck(cudaStreamBeginCapture(s.value,cudaStreamCaptureModeThreadLocal));pair();ck(cudaStreamEndCapture(s.value,&g));
    size_t count=0;ck(cudaGraphGetNodes(g,nullptr,&count));insist(count==1,"one pair kernel node");
    cudaGraphNode_t node{};ck(cudaGraphGetNodes(g,&node,&count));cudaKernelNodeParams params{};ck(cudaGraphKernelNodeGetParams(node,&params));
    const char* name=nullptr;ck(cudaFuncGetName(&name,params.func));insist(strstr(name,"_pair")!=nullptr,"the pair kernel ran");
    ck(cudaGraphDestroy(g));
    printf("PASS pair rank=%d kind=%d na=%d nb=%d k=%d m=%d bits=1 one_launch=1 symbol=%s\n",rank,kind,na,nb,k,m,name);fflush(stdout);
}
// memra #710: the shared expert's owner-gated launches. With the owner word set, each output has
// the bits of its ungated launch (the pair, the M-row launch into a plane at an offset, SwiGLU
// then the bf16 pack, the bf16 pack); with it clear, no output byte moves.
static void gated_case(int rank,int na,int k,int m){
    ck(cudaSetDevice(rank));Stream s;
    Case a=synthetic(rank,0,na,k),b=synthetic(rank,0,na,k),d=synthetic(rank,0,k,na);
    {auto w=b.w;const size_t count=w.size();for(size_t i=0;i<count;++i)b.w[i]=w[count-1-i];}
    std::vector<uint8_t> xs(size_t(m)*k*2),hs(size_t(m)*na*2);
    for(int r=0;r<m;++r)for(int i=0;i<k;++i)memcpy(xs.data()+(size_t(r)*k+i)*2,a.x.data()+size_t((i+7*r)%k)*2,2);
    for(int r=0;r<m;++r)for(int i=0;i<na;++i)memcpy(hs.data()+(size_t(r)*na+i)*2,d.x.data()+size_t((i+5*r)%na)*2,2);
    Guard<uint8_t> wa(a.w.size(),s.value),wb(b.w.size(),s.value),wd(d.w.size(),s.value),x(xs.size(),s.value),h(hs.size(),s.value);
    Guard<float> sa(a.sc.size(),s.value),sb(b.sc.size(),s.value),sd(d.sc.size(),s.value);
    wa.upload(a.w);wb.upload(b.w);wd.upload(d.w);x.upload(xs);h.upload(hs);sa.upload(a.sc);sb.upload(b.sc);sd.upload(d.sc);
    Guard<int> on(1,s.value),off(1,s.value);on.upload({1});off.upload({0});
    const size_t plane=size_t(m)*k,skip=size_t(3)*k;  // the shared rows after three routed rows
    Guard<float> ya1(size_t(m)*na,s.value),yb1(size_t(m)*na,s.value),ya2(size_t(m)*na,s.value),yb2(size_t(m)*na,s.value);
    Guard<float> yd1(plane,s.value),yd2(skip+plane,s.value),yd3(skip+plane,s.value);
    Guard<float> ya3(size_t(m)*na,s.value),yb3(size_t(m)*na,s.value);
    api(memra_dsv4_dense_fast_set_for_gate(1));
    api(memra_dsv4_gemv_fp8_m_pair(wa.data(),sa.data(),a.sc_cols,ya1.data(),na,0,wb.data(),sb.data(),b.sc_cols,yb1.data(),na,0,x.data(),m,k,0,s.value));
    api(memra_dsv4_gemv_fp8_m_pair_gated(wa.data(),sa.data(),a.sc_cols,ya2.data(),na,wb.data(),sb.data(),b.sc_cols,yb2.data(),na,x.data(),m,k,on.data(),s.value));
    api(memra_dsv4_gemv_fp8_m_pair_gated(wa.data(),sa.data(),a.sc_cols,ya3.data(),na,wb.data(),sb.data(),b.sc_cols,yb3.data(),na,x.data(),m,k,off.data(),s.value));
    api(memra_dsv4_gemv_fp8_m(wd.data(),sd.data(),d.sc_cols,h.data(),yd1.data(),m,k,na,0,0,s.value));
    api(memra_dsv4_gemv_fp8_m_gated(wd.data(),sd.data(),d.sc_cols,h.data(),yd2.data()+skip,m,k,na,k,on.data(),s.value));
    api(memra_dsv4_gemv_fp8_m_gated(wd.data(),sd.data(),d.sc_cols,h.data(),yd3.data()+skip,m,k,na,k,off.data(),s.value));
    ck(cudaStreamSynchronize(s.value));
    equal(ya1.output(size_t(m)*na),ya2.output(size_t(m)*na));equal(yb1.output(size_t(m)*na),yb2.output(size_t(m)*na));
    {auto full=yd2.output(skip+plane);std::vector<float> rows(full.begin()+skip,full.end());equal(yd1.output(plane),rows);}
    ya3.immutable();yb3.immutable();yd3.immutable();
    // SwiGLU then the pack, against the two launches it fuses; values past the limit clamp.
    const long n=long(m)*na;const float limit=7.0f;std::vector<float> g(n),u(n);
    for(long i=0;i<n;++i){g[i]=float(int((i*29+3)%211)-105)/9.0f;u[i]=float(int((i*31+11)%197)-98)/8.0f;}
    Guard<float> gg(n,s.value),uu(n,s.value),f32(n,s.value);gg.upload(g);uu.upload(u);
    Guard<uint16_t> p1(n,s.value),p2(n,s.value),p3(n,s.value),c1(n,s.value),c2(n,s.value),c3(n,s.value);
    api(memra_dsv4_swiglu(gg.data(),uu.data(),f32.data(),m,na,limit,nullptr,s.value));
    api(memra_dsv4_cvt_bf16(f32.data(),p1.data(),n,s.value));
    api(memra_dsv4_swiglu_bf16_gated(gg.data(),uu.data(),p2.data(),m,na,limit,on.data(),s.value));
    api(memra_dsv4_swiglu_bf16_gated(gg.data(),uu.data(),p3.data(),m,na,limit,off.data(),s.value));
    api(memra_dsv4_cvt_bf16(gg.data(),c1.data(),n,s.value));
    api(memra_dsv4_cvt_bf16_gated(gg.data(),c2.data(),n,on.data(),s.value));
    api(memra_dsv4_cvt_bf16_gated(gg.data(),c3.data(),n,off.data(),s.value));
    ck(cudaStreamSynchronize(s.value));
    insist(p1.output(n)==p2.output(n),"gated SwiGLU pack bits");insist(c1.output(n)==c2.output(n),"gated pack bits");
    p3.immutable();c3.immutable();
    printf("PASS gated rank=%d na=%d k=%d m=%d bits=1 owner_off_untouched=1\n",rank,na,k,m);fflush(stdout);
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
    // The pairs the TP/EP step launches: shared-expert gate and up, wq_a and wkv, the compressor's
    // kv and gate dots, at the one-token, B-row and verify widths.
    for(int rank=0;rank<2;++rank)for(int m:{1,2,4,6}){
        pair_case(rank,0,2048,2048,4096,m);pair_case(rank,0,1024,512,4096,m);
        pair_case(rank,2,1024,1024,4096,m);pair_case(rank,2,256,256,4096,m);}
    // The shared expert's owner-gated launches at its shape (inter 2048, hidden 4096).
    for(int rank=0;rank<2;++rank)for(int m:{1,2,4,8})gated_case(rank,2048,4096,m);
    puts("PASS dense_fast_components");return 0;
}catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}

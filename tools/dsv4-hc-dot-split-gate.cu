// Real-operand HC24 split components, same standalone build flags as dense-fast.
#define main hc_dense_fast_unused_main
#include "dsv4-dense-exact-tail-gate.cu"
#undef main
#include <algorithm>
#include <filesystem>
#include <fstream>
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

struct HcGraph {
    cudaGraph_t graph{}; cudaGraphExec_t exec{}; cudaStream_t stream;
    template<class F> HcGraph(cudaStream_t s, F enqueue, int slices):stream(s) {
        ck(cudaStreamSynchronize(s));
        ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal)); enqueue();
        ck(cudaStreamEndCapture(s,&graph)); ck(cudaGraphInstantiate(&exec,graph,0));
        size_t count=0; ck(cudaGraphGetNodes(graph,nullptr,&count));
        insist(count==size_t(slices?2:1),"component graph node count");
        std::vector<cudaGraphNode_t> nodes(count); ck(cudaGraphGetNodes(graph,nodes.data(),&count));
        int partials=0,reductions=0,control=0;
        for(auto node:nodes) {
            cudaKernelNodeParams p{}; ck(cudaGraphKernelNodeGetParams(node,&p));
            const char* name=nullptr; ck(cudaFuncGetName(&name,p.func));
            partials+=strstr(name,"dsv4_hc_dot_split_partial_kernel")!=nullptr;
            reductions+=strstr(name,"dsv4_hc_dot_split_reduce_kernel")!=nullptr;
            control+=strstr(name,"dsv4_dense_fast_dots_kernel")!=nullptr;
            cudaFuncAttributes attr{}; ck(cudaFuncGetAttributes(&attr,p.func));
            int blocks=0; ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,p.func,p.blockDim.x,0));
            cudaDeviceProp prop{}; int dev; ck(cudaGetDevice(&dev)); ck(cudaGetDeviceProperties(&prop,dev));
            printf("RESOURCE rank=%d slices=%d sm_count=%d grid=%u threads=%u registers=%d active_blocks_per_sm=%d grid_sm_coverage=%.6f symbol=%s\n",dev,slices,prop.multiProcessorCount,p.gridDim.x,p.blockDim.x,attr.numRegs,blocks,double(std::min(p.gridDim.x,unsigned(prop.multiProcessorCount)))/prop.multiProcessorCount,name);
        }
        insist(partials==int(slices!=0)&&reductions==int(slices!=0)&&control==int(slices==0),"actual graph functions");
    }
    ~HcGraph(){cudaStreamSynchronize(stream);cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);}
    void launch(){ck(cudaGraphLaunch(exec,stream));}
};
static void hc_case(Case& c,bool timing) {
    insist(c.kind==1&&c.n==24&&c.k==16384,"HC24 F32 only");
    ck(cudaSetDevice(c.rank)); Stream s;
    Guard<uint8_t> w(c.w.size(),s.value),x(c.x.size(),s.value);
    Guard<float> a(31,s.value),b(31,s.value),partial(24*32+7,s.value);
    w.upload(c.w); x.upload(c.x);
    api(memra_dsv4_dense_fast_set_for_gate(1));
    HcGraph control(s.value,[&]{api(memra_dsv4_dense_exact_tail_dots((float*)x.data(),w.data(),0,a.data(),1,24,16384,s.value));},0);
    control.launch(); auto reference=a.output(24);
    for(int slices:{8,16,32}) {
        api(memra_dsv4_hc_dot_split_set_for_gate(slices));
        HcGraph candidate(s.value,[&]{api(memra_dsv4_hc_dot_split((float*)x.data(),(float*)w.data(),partial.data(),24*32,b.data(),24,16384,s.value));},slices);
        // Retained functions must ignore later host selection; changing inputs must
        // not reuse stale partials. Repeats compare raw bits within this S class.
        api(memra_dsv4_hc_dot_split_set_for_gate(0));
        for(int generation=0;generation<2;++generation) {
            auto changed=c.x;
            if(generation)for(size_t i=0;i<changed.size();i+=4)changed[i+3]^=0x80;
            x.upload(changed); a.reset(); b.reset(); partial.reset();
            control.launch(); candidate.launch(); reference=a.output(24); auto expected=b.output(24);
            partial.output(24*slices);
            double max_abs=0,max_rel=0,scale=0;
            for(size_t row=0;row<24;++row) {
                insist(std::isfinite(reference[row])&&std::isfinite(expected[row]),"finite output");
                double delta=std::abs(double(expected[row])-reference[row]);
                max_abs=std::max(max_abs,delta); scale=std::max(scale,std::abs(double(reference[row])));
                if(reference[row]!=0)max_rel=std::max(max_rel,delta/std::abs(double(reference[row])));
            }
            for(int repeat=0;repeat<20;++repeat){candidate.launch();equal(expected,b.output(24));}
            printf("DELTA rank=%d slices=%d generation=%d repeats=20 max_abs=%.12g max_relative=%.12g max_abs_over_max_reference=%.12g deterministic=1\n",c.rank,slices,generation,max_abs,max_rel,max_abs/std::max(scale,1e-30));
        }
        x.upload(c.x);
        if(timing) {
            void* flush=nullptr;ck(cudaMalloc(&flush,256ull<<20));cudaEvent_t start{},end{};
            ck(cudaEventCreate(&start));ck(cudaEventCreate(&end));
            for(int i=0;i<32;++i){control.launch();candidate.launch();}ck(cudaStreamSynchronize(s.value));
            for(int cold=0;cold<2;++cold)for(int rep=0;rep<50;++rep)for(int slot=0;slot<2;++slot){
                int on=(rep%2)^slot;
                if(cold)ck(cudaMemsetAsync(flush,rep,256ull<<20,s.value));
                ck(cudaEventRecord(start,s.value));if(on)candidate.launch();else control.launch();
                ck(cudaEventRecord(end,s.value));ck(cudaEventSynchronize(end));float ms=0;ck(cudaEventElapsedTime(&ms,start,end));
                size_t bytes=c.w.size()+c.x.size()+24*4+(on?2*24*slices*4:0);
                printf("TIME rank=%d slices=%d on=%d cold=%d rep=%d us=%.6f bytes=%zu tensor_gbs=%.6f\n",c.rank,slices,on,cold,rep,ms*1000,bytes,bytes/(ms*1e6));
            }
            ck(cudaFree(flush));ck(cudaEventDestroy(start));ck(cudaEventDestroy(end));
        }
        w.immutable();x.immutable();a.output(24);b.output(24);partial.output(24*slices);
    }
    // Raw launcher rejects unsupported shapes, undersized scratch, OFF and invalid S.
    for(int slices:{0,8,16,32}) {
        api(memra_dsv4_hc_dot_split_set_for_gate(slices));
        insist(memra_dsv4_hc_dot_split((float*)x.data(),(float*)w.data(),partial.data(),0,b.data(),24,16384,s.value)==40075,"scratch/OFF refusal");
        insist(memra_dsv4_hc_dot_split((float*)x.data(),(float*)w.data(),partial.data(),24*32,b.data(),4,16384,s.value)==40075,"shape refusal");
    }
    insist(memra_dsv4_hc_dot_split_set_for_gate(7)==40075,"invalid S refusal");
    printf("PASS hc_components rank=%d guards=1 inputs_immutable=1\n",c.rank);fflush(stdout);
}
int main(int argc,char** argv){try {
    insist(argc==2||argc==3,"usage: component operand-dir [--check-only]");
    bool timing=argc==2; if(argc==3)insist(std::string(argv[2])=="--check-only","option");
    int seen=0;
    for(auto& entry:std::filesystem::directory_iterator(argv[1])) {
        if(entry.path().extension()!=".bin")continue;
        auto c=read_case(entry.path());if(c.kind!=1||c.n!=24||c.k!=16384)continue;
        insist(!(seen&(1<<c.rank)),"one captured HC24 key per rank"); hc_case(c,timing);seen|=1<<c.rank;
    }
    insist(seen==3,"both real ranks required");puts("PASS hc_dot_split_components");return 0;
}catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}

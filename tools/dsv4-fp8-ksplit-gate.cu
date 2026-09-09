// Real-operand FP8 split components, same standalone build flags as dense-fast.
#define main fp8_dense_fast_unused_main
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

struct Fp8Graph {
    cudaGraph_t graph{}; cudaGraphExec_t exec{}; cudaStream_t stream;
    template<class F> Fp8Graph(cudaStream_t s, F enqueue, int slices):stream(s) {
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
            partials+=strstr(name,"dsv4_fp8_ksplit_partial_kernel")!=nullptr;
            reductions+=strstr(name,"dsv4_fp8_ksplit_reduce_kernel")!=nullptr;
            control+=strstr(name,"dsv4_dense_fast_fp8_kernel")!=nullptr;
            cudaFuncAttributes attr{}; ck(cudaFuncGetAttributes(&attr,p.func));
            int blocks=0; ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,p.func,p.blockDim.x,0));
            cudaDeviceProp prop{}; int dev; ck(cudaGetDevice(&dev)); ck(cudaGetDeviceProperties(&prop,dev));
            printf("RESOURCE rank=%d slices=%d sm_count=%d grid=%u threads=%u registers=%d active_blocks_per_sm=%d grid_sm_coverage=%.6f symbol=%s\n",dev,slices,prop.multiProcessorCount,p.gridDim.x,p.blockDim.x,attr.numRegs,blocks,double(std::min(p.gridDim.x,unsigned(prop.multiProcessorCount)))/prop.multiProcessorCount,name);
        }
        insist(partials==int(slices!=0)&&reductions==int(slices!=0)&&control==int(slices==0),"actual graph functions");
    }
    ~Fp8Graph(){cudaStreamSynchronize(stream);cudaGraphExecDestroy(exec);cudaGraphDestroy(graph);}
    void launch(){ck(cudaGraphLaunch(exec,stream));}
};
static void fp8_case(Case& c,bool timing) {
    insist(c.kind==0&&(c.n==512||c.n==1024)&&c.k==4096,"FP8 BF16-input target shapes only");
    ck(cudaSetDevice(c.rank)); Stream s;
    Guard<uint8_t> w(c.w.size(),s.value),x(c.x.size(),s.value);
    Guard<float> sc(c.sc.size(),s.value),a(c.n+7,s.value),b(c.n+7,s.value),partial(c.n*8+7,s.value);
    w.upload(c.w); x.upload(c.x); sc.upload(c.sc);
    api(memra_dsv4_dense_fast_set_for_gate(1));
    Fp8Graph control(s.value,[&]{api(memra_dsv4_dense_exact_tail_fp8(w.data(),sc.data(),c.sc_cols,x.data(),a.data(),1,c.n,c.k,0,0,s.value));},0);
    control.launch(); auto reference=a.output(c.n);
    for(int slices:{2,4,8}) {
        api(memra_dsv4_fp8_ksplit_set_for_gate(slices));
        Fp8Graph candidate(s.value,[&]{api(memra_dsv4_fp8_ksplit(w.data(),sc.data(),c.sc_cols,x.data(),partial.data(),c.n*8,b.data(),1,c.n,c.k,s.value));},slices);
        // Retained functions must ignore later host selection; changing inputs must
        // not reuse stale partials. Repeats compare raw bits within this S class.
        api(memra_dsv4_fp8_ksplit_set_for_gate(0));
        for(int generation=0;generation<2;++generation) {
            auto changed=c.x;
            if(generation)for(size_t i=0;i<changed.size();i+=2)changed[i+1]^=0x80;
            x.upload(changed); a.reset(); b.reset(); partial.reset();
            control.launch(); candidate.launch(); reference=a.output(c.n); auto expected=b.output(c.n);
            partial.output(c.n*slices);
            double max_abs=0,max_rel=0,scale=0;
            for(size_t row=0;row<size_t(c.n);++row) {
                insist(std::isfinite(reference[row])&&std::isfinite(expected[row]),"finite output");
                double delta=std::abs(double(expected[row])-reference[row]);
                max_abs=std::max(max_abs,delta); scale=std::max(scale,std::abs(double(reference[row])));
                if(reference[row]!=0)max_rel=std::max(max_rel,delta/std::abs(double(reference[row])));
            }
            for(int repeat=0;repeat<20;++repeat){candidate.launch();equal(expected,b.output(c.n));}
            printf("DELTA rank=%d n=%d slices=%d generation=%d repeats=20 max_abs=%.12g max_relative=%.12g max_abs_over_max_reference=%.12g deterministic=1\n",c.rank,c.n,slices,generation,max_abs,max_rel,max_abs/std::max(scale,1e-30));
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
                // Unique tensors including scratch once, not measured DRAM traffic.
                size_t bytes=c.w.size()+c.x.size()+c.sc.size()*4+c.n*4+(on?c.n*slices*4:0);
                printf("TIME rank=%d n=%d slices=%d on=%d cold=%d rep=%d us=%.6f bytes=%zu provisional_tensor_gbs=%.6f\n",c.rank,c.n,slices,on,cold,rep,ms*1000,bytes,bytes/(ms*1e6));
            }
            ck(cudaFree(flush));ck(cudaEventDestroy(start));ck(cudaEventDestroy(end));
        }
        w.immutable();x.immutable();sc.immutable();a.output(c.n);b.output(c.n);partial.output(c.n*slices);
    }
    // Raw refusals must launch no kernels or change output/scratch.
    auto before = b.output(c.n);
    for(int slices:{0,2,4,8}) {
        api(memra_dsv4_fp8_ksplit_set_for_gate(slices));
        auto launch=[&](int m,int n,int k,int len,int cols,const void* codes,const void* input,float* scratch){
            return memra_dsv4_fp8_ksplit(codes,sc.data(),cols,input,scratch,len,b.data(),m,n,k,s.value);
        };
        insist(launch(1,c.n,c.k,0,c.sc_cols,w.data(),x.data(),partial.data())==40075,"short scratch refusal");
        insist(launch(1,2048,c.k,c.n*8,c.sc_cols,w.data(),x.data(),partial.data())==40075,"shape refusal");
        insist(launch(2,c.n,c.k,c.n*8,c.sc_cols,w.data(),x.data(),partial.data())==40075,"M refusal");
        insist(launch(1,c.n,2048,c.n*8,c.sc_cols,w.data(),x.data(),partial.data())==40075,"K refusal");
        insist(launch(1,c.n,c.k,c.n*8,31,w.data(),x.data(),partial.data())==40075,"scale stride refusal");
        insist(launch(1,c.n,c.k,c.n*8,c.sc_cols,w.data()+1,x.data(),partial.data())==40075,"weight alignment refusal");
        insist(launch(1,c.n,c.k,c.n*8,c.sc_cols,w.data(),x.data()+2,partial.data())==40075,"input alignment refusal");
        insist(launch(1,c.n,c.k,c.n*8,c.sc_cols,w.data(),x.data(),nullptr)==40075,"null scratch refusal");
        if(!slices) insist(launch(1,c.n,c.k,c.n*8,c.sc_cols,w.data(),x.data(),partial.data())==40075,"OFF refusal");
    }
    insist(memra_dsv4_fp8_ksplit_set_for_gate(7)==40075,"invalid S refusal");
    equal(before,b.output(c.n));
    printf("PASS fp8_components rank=%d guards=1 inputs_immutable=1\n",c.rank);fflush(stdout);
}
int main(int argc,char** argv){try {
    api(memra_dsv4_fp8_ksplit_set_for_gate(0));
    insist(argc==2||argc==3,"usage: component operand-dir [--check-only]");
    bool timing=argc==2; if(argc==3)insist(std::string(argv[2])=="--check-only","option");
    int seen=0;
    for(auto& entry:std::filesystem::directory_iterator(argv[1])) {
        if(entry.path().extension()!=".bin")continue;
        auto c=read_case(entry.path());if(c.kind!=0||(c.n!=512&&c.n!=1024)||c.k!=4096)continue;
        int key=c.rank*2+(c.n==1024); insist(!(seen&(1<<key)),"one captured key per rank/shape"); fp8_case(c,timing);seen|=1<<key;
    }
    insist(seen==15,"both shapes on both ranks required");puts("PASS fp8_ksplit_components");return 0;
}catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}

// Exact score/eval/den/output and live-input graph checks. Optional warm microtimings.
#include "../crates/memra-engine/cu/dsv4_gpu.cu"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void check(cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
static void ok(int rc){if(rc)throw std::runtime_error("kernel rc="+std::to_string(rc));}
template<class T> struct Dev {
    T* p;size_t n;
    explicit Dev(size_t n):n(n){check(cudaMalloc(&p,n*sizeof(T)));check(cudaMemset(p,0,n*sizeof(T)));}
    ~Dev(){cudaFree(p);} Dev(const Dev&)=delete;
    void put(const std::vector<T>& v){if(v.size()>n)throw std::runtime_error("upload length");check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get(){std::vector<T> v(n);check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
static void same(const std::vector<float>& a,const std::vector<float>& b,const char* what){
    if(a.size()!=b.size())throw std::runtime_error("comparison length");
    for(size_t i=0;i<a.size();++i){uint32_t x,y;std::memcpy(&x,&a[i],4);std::memcpy(&y,&b[i],4);
        if(x!=y){fprintf(stderr,"MISMATCH %s index=%zu a=%08x b=%08x\n",what,i,x,y);throw std::runtime_error(what);}}
}
static float value(uint32_t i,uint32_t seed){
    uint32_t x=i*747796405u+seed;x^=x>>16;x*=2246822519u;x^=x>>13;
    return float(int(x%8191)-4095)/4096.0f;
}
struct Outputs {
    Dev<float> scores,evals,den,out;
    Outputs(int nq,int slots):scores((size_t)nq*64*slots+7),evals(scores.n),den(nq*64+7),out((size_t)nq*64*512+7){}
    void poison(){for(auto* a:{&scores,&evals,&den,&out})check(cudaMemset(a->p,0xa5,a->n*sizeof(float)));}
    void equal(Outputs& other){same(scores.get(),other.scores.get(),"scores");same(evals.get(),other.evals.get(),"evals");same(den.get(),other.den.get(),"den");same(out.get(),other.out.get(),"output");}
    void guards(){for(auto* a:{&scores,&evals,&den,&out}){auto v=a->get();for(size_t i=v.size()-7;i<v.size();++i){uint32_t x;std::memcpy(&x,&v[i],4);if(x!=0xa5a5a5a5u)throw std::runtime_error("output guard overwritten");}}}
};

static void cell(int nq,int slots,float scale,bool bench){
    const int hd=512,heads=64,kv_rows=9001,stride=slots+5;
    Dev<float> q((size_t)nq*heads*hd),kv((size_t)kv_rows*hd),sink(heads);
    Dev<int> ids((size_t)nq*stride);
    Outputs reference(nq,slots),candidate(nq,slots);
    std::vector<float> qh(q.n),kh(kv.n),sh(heads);
    std::vector<int> ih(ids.n);
    for(int h=0;h<heads;++h)sh[h]=value(h,13)*3.0f;sink.put(sh);
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto run=[&](bool tiled,Outputs& out){
        if(tiled)ok(memra_dsv4_sink_attn_dec_mq_f32acc_tiled(q.p,kv.p,ids.p,sink.p,
            out.scores.p,out.evals.p,out.den.p,out.out.p,nq,heads,hd,slots,stride,scale,stream));
        else ok(memra_dsv4_sink_attn_dec_mq_f32acc(q.p,kv.p,ids.p,sink.p,
            out.scores.p,out.evals.p,out.den.p,out.out.p,nq,heads,hd,slots,stride,scale,stream));
    };
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    for(int pattern=0;pattern<6;++pattern){
        for(size_t i=0;i<q.n;++i)qh[i]=pattern==4?(i%2?-0.0f:0.0f):value(uint32_t(i),71+pattern*17);
        for(size_t i=0;i<kv.n;++i)kh[i]=value(uint32_t(i),211+pattern*31);
        if(pattern==5){
            for(size_t i=0;i<q.n;++i)qh[i]*=std::ldexp(1.0f,int(i%7)-3);
            for(size_t i=0;i<kv.n;++i)kh[i]*=std::ldexp(1.0f,int(i%9)-4);
        }
        std::fill(ih.begin(),ih.end(),-1234567);
        for(int p=0;p<nq;++p)for(int k=0;k<slots;++k)
            ih[(size_t)p*stride+k]=pattern==2?-1:pattern==1&&k%7==0?-1:pattern==3?3:(p*71+k*17+3)%kv_rows;
        q.put(qh);kv.put(kh);ids.put(ih);reference.poison();candidate.poison();check(cudaDeviceSynchronize());
        run(false,reference);check(cudaStreamSynchronize(stream));
        if(pattern==0){
            run(true,candidate);check(cudaStreamSynchronize(stream));reference.equal(candidate);
            // Deliberately change one valid index; the data must make that observable.
            if(scale!=0.0f){auto bad=ih;bad[0]=(bad[0]+1)%kv_rows;ids.put(bad);check(cudaDeviceSynchronize());run(true,candidate);check(cudaStreamSynchronize(stream));
                auto a=reference.scores.get(),b=candidate.scores.get();if(!std::memcmp(a.data(),b.data(),a.size()*4))throw std::runtime_error("corrupted-index red control invisible");
                ids.put(ih);check(cudaDeviceSynchronize());}
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));run(true,candidate);
            check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,0));
        }
        check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
        reference.equal(candidate);reference.guards();candidate.guards();
        if(nq==1){
            candidate.poison();check(cudaDeviceSynchronize());
            ok(memra_dsv4_sink_attn_dec_f32acc(q.p,kv.p,ids.p,sink.p,candidate.scores.p,
                candidate.evals.p,candidate.den.p,candidate.out.p,heads,hd,slots,scale,stream));
            check(cudaStreamSynchronize(stream));reference.equal(candidate);
        }
    }
    for(auto shape:std::vector<std::vector<int>>{{0,64,512,slots,stride},{513,64,512,slots,stride},{1,63,512,slots,stride},
            {1,64,511,slots,stride},{1,64,512,0,stride},{1,64,512,slots,slots-1}})
        if(memra_dsv4_sink_scores_tiled_f32acc(q.p,kv.p,ids.p,candidate.scores.p,
            shape[0],shape[1],shape[2],shape[3],shape[4],scale,stream)!=40010)throw std::runtime_error("invalid shape not refused");
    printf("PASS nq=%d slots=%d stride=%d scale=%a patterns=6 graph_replays=6 scores/evals/den/output/guards plain_twin=%d\n",nq,slots,stride,scale,int(nq==1));
    if(bench){
        cudaEvent_t a,b;check(cudaEventCreate(&a));check(cudaEventCreate(&b));
        for(int rep=0;rep<5;++rep)for(bool tiled:{false,true,true,false}){
            run(tiled,candidate);check(cudaStreamSynchronize(stream));
            check(cudaEventRecord(a,stream));for(int i=0;i<8;++i)run(tiled,candidate);
            check(cudaEventRecord(b,stream));check(cudaEventSynchronize(b));float ms;check(cudaEventElapsedTime(&ms,a,b));
            printf("WARM_COMPONENT nq=%d slots=%d rep=%d tiled=%d microseconds=%.4f\n",nq,slots,rep,int(tiled),ms*1000.0f/8.0f);
        }
        check(cudaEventDestroy(a));check(cudaEventDestroy(b));
    }
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
}
int main(int argc,char** argv){try{
    bool bench=argc==2&&std::string(argv[1])=="--bench";
    if(argc>1&&!bench)throw std::runtime_error("usage: gate [--bench]");
    ok(memra_dsv4_sink_scores_tiled_init());
    printf("SHARED bytes=%d init_before_capture=true\n",DSV4_SCORE_SMEM);
    float scale=1.0f/std::sqrt(512.0f);
    for(int nq:{1,6,32})for(int slots:{1,31,32,33,128,640,8320})
        cell(nq,slots,scale,bench&&(slots==640||slots==8320));
    cell(128,33,scale,false);cell(512,33,scale,false);cell(1,33,0.0f,false);cell(6,33,-0.0f,false);
    puts("PASS tiled sink score and unchanged attention pipeline; model/performance admission remains separate");
}catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}

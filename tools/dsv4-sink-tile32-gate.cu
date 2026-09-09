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
    Outputs(int nq,int slots):scores((size_t)nq*32*slots+7),evals(scores.n),den(nq*32+7),out((size_t)nq*32*512+7){}
    void poison(){for(auto* a:{&scores,&evals,&den,&out})check(cudaMemset(a->p,0xa5,a->n*sizeof(float)));}
    void equal(Outputs& other){same(scores.get(),other.scores.get(),"scores");same(evals.get(),other.evals.get(),"evals");same(den.get(),other.den.get(),"den");same(out.get(),other.out.get(),"output");}
    void guards(){for(auto* a:{&scores,&evals,&den,&out}){auto v=a->get();for(size_t i=v.size()-7;i<v.size();++i){uint32_t x;std::memcpy(&x,&v[i],4);if(x!=0xa5a5a5a5u)throw std::runtime_error("output guard overwritten");}}}
};

static void cell(int nq,int slots,float scale,bool bench){
    const int hd=512,heads=32,kv_rows=9001,stride=slots+5;
    Dev<float> q((size_t)nq*heads*hd),kv((size_t)kv_rows*hd),sink(heads);
    Dev<int> ids((size_t)nq*stride);
    Outputs reference(nq,slots),candidate(nq,slots);
    std::vector<float> qh(q.n),kh(kv.n),sh(heads);
    std::vector<int> ih(ids.n);
    for(int h=0;h<heads;++h)sh[h]=value(h,13)*3.0f;sink.put(sh);
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto run=[&](bool tiled,Outputs& out){
        if(tiled)ok(memra_dsv4_sink_attn_dec_mq_f32acc_tiled32(q.p,kv.p,ids.p,sink.p,
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
    for(auto shape:std::vector<std::vector<int>>{{0,32,512,slots,stride},{513,32,512,slots,stride},{1,63,512,slots,stride},
            {1,32,511,slots,stride},{1,32,512,0,stride},{1,32,512,slots,slots-1}})
        if(memra_dsv4_sink_scores_tiled32_f32acc(q.p,kv.p,ids.p,candidate.scores.p,
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

#include <fstream>
#include <string>
template<class T> static std::vector<T> read(std::ifstream& f,size_t n){
    std::vector<T> v(n); f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T));
    if(!f)throw std::runtime_error("short operand tape");return v;
}
static void real(const std::string& dir,int rank,int layer){
    char name[128];snprintf(name,sizeof(name),"/rank%d-layer%02d.bin",rank,layer);
    std::ifstream f(dir+name,std::ios::binary);auto meta=read<uint32_t>(f,5);
    if(meta[0]!=(unsigned)rank||meta[1]!=(unsigned)layer||meta[3]<1||meta[3]>8320)throw std::runtime_error("bad capture header");
    int slots=meta[3];float scale;std::memcpy(&scale,&meta[4],4);
    auto qh=read<float>(f,32*512),sh=read<float>(f,32),kh=read<float>(f,slots*512);
    auto ih=read<int>(f,slots),original=read<int>(f,slots);
    if(f.peek()!=EOF)throw std::runtime_error("trailing capture data");
    for(int i=0;i<slots;++i)if(ih[i]!=(original[i]<0?-1:i))throw std::runtime_error("capture index mapping");
    Dev<float> q(qh.size()),kv(kh.size()),sink(32);Dev<int> ids(slots),pos(1);
    q.put(qh);kv.put(kh);sink.put(sh);ids.put(ih);pos.put({256});
    Outputs reference(1,slots),candidate(1,slots);
    Dev<unsigned char> scrub(128*1024*1024);
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto run=[&](bool tiled,Outputs& o,bool replay){
        if(replay){
            auto fn=tiled?memra_dsv4_replay_attention_tiled32:memra_dsv4_replay_attention;
            // Real ratio and live position drive the compressed tail.
            // Earlier replay positions stay within the captured allocation.
            int win=slots-(meta[2]?257/int(meta[2]):0);
            ok(fn(q.p,kv.p,ids.p,sink.p,o.scores.p,o.evals.p,o.den.p,o.out.p,pos.p,
                32,512,slots,slots,scale,win,meta[2],512,stream));
        }else{
            auto fn=tiled?memra_dsv4_sink_attn_dec_mq_f32acc_tiled32:memra_dsv4_sink_attn_dec_mq_f32acc;
            ok(fn(q.p,kv.p,ids.p,sink.p,o.scores.p,o.evals.p,o.den.p,o.out.p,1,32,512,slots,slots,scale,stream));
        }
    };
    cudaEvent_t a,b;check(cudaEventCreate(&a));check(cudaEventCreate(&b));
    for(bool cold:{false,true}){
        double sums[2]={};int counts[2]={};
        for(int rep=0;rep<12;++rep){bool tiled=rep%4==1||rep%4==2;
            auto& o=tiled?candidate:reference;o.poison();
            if(cold)check(cudaMemsetAsync(scrub.p,0,scrub.n,stream));
            check(cudaEventRecord(a,stream));run(tiled,o,false);check(cudaEventRecord(b,stream));check(cudaStreamSynchronize(stream));
            o.guards();float ms;check(cudaEventElapsedTime(&ms,a,b));
            if(rep>=4){sums[tiled]+=ms*1000;counts[tiled]++;}
            if(rep%4==3)reference.equal(candidate);
        }
        printf("REAL_COMPONENT rank=%d layer=%d ratio=%u slots=%d cold=%d off_us=%.6f on_us=%.6f n=%d bits_equal=true\n",rank,layer,meta[2],slots,int(cold),sums[0]/counts[0],sums[1]/counts[1],counts[0]);
    }
    // Capture once at position 256, then change the device position so the live
    // score stride and output trip count cannot be baked into the candidate.
    cudaGraph_t graph;cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));run(true,candidate,true);
    check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,0));
    for(int p:{256,255,128,127,1,0,256}){
        pos.put({p});reference.poison();candidate.poison();check(cudaDeviceSynchronize());
        run(false,reference,true);check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
        reference.equal(candidate);reference.guards();candidate.guards();
    }
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));
    check(cudaEventDestroy(a));check(cudaEventDestroy(b));check(cudaStreamDestroy(stream));
}
int main(int argc,char** argv){try{
    if(argc!=2)throw std::runtime_error("usage: sink-tile32-component <captured-dir>");
    for(int rank=0;rank<2;++rank){check(cudaSetDevice(rank));ok(memra_dsv4_sink_scores_tiled32_init());
        for(int slots:{1,31,32,33,128,640,8320})cell(1,slots,0.044194173f,false);
        cell(1,33,0.0f,false);cell(1,33,-0.0f,false);cell(6,33,0.044194173f,false);
        for(int layer=0;layer<43;++layer)real(argv[1],rank,layer);
    }
    puts("COMPONENT_PASS real_sites=86 ranks=2 layers=43 edges=20 live_position_graphs=86");return 0;
}catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}

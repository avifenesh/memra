// Model-free prerequisite, NOT a model coverage or performance gate.
// nvcc -std=c++17 -O2 -fmad=false -arch=sm_120a \
//   tools/dsv4-full-token-control-gate.cu -o <owned-target>/dsv4-full-token-control-gate
#include "../crates/memra-engine/cu/dsv4_replay_control.cuh"
#include "../crates/memra-engine/cu/tp_ar.cu"
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void ck(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static void require(bool ok, const char* why) {
    if (!ok) throw std::runtime_error(why);
}
constexpr unsigned window = 128, layers = 43, steps = 513;
struct Fixture {
    uint32_t pending4[8], pending128[128], ring[window];
    uint32_t snap4[8], snap128[128];
    uint32_t blocks4, blocks128, snap_blocks4, snap_blocks128;
    uint32_t emits4, emits128, commits, forwards, sampled_token;
    uint64_t sampled_uniform;
};
__global__ void snapshot(const Dsv4ReplayControl* c, Fixture* f) {
    if (c->invalid) return;
    for (int i=0; i<8; ++i) f->snap4[i]=f->pending4[i];
    for (int i=0; i<128; ++i) f->snap128[i]=f->pending128[i];
    f->snap_blocks4=f->blocks4; f->snap_blocks128=f->blocks128;
    f->pending4[c->c4_pending_slot]=c->input.token;
    f->pending128[c->c128_pending_slot]=c->input.token;
    ++f->forwards;
}
__global__ void emit4(const Dsv4ReplayControl* c, Fixture* f) {
    for (int i=0; i<4; ++i) f->pending4[i]=f->pending4[i+4];
    f->blocks4=c->c4_blocks; ++f->emits4;
}
__global__ void emit128(const Dsv4ReplayControl* c, Fixture* f) {
    f->blocks128=c->c128_blocks; ++f->emits128;
}
__global__ void producer(const Dsv4ReplayControl* c, float* x, int rank, int layer, int phase, int n) {
    for(int i=threadIdx.x;i<n;i+=blockDim.x)
        x[i]=float((c->input.token % 1024) + c->input.position + rank + layer + phase + i%127);
}
__global__ void finish(const Dsv4ReplayControl* c, Fixture* f) {
    f->ring[c->ring_slot]=c->input.token;
    f->sampled_token=c->input.token;
    f->sampled_uniform=c->input.uniform_bits;
    ++f->commits;
}
__global__ void rollback(Fixture* f) {
    for (int i=0; i<8; ++i) f->pending4[i]=f->snap4[i];
    for (int i=0; i<128; ++i) f->pending128[i]=f->snap128[i];
    f->blocks4=f->snap_blocks4; f->blocks128=f->snap_blocks128;
}

struct CleanupAudit {
    unsigned drained=0, released=0;
    int refusals[2]={-1,-1};
    bool order_error=false;
};
static bool cleanup_failed=false;
struct Rank {
    int device;
    cudaStream_t stream{};
    cudaGraph_t forward{}, commit{}, fault{};
    cudaGraphExec_t forward_exec{}, commit_exec{}, fault_exec{};
    Dsv4ReplayInput *input{}, *host{};
    Dsv4ReplayControl* control{};
    Fixture* fixture{};
    void* signal{};
    int* error{};
    float *partial{}, *sum{};
    bool quarantined=false, release_allowed=true;
    CleanupAudit* audit=nullptr;
    int width=1, blocks=1;
    long long spin_limit=2000000000LL;
    explicit Rank(int d, int n=1, int b=1, long long spin=2000000000LL):device(d),width(n),blocks(b),spin_limit(spin) {
        ck(cudaSetDevice(device)); ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        ck(cudaMalloc(&input,sizeof(*input))); ck(cudaMallocHost(&host,sizeof(*host)));
        ck(cudaMalloc(&control,sizeof(*control))); ck(cudaMalloc(&fixture,sizeof(*fixture)));
        ck(cudaMemset(fixture,0,sizeof(*fixture)));
        ck(cudaMalloc(&signal,memra_tp_ar_signal_bytes()));
        ck(cudaMemset(signal,0,memra_tp_ar_signal_bytes()));
        ck(cudaMalloc(&error,sizeof(int))); ck(cudaMemset(error,0,sizeof(int)));
        ck(cudaMalloc(&partial,width*sizeof(float))); ck(cudaMalloc(&sum,86*width*sizeof(float)));
        ck(cudaMemset(sum,0xff,86*width*sizeof(float)));
        ck(cudaStreamSynchronize(stream));
    }
    ~Rank() {
        // Pair owns the cross-device lifetime. A failed drain keeps BOTH planes
        // allocated until process teardown rather than freeing a possible peer target.
        if (!release_allowed) return;
        if (audit && audit->drained != 3) {
            audit->order_error=true; cleanup_failed=true;
            fprintf(stderr,"FAIL freeing rank %d before both pair drains\n",device);
            return;
        }
        cudaSetDevice(device); cudaStreamSynchronize(stream);
        if(audit) audit->released |= 1u << device;
        if(forward_exec) cudaGraphExecDestroy(forward_exec);
        if(commit_exec) cudaGraphExecDestroy(commit_exec);
        if(fault_exec) cudaGraphExecDestroy(fault_exec);
        if(forward) cudaGraphDestroy(forward);
        if(commit) cudaGraphDestroy(commit);
        if(fault) cudaGraphDestroy(fault);
        cudaFree(input); cudaFreeHost(host); cudaFree(control); cudaFree(fixture);
        cudaFree(signal); cudaFree(error); cudaFree(partial); cudaFree(sum);
        cudaStreamDestroy(stream);
    }
    Rank(const Rank&)=delete;
    Rank& operator=(const Rank&)=delete;
};

// Declared after its ranks are constructed and destroyed BEFORE either rank.
// This owner is mandatory even when the second graph submission throws.
struct Pair {
    CleanupAudit own_audit;
    CleanupAudit* audit;
    Rank a,b;
    explicit Pair(int width=1,int blocks=1,long long spin=2000000000LL,
                  CleanupAudit* external=nullptr)
        : audit(external?external:&own_audit),a(0,width,blocks,spin),b(1,width,blocks,spin) {
        a.audit=b.audit=audit;
    }
    ~Pair() {
        for(auto* r:{&a,&b}) {
            auto rc=cudaSetDevice(r->device);
            if(rc==cudaSuccess) rc=cudaStreamSynchronize(r->stream);
            if(rc==cudaSuccess) audit->drained |= 1u << r->device;
            else {
                cleanup_failed=true;
                fprintf(stderr,"FAIL pair cleanup rank=%d error=%s\n",r->device,cudaGetErrorString(rc));
            }
        }
        if(audit->drained!=3) a.release_allowed=b.release_allowed=false;
        else for(auto* r:{&a,&b}) {
            auto rc=cudaSetDevice(r->device);
            if(rc==cudaSuccess) rc=cudaMemcpy(&audit->refusals[r->device],r->error,
                                             sizeof(int),cudaMemcpyDeviceToHost);
            if(rc!=cudaSuccess) {
                cleanup_failed=true;
                fprintf(stderr,"FAIL cleanup refusal read rank=%d error=%s\n",r->device,cudaGetErrorString(rc));
            }
        }
    }
};

template<typename F,typename... Args>
cudaGraphNode_t kernel(cudaGraph_t g,cudaGraphNode_t tail,F fn,Args... args) {
    void* argv[]={&args...};
    cudaKernelNodeParams p{}; p.func=(void*)fn; p.gridDim=dim3(1); p.blockDim=dim3(1);
    p.kernelParams=argv;
    cudaGraphNode_t node{};
    ck(cudaGraphAddKernelNode(&node,g,tail?&tail:nullptr,tail?1:0,&p)); return node;
}
cudaGraphNode_t conditional(cudaGraph_t g,cudaGraphNode_t tail,
                           cudaGraphConditionalHandle handle,Rank& r,bool four) {
    cudaGraphNodeParams p{}; p.type=cudaGraphNodeTypeConditional;
    p.conditional.handle=handle; p.conditional.type=cudaGraphCondTypeIf; p.conditional.size=1;
    cudaGraphNode_t node{}; ck(cudaGraphAddNode(&node,g,&tail,nullptr,1,&p));
    if(four) kernel(p.conditional.phGraph_out[0],nullptr,emit4,r.control,r.fixture);
    else kernel(p.conditional.phGraph_out[0],nullptr,emit128,r.control,r.fixture);
    return node;
}

void build(Rank& r,Rank& peer,bool fault=false) {
    auto& graph=fault?r.fault:r.forward;
    auto& executable=fault?r.fault_exec:r.forward_exec;
    ck(cudaSetDevice(r.device)); ck(cudaGraphCreate(&graph,0));
    cudaGraphConditionalHandle c4{},c128{};
    ck(cudaGraphConditionalHandleCreate(&c4,graph,0,cudaGraphCondAssignDefault));
    ck(cudaGraphConditionalHandleCreate(&c128,graph,0,cudaGraphCondAssignDefault));
    auto tail=kernel(graph,nullptr,dsv4_replay_control_kernel,r.input,r.control,c4,c128);
    tail=kernel(graph,tail,snapshot,r.control,r.fixture);
    tail=conditional(graph,tail,c4,r,true);
    tail=conditional(graph,tail,c128,r,false);
    // Existing transport, 86 ordered joins per replay. These producers are
    // integer-exact fixtures, not substitutes for DSV4 layer arithmetic.
    ck(cudaStreamBeginCaptureToGraph(r.stream,graph,&tail,nullptr,1,cudaStreamCaptureModeRelaxed));
    for(int layer=0; layer<int(fault?1:layers); ++layer) for(int phase=0; phase<2; ++phase) {
        const int width=(r.width==24576 && phase==0)?4096:r.width;
        const int blocks=(r.width==24576 && phase==0)?1:r.blocks;
        producer<<<1,128,0,r.stream>>>(r.control,r.partial,r.device,layer,phase,width);
        const auto* p0=r.device==0?r.partial:peer.partial;
        const auto* p1=r.device==1?r.partial:peer.partial;
        require(memra_tp_ar_1stage(p0,p1,r.sum+(layer*2+phase)*r.width,r.signal,peer.signal,r.device,width,r.error,
                                  fault?5000000LL:r.spin_limit,blocks,r.stream)==0,"AR capture failed");
    }
    cudaGraph_t ended{}; ck(cudaStreamEndCapture(r.stream,&ended));
    require(ended==graph,"capture changed graph owner");
    ck(cudaGraphInstantiate(&executable,graph,0));
    if(!fault) {
    ck(cudaGraphCreate(&r.commit,0));
    kernel(r.commit,nullptr,finish,r.control,r.fixture);
    ck(cudaGraphInstantiate(&r.commit_exec,r.commit,0));
    }
    size_t n=0; ck(cudaGraphGetNodes(graph,nullptr,&n));
    require(n==4+(fault?1:layers)*4,"unexpected forward graph node count");
    printf("GRAPH rank=%d fault=%d forward_nodes=%zu conditional_bodies=2 ar_nodes=%u commit_nodes=%d\n",r.device,fault,n,(fault?1:layers)*2,fault?0:1);
}
template<typename T> T read(Rank& r,const T* p) {
    ck(cudaSetDevice(r.device)); T v{};
    ck(cudaMemcpyAsync(&v,p,sizeof(v),cudaMemcpyDeviceToHost,r.stream));
    ck(cudaStreamSynchronize(r.stream)); return v;
}
void upload(Rank& r,Dsv4ReplayInput in) {
    require(!r.quarantined,"quarantined replay refused");
    require(in.window==window && in.position<in.capacity && in.capacity<=0x7fffffff,"invalid control refused");
    ck(cudaSetDevice(r.device)); *r.host=in;
    ck(cudaMemcpyAsync(r.input,r.host,sizeof(in),cudaMemcpyHostToDevice,r.stream));
}
void launch(Rank& r,bool commit,bool fault=false) {
    require(!r.quarantined,"quarantined replay refused");
    ck(cudaSetDevice(r.device)); ck(cudaGraphLaunch(commit?r.commit_exec:(fault?r.fault_exec:r.forward_exec),r.stream));
}
void check_control(const Dsv4ReplayControl& c,const Dsv4ReplayInput& in) {
    require(c.input.token==in.token && c.input.position==in.position &&
            c.input.uniform_bits==in.uniform_bits,"frozen token/position/uniform");
    require(!c.invalid && c.ring_slot==in.position%in.window &&
            c.c4_pending_slot==4+in.position%4 && c.c128_pending_slot==in.position%128 &&
            c.c4_blocks==(in.position+1)/4 && c.c128_blocks==(in.position+1)/128 &&
            c.c4_rope_position==in.position/4*4 && c.c128_rope_position==in.position/128*128 &&
            c.indexer_count==((in.position+1)/4<512?(in.position+1)/4:512) &&
            c.attention_slots==in.window+c.indexer_count,"live control mismatch");
}
enum class Submission { Both, OnlyA, OnlyB, FailB };
bool step(Pair& pair,Dsv4ReplayInput in,int inject=-1,Submission submission=Submission::Both) {
    auto& a=pair.a; auto& b=pair.b;
    upload(a,in); upload(b,in);
    // The absent rank's snapshot is valid even if its graph never runs. Its
    // persistent state is unchanged and must not be restored from stale storage.
    std::array<bool,2> submitted={false,false};
    if(submission!=Submission::OnlyB) {launch(a,false,submission!=Submission::Both); submitted[0]=true;}
    if(submission==Submission::FailB) {
        // A host submission error can occur before the second CUDA call. This
        // injection throws at that exact boundary, with rank A already executing.
        throw std::runtime_error("deliberate second submission failure: before peer enqueue");
    }
    if(submission!=Submission::OnlyA) {launch(b,false,submission!=Submission::Both); submitted[1]=true;}
    if(inject>=0) {
        auto& r=inject?b:a; ck(cudaSetDevice(r.device));
        int code=40043; ck(cudaMemcpyAsync(r.error,&code,sizeof(code),cudaMemcpyHostToDevice,r.stream));
        ck(cudaStreamSynchronize(r.stream));
    }
    // Read BOTH words first. A start-barrier refusal leaves output unwritten;
    // neither controls nor sum payloads are correctness-readable on that path.
    std::array<int,2> errors={read(a,a.error),read(b,b.error)};
    if(errors!=std::array<int,2>{0,0}) {
        a.quarantined=b.quarantined=true;
        for(auto* r:{&a,&b}) if(submitted[r->device]) {
            ck(cudaSetDevice(r->device)); rollback<<<1,1,0,r->stream>>>(r->fixture);
            ck(cudaGetLastError()); ck(cudaStreamSynchronize(r->stream));
        }
        printf("REFUSED rank_words=[%d,%d] sums_validated=0 commits_launched=0\n",errors[0],errors[1]);
        return false;
    }
    check_control(read(a,a.control),in); check_control(read(b,b.control),in);
    for(auto* r:{&a,&b}) {
        std::vector<float> sums(86*r->width);
        ck(cudaSetDevice(r->device));
        ck(cudaMemcpy(sums.data(),r->sum,sums.size()*sizeof(float),cudaMemcpyDeviceToHost));
        for(unsigned layer=0;layer<layers;++layer) for(unsigned phase=0;phase<2;++phase)
            for(int i=0;i<((r->width==24576 && phase==0)?4096:r->width);++i)
                require(sums[(layer*2+phase)*r->width+i]==float(2*((in.token%1024)+in.position+layer+phase+i%127)+1),
                        "AR stale input or wrong order");
    }
    launch(a,true); launch(b,true);
    return true;
}

void check_fixture(Rank& r,const Fixture& expected,Dsv4ReplayInput in,unsigned accepted) {
    auto f=read(r,r.fixture);
    require(!memcmp(f.pending4,expected.pending4,sizeof(f.pending4)) &&
            !memcmp(f.pending128,expected.pending128,sizeof(f.pending128)) &&
            !memcmp(f.ring,expected.ring,sizeof(f.ring)),"cache fixture differs from host sequence");
    require(f.blocks4==expected.blocks4 && f.blocks128==expected.blocks128 &&
            f.emits4==expected.emits4 && f.emits128==expected.emits128 &&
            f.commits==accepted && f.forwards==accepted &&
            f.sampled_token==in.token && f.sampled_uniform==in.uniform_bits,"cadence/readback/counter mismatch");
    auto* seq=((const MemraArSignal*)r.signal)->seq;
    for(int block=0;block<r.blocks;++block)
        require(read(r,seq+block)==accepted*86,"AR epoch did not advance on device");
}
int main() try {
    int devices=0,driver=0,runtime=0;
    ck(cudaGetDeviceCount(&devices)); require(devices==2,"requires exactly two visible devices");
    ck(cudaDriverGetVersion(&driver)); ck(cudaRuntimeGetVersion(&runtime));
    printf("RUNTIME driver=%d runtime=%d headers=%d\n",driver,runtime,CUDART_VERSION);
    for(int d=0;d<2;++d) {
        ck(cudaSetDevice(d)); int can=0; ck(cudaDeviceCanAccessPeer(&can,d,1-d));
        require(can,"peer access unavailable");
        auto e=cudaDeviceEnablePeerAccess(1-d,0);
        if(e==cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError(); else ck(e);
    }
    {
        Pair pair; auto& a=pair.a; auto& b=pair.b; build(a,b); build(b,a); Fixture expected{};
        for(unsigned p=0;p<steps;++p) {
            Dsv4ReplayInput in{(p*7919+17)%129280,p,(0x3fe0000000000000ULL ^ (uint64_t(p)*0x9e3779b97f4a7c15ULL)),window,4096};
            expected.pending4[4+p%4]=in.token; expected.pending128[p%128]=in.token;
            if((p+1)%4==0) {
                for(int i=0;i<4;++i) expected.pending4[i]=expected.pending4[i+4];
                expected.blocks4=(p+1)/4; ++expected.emits4;
            }
            if((p+1)%128==0) {expected.blocks128=(p+1)/128; ++expected.emits128;}
            expected.ring[p%window]=in.token;
            require(step(pair,in),"positive replay refused");
            check_fixture(a,expected,in,p+1); check_fixture(b,expected,in,p+1);
        }
        printf("PASS live controls: steps=%u ranks=2 c4=128 c128=4 ring_wraps=4 ar_epochs_per_rank=%u\n",steps,steps*86);
    }
    for(int rank=0;rank<2;++rank) for(unsigned pos:{127u,255u,511u}) {
        Pair pair; auto& a=pair.a; auto& b=pair.b; build(a,b); build(b,a);
        Dsv4ReplayInput before{91,pos-1,0x3fe0000000000001ULL,window,4096};
        for(unsigned p=0;p<pos;++p) {
            auto prime=before; prime.position=p; prime.token=(p*7919+17)%129280;
            prime.uniform_bits=uint64_t(p)*0x9e3779b97f4a7c15ULL;
            require(step(pair,prime),"refusal prefix setup failed");
        }
        std::array<Fixture,2> saved={read(a,a.fixture),read(b,b.fixture)};
        for(const auto& f:saved) {
            require(f.blocks4==pos/4 && f.blocks4>0,"prefix did not seed C4 high-water");
            require(f.blocks128==pos/128 && (pos<128 || f.blocks128>0),"prefix did not seed C128 high-water");
        }
        auto refused=before; refused.position=pos; refused.token=104; refused.uniform_bits^=0xa5a5a5a5ffffffffULL;
        require(!step(pair,refused,rank),"fault did not refuse");
        for(auto* r:{&a,&b}) {
            auto f=read(*r,r->fixture); const auto& old=saved[r->device];
            require(!memcmp(f.ring,old.ring,sizeof(f.ring)) &&
                    !memcmp(f.pending4,old.pending4,sizeof(f.pending4)) &&
                    !memcmp(f.pending128,old.pending128,sizeof(f.pending128)) &&
                    f.blocks4==old.blocks4 && f.blocks128==old.blocks128 &&
                    f.commits==old.commits && f.sampled_token==old.sampled_token &&
                    f.sampled_uniform==old.sampled_uniform,"refusal changed committed state");
            bool rejected=false; try {upload(*r,refused);} catch(const std::runtime_error&) {rejected=true;}
            require(rejected,"retry was not quarantined");
        }
        printf("PASS refusal rank=%d position=%u no_commit=1 rollback_both=1 retry_quarantined=1\n",rank,pos);
    }
    // Exact default launch geometries from tp_ar::ar_blocks_for: attention
    // 4096 floats/1 block and 6 selected expert rows * 4096 /48 blocks.
    for(auto geometry:{std::pair<int,int>{4096,1},{24576,48}}) {
        Pair pair(geometry.first,geometry.second); auto& a=pair.a; auto& b=pair.b;
        build(a,b); build(b,a);
        for(unsigned p=0;p<8;++p) {
            Dsv4ReplayInput in{17+p*97,252+p,uint64_t(p)*0xd6e8feb86659fd93ULL,window,4096};
            require(step(pair,in),"production-geometry replay refused");
            for(auto* r:{&a,&b}) {
                auto* seq=((const MemraArSignal*)r->signal)->seq;
                for(int block=0;block<r->blocks;++block)
                    require(read(*r,seq+block)==(p+1)*(block==0?86:43),"production-geometry epoch mismatch");
            }
        }
        printf("PASS production geometry max_n=%d max_blocks=%d replay_tokens=8 joins_per_token=86\n",geometry.first,geometry.second);
    }
    for(int rank=0;rank<2;++rank) {
        Pair pair(4096,1); auto& a=pair.a; auto& b=pair.b;
        build(a,b); build(b,a); build(a,b,true); build(b,a,true);
        // Real prefix gives both checkpoints nonzero high-water marks. Keep the
        // short timeout only on failure graphs, without weakening positive graphs.
        for(unsigned p=0;p<255;++p) {
            Dsv4ReplayInput in{31+p,p,uint64_t(p)*0xd6e8feb86659fd93ULL,window,4096};
            require(step(pair,in),"timeout prefix failed");
        }
        auto saved=std::array<Fixture,2>{read(a,a.fixture),read(b,b.fixture)};
        // Poison sum outputs. A genuine start-barrier timeout cannot produce
        // them; checking sums before refusal would deterministically fail here.
        for(auto* r:{&a,&b}) {ck(cudaSetDevice(r->device)); ck(cudaMemset(r->sum,0xff,86*r->width*sizeof(float)));}
        Dsv4ReplayInput in{999,255,0xa5a5a5a5ffffffffULL,window,4096};
        require(!step(pair,in,-1,rank?Submission::OnlyB:Submission::OnlyA),"missing peer did not refuse");
        auto& launched=rank?b:a;
        require(read(launched,launched.error)==40043,"actual start refusal code missing");
        for(auto* r:{&a,&b}) {
            auto f=read(*r,r->fixture); auto old=saved[r->device];
            require(!memcmp(f.ring,old.ring,sizeof(f.ring)) &&
                    !memcmp(f.pending4,old.pending4,sizeof(f.pending4)) &&
                    !memcmp(f.pending128,old.pending128,sizeof(f.pending128)) &&
                    f.blocks4==old.blocks4 && f.blocks128==old.blocks128 && f.commits==old.commits,
                    "actual timeout failed rollback before commit");
            require(r->quarantined,"actual timeout did not quarantine both ranks");
        }
        printf("PASS actual start timeout rank=%d nonzero_high_water_restored=1 poisoned_sums_not_read=1\n",rank);
    }
    {
        CleanupAudit audit;
        bool caught=false;
        try {
            Pair pair(4096,1,2000000000LL,&audit); build(pair.a,pair.b,true); build(pair.b,pair.a,true);
            Dsv4ReplayInput in{17,127,0xa5a5a5a5ffffffffULL,window,4096};
            step(pair,in,-1,Submission::FailB);
        } catch(const std::runtime_error& e) {
            caught=std::string(e.what()).find("deliberate second submission failure:")==0;
            if(!caught) throw;
            printf("EXPECTED %s\n",e.what());
        }
        require(caught && audit.drained==3 && audit.released==3 && !audit.order_error &&
                audit.refusals[0]==40043 && audit.refusals[1]==0,
                "second submission failure did not drain both before either release");
        puts("PASS first_launch_success_second_launch_failure both_drained_before_free=1");
    }
    require(!cleanup_failed,"pair cleanup reported a CUDA error");
    puts("PASS model-free prerequisite only; model_layers=0 model_sampling=0 performance_rows=0");
    return 0;
} catch(const std::exception& e) {
    fprintf(stderr,"FAIL %s\n",e.what()); return 1;
}

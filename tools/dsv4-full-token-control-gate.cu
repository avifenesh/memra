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
__global__ void producer(const Dsv4ReplayControl* c, float* x, int rank, int layer, int phase) {
    x[0]=float((c->input.token % 1024) + c->input.position + rank + layer + phase);
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

struct Rank {
    int device;
    cudaStream_t stream{};
    cudaGraph_t forward{}, commit{};
    cudaGraphExec_t forward_exec{}, commit_exec{};
    Dsv4ReplayInput *input{}, *host{};
    Dsv4ReplayControl* control{};
    Fixture* fixture{};
    void* signal{};
    int* error{};
    float *partial{}, *sum{};
    bool quarantined=false;
    explicit Rank(int d):device(d) {
        ck(cudaSetDevice(device)); ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        ck(cudaMalloc(&input,sizeof(*input))); ck(cudaMallocHost(&host,sizeof(*host)));
        ck(cudaMalloc(&control,sizeof(*control))); ck(cudaMalloc(&fixture,sizeof(*fixture)));
        ck(cudaMemset(fixture,0,sizeof(*fixture)));
        ck(cudaMalloc(&signal,memra_tp_ar_signal_bytes()));
        ck(cudaMemset(signal,0,memra_tp_ar_signal_bytes()));
        ck(cudaMalloc(&error,sizeof(int))); ck(cudaMemset(error,0,sizeof(int)));
        ck(cudaMalloc(&partial,sizeof(float))); ck(cudaMalloc(&sum,86*sizeof(float)));
        ck(cudaStreamSynchronize(stream));
    }
    ~Rank() {
        cudaSetDevice(device); cudaStreamSynchronize(stream);
        if(forward_exec) cudaGraphExecDestroy(forward_exec);
        if(commit_exec) cudaGraphExecDestroy(commit_exec);
        if(forward) cudaGraphDestroy(forward);
        if(commit) cudaGraphDestroy(commit);
        cudaFree(input); cudaFreeHost(host); cudaFree(control); cudaFree(fixture);
        cudaFree(signal); cudaFree(error); cudaFree(partial); cudaFree(sum);
        cudaStreamDestroy(stream);
    }
    Rank(const Rank&)=delete;
    Rank& operator=(const Rank&)=delete;
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

void build(Rank& r,Rank& peer) {
    ck(cudaSetDevice(r.device)); ck(cudaGraphCreate(&r.forward,0));
    cudaGraphConditionalHandle c4{},c128{};
    ck(cudaGraphConditionalHandleCreate(&c4,r.forward,0,cudaGraphCondAssignDefault));
    ck(cudaGraphConditionalHandleCreate(&c128,r.forward,0,cudaGraphCondAssignDefault));
    auto tail=kernel(r.forward,nullptr,dsv4_replay_control_kernel,r.input,r.control,c4,c128);
    tail=kernel(r.forward,tail,snapshot,r.control,r.fixture);
    tail=conditional(r.forward,tail,c4,r,true);
    tail=conditional(r.forward,tail,c128,r,false);
    // Existing transport, 86 ordered joins per replay. These producers are
    // integer-exact fixtures, not substitutes for DSV4 layer arithmetic.
    ck(cudaStreamBeginCaptureToGraph(r.stream,r.forward,&tail,nullptr,1,cudaStreamCaptureModeRelaxed));
    for(int layer=0; layer<int(layers); ++layer) for(int phase=0; phase<2; ++phase) {
        producer<<<1,1,0,r.stream>>>(r.control,r.partial,r.device,layer,phase);
        const auto* p0=r.device==0?r.partial:peer.partial;
        const auto* p1=r.device==1?r.partial:peer.partial;
        require(memra_tp_ar_1stage(p0,p1,r.sum+layer*2+phase,r.signal,peer.signal,r.device,1,r.error,
                                  2000000000LL,1,r.stream)==0,"AR capture failed");
    }
    cudaGraph_t ended{}; ck(cudaStreamEndCapture(r.stream,&ended));
    require(ended==r.forward,"capture changed graph owner");
    ck(cudaGraphInstantiate(&r.forward_exec,r.forward,0));
    ck(cudaGraphCreate(&r.commit,0));
    kernel(r.commit,nullptr,finish,r.control,r.fixture);
    ck(cudaGraphInstantiate(&r.commit_exec,r.commit,0));
    size_t n=0; ck(cudaGraphGetNodes(r.forward,nullptr,&n));
    require(n==4+layers*4,"unexpected forward graph node count");
    printf("GRAPH rank=%d forward_nodes=%zu conditional_bodies=2 ar_nodes=86 commit_nodes=1\n",r.device,n);
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
void launch(Rank& r,bool commit) {
    require(!r.quarantined,"quarantined replay refused");
    ck(cudaSetDevice(r.device)); ck(cudaGraphLaunch(commit?r.commit_exec:r.forward_exec,r.stream));
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
bool step(Rank& a,Rank& b,Dsv4ReplayInput in,int inject=-1) {
    upload(a,in); upload(b,in); launch(a,false); launch(b,false);
    // Both ranks are drained/read before ANY commit launch. Host feedback stays.
    std::array<int,2> errors={read(a,a.error),read(b,b.error)};
    check_control(read(a,a.control),in); check_control(read(b,b.control),in);
    for(auto* r:{&a,&b}) {
        auto sums=read(*r,(const std::array<float,86>*)r->sum);
        for(unsigned layer=0;layer<layers;++layer) for(unsigned phase=0;phase<2;++phase)
            require(sums[layer*2+phase]==float(2*((in.token%1024)+in.position+layer+phase)+1),
                    "AR stale input or wrong order");
    }
    if(inject>=0) {
        auto& r=inject?b:a; ck(cudaSetDevice(r.device));
        int code=40043; ck(cudaMemcpyAsync(r.error,&code,sizeof(code),cudaMemcpyHostToDevice,r.stream));
        errors[inject]=read(r,r.error);
    }
    if(errors!=std::array<int,2>{0,0}) {
        for(auto* r:{&a,&b}) {
            ck(cudaSetDevice(r->device)); rollback<<<1,1,0,r->stream>>>(r->fixture);
            ck(cudaStreamSynchronize(r->stream)); r->quarantined=true;
        }
        return false;
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
    require(read(r,seq)==accepted*86,"AR epoch did not advance on device");
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
        Rank a(0),b(1); build(a,b); build(b,a); Fixture expected{};
        for(unsigned p=0;p<steps;++p) {
            Dsv4ReplayInput in{(p*7919+17)%129280,p,0x3fe0000000000000ULL+p*104729,window,4096};
            expected.pending4[4+p%4]=in.token; expected.pending128[p%128]=in.token;
            if((p+1)%4==0) {
                for(int i=0;i<4;++i) expected.pending4[i]=expected.pending4[i+4];
                expected.blocks4=(p+1)/4; ++expected.emits4;
            }
            if((p+1)%128==0) {expected.blocks128=(p+1)/128; ++expected.emits128;}
            expected.ring[p%window]=in.token;
            require(step(a,b,in),"positive replay refused");
            check_fixture(a,expected,in,p+1); check_fixture(b,expected,in,p+1);
        }
        printf("PASS live controls: steps=%u ranks=2 c4=128 c128=4 ring_wraps=4 ar_epochs_per_rank=%u\n",steps,steps*86);
    }
    for(int rank=0;rank<2;++rank) for(unsigned pos:{127u,255u,511u}) {
        Rank a(0),b(1); build(a,b); build(b,a);
        Dsv4ReplayInput before{91,pos-1,0x3fe0000000000001ULL,window,4096};
        require(step(a,b,before),"refusal setup failed");
        std::array<Fixture,2> saved={read(a,a.fixture),read(b,b.fixture)};
        auto refused=before; refused.position=pos; refused.token=104; ++refused.uniform_bits;
        require(!step(a,b,refused,rank),"fault did not refuse");
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
    puts("PASS model-free prerequisite only; model_layers=0 model_sampling=0 performance_rows=0");
    return 0;
} catch(const std::exception& e) {
    fprintf(stderr,"FAIL %s\n",e.what()); return 1;
}

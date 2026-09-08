// Model-free cadence capture prerequisite. Same kernels as the runtime; no rate claim.
// nvcc -t 2 -std=c++17 -O2 -fmad=false -arch=sm_120a \
//   tools/dsv4-replay-cadence-gate.cu -lcublasLt -lcublas -ldl -o <owned-target>/gate
#include "../crates/memra-engine/cu/dsv4_gpu.cu"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

static void check(cudaError_t rc) {
    if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
}
static void kernel_check(int rc) {
    if (rc) throw std::runtime_error("kernel rc=" + std::to_string(rc));
}
static void insist(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
struct Arena {
    cudaStream_t stream{};
    std::vector<void*> buffers;
    void* graphs[4]{};
    void* executable[4]{};
    Arena() { check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
    ~Arena() {
        if (memra_dsv4_replay_abort(stream) || cudaStreamSynchronize(stream) != cudaSuccess) {
            fprintf(stderr, "FATAL cadence component completion unproven; retaining buffers via abort\n");
            std::abort();
        }
        for (int i=0;i<4;++i) memra_dsv4_replay_destroy(graphs[i], executable[i], stream);
        for (auto p:buffers) cudaFree(p);
        cudaStreamDestroy(stream);
    }
    template<class T> T* alloc(size_t n) {
        T* p=nullptr; check(cudaMalloc(&p,n*sizeof(T))); buffers.push_back(p);
        check(cudaMemsetAsync(p,0,n*sizeof(T),stream)); return p;
    }
    template<class T> void write(T* p,const std::vector<T>& v) {
        check(cudaMemcpyAsync(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice,stream));
        check(cudaStreamSynchronize(stream));
    }
    template<class T> std::vector<T> read(const T* p,size_t n) {
        std::vector<T> v(n);
        check(cudaMemcpyAsync(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream)); return v;
    }
    void equal(const float* a,const float* b,size_t n,const char* why) {
        auto x=read(a,n),y=read(b,n); insist(!memcmp(x.data(),y.data(),n*sizeof(float)),why);
    }
};
static std::vector<float> values(size_t n,int salt) {
    std::vector<float> v(n);
    for (size_t i=0;i<n;++i) v[i]=float(int((i*17+salt*13)%193)-96)/128.0f;
    return v;
}

static void compressor(int ratio,int dim,bool rotate) {
    Arena a;
    const bool overlap=ratio==4;
    const int latent=overlap?2*dim:dim, pending_rows=overlap?8:128, rd=64;
    const int store_rows=512/ratio+1;
    auto input=a.alloc<uint64_t>(3), counts=a.alloc<uint64_t>(4);
    auto kv=a.alloc<float>(latent), sc=a.alloc<float>(latent);
    auto ape=a.alloc<float>(ratio*latent), norm=a.alloc<float>(dim), rope=a.alloc<float>(512*rd);
    a.write(ape,values(ratio*latent,7)); a.write(norm,std::vector<float>(dim,1));
    a.write(rope,values(512*rd,13));
    struct Plane { float *kv,*sc,*emit,*shift,*store; int *token,*pos,*slot; };
    auto plane=[&]() {
        Plane p{a.alloc<float>(pending_rows*latent),a.alloc<float>(pending_rows*latent),
            a.alloc<float>((overlap?2:1)*dim),a.alloc<float>(ratio*latent),
            a.alloc<float>(store_rows*dim),a.alloc<int>(1),a.alloc<int>(1),a.alloc<int>(1)};
        // Nonzero prior state makes an accidental clear or inactive emission visible.
        a.write(p.kv,values(pending_rows*latent,31));
        a.write(p.sc,values(pending_rows*latent,37));
        a.write(p.store,values(store_rows*dim,41));
        return p;
    };
    auto full=plane(), candidate=plane();
    unsigned long long nodes[4]{};
    for (int variant=0;variant<4;++variant) {
        auto& p=variant==0?full:candidate;
        const bool emits=variant==0 || variant==3 || (variant==2 && ratio==4);
        check(cudaStreamSynchronize(a.stream));
        kernel_check(memra_dsv4_replay_capture_begin(&a.graphs[variant],a.stream));
        kernel_check(memra_dsv4_replay_input(input,p.token,p.pos,p.slot,128,counts+variant,a.stream));
        kernel_check(memra_dsv4_replay_copy_row(kv,p.kv,p.pos,latent,ratio,overlap?ratio:0,0,a.stream));
        kernel_check(memra_dsv4_replay_copy_row(sc,p.sc,p.pos,latent,ratio,overlap?ratio:0,0,a.stream));
        if (emits) kernel_check(memra_dsv4_replay_compressor_emit(p.kv,p.sc,ape,p.emit,norm,rope,
            p.store,p.shift,p.pos,ratio,dim,latent,overlap,rotate,0,rd,0,1e-6f,
            powf(float(dim),-0.5f),a.stream));
        kernel_check(memra_dsv4_replay_capture_end(a.graphs[variant],&a.executable[variant],a.stream));
        unsigned long long census[7]{};
        kernel_check(memra_dsv4_replay_census(a.graphs[variant],census));
        nodes[variant]=census[1];
        insist(census[6]==0 && census[1]==3+(emits?(5+int(rotate)+4*int(overlap)):0),"cadence node omission");
    }
    insist(a.read(counts,4)==std::vector<uint64_t>(4,0),"capture executed controls");
    a.equal(full.kv,candidate.kv,pending_rows*latent,"capture mutated pending state");
    a.equal(full.store,candidate.store,store_rows*dim,"capture mutated store");
    std::vector<int> positions;
    for(int p=0;p<512;++p) positions.push_back(p);
    // These address tests intentionally reuse nonzero planes at decreasing positions.
    // They do not claim a valid whole-model prefix without a prefix restore.
    for(int p:{300,3,127,128,511,4,383,259,256}) positions.push_back(p);
    std::vector<uint64_t> expected(4,0);
    for(size_t step=0;step<positions.size();++step) {
        int pos=positions[step], variant=(pos+1)%128==0?3:((pos+1)%4==0?2:1);
        unsigned token=unsigned((step*37+13)%4096);
        uint64_t uniform=0x6a09e667f3bcc909ull ^ ((step+1)*0x9e3779b97f4a7c15ull);
        std::vector<uint64_t> words{token|(static_cast<uint64_t>(pos)<<32),uniform,0};
        a.write(input,words); a.write(kv,values(latent,int(step))); a.write(sc,values(latent,int(step)+7));
        kernel_check(memra_dsv4_replay_launch(a.executable[0],a.stream));
        kernel_check(memra_dsv4_replay_launch(a.executable[variant],a.stream));
        a.equal(full.kv,candidate.kv,pending_rows*latent,"pending KV differs from full replay");
        a.equal(full.sc,candidate.sc,pending_rows*latent,"pending score differs from full replay");
        a.equal(full.store,candidate.store,store_rows*dim,"emitted store differs from full replay");
        insist(a.read(input,3)==words,"full 64-bit uniform/input freshness");
        for(auto* p:{&full,&candidate}) {
            insist(a.read(p->token,1)[0]==int(token),"token frozen");
            insist(a.read(p->pos,1)[0]==pos,"position frozen");
            insist(a.read(p->slot,1)[0]==pos%128,"ring slot frozen");
        }
        ++expected[0]; ++expected[variant];
        insist(a.read(counts,4)==expected,"actual variant counter differs");
    }
    printf("PASS cadence compressor ratio=%d dim=%d rotate=%d positions=%zu captures=4 nodes=[%llu,%llu,%llu,%llu] replays=[%llu,%llu,%llu,%llu] full_replay_bit_equal=1 decreasing_positions=1 full_u64_uniform_fresh=1 actual_sampling=0 model_layers=0\n",
        ratio,dim,int(rotate),positions.size(),nodes[0],nodes[1],nodes[2],nodes[3],
        static_cast<unsigned long long>(expected[0]),static_cast<unsigned long long>(expected[1]),
        static_cast<unsigned long long>(expected[2]),static_cast<unsigned long long>(expected[3]));
}
int main() try {
    int n=0;check(cudaGetDeviceCount(&n));insist(n==2,"exact visible pair required");
    for(int rank=0;rank<2;++rank) {
        check(cudaSetDevice(rank));
        compressor(4,512,false); compressor(128,512,false); compressor(4,128,true);
        printf("PASS rank=%d cadence real-compressor component only\n",rank);
    }
    return 0;
} catch(const std::exception& e) { fprintf(stderr,"FAIL %s\n",e.what());return 1; }

// Same existing device-f64-exp-tree-cdf-v1 sampler, by-value vs live pointer.
#define main dsv4_live_gate_unused_main
#include "dsv4-replay-live-kernel-gate.cu"
#undef main
#include "../crates/memra-engine/cu/dsv4_sampler.cu"
#include <set>

void sampler(int n){
    Arena a; auto logits=a.alloc<float>(n); auto uniform=a.alloc<double>(1);
    auto counts=a.alloc<int>(n); a.write(logits,values(n,17));
    struct Scratch{float* values;uint64_t *keys0,*keys1;double *prefix,*blocks;unsigned* result;};
    auto alloc=[&](){return Scratch{a.alloc<float>(n),a.alloc<uint64_t>(n),a.alloc<uint64_t>(n),
        a.alloc<double>(n),a.alloc<double>((n+255)/256+2),a.alloc<unsigned>(3)};};
    auto eager=alloc(),graph=alloc();
    a.write(eager.result,std::vector<unsigned>{0,0,0x5a17cafe});
    a.write(graph.result,std::vector<unsigned>{0,0,0x5a17cafe});
    a.begin();
    kernel_check(memra_dsv4_sample_device_replay(logits,graph.values,graph.keys0,graph.keys1,
        graph.prefix,graph.blocks,counts,graph.result,n,n,1.0,1.0,uniform,a.stream));
    a.end();
    std::set<unsigned> tokens;
    for(double u:{0.0,0.01,0.234567891234567,0.7777777712345,0.98,0.9999999999999999,0.51,0.11}){
        a.write(uniform,std::vector<double>{u});
        kernel_check(memra_dsv4_sample_device(logits,eager.values,eager.keys0,eager.keys1,
            eager.prefix,eager.blocks,counts,eager.result,n,n,1.0,1.0,u,1,0,0,a.stream));
        a.replay();
        a.equal(eager.result,graph.result,3,"sampler token/refusal/canary differs");
        a.equal(eager.prefix,graph.prefix,n,"sampler prefix tree differs");
        a.equal(eager.blocks,graph.blocks,(n+255)/256+2,"sampler block sums differ");
        auto result=a.read(graph.result,3);
        insist(result[0]<unsigned(n) && result[1]==0 && result[2]==0x5a17cafe,"sampler result/canary invalid");
        tokens.insert(result[0]);
    }
    insist(tokens.size()>1,"constant-logit changing-uniform test was vacuous");
    printf("PASS real sampler n=%d draws=8 distinct_tokens=%zu fixed_logits=1 capture_count=1 uniform_pointer_fresh=1\n",n,tokens.size());
}
int main() try {
    int n=0; check(cudaGetDeviceCount(&n)); insist(n==2,"requires exact visible pair");
    for(int rank=0;rank<2;++rank){check(cudaSetDevice(rank)); sampler(257); sampler(129280); printf("PASS sampler rank=%d\n",rank);}
    return 0;
}catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}

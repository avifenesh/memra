// Real-kernel controls prerequisite. No model weights, throughput, or serving claim.
// nvcc -t 2 -std=c++17 -O2 -fmad=false -arch=sm_120a \
//   tools/dsv4-replay-live-kernel-gate.cu -lcublasLt -lcublas -ldl -o <owned-target>/gate
#include "../crates/memra-engine/cu/dsv4_gpu.cu"
#include <cstdio>
#include <climits>
#include <stdexcept>
#include <string>

static void check(cudaError_t e) {
    if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
static void kernel_check(int e) {
    if(e) throw std::runtime_error("kernel rc="+std::to_string(e));
}
static void insist(bool ok,const char* why) {if(!ok) throw std::runtime_error(why);}
struct Arena {
    cudaStream_t stream{};
    std::vector<void*> allocations;
    std::vector<std::pair<void*,void*>> graphs;
    Arena(){check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));}
    ~Arena(){
        // No peer pointers in this single-device component. End an incomplete
        // capture before draining; executable graphs die before all their buffers.
        memra_dsv4_replay_abort(stream);
        cudaStreamSynchronize(stream);
        for(auto [graph,exec]:graphs) memra_dsv4_replay_destroy(graph,exec,stream);
        for(auto p:allocations) cudaFree(p);
        cudaStreamDestroy(stream);
    }
    template<class T> T* alloc(size_t n){
        T* p=nullptr; check(cudaMalloc(&p,n*sizeof(T))); allocations.push_back(p);
        check(cudaMemsetAsync(p,0,n*sizeof(T),stream)); return p;
    }
    template<class T> void write(T* p,const std::vector<T>& v){
        check(cudaMemcpyAsync(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice,stream));
        check(cudaStreamSynchronize(stream));
    }
    template<class T> std::vector<T> read(const T* p,size_t n){
        std::vector<T> v(n); check(cudaMemcpyAsync(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream)); return v;
    }
    template<class T> void equal(const T* a,const T* b,size_t n,const char* why){
        auto va=read(a,n),vb=read(b,n); insist(!memcmp(va.data(),vb.data(),n*sizeof(T)),why);
    }
    void begin(){
        check(cudaStreamSynchronize(stream)); graphs.push_back({nullptr,nullptr});
        kernel_check(memra_dsv4_replay_capture_begin(&graphs.back().first,stream));
    }
    void end(){
        kernel_check(memra_dsv4_replay_capture_end(graphs.back().first,&graphs.back().second,stream));
        unsigned long long census[7]{};
        kernel_check(memra_dsv4_replay_census(graphs.back().first,census));
        insist(census[0]>0 && census[1]>0 && census[6]==0,"non-flat or empty graph census");
    }
    void replay(){kernel_check(memra_dsv4_replay_launch(graphs.back().second,stream));}
};
static std::vector<float> values(size_t n,int salt){
    std::vector<float> v(n);
    for(size_t i=0;i<n;++i) v[i]=float(int((i*17+salt*13)%193)-96)/128.0f;
    return v;
}

void compressor(int ratio,int dim,bool rotate){
    Arena a; const int overlap=ratio==4,latent=overlap?2*dim:dim,rows=overlap?2*ratio:ratio;
    const int positions=513,rd=64,store_rows=positions/ratio+1;
    auto pos=a.alloc<int>(1); auto kv=a.alloc<float>(latent),sc=a.alloc<float>(latent);
    auto ape=a.alloc<float>(ratio*latent),norm=a.alloc<float>(dim),cs=a.alloc<float>(positions*rd);
    a.write(ape,values(ratio*latent,7)); a.write(norm,std::vector<float>(dim,1));
    auto table=values(positions*rd,13); a.write(cs,table);
    struct Plane{float *kv,*sc,*emit,*shift,*store;};
    auto plane=[&](){return Plane{a.alloc<float>(rows*latent),a.alloc<float>(rows*latent),
        a.alloc<float>((overlap?2:1)*dim),a.alloc<float>(ratio*latent),
        a.alloc<float>(store_rows*dim)};};
    auto eager=plane(),graph=plane();
    auto emission=[&](Plane& p,int host_pos){
        kernel_check(memra_dsv4_compressor_pool(p.kv,p.sc,ape,p.emit,overlap?2:1,ratio,dim,latent,overlap,a.stream));
        float* row=p.emit+(overlap?dim:0);
        kernel_check(memra_dsv4_rmsnorm_f32acc(row,norm,row,1,dim,1e-6f,a.stream));
        kernel_check(memra_dsv4_rope_at(row,1,dim,rd,cs,host_pos/ratio*ratio,0,a.stream));
        if(rotate){
            kernel_check(memra_dsv4_hadamard(row,1,dim,powf(float(dim),-0.5f),a.stream));
            kernel_check(memra_dsv4_fp4_act_quant(row,1,dim,dim,a.stream));
        }else kernel_check(memra_dsv4_act_quant(row,1,dim,dim-rd,64,0,a.stream));
        check(cudaMemcpyAsync(p.store+(host_pos/ratio)*dim,row,dim*sizeof(float),cudaMemcpyDeviceToDevice,a.stream));
        if(overlap) for(float* pending:{p.kv,p.sc}){
            check(cudaMemcpyAsync(p.shift,pending+ratio*latent,ratio*latent*sizeof(float),cudaMemcpyDeviceToDevice,a.stream));
            check(cudaMemcpyAsync(pending,p.shift,ratio*latent*sizeof(float),cudaMemcpyDeviceToDevice,a.stream));
        }
    };
    // The model path captures only after priming. Warm the identical emission
    // functions on disposable eager scratch, then restore it before the oracle
    // sequence. Capture itself still executes no cache mutation.
    emission(eager,ratio-1);
    check(cudaStreamSynchronize(a.stream));
    check(cudaMemsetAsync(eager.kv,0,rows*latent*sizeof(float),a.stream));
    check(cudaMemsetAsync(eager.sc,0,rows*latent*sizeof(float),a.stream));
    check(cudaMemsetAsync(eager.emit,0,(overlap?2:1)*dim*sizeof(float),a.stream));
    check(cudaMemsetAsync(eager.shift,0,ratio*latent*sizeof(float),a.stream));
    check(cudaMemsetAsync(eager.store,0,store_rows*dim*sizeof(float),a.stream));
    a.begin();
    kernel_check(memra_dsv4_replay_copy_row(kv,graph.kv,pos,latent,ratio,overlap?ratio:0,0,a.stream));
    kernel_check(memra_dsv4_replay_copy_row(sc,graph.sc,pos,latent,ratio,overlap?ratio:0,0,a.stream));
    kernel_check(memra_dsv4_replay_compressor_emit(graph.kv,graph.sc,ape,graph.emit,norm,cs,
        graph.store,graph.shift,pos,ratio,dim,latent,overlap,rotate,0,rd,0,1e-6f,
        powf(float(dim),-0.5f),a.stream));
    a.end();
    auto initial=a.read(graph.store,store_rows*dim);
    for(float v:initial) insist(v==0.0f,"capture executed compressor body");
    for(int p=0;p<positions;++p){
        a.write(pos,std::vector<int>{p}); a.write(kv,values(latent,p)); a.write(sc,values(latent,p+7));
        int slot=(overlap?ratio:0)+p%ratio;
        check(cudaMemcpyAsync(eager.kv+slot*latent,kv,latent*sizeof(float),cudaMemcpyDeviceToDevice,a.stream));
        check(cudaMemcpyAsync(eager.sc+slot*latent,sc,latent*sizeof(float),cudaMemcpyDeviceToDevice,a.stream));
        if((p+1)%ratio==0) emission(eager,p);
        a.replay();
        {
            a.equal(eager.kv,graph.kv,rows*latent,"pending KV mismatch");
            a.equal(eager.sc,graph.sc,rows*latent,"pending score mismatch");
            a.equal(eager.store,graph.store,store_rows*dim,"compressed store mismatch");
        }
        if((p+1)%ratio==0){
            auto row=a.read(graph.store+(p/ratio)*dim,dim);
            bool nonzero=false; for(float v:row) nonzero|=v!=0.0f;
            insist(nonzero,"compressor emission did not engage");
        }
    }
    printf("PASS compressor ratio=%d dim=%d rotate=%d positions=513 real_pool_norm_rope_quant=1 capture_count=1 uniform_device_predicate=1\n",ratio,dim,rotate);
}

void attention(int ratio,bool fine){
    Arena a; const int win=128,limit=512,nb_max=ratio?limit/ratio:0;
    const int heads=32,hd=512,ih=64,ihd=128,topk=fine?512:INT_MAX;
    const int slots_max=win+nb_max,trans_base=slots_max+8;
    auto pos=a.alloc<int>(1);
    auto q=a.alloc<float>(heads*hd),kv=a.alloc<float>((trans_base+1)*hd);
    auto sink=a.alloc<float>(heads),iq=a.alloc<float>(ih*ihd),ikv=a.alloc<float>((limit/4)*ihd),w=a.alloc<float>(ih);
    a.write(kv,values((trans_base+1)*hd,23)); a.write(ikv,values((limit/4)*ihd,29));
    a.write(sink,values(heads,7)); a.write(w,values(ih,5));
    struct Plane{int* idx;float *score,*eval,*den,*out,*iscore;};
    auto plane=[&](){return Plane{a.alloc<int>(slots_max),a.alloc<float>(heads*slots_max),
        a.alloc<float>(heads*slots_max),a.alloc<float>(heads),a.alloc<float>(heads*hd),a.alloc<float>(limit/4)};};
    auto eager=plane(),graph=plane(); const float scale=powf(float(hd),-0.5f),iscale=0.011048543f;
    // q is arbitrary test data and BOTH arms below read it through the same scorer, which now
    // takes [nq][hd][heads]. This gate compares a captured graph against an eager replay, so
    // the layout cancels: it is not staged here, and staging it would change nothing.
    a.begin();
    kernel_check(memra_dsv4_replay_indices(graph.idx,pos,win,ratio,slots_max,slots_max,trans_base,fine,topk,a.stream));
    if(fine) kernel_check(memra_dsv4_replay_indexer(iq,ikv,w,iscale,graph.iscore,graph.idx+win,pos,ih,ihd,nb_max,ratio,topk,win,a.stream));
    kernel_check(memra_dsv4_replay_attention(q,kv,graph.idx,sink,graph.score,graph.eval,graph.den,graph.out,pos,
        heads,hd,slots_max,slots_max,scale,win,ratio,topk,a.stream));
    a.end();
    for(int p:{0,1,2,3,4,126,127,128,255,256,383,511,300,4}){
        a.write(pos,std::vector<int>{p}); a.write(q,values(heads*hd,p)); a.write(iq,values(ih*ihd,p+2));
        int nb=ratio?(p+1)/ratio:0,slots=win+nb;
        if(fine) kernel_check(memra_dsv4_build_idx_redirect(eager.idx,p,win,0,slots,p,trans_base,a.stream));
        else kernel_check(memra_dsv4_build_idx_redirect_m(eager.idx,p,1,win,ratio,slots,slots_max,trans_base,0,a.stream));
        if(fine && nb){
            kernel_check(memra_dsv4_indexer_score_f32acc(iq,ikv,w,iscale,eager.iscore,1,ih,ihd,nb,ratio,nb,a.stream));
            kernel_check(memra_dsv4_topk_idx_numeric(eager.iscore,nb,nb,win,eager.idx+win,a.stream));
        }
        kernel_check(memra_dsv4_sink_attn_dec_mq_f32acc(q,kv,eager.idx,sink,eager.score,eager.eval,eager.den,eager.out,
            1,heads,hd,slots,slots_max,scale,a.stream));
        a.replay();
        a.equal(eager.idx,graph.idx,slots,"redirect/selector mismatch");
        if(fine && nb) a.equal(eager.iscore,graph.iscore,nb,"indexer score mismatch");
        a.equal(eager.score,graph.score,heads*slots,"attention scores mismatch");
        a.equal(eager.eval,graph.eval,heads*slots,"attention eval mismatch");
        a.equal(eager.den,graph.den,heads,"attention denominator mismatch");
        a.equal(eager.out,graph.out,heads*hd,"attention output mismatch");
    }
    printf("PASS attention ratio=%d fine=%d changing_bounds_and_positions=14 capture_count=1 all_intermediates_bit_equal=1\n",ratio,fine);
}
int main() try {
    int n=0;check(cudaGetDeviceCount(&n));insist(n==2,"requires exact visible pair");
    for(int rank=0;rank<2;++rank){
        check(cudaSetDevice(rank));
        compressor(4,512,false); compressor(128,512,false); compressor(4,128,true);
        attention(0,false); attention(128,false); attention(4,true);
        printf("PASS rank=%d real-kernel controls only model_layers=0\n",rank);
    }
    return 0;
}catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}

// Correctness only: partitioned CSR and local-bank MMA versus global-bank MMA.
// Graph coverage is routing metadata only; no full-MoE graph claim.
#include "../crates/memra-engine/cu/dsv4_gpu.cu"
#include "../crates/memra-engine/cu/moe_f16_grouped.cu"
#include <algorithm>
#include <climits>
#include <cstdio>
#include <stdexcept>
#include <vector>

static void check(cudaError_t rc) { if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc)); }
static void ok(int rc) { if (rc) throw std::runtime_error("kernel rc=" + std::to_string(rc)); }
template<class T> struct Dev {
    T* p; size_t n;
    explicit Dev(size_t size) : n(size) { check(cudaMalloc(&p,n*sizeof(T))); check(cudaMemset(p,0,n*sizeof(T))); }
    ~Dev(){cudaFree(p);}
    Dev(const Dev&)=delete;
    void put(const std::vector<T>& v){if(v.size()>n)throw std::runtime_error("upload size");check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
    std::vector<T> get(){std::vector<T> v(n);check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
template<class T> static void same(const std::vector<T>& a,const std::vector<T>& b,const char* what){
    if(a.size()!=b.size()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T)))throw std::runtime_error(what);
}
template<class T> static std::vector<T> guarded(size_t n) {
    std::vector<T> v(n); std::memset(v.data(),0xa5,n*sizeof(T)); return v;
}

static void cell(int rows,int ne,int first,int count,int cross,int tail,int kdim=128,int ndim=70) {
    const int topk=std::min(6,ne),slots=rows*topk,guard=7;
    Dev<int> selected(slots),counts(count+guard),offsets(count+1+guard),ids(count+guard),
        pairs(slots+guard),tokens(slots+guard),status(1+guard),half_status(slots),global_ids(count);
    Dev<float> weights(slots),s2(ne*3),rw(slots+guard),s1(slots+guard),sdown(slots+guard),s3(slots+guard),
        as(rows*kdim/128),rs(slots),out(slots*ndim+guard),ref(slots*ndim+guard);
    Dev<uint8_t> codes(rows*kdim),w(ne*3*ndim*kdim/2),sc(ne*3*ndim*kdim/16);
    Dev<__half> act(slots*kdim);
    Dev<unsigned long long> local_table(count*6),full_table(ne*6);
    std::vector<float> weights_h(slots),s2_h(ne*3);
    std::vector<uint8_t> codes_h(codes.n),w_h(w.n),sc_h(sc.n);
    for(size_t i=0;i<w_h.size();++i){
        size_t expert=(i/(ndim*kdim/2))%ne,projection=i/(ne*ndim*kdim/2);
        w_h[i]=uint8_t((i*37+17+expert*13+projection*59)%256);
    }
    for(size_t i=0;i<sc_h.size();++i){
        size_t expert=(i/(ndim*kdim/16))%ne,projection=i/(ne*ndim*kdim/16);
        sc_h[i]=uint8_t((5+(i+expert*7+projection*3)%4)<<3);
    }
    for(size_t i=0;i<codes_h.size();++i)codes_h[i]=uint8_t((i*53+7)%126);
    for(int p=0;p<slots;++p)weights_h[p]=p%5 ? float(p%19)/32.0f : -0.0f;
    for(int e=0;e<ne*3;++e)s2_h[e]=std::ldexp(1.0f,e%9-8);
    weights.put(weights_h);s2.put(s2_h);codes.put(codes_h);as.put(std::vector<float>(as.n,1.0f));w.put(w_h);sc.put(sc_h);
    std::vector<unsigned long long> full(ne*6),local(count*6);
    std::vector<int> gids(count);
    for(int proj=0;proj<3;++proj)for(int e=0;e<ne;++e){
        full[(proj*2)*ne+e]=(unsigned long long)(w.p+(proj*ne+e)*ndim*kdim/2);
        full[(proj*2+1)*ne+e]=(unsigned long long)(sc.p+(proj*ne+e)*ndim*kdim/16);
    }
    for(int e=0;e<count;++e){gids[e]=first+e;for(int plane=0;plane<6;++plane)local[plane*count+e]=full[plane*ne+first+e];}
    local_table.put(local);full_table.put(full);global_ids.put(gids);
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto route=[&](int begin,int size,int k){return memra_dsv4_grouped_routes_partition(
        selected.p,weights.p,s2.p,counts.p,offsets.p,ids.p,pairs.p,tokens.p,rw.p,
        s1.p,sdown.p,s3.p,status.p,slots,ne,begin,size,k,stream);};
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    int empty=0;
    for(int pattern=0;pattern<7;++pattern){
        int outside=first>0?first-1:first+count<ne?first+count:0;
        std::vector<int> sel(slots);
        for(int p=0;p<slots;++p)sel[p]=pattern==0?first:pattern==1?outside:pattern==2?first+count-1:
            pattern==3?p%ne:pattern==4?(p%7?outside:first+count-1):pattern==5?(p*97+13)%ne:(p%2?outside:first);
        auto ec=guarded<int>(counts.n),eo=guarded<int>(offsets.n),ei=guarded<int>(ids.n),
            ep=guarded<int>(pairs.n),et=guarded<int>(tokens.n),es=guarded<int>(status.n);
        auto ew=guarded<float>(rw.n),e1=guarded<float>(s1.n),e2=guarded<float>(sdown.n),e3=guarded<float>(s3.n);
        int live=0,max_m=0;eo[0]=0;es[0]=0;
        for(int e=0;e<count;++e){
            int before=live;ei[e]=e;
            for(int p=0;p<slots;++p)if(sel[p]==first+e){
                ep[live]=p;et[live]=p/topk;ew[live]=weights_h[p];e1[live]=s2_h[(first+e)*3];
                e2[live]=s2_h[(first+e)*3+1];e3[live]=s2_h[(first+e)*3+2];++live;
            }
            ec[e]=live-before;max_m=std::max(max_m,ec[e]);eo[e+1]=live;
        }
        if(!live)++empty;
        selected.put(sel);
        for(auto* v:{&counts,&offsets,&ids,&pairs,&tokens,&status})check(cudaMemset(v->p,0xa5,v->n*sizeof(int)));
        for(auto* v:{&rw,&s1,&sdown,&s3})check(cudaMemset(v->p,0xa5,v->n*sizeof(float)));
        check(cudaDeviceSynchronize());
        if(pattern==0){
            ok(route(first,count,topk));check(cudaStreamSynchronize(stream));
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));ok(route(first,count,topk));
            check(cudaStreamEndCapture(stream,&graph));check(cudaGraphInstantiate(&executable,graph,0));
        }
        check(cudaGraphLaunch(executable,stream));check(cudaStreamSynchronize(stream));
        same(counts.get(),ec,"partition counts and guard");same(offsets.get(),eo,"partition offsets/live count and guard");
        same(ids.get(),ei,"local group ids and guard");same(pairs.get(),ep,"original slots and untouched tail");
        same(tokens.get(),et,"source token ids and untouched tail");same(status.get(),es,"valid global-id status and guard");
        same(rw.get(),ew,"route weight bits");same(s1.get(),e1,"global scale1 bits");
        same(sdown.get(),e2,"global scale2 bits");same(s3.get(),e3,"global scale3 bits");
        if(live){
            ok(memra_dsv4_fp8_gather_half(codes.p,as.p,tokens.p,act.p,rs.p,half_status.p,live,kdim,stream));
            check(cudaStreamSynchronize(stream));
            auto hs=half_status.get();for(int p=0;p<live;++p)if(hs[p])throw std::runtime_error("half transport rejection");
        }
        for(int proj=0;proj<3;++proj){
            check(cudaMemset(out.p,0xa5,out.n*sizeof(float)));check(cudaMemset(ref.p,0xa5,ref.n*sizeof(float)));
            check(cudaDeviceSynchronize());
            // The device-count visitor must also safely do no work for an empty rank.
            ok(memra_moe_kq_gemm_sk(local_table.p,proj,count,ids.p,act.p,out.p,rs.p,offsets.p,nullptr,
                count,slots,kdim,ndim,QT_NVFP4_MODELOPT,cross,tail,kdim/2,stream));
            if(live)ok(memra_moe_kq_gemm_sk(full_table.p,proj,ne,global_ids.p,act.p,ref.p,rs.p,offsets.p,eo.data(),
                count,max_m,kdim,ndim,QT_NVFP4_MODELOPT,cross,tail,kdim/2,stream));
            check(cudaStreamSynchronize(stream));
            auto got=out.get(),expected=ref.get();same(got,expected,"local-bank versus global-bank matrix output");
            auto tail_bits=guarded<float>(got.size()-live*ndim);
            same(std::vector<float>(got.begin()+live*ndim,got.end()),tail_bits,"independent unowned/output tail guard");
            if(pattern==0&&proj==0&&ne>1){
                // A wrong half-bank must be visible, not masked by periodic fixture weights.
                auto broken=local;int wrong_expert=(first+ne/2)%ne;
                for(int plane=0;plane<6;++plane)broken[plane*count]=full[plane*ne+wrong_expert];
                local_table.put(broken);check(cudaDeviceSynchronize());
                ok(memra_moe_kq_gemm_sk(local_table.p,proj,count,ids.p,act.p,out.p,rs.p,offsets.p,nullptr,
                    count,slots,kdim,ndim,QT_NVFP4_MODELOPT,cross,tail,kdim/2,stream));
                check(cudaStreamSynchronize(stream));auto wrong=out.get();
                if(!std::memcmp(wrong.data(),expected.data(),live*ndim*sizeof(float)))
                    throw std::runtime_error("corrupted shard pointer table was invisible to the fixture");
                local_table.put(local);check(cudaDeviceSynchronize());
            }
        }
    }
    if(count<ne&&!empty)throw std::runtime_error("empty-rank control did not engage");
    for(int bad:{-1,ne,INT_MAX}){
        std::vector<int> invalid(slots,first);invalid.back()=bad;selected.put(invalid);check(cudaDeviceSynchronize());
        ok(route(first,count,topk));check(cudaStreamSynchronize(stream));if(status.get()[0]!=1)throw std::runtime_error("invalid GLOBAL id not rejected");
    }
    selected.put(std::vector<int>(slots,first));check(cudaDeviceSynchronize());ok(route(first,count,topk));
    check(cudaStreamSynchronize(stream));if(status.get()[0]!=0)throw std::runtime_error("invalid status was not cleared");
    for(auto shape:std::vector<std::vector<int>>{{-1,count,topk},{ne,count,topk},{first,0,topk},{first,ne+1,topk},{first,count,0},{first,count,ne+1}})
        if(route(shape[0],shape[1],shape[2])!=40004)throw std::runtime_error("invalid partition shape accepted");
    check(cudaGraphExecDestroy(executable));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
    printf("PASS rows=%d global=%d first=%d count=%d cross=%d tail=%d kdim=%d ndim=%d patterns=7 empty=%d projections=3 metadata_graph=7 invalid_ids=3 pointer_red=%d\n",
        rows,ne,first,count,cross,tail,kdim,ndim,empty,int(ne>1));
}
int main(){try{
    for(int rows:{1,32,512})for(int first:{0,128,255})for(int cross:{1,64,INT_MAX})for(int tail:{0,1})
        cell(rows,256,first,first==255?1:128,cross,tail);
    cell(33,8,3,2,64,1);cell(1,1,0,1,64,1);
    for(int rows:{1,6,32}) {
        const int cross=rows==32 ? 64 : INT_MAX;
        cell(rows,8,0,4,cross,1,4096,2048);
        cell(rows,8,0,4,cross,1,2048,4096);
    }
    puts("PASS partitioned GPU CSR and local/global matrix identity; full matrix-EP integration remains unqualified");
}catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}

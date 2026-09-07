// Correctness only: stable routing, device-count grouped MMA, live graph replay.
#include "../crates/memra-engine/cu/dsv4_gpu.cu"
#include "../crates/memra-engine/cu/moe_f16_grouped.cu"
#include <algorithm>
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

static void cell(int rows,int ne,int topk,int cross,int tail,bool graph) {
    const int slots=rows*topk,kdim=128,ndim=70,guard=7;
    Dev<int> selected(slots),counts(ne),offsets(ne+1),ids(ne),pairs(slots),tokens(slots),status(1),half_status(slots);
    Dev<float> weights(slots),s2(ne*3),rw(slots),s1(slots),sdown(slots),s3(slots),as(rows),rs(slots),out(slots*ndim+guard),ref(slots*ndim+guard);
    Dev<uint8_t> codes(rows*kdim),w(ne*3*ndim*kdim/2),sc(ne*3*ndim*kdim/16);
    Dev<__half> act(slots*kdim);
    Dev<unsigned long long> table(ne*6);
    std::vector<float> weights_h(slots),s2_h(ne*3),ones(rows,1.0f);
    std::vector<uint8_t> codes_h(rows*kdim),w_h(w.n),sc_h(sc.n);
    for(size_t i=0;i<w_h.size();++i)w_h[i]=(i*37+17)%256;
    for(size_t i=0;i<sc_h.size();++i)sc_h[i]=uint8_t((5+i%4)<<3);
    for(size_t i=0;i<codes_h.size();++i)codes_h[i]=uint8_t((i*53+7)%126);
    for(int p=0;p<slots;++p)weights_h[p]=p%5 ? float(p%19)/32.0f : -0.0f;
    for(int e=0;e<ne*3;++e)s2_h[e]=std::ldexp(1.0f,e%9-8);
    weights.put(weights_h);s2.put(s2_h);codes.put(codes_h);as.put(ones);w.put(w_h);sc.put(sc_h);
    std::vector<unsigned long long> tab(ne*6);
    for(int proj=0;proj<3;++proj)for(int e=0;e<ne;++e){
        tab[(proj*2)*ne+e]=(unsigned long long)(w.p+(proj*ne+e)*ndim*kdim/2);
        tab[(proj*2+1)*ne+e]=(unsigned long long)(sc.p+(proj*ne+e)*ndim*kdim/16);
    }
    table.put(tab);
    cudaStream_t stream;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto route=[&](){ok(memra_dsv4_grouped_routes(selected.p,weights.p,s2.p,counts.p,offsets.p,ids.p,
        pairs.p,tokens.p,rw.p,s1.p,sdown.p,s3.p,status.p,slots,ne,topk,stream));};
    auto run=[&](){
        route();
        ok(memra_dsv4_fp8_gather_half(codes.p,as.p,tokens.p,act.p,rs.p,half_status.p,slots,kdim,stream));
        ok(memra_moe_kq_gemm_sk(table.p,0,ne,ids.p,act.p,out.p,rs.p,offsets.p,nullptr,
            ne,slots,kdim,ndim,QT_NVFP4_MODELOPT,cross,tail,kdim/2,stream));
    };
    cudaGraph_t captured=nullptr;cudaGraphExec_t executable=nullptr;
    for(int pattern=0;pattern<5;++pattern){
        std::vector<int> sel(slots),expected_offsets{0},expected_pairs,expected_tokens,expected_ids(ne);
        std::vector<float> expected_w,expected_s1,expected_s2,expected_s3;
        for(int p=0;p<slots;++p)sel[p]=pattern==0?ne-1:pattern==1?0:pattern==2?p%ne:pattern==3?(p%11?0:ne-1):(p*97+13)%ne;
        for(int e=0;e<ne;++e){
            expected_ids[e]=e;
            for(int p=0;p<slots;++p)if(sel[p]==e){expected_pairs.push_back(p);expected_tokens.push_back(p/topk);
                expected_w.push_back(weights_h[p]);expected_s1.push_back(s2_h[e*3]);
                expected_s2.push_back(s2_h[e*3+1]);expected_s3.push_back(s2_h[e*3+2]);}
            expected_offsets.push_back(expected_pairs.size());
        }
        selected.put(sel);
        check(cudaMemset(out.p,0xa5,out.n*sizeof(float)));check(cudaMemset(ref.p,0xa5,ref.n*sizeof(float)));
        check(cudaDeviceSynchronize()); // default-stream inputs precede non-blocking work
        if(graph&&pattern==0){
            run();check(cudaStreamSynchronize(stream)); // attributes and kernel modules ready
            check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));run();
            check(cudaStreamEndCapture(stream,&captured));
            check(cudaGraphInstantiate(&executable,captured,0));
        }
        if(graph)check(cudaGraphLaunch(executable,stream));else run();
        check(cudaStreamSynchronize(stream));
        same(offsets.get(),expected_offsets,"CSR offsets");same(ids.get(),expected_ids,"expert ids");
        same(pairs.get(),expected_pairs,"stable pairs");same(tokens.get(),expected_tokens,"token ids");
        same(rw.get(),expected_w,"route weight bits");same(s1.get(),expected_s1,"scale1 bits");
        same(sdown.get(),expected_s2,"scale2 bits");same(s3.get(),expected_s3,"scale3 bits");
        if(status.get()[0])throw std::runtime_error("spurious route rejection");
        for(int v:half_status.get())if(v)throw std::runtime_error("half transport rejection");
        int max_m=0;for(int e=0;e<ne;++e)max_m=std::max(max_m,expected_offsets[e+1]-expected_offsets[e]);
        ok(memra_moe_kq_gemm_sk(table.p,0,ne,ids.p,act.p,ref.p,rs.p,offsets.p,expected_offsets.data(),
            ne,max_m,kdim,ndim,QT_NVFP4_MODELOPT,cross,tail,kdim/2,stream));
        check(cudaStreamSynchronize(stream));
        auto got=out.get(),expected=ref.get();same(got,expected,"device-count MMA output");
        for(size_t at=slots*ndim;at<got.size();++at){
            uint32_t a,b;std::memcpy(&a,&got[at],4);std::memcpy(&b,&expected[at],4);
            if(a!=0xa5a5a5a5u||b!=0xa5a5a5a5u)throw std::runtime_error("matrix tail guard overwritten");
        }
    }
    for(int bad:{-1,ne}){
        std::vector<int> invalid(slots,0);invalid[slots-1]=bad;selected.put(invalid);check(cudaDeviceSynchronize());
        route();check(cudaStreamSynchronize(stream));if(status.get()[0]!=1)throw std::runtime_error("invalid route not rejected");
    }
    if(executable)check(cudaGraphExecDestroy(executable));if(captured)check(cudaGraphDestroy(captured));
    check(cudaStreamDestroy(stream));
    printf("PASS rows=%d experts=%d topk=%d cross=%d tail=%d graph=%d patterns=5 invalid=2\n",rows,ne,topk,cross,tail,int(graph));
}
int main(){try{
    for(int rows:{1,6,32,33,128,512})for(int cross:{1,64,0x7fffffff})for(int tail:{0,1})cell(rows,8,6,cross,tail,false);
    for(int cross:{1,64,0x7fffffff})for(int tail:{0,1})cell(33,256,6,cross,tail,true);
    cell(1,1,1,64,1,true);
    puts("PASS stable GPU CSR, all three matrix visitors, live-routing graph replay and invalid-id guards");
}catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}}

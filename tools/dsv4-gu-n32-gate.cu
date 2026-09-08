// Standalone exact N32 falsification over frozen native routed GU operands.
// Build with the real grouped TU and link dsv4_gpu.cu's unchanged FP8/scale/scatter
// helpers. No synthetic-control replacement, no scalar half transport shortcut.
#include "../crates/memra-engine/cu/moe_f16_grouped.cu"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <stdexcept>
#include <limits>

extern "C" int memra_dsv4_act_quant_fp8(const float*,void*,float*,int,int,void*);
extern "C" int memra_dsv4_fp8_gather_half(const void*,const float*,const int*,void*,float*,int*,int,int,void*);
extern "C" int memra_dsv4_scale_rows(float*,const float*,int,int,void*);
extern "C" int memra_dsv4_scatter_rows(const float*,float*,const int*,int,int,void*);
static void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
static void rc(int e){if(e)throw std::runtime_error("FFI refusal "+std::to_string(e));}
static void require(bool b,const char* s){if(!b)throw std::runtime_error(s);}
constexpr int NE=128,K=4096,N=2048,MAXR=6,G=64;
constexpr uint32_t GUARD=0x7fc13579;
struct Row { uint32_t group,eid,pair; float scales[5]; std::vector<uint16_t> a; std::vector<uint8_t> w[6]; };
struct Fixture {int rank,call;float limit;std::vector<Row> rows;std::string name;};
static Fixture load(const std::filesystem::path& path){
    std::ifstream f(path,std::ios::binary);require(bool(f),"fixture open");
    auto read=[&](void* p,size_t n){f.read((char*)p,n);require(bool(f),"fixture truncated");};
    uint32_t h[8];read(h,sizeof(h));
    require(h[0]==0x4e333247 && h[1]==1 && h[2]<2 && h[4]==NE && h[5]==K && h[6]==N && h[7]<=6,"fixture header");
    Fixture x{int(h[2]),int(h[3]),0,{},path.filename().string()};read(&x.limit,4);
    for(unsigned i=0;i<h[7];++i){Row r;uint32_t meta[3];read(meta,12);r.group=meta[0];r.eid=meta[1];r.pair=meta[2];
        require(r.group<NE && r.eid<NE && r.pair<6,"route metadata");
        read(r.scales,sizeof(r.scales));r.a.resize(K);read(r.a.data(),K*2);
        for(int p=0;p<6;++p){r.w[p].resize(size_t(K)*N/(p%2?16:2));read(r.w[p].data(),r.w[p].size());}
        x.rows.push_back(std::move(r));
    }
    require(f.peek()==EOF,"fixture trailing bytes");return x;
}
struct Device {
    cudaStream_t stream{};std::vector<void*> allocations;
    Device(){ck(cudaStreamCreate(&stream));}
    ~Device(){cudaStreamSynchronize(stream);for(void* p:allocations)cudaFree(p);cudaStreamDestroy(stream);}
    template<class T>T* alloc(size_t n){T* p=nullptr;ck(cudaMalloc(&p,std::max(size_t(1),n)*sizeof(T)));allocations.push_back(p);return p;}
    template<class T>T* upload(const std::vector<T>& x){T* p=alloc<T>(x.size());if(!x.empty()){ck(cudaMemcpyAsync(p,x.data(),x.size()*sizeof(T),cudaMemcpyHostToDevice,stream));ck(cudaStreamSynchronize(stream));}return p;}
    template<class T>std::vector<T> read(T* p,size_t n){std::vector<T> x(n);ck(cudaMemcpyAsync(x.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost,stream));ck(cudaStreamSynchronize(stream));return x;}
};
struct Payload {
    Device dev;int live;float limit;unsigned long long* table;int *ids,*off,*pairs;__half* a;
    float *rs,*mg,*mu,*rw,*md,*h[2],*out[2];unsigned* visits;int grid;
    Payload(const Fixture& f):live(f.rows.size()),limit(f.limit){
        std::vector<unsigned long long> tab(NE*6);std::vector<int> ii(NE),oo(NE+1),pp(MAXR);std::vector<uint16_t> aa(MAXR*K,0);
        std::vector<float> sc[5];for(auto& s:sc)s.resize(MAXR,1.f);
        std::vector<Row const*> order;for(auto& r:f.rows)order.push_back(&r);
        std::sort(order.begin(),order.end(),[](auto a,auto b){return a->group<b->group;});
        int row=0;for(int g=0;g<NE;++g){ii[g]=g;oo[g]=row;
            if(row<live && int(order[row]->group)==g){const Row& r=*order[row];ii[g]=r.eid;
                for(int p=0;p<6;++p){require(tab[p*NE+r.eid]==0,"duplicate expert fixture");tab[p*NE+r.eid]=(unsigned long long)dev.upload(r.w[p]);}
                std::copy(r.a.begin(),r.a.end(),aa.begin()+row*K);for(int p=0;p<5;++p)sc[p][row]=r.scales[p];pp[row]=r.pair;++row;
            }}
        require(row==live,"duplicate group fixture");oo[NE]=row;
        table=dev.upload(tab);ids=dev.upload(ii);off=dev.upload(oo);pairs=dev.upload(pp);a=(__half*)dev.upload(aa);
        rs=dev.upload(sc[0]);mg=dev.upload(sc[1]);mu=dev.upload(sc[2]);rw=dev.upload(sc[3]);md=dev.upload(sc[4]);
        for(int i=0;i<2;++i){h[i]=(float*)dev.upload(std::vector<uint32_t>(MAXR*N+2*G,GUARD));out[i]=(float*)dev.upload(std::vector<uint32_t>(MAXR*K+2*G,GUARD));}
        visits=dev.alloc<unsigned>(1);ck(cudaMemsetAsync(visits,0,4,dev.stream));
        int d,sms,occ;ck(cudaGetDevice(&d));ck(cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,d));
        ck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,dsv4_gu_n32_kernel,64,1024));require(occ>0,"candidate occupancy");grid=sms*occ;
        ck(cudaStreamSynchronize(dev.stream));
    }
    void launch(int arm,bool count=false){
        if(arm==0){rc(memra_moe_kq_gemm_sk_gu_m1_half2(table,NE,ids,a,h[0]+G,rs,mg,mu,rw,off,NE,K,N,limit,K/2,dev.stream));}
        else {dsv4_gu_n32_kernel<<<grid,dim3(32,2,1),1024,dev.stream>>>(table,NE,ids,K/2,a,h[1]+G,rs,mg,mu,rw,off,NE,K,N,-1,limit,count?visits:nullptr);ck(cudaGetLastError());}
    }
    void downstream(int arm){
        // Keep FP8-QAT, normalized-half mirror, actual down FFI, macro2 and
        // original-slot scatter identical to GroupedWork::down.
        if(!live)return;
        auto codes=dev.alloc<uint8_t>(live*N);auto scales=dev.alloc<float>(live*N/128);
        auto half=dev.alloc<uint16_t>(live*N);auto rowscale=dev.alloc<float>(live);auto status=dev.alloc<int>(live);
        auto contribution=dev.alloc<float>(live*K);
        rc(memra_dsv4_act_quant_fp8(h[arm]+G,codes,scales,live,N,dev.stream));
        rc(memra_dsv4_fp8_gather_half(codes,scales,nullptr,half,rowscale,status,live,N,dev.stream));
        auto st=dev.read(status,live);require(std::all_of(st.begin(),st.end(),[](int x){return x==0;}),"FP8 mirror exactness");
        rc(memra_moe_kq_gemm_sk_m1_half2(table,NE,ids,half,contribution,rowscale,off,NE,N,K,N/2,dev.stream));
        rc(memra_dsv4_scale_rows(contribution,md,live,K,dev.stream));
        ck(cudaMemsetAsync(out[arm]+G,0,MAXR*K*4,dev.stream));
        rc(memra_dsv4_scatter_rows(contribution,out[arm]+G,pairs,live,K,dev.stream));
    }
    void exact(){
        auto x=dev.read((uint32_t*)h[0],MAXR*N+2*G);auto y=dev.read((uint32_t*)h[1],MAXR*N+2*G);
        require(x==y,"GU H bit mismatch");
        for(int i=0;i<G;++i)require(x[i]==GUARD && x[G+MAXR*N+i]==GUARD,"H guard");
        for(int i=live*N;i<MAXR*N;++i)require(x[G+i]==GUARD,"H inactive tail write");
        for(int i=0;i<live*N;++i){float v;memcpy(&v,&x[G+i],4);require(std::isfinite(v),"H not finite");}
    }
    void check_down(){
        downstream(0);downstream(1);auto x=dev.read((uint32_t*)out[0],MAXR*K+2*G);auto y=dev.read((uint32_t*)out[1],MAXR*K+2*G);
        require(x==y,"complete contribution bit mismatch");for(int i=0;i<G;++i)require(x[i]==GUARD && x[G+MAXR*K+i]==GUARD,"contribution guard");
    }
};
static __global__ void flush_cache(uint4* p,size_t n){for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x)p[i]=make_uint4(i,i+1,i+2,i+3);}
static float timed(Payload& p,int arm,cudaEvent_t start,cudaEvent_t end){ck(cudaEventRecord(start,p.dev.stream));p.launch(arm);ck(cudaEventRecord(end,p.dev.stream));ck(cudaEventSynchronize(end));float ms;ck(cudaEventElapsedTime(&ms,start,end));return ms*1000;}
static void cell(const Fixture& f,bool correctness_only){
    ck(cudaSetDevice(f.rank));Payload p(f);auto before=memra_moe_kq_gemm_sk_gu_half2_dispatches();
    p.launch(0);p.launch(1,true);p.exact();p.check_down();auto visits=p.dev.read(p.visits,1)[0];require(visits==unsigned(p.live*64),"dynamic useful tile count");
    cudaFuncAttributes a,b;ck(cudaFuncGetAttributes(&a,moe_kq_sktail_gu_kernel<108,true,true>));ck(cudaFuncGetAttributes(&b,dsv4_gu_n32_kernel));
    printf("EXACT name=%s rank=%d live=%d H_bits=%d contribution_bits=%d useful_tiles_current=%d useful_tiles_n32=%u K64=64 control_reg=%d candidate_reg=%d control_static=%zu candidate_static=%zu dynamic=1024 control_local=%zu candidate_local=%zu candidate_grid=%d guards=true\n",f.name.c_str(),f.rank,p.live,p.live*N,MAXR*K,p.live*32,visits,a.numRegs,b.numRegs,a.sharedSizeBytes,b.sharedSizeBytes,a.localSizeBytes,b.localSizeBytes,p.grid);
    if(!correctness_only){
        cudaEvent_t start,end;ck(cudaEventCreate(&start));ck(cudaEventCreate(&end));
        int l2;ck(cudaDeviceGetAttribute(&l2,cudaDevAttrL2CacheSize,f.rank));size_t bytes=size_t(l2)*2;
        auto flush=p.dev.alloc<uint4>(bytes/16);
        for(const char* regime:{"warm","cold2xL2"}){
            for(int i=0;i<2;++i){p.launch(0);p.launch(1);}ck(cudaStreamSynchronize(p.dev.stream));
            for(int cycle=0;cycle<3;++cycle)for(int index=0;index<4;++index){int arm=(index==1 || index==2);
                if(regime[0]=='c'){flush_cache<<<1024,256,0,p.dev.stream>>>(flush,bytes/16);ck(cudaGetLastError());ck(cudaStreamSynchronize(p.dev.stream));}
                float us=timed(p,arm,start,end);p.exact();
                printf("TIMING name=%s rank=%d live=%d regime=%s cycle=%d index=%d arm=%s us=%.6f cold_bytes=%zu\n",f.name.c_str(),f.rank,p.live,regime,cycle,index,arm?"n32":"current",us,regime[0]=='c'?bytes:0);
            }
        }
        ck(cudaEventDestroy(start));ck(cudaEventDestroy(end));
    }
    require(memra_moe_kq_gemm_sk_gu_half2_dispatches()>before,"actual current FFI not engaged");fflush(stdout);
}
int main(int argc,char** argv){try{
    require(argc==2 || (argc==3 && std::string(argv[2])=="--correctness-only"),"usage: gate fixture-directory [--correctness-only]");
    require(!getenv("MEMRA_DSV4_GU_N32") || std::string(getenv("MEMRA_DSV4_GU_N32"))=="0","control FFI must be OFF");
    std::vector<std::filesystem::path> files;for(auto& p:std::filesystem::directory_iterator(argv[1]))if(p.path().extension()==".gun32")files.push_back(p.path());std::sort(files.begin(),files.end());
    require(files.size()==16,"require eight frozen real calls per rank");int seen[2]={};std::vector<Fixture> seeds;
    for(auto& file:files){Fixture f=load(file);++seen[f.rank];if(seeds.size()<2 || seeds.back().rank!=f.rank){if(!f.rows.empty())seeds.push_back(f);}cell(f,argc==3);}
    require(seen[0]==8 && seen[1]==8,"both ranks required");
    for(int rank=0;rank<2;++rank){auto it=std::find_if(seeds.begin(),seeds.end(),[rank](auto& f){return f.rank==rank && !f.rows.empty();});require(it!=seeds.end(),"boundary seed absent");
        Fixture empty=*it;empty.rows.clear();empty.name="derived-empty";cell(empty,argc==3);
        Fixture six=*it;std::vector<Row> original=six.rows;six.rows.clear();for(int i=0;i<6;++i){Row r=original[i%original.size()];r.group=i*17;r.eid=r.group;r.pair=5-i;six.rows.push_back(std::move(r));}six.name="derived-six-real-payloads";cell(six,argc==3);
    }
    printf("PASS GU_N32 frozen_real_calls=16 boundary_cells=4 H_bit_identity=true complete_contribution_identity=true full_K=true no_full_model_rate=true\n");return 0;
}catch(const std::exception& e){fprintf(stderr,"FAIL GU_N32: %s\n",e.what());return 1;}}

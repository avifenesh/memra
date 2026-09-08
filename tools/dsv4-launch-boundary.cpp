// Gate-only linker wrappers. No CUPTI, NVTX, runtime source or kernel changes.
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

namespace {
void ck(cudaError_t rc, const char* op) {
    if (rc != cudaSuccess) { std::fprintf(stderr,"BOUNDARY_FATAL %s: %s\n",op,cudaGetErrorString(rc)); std::abort(); }
}
void require(bool ok, const char* message) {
    if (!ok) { std::fprintf(stderr,"BOUNDARY_FATAL %s\n",message); std::abort(); }
}
uint64_t now_ns() {
    timespec t{};
    require(clock_gettime(CLOCK_MONOTONIC_RAW,&t)==0,"monotonic clock");
    return uint64_t(t.tv_sec)*1000000000ULL+uint64_t(t.tv_nsec);
}
struct Rank {
    cudaStream_t stream{};
    cudaEvent_t base{}, before{}, start{}, reply{}, ar_before{}, ar_after{};
    uint64_t host_before{}, host_after{};
    unsigned calls{};
};
struct Entry { cudaGraphExec_t original{}, instrumented{}; cudaGraph_t graph{}; int rank{}; };
std::array<Rank,2> ranks;
std::vector<Entry> entries;
bool active=false;
int saved_device=0;

void init_rank(int rank, cudaStream_t stream) {
    auto& r=ranks.at(rank);
    if (r.stream) { require(r.stream==stream,"changed rank stream"); return; }
    r.stream=stream;
    for(auto event:{&r.base,&r.before,&r.start,&r.reply,&r.ar_before,&r.ar_after}) ck(cudaEventCreate(event),"create event");
}
float elapsed(cudaEvent_t a,cudaEvent_t b) { float ms=0; ck(cudaEventElapsedTime(&ms,a,b),"elapsed same-device events"); return ms; }
}

extern "C" int __real_memra_dsv4_replay_capture_end(void*,void**,void*);
extern "C" int __real_memra_dsv4_replay_launch(void*,void*);
extern "C" int __real_memra_dsv4_replay_destroy(void*,void*,void*);
extern "C" int __wrap_memra_dsv4_replay_capture_end(void* graph,void** executable,void* raw_stream) {
    const int rc=__real_memra_dsv4_replay_capture_end(graph,executable,raw_stream);
    if(rc) return rc;
    size_t count=0; ck(cudaGraphGetNodes((cudaGraph_t)graph,nullptr,&count),"node count");
    std::vector<cudaGraphNode_t> nodes(count); ck(cudaGraphGetNodes((cudaGraph_t)graph,nodes.data(),&count),"nodes");
    unsigned embeds=0,kernels=0,ars=0;
    for(auto node:nodes) {
        cudaGraphNodeType type; ck(cudaGraphNodeGetType(node,&type),"node type");
        if(type!=cudaGraphNodeTypeKernel) continue;
        ++kernels; cudaKernelNodeParams params{}; const char* name=nullptr;
        ck(cudaGraphKernelNodeGetParams(node,&params),"kernel params"); ck(cudaFuncGetName(&name,params.func),"kernel name");
        embeds+=std::strstr(name,"dsv4_embed_rows_kernel")!=nullptr;
        ars+=std::strstr(name,"memra_tp_ar_1stage_kernel")!=nullptr;
    }
    if(!embeds) return rc; // Commit/head graph is untouched.
    require(embeds==1 && ars==86,"not a complete forward graph");
    int rank; ck(cudaGetDevice(&rank),"rank"); require(rank==0 || rank==1,"rank outside pair");
    init_rank(rank,(cudaStream_t)raw_stream);
    Entry e{}; e.original=(cudaGraphExec_t)*executable; e.rank=rank;
    ck(cudaGraphClone(&e.graph,(cudaGraph_t)graph),"clone forward");
    size_t roots_count=0; ck(cudaGraphGetRootNodes(e.graph,nullptr,&roots_count),"roots count");
    std::vector<cudaGraphNode_t> roots(roots_count); ck(cudaGraphGetRootNodes(e.graph,roots.data(),&roots_count),"roots");
    require(roots_count>0,"no graph roots");
    // The pinned single-stream forward is a chain up to its first collective.
    require(roots_count==1,"first-AR event bracket requires one original root");
    auto first_ar=roots[0];
    for(size_t walked=0;walked<count;++walked) {
        cudaGraphNodeType type;ck(cudaGraphNodeGetType(first_ar,&type),"first-AR node type");
        if(type==cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams params{};const char* name=nullptr;
            ck(cudaGraphKernelNodeGetParams(first_ar,&params),"first-AR params");ck(cudaFuncGetName(&name,params.func),"first-AR name");
            if(std::strstr(name,"memra_tp_ar_1stage_kernel")) break;
        }
        size_t n=0;ck(cudaGraphNodeGetDependentNodes(first_ar,nullptr,nullptr,&n),"walk first AR");
        require(n==1 && walked+1<count,"first AR not on unambiguous chain");
        cudaGraphNode_t next;ck(cudaGraphNodeGetDependentNodes(first_ar,&next,nullptr,&n),"next before first AR");first_ar=next;
    }
    size_t nd=0,ns=0;
    ck(cudaGraphNodeGetDependencies(first_ar,nullptr,nullptr,&nd),"AR predecessors");
    ck(cudaGraphNodeGetDependentNodes(first_ar,nullptr,nullptr,&ns),"AR successors");
    std::vector<cudaGraphNode_t> deps(nd),succ(ns);
    ck(cudaGraphNodeGetDependencies(first_ar,deps.data(),nullptr,&nd),"AR dependency list");
    ck(cudaGraphNodeGetDependentNodes(first_ar,succ.data(),nullptr,&ns),"AR successor list");
    cudaGraphNode_t ar_before,ar_after;
    ck(cudaGraphAddEventRecordNode(&ar_before,e.graph,deps.data(),nd,ranks[rank].ar_before),"AR before event");
    ck(cudaGraphAddDependencies(e.graph,&ar_before,&first_ar,nullptr,1),"AR waits for before event");
    ck(cudaGraphAddEventRecordNode(&ar_after,e.graph,&first_ar,1,ranks[rank].ar_after),"AR after event");
    std::vector<cudaGraphNode_t> afters(ns,ar_after);
    ck(cudaGraphAddDependencies(e.graph,afters.data(),succ.data(),nullptr,ns),"successors wait for AR after event");
    cudaGraphNode_t marker;
    ck(cudaGraphAddEventRecordNode(&marker,e.graph,nullptr,0,ranks[rank].start),"first-node event");
    std::vector<cudaGraphNode_t> markers(roots_count,marker);
    ck(cudaGraphAddDependencies(e.graph,markers.data(),roots.data(),nullptr,roots_count),"marker precedes original roots");
    size_t instrumented_count=0; ck(cudaGraphGetNodes(e.graph,nullptr,&instrumented_count),"instrumented census");
    require(instrumented_count==count+3,"instrumented census mismatch");
    ck(cudaGraphInstantiate(&e.instrumented,e.graph,0),"instantiate diagnostic clone");
    entries.push_back(e);
    std::printf("BOUNDARY_CLONE {\"rank\":%d,\"original_nodes\":%zu,\"kernels\":%u,\"ar\":%u,\"added_event_nodes\":3,\"original_edges_preserved\":true}\n",rank,count,kernels,ars);
    return rc;
}
extern "C" int __wrap_memra_dsv4_replay_launch(void* executable,void* raw_stream) {
    if(!active) return __real_memra_dsv4_replay_launch(executable,raw_stream);
    auto it=std::find_if(entries.begin(),entries.end(),[&](const Entry& e){return e.original==(cudaGraphExec_t)executable;});
    if(it==entries.end()) return __real_memra_dsv4_replay_launch(executable,raw_stream);
    auto& r=ranks[it->rank]; require(r.calls==0 && r.stream==(cudaStream_t)raw_stream,"forward launch repeated or stream changed");
    ck(cudaEventRecord(r.before,r.stream),"event immediately before graph launch");
    r.host_before=now_ns();
    const auto rc=cudaGraphLaunch(it->instrumented,r.stream);
    r.host_after=now_ns();
    ++r.calls;
    return rc==cudaSuccess ? 0 : 10000+(int)rc;
}
extern "C" int __wrap_memra_dsv4_replay_destroy(void* graph,void* executable,void* stream) {
    auto it=std::find_if(entries.begin(),entries.end(),[&](const Entry& e){return e.original==(cudaGraphExec_t)executable;});
    if(it!=entries.end()) {
        ck(cudaGraphExecDestroy(it->instrumented),"destroy instrumented executable");
        ck(cudaGraphDestroy(it->graph),"destroy instrumented graph"); entries.erase(it);
    }
    return __real_memra_dsv4_replay_destroy(graph,executable,stream);
}
extern "C" void memra_launch_boundary_prepare() {
    require(entries.size()==6,"expected both ranks' three cadence variants");
    int saved; ck(cudaGetDevice(&saved),"save context");
    for(const auto& e:entries) { ck(cudaSetDevice(e.rank),"upload rank"); ck(cudaGraphUpload(e.instrumented,ranks[e.rank].stream),"upload diagnostic graph outside rows"); }
    for(int rank=0;rank<2;++rank) {ck(cudaSetDevice(rank),"drain rank");ck(cudaStreamSynchronize(ranks[rank].stream),"prepare drain");}
    ck(cudaSetDevice(saved),"restore context");
}
extern "C" void memra_launch_boundary_begin() {
    require(!active,"nested diagnostic step");
    ck(cudaGetDevice(&saved_device),"save device");
    for(auto& r:ranks) { require(r.stream,"rank not captured");r.calls=0; }
    // A0 <= B0 <= A1. Their unknown one-way offset d is in [0,A1-A0].
    // ElapsedTime is used ONLY within one GPU. No symmetry assumption.
    ck(cudaSetDevice(0),"calibration rank0"); ck(cudaEventRecord(ranks[0].base,ranks[0].stream),"A0");
    ck(cudaSetDevice(1),"calibration rank1"); ck(cudaStreamWaitEvent(ranks[1].stream,ranks[0].base,0),"wait A0"); ck(cudaEventRecord(ranks[1].base,ranks[1].stream),"B0");
    ck(cudaSetDevice(0),"calibration reply rank0"); ck(cudaStreamWaitEvent(ranks[0].stream,ranks[1].base,0),"wait B0"); ck(cudaEventRecord(ranks[0].reply,ranks[0].stream),"A1");
    ck(cudaEventSynchronize(ranks[0].reply),"finish handshake outside launches");
    ck(cudaSetDevice(saved_device),"restore device before decode");
    active=true;
}
extern "C" void memra_launch_boundary_finish(unsigned position) {
    require(active && ranks[0].calls==1 && ranks[1].calls==1,"missing rank launch"); active=false;
    float start[2],before[2],queued[2],ar[2];
    for(int rank=0;rank<2;++rank) {
        ck(cudaSetDevice(rank),"read event rank"); ck(cudaEventSynchronize(ranks[rank].start),"start event complete");
        ck(cudaEventSynchronize(ranks[rank].ar_after),"first AR event complete");
        ar[rank]=elapsed(ranks[rank].ar_before,ranks[rank].ar_after);
        start[rank]=elapsed(ranks[rank].base,ranks[rank].start);
        before[rank]=elapsed(ranks[rank].base,ranks[rank].before);
        queued[rank]=elapsed(ranks[rank].before,ranks[rank].start);
    }
    ck(cudaSetDevice(0),"read calibration"); const float roundtrip=elapsed(ranks[0].base,ranks[0].reply);
    const double lower=double(start[1])-double(start[0]);
    std::printf("BOUNDARY_ROW {\"position\":%u,\"host_before_ns\":[%llu,%llu],\"host_after_ns\":[%llu,%llu],\"host_launch_ms\":[%.9f,%.9f],\"calibration_roundtrip_ms\":%.9f,\"start_from_local_base_ms\":[%.9f,%.9f],\"prelaunch_from_local_base_ms\":[%.9f,%.9f],\"prelaunch_to_graph_start_ms\":[%.9f,%.9f],\"rank1_minus_rank0_start_lower_ms\":%.9f,\"rank1_minus_rank0_start_upper_ms\":%.9f,\"rank1_minus_rank0_start_midpoint_ms\":%.9f,\"event_resolution_us_approx\":0.5,\"first_ar_wait\":null,\"first_ar_event_residence_ms\":[%.9f,%.9f]}\n",position,
        (unsigned long long)ranks[0].host_before,(unsigned long long)ranks[1].host_before,(unsigned long long)ranks[0].host_after,(unsigned long long)ranks[1].host_after,
        double(ranks[0].host_after-ranks[0].host_before)/1e6,double(ranks[1].host_after-ranks[1].host_before)/1e6,roundtrip,start[0],start[1],before[0],before[1],queued[0],queued[1],lower,lower+roundtrip,lower+roundtrip/2,ar[0],ar[1]);
    ck(cudaSetDevice(saved_device),"restore device after readback");
}
extern "C" void memra_launch_boundary_cleanup() {
    require(entries.empty() && !active,"cleanup before captured states drop");
    int saved;ck(cudaGetDevice(&saved),"cleanup saved device");
    for(int rank=0;rank<2;++rank) {
        ck(cudaSetDevice(rank),"cleanup rank");
        for(auto event:{ranks[rank].base,ranks[rank].before,ranks[rank].start,ranks[rank].reply,ranks[rank].ar_before,ranks[rank].ar_after}) ck(cudaEventDestroy(event),"cleanup event");
    }
    ck(cudaSetDevice(saved),"cleanup restore device");
}

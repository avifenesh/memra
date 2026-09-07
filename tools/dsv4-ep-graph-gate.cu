// Standalone two-device EP fork/join CUDA-graph micro-gate.
//
// This probe does not load a model or touch engine/runtime sources. It uses
// the real CUDA topology that DSV4 EP relies on:
//
//   owner stream: P2P copy -> tx event
//   peer stream:  wait tx -> peer kernel -> P2P return -> rx event
//   owner stream: wait rx -> merge kernel
//
// The first capture attempt used cudaMemcpyPeerAsync inside stream capture and
// is intentionally not retained: CUDA rejected that API with
// cudaErrorStreamCaptureUnsupported on the pair. This revision builds an
// explicit cudaGraphAddNode graph, assigning device execution contexts to the
// kernel/memcpy nodes and explicit event record/wait edges. Eager and replay
// paths use fresh input words, bit guards, and an event-dependent transform.
// The program reports node-type/P2P census and emits a graph DOT file when run.
// It is intentionally not executed by this lane.
//
// Compile/link only:
//   /usr/local/cuda-13.1/bin/nvcc -std=c++17 -O3 \
//     -gencode arch=compute_120a,code=sm_120a \
//     tools/dsv4-ep-graph-gate.cu -o target/dsv4-ep-graph-gate
// Later node-placement trace:
//   nsys profile --trace=cuda,nvtx,osrt --stats=true \
//     -o target/dsv4-ep-graph-r2-nsys target/dsv4-ep-graph-gate-r2 --trace

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace dsv4_ep_graph_gate {

constexpr int kOwner = 0;
constexpr int kPeer = 1;
constexpr std::size_t kWords = 1024;
constexpr std::size_t kGuardWords = 16;
constexpr std::uint32_t kGuard = 0x7fc01234u;
constexpr std::size_t kBytes = kWords * sizeof(std::uint32_t);

static void check(cudaError_t status, const char* call) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(call) + ": " + cudaGetErrorString(status));
}

static std::uint32_t mix_word(std::uint32_t seed, std::size_t i) {
    std::uint32_t x = seed + static_cast<std::uint32_t>(i) * 0x9e3779b9u;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    return x * 0x846ca68bu + 0x13579bdu;
}

__global__ void peer_transform(const std::uint32_t* src, std::uint32_t* dst,
                               std::size_t words) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < words) dst[i] = src[i] + 0x01010101u + static_cast<std::uint32_t>(i * 3u);
}

__global__ void owner_merge(const std::uint32_t* returned, const std::uint32_t* input,
                            std::uint32_t* output, std::size_t words) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < words) output[i] = returned[i] ^ input[i];
}

struct Buffers {
    std::uint32_t* owner_input = nullptr;
    std::uint32_t* peer_work = nullptr;
    std::uint32_t* owner_return = nullptr;
    std::uint32_t* owner_output = nullptr;
    cudaStream_t owner_stream = nullptr;
    cudaStream_t peer_stream = nullptr;
    cudaEvent_t tx_done = nullptr;
    cudaEvent_t rx_done = nullptr;

    ~Buffers() {
        if (owner_stream) cudaStreamSynchronize(owner_stream);
        if (peer_stream) cudaStreamSynchronize(peer_stream);
        if (tx_done) cudaEventDestroy(tx_done);
        if (rx_done) cudaEventDestroy(rx_done);
        if (owner_stream) cudaStreamDestroy(owner_stream);
        if (peer_stream) cudaStreamDestroy(peer_stream);
        cudaSetDevice(kOwner);
        if (owner_input) cudaFree(owner_input);
        if (owner_return) cudaFree(owner_return);
        if (owner_output) cudaFree(owner_output);
        cudaSetDevice(kPeer);
        if (peer_work) cudaFree(peer_work);
    }
};

static void allocate(Buffers& b) {
    int count = 0;
    check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count < 2) throw std::runtime_error("EP graph requires two visible CUDA devices");
    int can01 = 0, can10 = 0;
    check(cudaDeviceCanAccessPeer(&can01, kOwner, kPeer), "cudaDeviceCanAccessPeer(0,1)");
    check(cudaDeviceCanAccessPeer(&can10, kPeer, kOwner), "cudaDeviceCanAccessPeer(1,0)");
    if (!can01 || !can10)
        throw std::runtime_error("EP graph P2P unavailable in one direction");

    check(cudaSetDevice(kOwner), "cudaSetDevice(owner)");
    cudaError_t enable01 = cudaDeviceEnablePeerAccess(kPeer, 0);
    if (enable01 != cudaSuccess && enable01 != cudaErrorPeerAccessAlreadyEnabled)
        check(enable01, "cudaDeviceEnablePeerAccess(owner->peer)");
    check(cudaStreamCreateWithFlags(&b.owner_stream, cudaStreamNonBlocking),
          "cudaStreamCreate(owner)");
    check(cudaMalloc(reinterpret_cast<void**>(&b.owner_input),
                     (kWords + kGuardWords) * sizeof(std::uint32_t)),
          "cudaMalloc(owner input)");
    check(cudaMalloc(reinterpret_cast<void**>(&b.owner_return),
                     (kWords + kGuardWords) * sizeof(std::uint32_t)),
          "cudaMalloc(owner return)");
    check(cudaMalloc(reinterpret_cast<void**>(&b.owner_output),
                     (kWords + kGuardWords) * sizeof(std::uint32_t)),
          "cudaMalloc(owner output)");
    check(cudaEventCreateWithFlags(&b.tx_done, cudaEventDisableTiming),
          "cudaEventCreate(tx owner)");

    check(cudaSetDevice(kPeer), "cudaSetDevice(peer)");
    cudaError_t enable10 = cudaDeviceEnablePeerAccess(kOwner, 0);
    if (enable10 != cudaSuccess && enable10 != cudaErrorPeerAccessAlreadyEnabled)
        check(enable10, "cudaDeviceEnablePeerAccess(peer->owner)");
    check(cudaStreamCreateWithFlags(&b.peer_stream, cudaStreamNonBlocking),
          "cudaStreamCreate(peer)");
    check(cudaMalloc(reinterpret_cast<void**>(&b.peer_work),
                     (kWords + kGuardWords) * sizeof(std::uint32_t)),
          "cudaMalloc(peer work)");
    check(cudaEventCreateWithFlags(&b.rx_done, cudaEventDisableTiming),
          "cudaEventCreate(rx peer)");
}

static std::vector<std::uint32_t> guard_buffer() {
    return std::vector<std::uint32_t>(kWords + kGuardWords, kGuard);
}

static void reset_buffers(Buffers& b, const std::vector<std::uint32_t>& input) {
    const auto guard = guard_buffer();
    check(cudaSetDevice(kOwner), "cudaSetDevice(reset owner)");
    check(cudaMemcpyAsync(b.owner_input, input.data(), kWords * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, b.owner_stream), "input H2D");
    check(cudaMemcpyAsync(b.owner_input + kWords, guard.data() + kWords,
                          kGuardWords * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
                          b.owner_stream), "input guard H2D");
    check(cudaMemcpyAsync(b.owner_return, guard.data(), guard.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, b.owner_stream), "return guard H2D");
    check(cudaMemcpyAsync(b.owner_output, guard.data(), guard.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, b.owner_stream), "output guard H2D");
    check(cudaSetDevice(kPeer), "cudaSetDevice(reset peer)");
    check(cudaMemcpyAsync(b.peer_work, guard.data(), guard.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, b.peer_stream), "peer guard H2D");
    // The guard/input host vectors are temporary. Drain BOTH streams before
    // they leave scope, and leave the host on the owner device for the next
    // graph launch. This also removes any reset-vs-graph peer-stream race.
    check(cudaStreamSynchronize(b.owner_stream), "reset owner sync");
    check(cudaStreamSynchronize(b.peer_stream), "reset peer sync");
    check(cudaSetDevice(kOwner), "cudaSetDevice(reset complete owner)");
}

static std::vector<std::uint32_t> read_owner_output(Buffers& b) {
    std::vector<std::uint32_t> out(kWords + kGuardWords);
    check(cudaSetDevice(kOwner), "cudaSetDevice(read owner)");
    check(cudaMemcpyAsync(out.data(), b.owner_output, out.size() * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost, b.owner_stream), "output D2H");
    check(cudaStreamSynchronize(b.owner_stream), "output D2H sync");
    return out;
}

static std::vector<std::uint32_t> read_device_words(int device, std::uint32_t* ptr,
                                                    cudaStream_t stream) {
    std::vector<std::uint32_t> out(kWords + kGuardWords);
    check(cudaSetDevice(device), "cudaSetDevice(read device)");
    check(cudaMemcpyAsync(out.data(), ptr, out.size() * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost, stream), "device words D2H");
    check(cudaStreamSynchronize(stream), "device words sync");
    return out;
}

static void check_output(const std::vector<std::uint32_t>& got,
                         const std::vector<std::uint32_t>& input) {
    for (std::size_t i = 0; i < kWords; ++i) {
        const std::uint32_t peer = input[i] + 0x01010101u + static_cast<std::uint32_t>(i * 3u);
        const std::uint32_t expected = peer ^ input[i];
        if (got[i] != expected)
            throw std::runtime_error("EP output mismatch at word " + std::to_string(i));
    }
    for (std::size_t i = kWords; i < got.size(); ++i) {
        if (got[i] != kGuard) throw std::runtime_error("EP output guard changed");
    }
}

static void check_words(const char* label, const std::vector<std::uint32_t>& got,
                        const std::vector<std::uint32_t>& expected) {
    if (got.size() != expected.size()) throw std::runtime_error(std::string(label) + " size");
    for (std::size_t i = 0; i < kWords; ++i) {
        if (got[i] != expected[i])
            throw std::runtime_error(std::string(label) + " mismatch at word " + std::to_string(i));
    }
    for (std::size_t i = kWords; i < got.size(); ++i) {
        if (got[i] != kGuard)
            throw std::runtime_error(std::string(label) + " guard changed");
    }
}

static void check_all_buffers(Buffers& b, const std::vector<std::uint32_t>& input) {
    std::vector<std::uint32_t> transformed(kWords + kGuardWords, kGuard);
    for (std::size_t i = 0; i < kWords; ++i)
        transformed[i] = input[i] + 0x01010101u + static_cast<std::uint32_t>(i * 3u);
    const auto owner_input = read_device_words(kOwner, b.owner_input, b.owner_stream);
    const auto peer_work = read_device_words(kPeer, b.peer_work, b.peer_stream);
    const auto owner_return = read_device_words(kOwner, b.owner_return, b.owner_stream);
    check_words("owner input", owner_input, [&] {
        auto expected = guard_buffer();
        std::copy(input.begin(), input.end(), expected.begin());
        return expected;
    }());
    check_words("peer work", peer_work, transformed);
    check_words("owner return", owner_return, transformed);
}

static void enqueue_eager(Buffers& b) {
    check(cudaSetDevice(kOwner), "cudaSetDevice(eager owner)");
    check(cudaMemcpyPeerAsync(b.peer_work, kPeer, b.owner_input, kOwner,
                              kBytes, b.owner_stream), "eager owner-to-peer P2P");
    check(cudaEventRecord(b.tx_done, b.owner_stream), "eager tx event record");
    check(cudaSetDevice(kPeer), "cudaSetDevice(eager peer)");
    check(cudaStreamWaitEvent(b.peer_stream, b.tx_done, 0), "eager peer tx wait");
    peer_transform<<<(kWords + 255) / 256, 256, 0, b.peer_stream>>>(
        b.peer_work, b.peer_work, kWords);
    check(cudaGetLastError(), "eager peer transform");
    check(cudaMemcpyPeerAsync(b.owner_return, kOwner, b.peer_work, kPeer,
                              kBytes, b.peer_stream), "eager peer-to-owner P2P");
    check(cudaEventRecord(b.rx_done, b.peer_stream), "eager rx event record");
    check(cudaSetDevice(kOwner), "cudaSetDevice(eager merge)");
    check(cudaStreamWaitEvent(b.owner_stream, b.rx_done, 0), "eager owner rx wait");
    owner_merge<<<(kWords + 255) / 256, 256, 0, b.owner_stream>>>(
        b.owner_return, b.owner_input, b.owner_output, kWords);
    check(cudaGetLastError(), "eager owner merge");
    check(cudaStreamSynchronize(b.owner_stream), "eager owner completion");
}

static cudaGraphNode_t add_kernel_node(cudaGraph_t graph, cudaGraphNode_t dependency,
                                       void* func, cudaExecutionContext_t context,
                                       void** args, const char* call) {
    cudaGraphNodeParams params{};
    params.type = cudaGraphNodeTypeKernel;
    params.kernel.func = func;
    params.kernel.gridDim = dim3((kWords + 255) / 256, 1, 1);
    params.kernel.blockDim = dim3(256, 1, 1);
    params.kernel.sharedMemBytes = 0;
    params.kernel.kernelParams = args;
    params.kernel.ctx = context;
    cudaGraphNode_t node = nullptr;
    check(cudaGraphAddNode(&node, graph, dependency ? &dependency : nullptr, nullptr,
                           dependency ? 1 : 0, &params), call);
    return node;
}

static cudaGraphNode_t add_memcpy_node(cudaGraph_t graph, cudaGraphNode_t dependency,
                                       void* src, int src_device, void* dst, int dst_device,
                                       cudaExecutionContext_t context, const char* call) {
    cudaMemcpy3DParms copy{};
    copy.srcPtr = make_cudaPitchedPtr(src, kBytes, kWords, 1);
    copy.dstPtr = make_cudaPitchedPtr(dst, kBytes, kWords, 1);
    copy.extent = make_cudaExtent(kBytes, 1, 1);
    // Explicit graph nodes use CUDA's UVA-aware default kind. The source and
    // destination pointer attributes identify the two devices; this avoids the
    // stream-capture-prohibited cudaMemcpyPeerAsync call.
    copy.kind = cudaMemcpyDefault;
    (void)src_device;
    (void)dst_device;
    cudaGraphNodeParams params{};
    params.type = cudaGraphNodeTypeMemcpy;
    params.memcpy.flags = 0;
    params.memcpy.reserved = 0;
    params.memcpy.ctx = context;
    params.memcpy.copyParams = copy;
    cudaGraphNode_t node = nullptr;
    check(cudaGraphAddNode(&node, graph, dependency ? &dependency : nullptr, nullptr,
                           dependency ? 1 : 0, &params), call);
    return node;
}

static cudaGraphNode_t add_event_node(cudaGraph_t graph, cudaGraphNode_t dependency,
                                      cudaGraphNodeType type, cudaEvent_t event,
                                      const char* call) {
    cudaGraphNodeParams params{};
    params.type = type;
    if (type == cudaGraphNodeTypeEventRecord) params.eventRecord.event = event;
    else params.eventWait.event = event;
    cudaGraphNode_t node = nullptr;
    check(cudaGraphAddNode(&node, graph, dependency ? &dependency : nullptr, nullptr,
                           dependency ? 1 : 0, &params), call);
    return node;
}

static cudaGraph_t build_explicit_fork_join(Buffers& b) {
    cudaGraph_t graph = nullptr;
    check(cudaSetDevice(kOwner), "cudaSetDevice(explicit graph owner)");
    cudaExecutionContext_t owner_ctx = nullptr, peer_ctx = nullptr;
    check(cudaDeviceGetExecutionCtx(&owner_ctx, kOwner), "cudaDeviceGetExecutionCtx(owner)");
    check(cudaDeviceGetExecutionCtx(&peer_ctx, kPeer), "cudaDeviceGetExecutionCtx(peer)");
    check(cudaGraphCreate(&graph, 0), "cudaGraphCreate(explicit)");
    try {
        std::size_t words = kWords;
        cudaGraphNode_t tx_copy = add_memcpy_node(
            graph, nullptr, b.owner_input, kOwner, b.peer_work, kPeer, owner_ctx,
            "cudaGraphAddNode(owner-to-peer memcpy)");
        cudaGraphNode_t tx_record = add_event_node(
            graph, tx_copy, cudaGraphNodeTypeEventRecord, b.tx_done,
            "cudaGraphAddNode(tx record)");
        cudaGraphNode_t tx_wait = add_event_node(
            graph, tx_record, cudaGraphNodeTypeWaitEvent, b.tx_done,
            "cudaGraphAddNode(peer tx wait)");

        void* peer_args[] = {&b.peer_work, &b.peer_work, &words};
        cudaGraphNode_t peer_kernel = add_kernel_node(
            graph, tx_wait, reinterpret_cast<void*>(peer_transform), peer_ctx, peer_args,
            "cudaGraphAddNode(peer transform)");
        cudaGraphNode_t rx_copy = add_memcpy_node(
            graph, peer_kernel, b.peer_work, kPeer, b.owner_return, kOwner, peer_ctx,
            "cudaGraphAddNode(peer-to-owner memcpy)");
        cudaGraphNode_t rx_record = add_event_node(
            graph, rx_copy, cudaGraphNodeTypeEventRecord, b.rx_done,
            "cudaGraphAddNode(rx record)");
        cudaGraphNode_t rx_wait = add_event_node(
            graph, rx_record, cudaGraphNodeTypeWaitEvent, b.rx_done,
            "cudaGraphAddNode(owner rx wait)");

        void* owner_args[] = {&b.owner_return, &b.owner_input, &b.owner_output,
                              &words};
        (void)add_kernel_node(
            graph, rx_wait, reinterpret_cast<void*>(owner_merge), owner_ctx, owner_args,
            "cudaGraphAddNode(owner merge)");
        return graph;
    } catch (...) {
        cudaGraphDestroy(graph);
        throw;
    }
}

static void graph_census(cudaGraph_t graph) {
    std::size_t count = 0;
    check(cudaGraphGetNodes(graph, nullptr, &count), "graph node count");
    std::vector<cudaGraphNode_t> nodes(count);
    check(cudaGraphGetNodes(graph, nodes.data(), &count), "graph nodes");
    int kernels = 0, copies = 0, records = 0, waits = 0, other = 0;
    int peer_transform_kernels = 0, owner_merge_kernels = 0;
    int kernel_param_fail = 0;
    int p2p_owner_peer = 0, p2p_peer_owner = 0, memcpy_param_fail = 0;
    int pointer_attr_fail = 0;
    for (cudaGraphNode_t node : nodes) {
        cudaGraphNodeType type{};
        check(cudaGraphNodeGetType(node, &type), "graph node type");
        switch (type) {
            case cudaGraphNodeTypeKernel: {
                ++kernels;
                cudaKernelNodeParams params{};
                if (cudaGraphKernelNodeGetParams(node, &params) != cudaSuccess) {
                    ++kernel_param_fail;
                    break;
                }
                if (params.func == reinterpret_cast<void*>(peer_transform))
                    ++peer_transform_kernels;
                if (params.func == reinterpret_cast<void*>(owner_merge))
                    ++owner_merge_kernels;
                break;
            }
            case cudaGraphNodeTypeMemcpy: {
                ++copies;
                cudaMemcpy3DParms params{};
                if (cudaGraphMemcpyNodeGetParams(node, &params) != cudaSuccess) {
                    ++memcpy_param_fail;
                    break;
                }
                cudaPointerAttributes src_attr{}, dst_attr{};
                const cudaError_t src_status = params.srcPtr.ptr
                    ? cudaPointerGetAttributes(&src_attr, params.srcPtr.ptr)
                    : cudaErrorInvalidValue;
                const cudaError_t dst_status = params.dstPtr.ptr
                    ? cudaPointerGetAttributes(&dst_attr, params.dstPtr.ptr)
                    : cudaErrorInvalidValue;
                if (src_status != cudaSuccess || dst_status != cudaSuccess) {
                    ++pointer_attr_fail;
                } else if (src_attr.type == cudaMemoryTypeDevice
                           && dst_attr.type == cudaMemoryTypeDevice) {
                    if (src_attr.device == kOwner && dst_attr.device == kPeer)
                        ++p2p_owner_peer;
                    if (src_attr.device == kPeer && dst_attr.device == kOwner)
                        ++p2p_peer_owner;
                }
                break;
            }
            case cudaGraphNodeTypeEventRecord: ++records; break;
            case cudaGraphNodeTypeWaitEvent: ++waits; break;
            default: ++other; break;
        }
    }
    if (kernel_param_fail || memcpy_param_fail || pointer_attr_fail) {
        throw std::runtime_error(
            "CENSUS_UNVERIFIED API limitation: cudaGraphKernelNodeGetParams="
            + std::to_string(kernel_param_fail)
            + " cudaGraphMemcpyNodeGetParams=" + std::to_string(memcpy_param_fail)
            + " cudaPointerGetAttributes=" + std::to_string(pointer_attr_fail));
    }
    if (peer_transform_kernels != 1 || owner_merge_kernels != 1) {
        throw std::runtime_error(
            "CENSUS_UNVERIFIED expected kernels missing: peer_transform="
            + std::to_string(peer_transform_kernels)
            + " owner_merge=" + std::to_string(owner_merge_kernels));
    }
    if (copies != 2 || p2p_owner_peer != 1 || p2p_peer_owner != 1) {
        throw std::runtime_error(
            "CENSUS_UNVERIFIED expected P2P copies owner->peer=1 peer->owner=1 total=2; got "
            "owner->peer=" + std::to_string(p2p_owner_peer)
            + " peer->owner=" + std::to_string(p2p_peer_owner)
            + " total=" + std::to_string(copies));
    }
    std::printf("GRAPH_CENSUS total=%zu kernels=%d memcpy=%d p2p_owner_peer=%d "
                "p2p_peer_owner=%d peer_transform=%d owner_merge=%d "
                "event_record=%d event_wait=%d other=%d owner_device=%d peer_device=%d "
                "path=owner_copy_tx_peer_wait_transform_return_rx_owner_wait_merge\n",
                count, kernels, copies, p2p_owner_peer, p2p_peer_owner,
                peer_transform_kernels, owner_merge_kernels, records, waits, other,
                kOwner, kPeer);
    (void)cudaGraphDebugDotPrint(graph, "target/dsv4-ep-graph-census.dot", 0);
}

static int run_gate(bool trace) {
    Buffers b;
    allocate(b);
    std::vector<std::uint32_t> input_a(kWords), input_b(kWords);
    for (std::size_t i = 0; i < kWords; ++i) {
        input_a[i] = mix_word(0x11110000u, i);
        input_b[i] = mix_word(0x77770000u, i);
    }
    std::puts("EP_GATE P2P checked; capturing one owner-origin cross-device fork/join graph");

    if (trace) std::puts("TRACE_MARKER phase=eager_a owner_device=0 peer_device=1");
    reset_buffers(b, input_a);
    enqueue_eager(b);
    check_all_buffers(b, input_a);
    const auto eager_a = read_owner_output(b);
    check_output(eager_a, input_a);

    if (trace) std::puts("TRACE_MARKER phase=capture_and_graph_a owner_device=0 peer_device=1");
    reset_buffers(b, input_a);
    cudaGraph_t graph = build_explicit_fork_join(b);
    graph_census(graph);
    cudaGraphExec_t exec = nullptr;
    check(cudaSetDevice(kOwner), "cudaSetDevice(instantiate)");
    check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0), "cudaGraphInstantiate");
    check(cudaGraphUpload(exec, b.owner_stream), "cudaGraphUpload");
    check(cudaSetDevice(kOwner), "cudaSetDevice(graph launch A)");
    check(cudaGraphLaunch(exec, b.owner_stream), "cudaGraphLaunch A");
    check(cudaStreamSynchronize(b.owner_stream), "graph A sync");
    check_all_buffers(b, input_a);
    const auto graph_a = read_owner_output(b);
    check_output(graph_a, input_a);
    if (graph_a != eager_a) throw std::runtime_error("graph A differs from eager A");

    if (trace) std::puts("TRACE_MARKER phase=graph_b_replay owner_device=0 peer_device=1");
    reset_buffers(b, input_b);
    check(cudaSetDevice(kOwner), "cudaSetDevice(graph launch B)");
    check(cudaGraphLaunch(exec, b.owner_stream), "cudaGraphLaunch B");
    check(cudaStreamSynchronize(b.owner_stream), "graph B sync");
    check_all_buffers(b, input_b);
    const auto graph_b = read_owner_output(b);
    check_output(graph_b, input_b);
    if (trace) std::puts("TRACE_MARKER phase=eager_b owner_device=0 peer_device=1");
    reset_buffers(b, input_b);
    enqueue_eager(b);
    check_all_buffers(b, input_b);
    const auto eager_b = read_owner_output(b);
    check_output(eager_b, input_b);
    if (graph_b != eager_b) throw std::runtime_error("graph B differs from eager B");

    check(cudaGraphExecDestroy(exec), "cudaGraphExecDestroy");
    check(cudaGraphDestroy(graph), "cudaGraphDestroy");
    std::puts("PASS EP fork/join graph eager/replay identity, fresh inputs, guards and event ordering");
    return 0;
}

} // namespace dsv4_ep_graph_gate

int main(int argc, char** argv) {
    try {
        const bool trace = argc > 1 && std::string(argv[1]) == "--trace";
        return dsv4_ep_graph_gate::run_gate(trace);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}

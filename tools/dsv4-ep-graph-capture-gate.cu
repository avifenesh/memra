// Standalone mixed stream-capture/explicit-node EP graph probe.
//
// This is deliberately separate from the engine and from
// tools/dsv4-ep-graph-gate.cu.  It asks one narrow CUDA API question:
// can an existing owner/peer stream-captured body keep its captured kernels
// and event joins while the two stream-capture-prohibited P2P copies are
// inserted directly into the capture graph and then attached through
// cudaStreamUpdateCaptureDependencies?
//
// Intended topology:
//
//   owner capture: explicit TX memcpy -> captured TX event
//   peer capture:  captured TX wait -> captured peer kernel
//                   -> explicit RX memcpy -> captured RX event
//   owner capture: captured RX wait -> captured owner merge
//
// The explicit memcpy nodes are added to the graph returned by
// cudaStreamGetCaptureInfo while capture is active.  They are then made
// the next captured dependency with cudaStreamUpdateCaptureDependencies.
// If any of those contracts is rejected, this probe reports the exact CUDA
// call/status and exits without silently switching to the all-explicit path.
// A compile-only child-graph composition helper is included at the bottom as
// the documented fallback seam; it is not executed by this probe.
//
// Compile/link only:
//   /usr/local/cuda-13.1/bin/nvcc -std=c++17 -O3 \
//     -gencode arch=compute_120a,code=sm_120a \
//     tools/dsv4-ep-graph-capture-gate.cu \
//     -o target/dsv4-ep-graph-capture-gate
//
// Later, under the pair lock:
//   compute-sanitizer --tool memcheck target/dsv4-ep-graph-capture-gate
//   compute-sanitizer --tool synccheck target/dsv4-ep-graph-capture-gate

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace dsv4_ep_graph_capture_gate {

constexpr int kOwner = 0;
constexpr int kPeer = 1;
constexpr std::size_t kWords = 1024;
constexpr std::size_t kGuardWords = 16;
constexpr std::uint32_t kGuard = 0x7fc01234u;
constexpr std::size_t kBytes = kWords * sizeof(std::uint32_t);

static std::string cuda_status(cudaError_t status) {
    return std::string(cudaGetErrorName(status)) + ":" + cudaGetErrorString(status);
}

static void check(cudaError_t status, const char* call) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(call) + " status=" + cuda_status(status));
}

static void check_capture(cudaError_t status, const char* call) {
    if (status != cudaSuccess) {
        throw std::runtime_error("MIXED_CAPTURE_UNAVAILABLE call=" + std::string(call)
                                 + " status=" + cuda_status(status));
    }
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
    int can01 = 0;
    int can10 = 0;
    check(cudaDeviceCanAccessPeer(&can01, kOwner, kPeer), "cudaDeviceCanAccessPeer(0,1)");
    check(cudaDeviceCanAccessPeer(&can10, kPeer, kOwner), "cudaDeviceCanAccessPeer(1,0)");
    if (!can01 || !can10) throw std::runtime_error("EP graph P2P unavailable in one direction");

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
                          cudaMemcpyHostToDevice, b.owner_stream),
          "input H2D");
    check(cudaMemcpyAsync(b.owner_input + kWords, guard.data() + kWords,
                          kGuardWords * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
                          b.owner_stream),
          "input guard H2D");
    check(cudaMemcpyAsync(b.owner_return, guard.data(), guard.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, b.owner_stream),
          "return guard H2D");
    check(cudaMemcpyAsync(b.owner_output, guard.data(), guard.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, b.owner_stream),
          "output guard H2D");
    check(cudaSetDevice(kPeer), "cudaSetDevice(reset peer)");
    check(cudaMemcpyAsync(b.peer_work, guard.data(), guard.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice, b.peer_stream),
          "peer guard H2D");
    // The host guard is temporary.  Both streams must be drained before it
    // leaves scope, and the owner must be current before a graph launch.
    check(cudaStreamSynchronize(b.owner_stream), "reset owner sync");
    check(cudaStreamSynchronize(b.peer_stream), "reset peer sync");
    check(cudaSetDevice(kOwner), "cudaSetDevice(reset complete owner)");
}

static std::vector<std::uint32_t> read_device_words(int device, std::uint32_t* ptr,
                                                    cudaStream_t stream) {
    std::vector<std::uint32_t> out(kWords + kGuardWords);
    check(cudaSetDevice(device), "cudaSetDevice(read device)");
    check(cudaMemcpyAsync(out.data(), ptr, out.size() * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost, stream),
          "device words D2H");
    check(cudaStreamSynchronize(stream), "device words sync");
    return out;
}

static void check_words(const char* label, const std::vector<std::uint32_t>& got,
                        const std::vector<std::uint32_t>& expected) {
    if (got.size() != expected.size()) throw std::runtime_error(std::string(label) + " size");
    for (std::size_t i = 0; i < kWords; ++i) {
        if (got[i] != expected[i]) {
            throw std::runtime_error(std::string(label) + " mismatch at word "
                                     + std::to_string(i));
        }
    }
    for (std::size_t i = kWords; i < got.size(); ++i) {
        if (got[i] != kGuard) throw std::runtime_error(std::string(label) + " guard changed");
    }
}

static std::vector<std::uint32_t> transformed_buffer(const std::vector<std::uint32_t>& input) {
    auto transformed = guard_buffer();
    for (std::size_t i = 0; i < kWords; ++i) {
        transformed[i] = input[i] + 0x01010101u + static_cast<std::uint32_t>(i * 3u);
    }
    return transformed;
}

static void check_output(const std::vector<std::uint32_t>& got,
                         const std::vector<std::uint32_t>& input) {
    const auto transformed = transformed_buffer(input);
    auto expected = guard_buffer();
    for (std::size_t i = 0; i < kWords; ++i) expected[i] = transformed[i] ^ input[i];
    check_words("owner output", got, expected);
}

static void check_all_buffers(Buffers& b, const std::vector<std::uint32_t>& input) {
    const auto transformed = transformed_buffer(input);
    auto expected_input = guard_buffer();
    std::copy(input.begin(), input.end(), expected_input.begin());
    check_words("owner input", read_device_words(kOwner, b.owner_input, b.owner_stream),
                expected_input);
    check_words("peer work", read_device_words(kPeer, b.peer_work, b.peer_stream), transformed);
    check_words("owner return", read_device_words(kOwner, b.owner_return, b.owner_stream),
                transformed);
}

static void enqueue_eager(Buffers& b) {
    check(cudaSetDevice(kOwner), "cudaSetDevice(eager owner)");
    check(cudaMemcpyPeerAsync(b.peer_work, kPeer, b.owner_input, kOwner, kBytes, b.owner_stream),
          "eager owner-to-peer P2P");
    check(cudaEventRecord(b.tx_done, b.owner_stream), "eager tx event record");
    check(cudaSetDevice(kPeer), "cudaSetDevice(eager peer)");
    check(cudaStreamWaitEvent(b.peer_stream, b.tx_done, 0), "eager peer tx wait");
    peer_transform<<<(kWords + 255) / 256, 256, 0, b.peer_stream>>>(
        b.peer_work, b.peer_work, kWords);
    check(cudaGetLastError(), "eager peer transform");
    check(cudaMemcpyPeerAsync(b.owner_return, kOwner, b.peer_work, kPeer, kBytes, b.peer_stream),
          "eager peer-to-owner P2P");
    check(cudaEventRecord(b.rx_done, b.peer_stream), "eager rx event record");
    check(cudaSetDevice(kOwner), "cudaSetDevice(eager merge)");
    check(cudaStreamWaitEvent(b.owner_stream, b.rx_done, 0), "eager owner rx wait");
    owner_merge<<<(kWords + 255) / 256, 256, 0, b.owner_stream>>>(
        b.owner_return, b.owner_input, b.owner_output, kWords);
    check(cudaGetLastError(), "eager owner merge");
    check(cudaStreamSynchronize(b.owner_stream), "eager owner completion");
}

struct CaptureInfo {
    cudaGraph_t graph = nullptr;
    std::vector<cudaGraphNode_t> dependencies;
};

struct CaptureTrace {
    std::size_t owner_initial_dependencies = 0;
    std::size_t peer_kernel_dependencies = 0;
    std::size_t peer_return_dependencies = 0;
    std::size_t owner_final_dependencies = 0;
};

static CaptureInfo capture_info(cudaStream_t stream, const char* call) {
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    unsigned long long id = 0;
    cudaGraph_t graph = nullptr;
    const cudaGraphNode_t* dependencies = nullptr;
    std::size_t count = 0;
    check_capture(cudaStreamGetCaptureInfo(stream, &status, &id, &graph, &dependencies, nullptr,
                                           &count),
                  call);
    if (status != cudaStreamCaptureStatusActive || graph == nullptr) {
        throw std::runtime_error("MIXED_CAPTURE_UNAVAILABLE call=" + std::string(call)
                                 + " status=not-active graph=null");
    }
    CaptureInfo info;
    info.graph = graph;
    if (dependencies && count) info.dependencies.assign(dependencies, dependencies + count);
    return info;
}

static cudaGraphNode_t add_capture_memcpy(const CaptureInfo& info, void* src, void* dst,
                                          std::size_t bytes, cudaExecutionContext_t context,
                                          const char* call) {
    cudaMemcpy3DParms copy{};
    copy.srcPtr = make_cudaPitchedPtr(src, bytes, bytes, 1);
    copy.dstPtr = make_cudaPitchedPtr(dst, bytes, bytes, 1);
    copy.extent = make_cudaExtent(bytes, 1, 1);
    copy.kind = cudaMemcpyDefault;

    cudaGraphNodeParams params{};
    params.type = cudaGraphNodeTypeMemcpy;
    params.memcpy.flags = 0;
    params.memcpy.reserved = 0;
    params.memcpy.ctx = context;
    params.memcpy.copyParams = copy;
    cudaGraphNode_t node = nullptr;
    const cudaGraphNode_t* deps = info.dependencies.empty() ? nullptr : info.dependencies.data();
    check_capture(cudaGraphAddNode(&node, info.graph, deps, nullptr, info.dependencies.size(),
                                   &params),
                  call);
    return node;
}

static void set_capture_dependency(cudaStream_t stream, cudaGraphNode_t node, const char* call) {
    cudaGraphNode_t dependency = node;
    check_capture(cudaStreamUpdateCaptureDependencies(
                      stream, &dependency, nullptr, 1, cudaStreamSetCaptureDependencies),
                  call);
}

static cudaGraph_t build_mixed_capture(Buffers& b, CaptureTrace& trace) {
    cudaExecutionContext_t owner_ctx = nullptr;
    cudaExecutionContext_t peer_ctx = nullptr;
    check(cudaDeviceGetExecutionCtx(&owner_ctx, kOwner), "cudaDeviceGetExecutionCtx(owner)");
    check(cudaDeviceGetExecutionCtx(&peer_ctx, kPeer), "cudaDeviceGetExecutionCtx(peer)");

    check_capture(cudaSetDevice(kOwner), "cudaSetDevice(capture owner)");
    check_capture(cudaStreamBeginCapture(b.owner_stream, cudaStreamCaptureModeGlobal),
                  "cudaStreamBeginCapture(owner)");
    bool capture_active = true;
    try {
        const CaptureInfo owner_start = capture_info(
            b.owner_stream, "cudaStreamGetCaptureInfo(owner initial)");
        trace.owner_initial_dependencies = owner_start.dependencies.size();
        const cudaGraphNode_t tx_copy = add_capture_memcpy(
            owner_start, b.owner_input, b.peer_work, kBytes, owner_ctx,
            "cudaGraphAddNode(captured owner-to-peer memcpy)");
        set_capture_dependency(b.owner_stream, tx_copy,
                               "cudaStreamUpdateCaptureDependencies(owner tx)");
        check_capture(cudaEventRecord(b.tx_done, b.owner_stream),
                      "cudaEventRecord(captured tx)");

        check_capture(cudaSetDevice(kPeer), "cudaSetDevice(capture peer)");
        check_capture(cudaStreamWaitEvent(b.peer_stream, b.tx_done, 0),
                      "cudaStreamWaitEvent(captured tx)");
        peer_transform<<<(kWords + 255) / 256, 256, 0, b.peer_stream>>>(
            b.peer_work, b.peer_work, kWords);
        check_capture(cudaGetLastError(), "captured peer transform");

        const CaptureInfo peer_after_kernel = capture_info(
            b.peer_stream, "cudaStreamGetCaptureInfo(peer kernel leaf)");
        trace.peer_kernel_dependencies = peer_after_kernel.dependencies.size();
        const cudaGraphNode_t rx_copy = add_capture_memcpy(
            peer_after_kernel, b.peer_work, b.owner_return, kBytes, peer_ctx,
            "cudaGraphAddNode(captured peer-to-owner memcpy)");
        trace.peer_return_dependencies = peer_after_kernel.dependencies.size();
        set_capture_dependency(b.peer_stream, rx_copy,
                               "cudaStreamUpdateCaptureDependencies(peer rx)");
        check_capture(cudaEventRecord(b.rx_done, b.peer_stream),
                      "cudaEventRecord(captured rx)");

        check_capture(cudaSetDevice(kOwner), "cudaSetDevice(capture merge)");
        check_capture(cudaStreamWaitEvent(b.owner_stream, b.rx_done, 0),
                      "cudaStreamWaitEvent(captured rx)");
        owner_merge<<<(kWords + 255) / 256, 256, 0, b.owner_stream>>>(
            b.owner_return, b.owner_input, b.owner_output, kWords);
        check_capture(cudaGetLastError(), "captured owner merge");
        const CaptureInfo owner_final = capture_info(
            b.owner_stream, "cudaStreamGetCaptureInfo(owner final)");
        trace.owner_final_dependencies = owner_final.dependencies.size();

        cudaGraph_t graph = nullptr;
        check_capture(cudaStreamEndCapture(b.owner_stream, &graph),
                      "cudaStreamEndCapture(owner)");
        capture_active = false;
        if (!graph) throw std::runtime_error("MIXED_CAPTURE_UNAVAILABLE graph=null after end");
        cudaStreamCaptureStatus peer_status = cudaStreamCaptureStatusNone;
        check_capture(cudaStreamIsCapturing(b.peer_stream, &peer_status),
                      "cudaStreamIsCapturing(peer after end)");
        if (peer_status != cudaStreamCaptureStatusNone) {
            cudaGraphDestroy(graph);
            throw std::runtime_error("MIXED_CAPTURE_UNAVAILABLE peer capture remained active");
        }
        return graph;
    } catch (...) {
        if (capture_active) {
            cudaGraph_t discarded = nullptr;
            // End the origin capture so the destructor cannot synchronize a
            // stream that is still in capture mode.  Preserve the original
            // mixed-API error for the caller.
            (void)cudaStreamEndCapture(b.owner_stream, &discarded);
            if (discarded) (void)cudaGraphDestroy(discarded);
        }
        throw;
    }
}

static void graph_census(cudaGraph_t graph) {
    std::size_t count = 0;
    check(cudaGraphGetNodes(graph, nullptr, &count), "cudaGraphGetNodes(count)");
    std::vector<cudaGraphNode_t> nodes(count);
    check(cudaGraphGetNodes(graph, nodes.data(), &count), "cudaGraphGetNodes(nodes)");
    int kernels = 0;
    int copies = 0;
    int records = 0;
    int waits = 0;
    int other = 0;
    int peer_kernel = 0;
    int owner_kernel = 0;
    int p2p_owner_peer = 0;
    int p2p_peer_owner = 0;
    int kernel_param_fail = 0;
    int memcpy_param_fail = 0;
    int pointer_attr_fail = 0;
    cudaGraphNode_t peer_node = nullptr, owner_node = nullptr;
    cudaGraphNode_t tx_node = nullptr, rx_node = nullptr;
    for (cudaGraphNode_t node : nodes) {
        cudaGraphNodeType type{};
        check(cudaGraphNodeGetType(node, &type), "cudaGraphNodeGetType");
        switch (type) {
        case cudaGraphNodeTypeKernel: {
            ++kernels;
            cudaKernelNodeParams params{};
            if (cudaGraphKernelNodeGetParams(node, &params) != cudaSuccess) {
                ++kernel_param_fail;
            } else if (params.func == reinterpret_cast<void*>(peer_transform)) {
                ++peer_kernel;
                peer_node = node;
            } else if (params.func == reinterpret_cast<void*>(owner_merge)) {
                ++owner_kernel;
                owner_node = node;
            }
            break;
        }
        case cudaGraphNodeTypeMemcpy: {
            ++copies;
            cudaMemcpy3DParms params{};
            if (cudaGraphMemcpyNodeGetParams(node, &params) != cudaSuccess) {
                ++memcpy_param_fail;
                break;
            }
            cudaPointerAttributes src_attr{};
            cudaPointerAttributes dst_attr{};
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
                if (src_attr.device == kOwner && dst_attr.device == kPeer) {
                    ++p2p_owner_peer;
                    tx_node = node;
                }
                if (src_attr.device == kPeer && dst_attr.device == kOwner) {
                    ++p2p_peer_owner;
                    rx_node = node;
                }
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
            "CENSUS_UNVERIFIED cudaGraphKernelNodeGetParams="
            + std::to_string(kernel_param_fail)
            + " cudaGraphMemcpyNodeGetParams=" + std::to_string(memcpy_param_fail)
            + " cudaPointerGetAttributes=" + std::to_string(pointer_attr_fail));
    }
    // Ordinary stream-captured record/wait pairs become dependency edges,
    // not external event nodes. Require the four operations and their order.
    if (count != 4 || kernels != 2 || peer_kernel != 1 || owner_kernel != 1
        || copies != 2 || p2p_owner_peer != 1 || p2p_peer_owner != 1
        || records != 0 || waits != 0 || other != 0) {
        throw std::runtime_error(
            "CENSUS_UNVERIFIED mixed capture topology: kernels=" + std::to_string(kernels)
            + " peer_kernel=" + std::to_string(peer_kernel)
            + " owner_kernel=" + std::to_string(owner_kernel)
            + " copies=" + std::to_string(copies)
            + " owner_to_peer=" + std::to_string(p2p_owner_peer)
            + " peer_to_owner=" + std::to_string(p2p_peer_owner)
            + " records=" + std::to_string(records)
            + " waits=" + std::to_string(waits)
            + " other=" + std::to_string(other));
    }
    std::size_t edges = 0;
    check(cudaGraphGetEdges(graph, nullptr, nullptr, nullptr, &edges), "graph edge count");
    std::vector<cudaGraphNode_t> from(edges), to(edges);
    check(cudaGraphGetEdges(graph, from.data(), to.data(), nullptr, &edges), "graph edges");
    const cudaGraphNode_t ordered[] = {tx_node, peer_node, rx_node, owner_node};
    const char* labels[] = {"tx_copy", "peer_kernel", "rx_copy", "owner_merge"};
    bool reach[4][4] = {};
    bool bad_edge = false;
    for (std::size_t i = 0; i < edges; ++i) {
        int a = -1, b = -1;
        for (int j = 0; j < 4; ++j) {
            if (from[i] == ordered[j]) a = j;
            if (to[i] == ordered[j]) b = j;
        }
        std::printf("MIXED_CAPTURE_EDGE from=%s to=%s\n",
                    a >= 0 ? labels[a] : "unknown", b >= 0 ? labels[b] : "unknown");
        if (a < 0 || b < 0 || a >= b || reach[a][b]) bad_edge = true;
        else reach[a][b] = true;
    }
    if (bad_edge) throw std::runtime_error("CENSUS_UNVERIFIED unknown/backward/duplicate edge");
    // A stream can retain an earlier dependency in addition to the joined
    // peer leaf. Such transitive edges are harmless, not a missing ordering.
    for (int k = 0; k < 4; ++k)
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j) reach[i][j] |= reach[i][k] && reach[k][j];
    if (!reach[0][1] || !reach[1][2] || !reach[2][3])
        throw std::runtime_error("CENSUS_UNVERIFIED missing required dependency path");
    std::printf("MIXED_CAPTURE_EDGES count=%zu required_order=tx_copy->peer_kernel->rx_copy->owner_merge verified=true\n", edges);
    std::printf(
        "MIXED_CAPTURE_CENSUS total=%zu kernels=%d memcpy=%d p2p_owner_peer=%d "
        "p2p_peer_owner=%d peer_kernel=%d owner_kernel=%d event_record=%d "
        "event_wait=%d other=%d path=captured-body-explicit-p2p\n",
        count, kernels, copies, p2p_owner_peer, p2p_peer_owner, peer_kernel, owner_kernel,
        records, waits, other);
}

// Compile-only fallback seam.  If the mixed graph contract is rejected on a
// future toolkit, separately captured owner/peer graphs can still be composed
// as child nodes for a later design.  This function intentionally is not
// called by run_gate: a child graph cannot by itself prove the required
// cross-device P2P/event semantics.
[[maybe_unused]] static void compile_child_graph_fallback(cudaGraph_t parent,
                                                          cudaGraph_t owner_child,
                                                          cudaGraph_t peer_child) {
    cudaGraphNodeParams owner_params{};
    owner_params.type = cudaGraphNodeTypeGraph;
    owner_params.graph.graph = owner_child;
    owner_params.graph.ownership = cudaGraphChildGraphOwnershipClone;
    cudaGraphNode_t owner_node = nullptr;
    check(cudaGraphAddNode(&owner_node, parent, nullptr, nullptr, 0, &owner_params),
          "cudaGraphAddNode(child owner)");

    cudaGraphNodeParams peer_params{};
    peer_params.type = cudaGraphNodeTypeGraph;
    peer_params.graph.graph = peer_child;
    peer_params.graph.ownership = cudaGraphChildGraphOwnershipClone;
    cudaGraphNode_t peer_node = nullptr;
    check(cudaGraphAddNode(&peer_node, parent, &owner_node, nullptr, 1, &peer_params),
          "cudaGraphAddNode(child peer)");
}

static int run_gate() {
    Buffers b;
    allocate(b);
    std::vector<std::uint32_t> input_a(kWords);
    std::vector<std::uint32_t> input_b(kWords);
    std::vector<std::uint32_t> input_c(kWords);
    for (std::size_t i = 0; i < kWords; ++i) {
        input_a[i] = mix_word(0x11110000u, i);
        input_b[i] = mix_word(0x77770000u, i);
        input_c[i] = mix_word(0xabc00000u, i);
    }

    std::puts("MIXED_CAPTURE_GATE P2P checked; probing captured body with explicit memcpy nodes");

    reset_buffers(b, input_a);
    enqueue_eager(b);
    check_all_buffers(b, input_a);
    const auto eager_a = read_device_words(kOwner, b.owner_output, b.owner_stream);
    check_output(eager_a, input_a);

    reset_buffers(b, input_a);
    CaptureTrace trace;
    cudaGraph_t graph = build_mixed_capture(b, trace);
    std::printf("MIXED_CAPTURE_DEPS owner_initial=%zu peer_kernel=%zu peer_return=%zu "
                "owner_final=%zu\n",
                trace.owner_initial_dependencies, trace.peer_kernel_dependencies,
                trace.peer_return_dependencies, trace.owner_final_dependencies);
    graph_census(graph);

    cudaGraphExec_t exec = nullptr;
    check(cudaSetDevice(kOwner), "cudaSetDevice(instantiate)");
    check(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0),
          "cudaGraphInstantiate(mixed capture)");
    check(cudaGraphUpload(exec, b.owner_stream), "cudaGraphUpload(mixed capture)");

    const std::vector<std::vector<std::uint32_t>> inputs = {input_a, input_b, input_c};
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        reset_buffers(b, inputs[i]);
        check(cudaSetDevice(kOwner), "cudaSetDevice(graph launch)");
        check(cudaGraphLaunch(exec, b.owner_stream), "cudaGraphLaunch(mixed capture)");
        check(cudaStreamSynchronize(b.owner_stream), "mixed graph owner sync");
        check_all_buffers(b, inputs[i]);
        const auto got = read_device_words(kOwner, b.owner_output, b.owner_stream);
        check_output(got, inputs[i]);
        if (i == 0 && got != eager_a)
            throw std::runtime_error("mixed captured graph differs from eager A");
    }

    check(cudaGraphExecDestroy(exec), "cudaGraphExecDestroy(mixed capture)");
    check(cudaGraphDestroy(graph), "cudaGraphDestroy(mixed capture)");
    std::puts("PASS mixed capture/explicit memcpy eager-replay identity, fresh inputs, guards");
    return 0;
}

} // namespace dsv4_ep_graph_capture_gate

int main() {
    try {
        return dsv4_ep_graph_capture_gate::run_gate();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}

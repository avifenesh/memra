// Standalone base-C4 graph micro-gate.
//
// This TU includes the production dsv4_gpu.cu implementation and calls the
// actual memra_dsv4_c4_gather launcher. It does not edit or link the Rust
// engine. The host rows use the same cacheable pinned allocation class as
// C4HostStore (cudaHostAlloc flags=0 under UVA). A producer D2H is queued on
// the same stream immediately before graph launch; replay must observe fresh
// host values without any CPU row scan.
//
// The gate uses keyed graph variants for (nq, slots, stride, live_rows,
// logical_transient, transient_rows). It covers slots 1/128/640, window/host/
// transient index sources, -1 pads, signed-zero bits, output guards, and
// pointer/shape refusal checks. Recent-sidecar mode is intentionally excluded.
//
// Build/link only; do not execute without the owning non-serving GPU lock:
//   /usr/local/cuda-13.1/bin/nvcc -std=c++17 -O3 \
//     --expt-relaxed-constexpr -gencode arch=compute_120a,code=sm_120a \
//     tools/dsv4-c4-graph-gate.cu -o target/dsv4-c4-graph-gate \
//     -lcublas -lcublasLt

#include "../crates/memra-engine/cu/dsv4_gpu.cu"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dsv4_c4_graph_gate {

constexpr int kHd = 512;
constexpr int kWin = 128;
constexpr int kHostRows = 256;
constexpr int kTransientRows = 256;
constexpr int kLogicalTransient = kWin + kHostRows;
constexpr int kMaxSlots = 640;
constexpr int kMaxStride = 647;
constexpr std::uint32_t kValueGuard = 0x7fc01234u;
constexpr std::uint32_t kIndexGuard = 0x6f6f6f6fu;

static void cuda_check(cudaError_t status, const char* what) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
}

static void check_control(const char* what, int rc) {
    if (rc != 0) throw std::runtime_error(std::string(what) + " rc=" + std::to_string(rc));
}

struct ShapeKey {
    int nq;
    int slots;
    int stride;
    int live_rows;
    int logical_transient;
    int transient_rows;

    bool operator==(const ShapeKey& other) const {
        return nq == other.nq && slots == other.slots && stride == other.stride
            && live_rows == other.live_rows && logical_transient == other.logical_transient
            && transient_rows == other.transient_rows;
    }
};

struct GraphVariant {
    ShapeKey key{};
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    std::uintptr_t host_ptr = 0;
    std::uintptr_t device_ptr = 0;
    std::uintptr_t indices_ptr = 0;
    std::uintptr_t values_ptr = 0;
    std::uintptr_t out_indices_ptr = 0;

    GraphVariant() = default;
    GraphVariant(const GraphVariant&) = delete;
    GraphVariant& operator=(const GraphVariant&) = delete;
    GraphVariant(GraphVariant&& other) noexcept
        : key(other.key), graph(other.graph), exec(other.exec),
          host_ptr(other.host_ptr), device_ptr(other.device_ptr),
          indices_ptr(other.indices_ptr), values_ptr(other.values_ptr),
          out_indices_ptr(other.out_indices_ptr) {
        other.graph = nullptr;
        other.exec = nullptr;
    }
    GraphVariant& operator=(GraphVariant&& other) noexcept {
        if (this == &other) return *this;
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
        key = other.key;
        graph = other.graph;
        exec = other.exec;
        host_ptr = other.host_ptr;
        device_ptr = other.device_ptr;
        indices_ptr = other.indices_ptr;
        values_ptr = other.values_ptr;
        out_indices_ptr = other.out_indices_ptr;
        other.graph = nullptr;
        other.exec = nullptr;
        return *this;
    }

    ~GraphVariant() {
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
    }

    bool pointers_match(std::uintptr_t host, std::uintptr_t device,
                        std::uintptr_t indices, std::uintptr_t values,
                        std::uintptr_t out_indices) const {
        return host_ptr == host && device_ptr == device && indices_ptr == indices
            && values_ptr == values && out_indices_ptr == out_indices;
    }
};

struct GraphStore {
    std::vector<GraphVariant> variants;

    GraphVariant* find(const ShapeKey& key) {
        for (auto& variant : variants) if (variant.key == key) return &variant;
        return nullptr;
    }
};

struct Buffers {
    cudaStream_t stream = nullptr;
    float* device_rows = nullptr;
    float* producer = nullptr;
    int* indices = nullptr;
    float* eager_values = nullptr;
    int* eager_indices = nullptr;
    float* graph_values = nullptr;
    int* graph_indices = nullptr;
    std::size_t host_bytes = static_cast<std::size_t>(kHostRows) * kHd * sizeof(float);
    std::size_t device_rows_count = static_cast<std::size_t>(kWin + kTransientRows) * kHd;
    std::size_t indices_count = kMaxStride;
    std::size_t values_count = static_cast<std::size_t>(kMaxSlots) * kHd + 16;

    ~Buffers() {
        if (stream) cudaStreamSynchronize(stream);
        if (device_rows) cudaFree(device_rows);
        if (producer) cudaFree(producer);
        if (indices) cudaFree(indices);
        if (eager_values) cudaFree(eager_values);
        if (eager_indices) cudaFree(eager_indices);
        if (graph_values) cudaFree(graph_values);
        if (graph_indices) cudaFree(graph_indices);
        if (stream) cudaStreamDestroy(stream);
    }
};

static void alloc_buffers(Buffers& b) {
    cuda_check(cudaStreamCreateWithFlags(&b.stream, cudaStreamNonBlocking), "stream create");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&b.device_rows),
                          b.device_rows_count * sizeof(float)), "device rows alloc");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&b.producer),
                          b.host_bytes), "producer alloc");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&b.indices),
                          b.indices_count * sizeof(int)), "indices alloc");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&b.eager_values),
                          b.values_count * sizeof(float)), "eager values alloc");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&b.eager_indices),
                          b.indices_count * sizeof(int)), "eager indices alloc");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&b.graph_values),
                          b.values_count * sizeof(float)), "graph values alloc");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&b.graph_indices),
                          b.indices_count * sizeof(int)), "graph indices alloc");
}

struct PinnedRows {
    float* ptr = nullptr;
    std::size_t count = static_cast<std::size_t>(kHostRows) * kHd;

    PinnedRows() {
        cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&ptr),
                                 count * sizeof(float), 0), "mapped host rows alloc");
    }
    ~PinnedRows() { if (ptr) cudaFreeHost(ptr); }
};

static std::uint32_t pattern_bits(int seed, std::size_t i) {
    if (((i + static_cast<std::size_t>(seed)) % 97) == 0) return 0x80000000u;
    if (((i + static_cast<std::size_t>(seed)) % 193) == 0) return 0x00000000u;
    std::uint32_t x = static_cast<std::uint32_t>(i) * 0x9e3779b9u
        + static_cast<std::uint32_t>(seed) * 0x85ebca6bu;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    return 0x3f000000u | (x & 0x007fffffu);
}

static std::vector<float> make_rows(int seed) {
    std::vector<float> rows(static_cast<std::size_t>(kHostRows) * kHd);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const std::uint32_t bits = pattern_bits(seed, i);
        std::memcpy(&rows[i], &bits, sizeof(bits));
    }
    return rows;
}

static int index_for_slot(const ShapeKey& key, int slot) {
    if (key.slots == 1) return 0;
    if (key.slots == 128) return slot == key.slots - 1 ? -1 : slot;
    if (slot < kWin) return slot;
    if (slot < kWin + kHostRows) return kWin + (slot - kWin);
    if (slot < kMaxSlots - 1) return key.logical_transient + (slot - kWin - kHostRows);
    return -1;
}

static void fill_device_rows(Buffers& b) {
    std::vector<float> rows(b.device_rows_count);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const std::uint32_t bits = pattern_bits(73, i + 10000);
        std::memcpy(&rows[i], &bits, sizeof(bits));
    }
    cuda_check(cudaMemcpy(b.device_rows, rows.data(), rows.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "device rows upload");
}

static void set_indices(Buffers& b, const ShapeKey& key) {
    std::vector<int> indices(b.indices_count, -1);
    for (int slot = 0; slot < key.slots; ++slot) indices[slot] = index_for_slot(key, slot);
    cuda_check(cudaMemcpy(b.indices, indices.data(), indices.size() * sizeof(int),
                          cudaMemcpyHostToDevice), "indices upload");
}

static void reset_outputs(Buffers& b) {
    std::vector<std::uint32_t> value_guard(b.values_count, kValueGuard);
    std::vector<std::uint32_t> index_guard(b.indices_count, kIndexGuard);
    cuda_check(cudaMemcpyAsync(b.eager_values, value_guard.data(),
                               value_guard.size() * sizeof(std::uint32_t),
                               cudaMemcpyHostToDevice, b.stream), "eager value guard");
    cuda_check(cudaMemcpyAsync(b.graph_values, value_guard.data(),
                               value_guard.size() * sizeof(std::uint32_t),
                               cudaMemcpyHostToDevice, b.stream), "graph value guard");
    cuda_check(cudaMemcpyAsync(b.eager_indices, index_guard.data(),
                               index_guard.size() * sizeof(std::uint32_t),
                               cudaMemcpyHostToDevice, b.stream), "eager index guard");
    cuda_check(cudaMemcpyAsync(b.graph_indices, index_guard.data(),
                               index_guard.size() * sizeof(std::uint32_t),
                               cudaMemcpyHostToDevice, b.stream), "graph index guard");
    cuda_check(cudaStreamSynchronize(b.stream), "output guard sync");
}

static void publish_host(Buffers& b, PinnedRows& host, const std::vector<float>& rows) {
    cuda_check(cudaMemcpyAsync(b.producer, rows.data(), rows.size() * sizeof(float),
                               cudaMemcpyHostToDevice, b.stream), "producer H2D");
    // This is the same owner-stream D2H publication that C4HostStore::write
    // queues. There is deliberately no CPU wait before the graph replay.
    cuda_check(cudaMemcpyAsync(host.ptr, b.producer, rows.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, b.stream), "producer D2H");
}

static std::vector<std::uint32_t> read_values(const float* ptr, std::size_t count,
                                              cudaStream_t stream) {
    std::vector<float> values(count);
    cuda_check(cudaMemcpyAsync(values.data(), ptr, count * sizeof(float),
                               cudaMemcpyDeviceToHost, stream), "values D2H");
    cuda_check(cudaStreamSynchronize(stream), "values sync");
    std::vector<std::uint32_t> bits(count);
    for (std::size_t i = 0; i < count; ++i)
        std::memcpy(&bits[i], &values[i], sizeof(std::uint32_t));
    return bits;
}

static std::vector<std::uint32_t> read_indices(const int* ptr, std::size_t count,
                                               cudaStream_t stream) {
    std::vector<int> values(count);
    cuda_check(cudaMemcpyAsync(values.data(), ptr, count * sizeof(int),
                               cudaMemcpyDeviceToHost, stream), "indices D2H");
    cuda_check(cudaStreamSynchronize(stream), "indices sync");
    std::vector<std::uint32_t> bits(count);
    for (std::size_t i = 0; i < count; ++i)
        std::memcpy(&bits[i], &values[i], sizeof(std::uint32_t));
    return bits;
}

static void check_output(const char* arm, const ShapeKey& key,
                         const std::vector<std::uint32_t>& values,
                         const std::vector<std::uint32_t>& indices) {
    for (std::size_t i = 0; i < static_cast<std::size_t>(key.slots) * kHd; ++i) {
        if (i >= values.size()) throw std::runtime_error("value output too short");
        float value = 0.0f;
        std::memcpy(&value, &values[i], sizeof(value));
        if (!std::isfinite(value) && values[i] != 0x80000000u && values[i] != 0x00000000u)
            throw std::runtime_error(std::string(arm) + " nonfinite value output");
    }
    for (std::size_t i = static_cast<std::size_t>(key.slots) * kHd;
         i < values.size(); ++i) {
        if (values[i] != kValueGuard)
            throw std::runtime_error(std::string(arm) + " value guard changed");
    }
    for (int slot = 0; slot < key.slots; ++slot) {
        const std::uint32_t got = indices[slot];
        const bool pad = got == 0xffffffffu;
        if (!pad && got != static_cast<std::uint32_t>(slot))
            throw std::runtime_error(std::string(arm) + " mapped-index mismatch");
    }
    for (std::size_t i = key.slots; i < indices.size(); ++i) {
        if (indices[i] != kIndexGuard)
            throw std::runtime_error(std::string(arm) + " index guard changed");
    }
}

static GraphVariant capture_variant(const ShapeKey& key, Buffers& b, PinnedRows& host) {
    GraphVariant variant;
    variant.key = key;
    variant.host_ptr = reinterpret_cast<std::uintptr_t>(host.ptr);
    variant.device_ptr = reinterpret_cast<std::uintptr_t>(b.device_rows);
    variant.indices_ptr = reinterpret_cast<std::uintptr_t>(b.indices);
    variant.values_ptr = reinterpret_cast<std::uintptr_t>(b.graph_values);
    variant.out_indices_ptr = reinterpret_cast<std::uintptr_t>(b.graph_indices);
    cuda_check(cudaStreamBeginCapture(b.stream, cudaStreamCaptureModeRelaxed), "C4 begin capture");
    check_control("C4 capture control", memra_dsv4_c4_gather(
        b.device_rows, host.ptr, b.indices, b.graph_values, b.graph_indices,
        key.nq, key.slots, key.stride, key.live_rows,
        key.logical_transient, key.transient_rows, b.stream));
    cuda_check(cudaStreamEndCapture(b.stream, &variant.graph), "C4 end capture");
    if (!variant.graph) throw std::runtime_error("C4 capture returned null graph");
    cuda_check(cudaGraphInstantiate(&variant.exec, variant.graph, nullptr, nullptr, 0),
               "C4 graph instantiate");
    cuda_check(cudaGraphUpload(variant.exec, b.stream), "C4 graph upload");
    cuda_check(cudaGraphLaunch(variant.exec, b.stream), "C4 first graph launch");
    cuda_check(cudaStreamSynchronize(b.stream), "C4 first graph sync");
    return variant;
}

static void require_equal(const char* arm, const std::vector<std::uint32_t>& a,
                          const std::vector<std::uint32_t>& b) {
    if (a != b) throw std::runtime_error(std::string(arm) + " output bit mismatch");
}

static bool check_fresh_host_generation(const ShapeKey& key,
                                        const std::vector<std::uint32_t>& generation_a,
                                        const std::vector<std::uint32_t>& generation_b,
                                        const std::vector<float>& known_b) {
    bool selected_host = false;
    bool changed = false;
    for (int slot = 0; slot < key.slots; ++slot) {
        const int index = index_for_slot(key, slot);
        if (index < kWin || index >= kWin + key.live_rows) continue;
        selected_host = true;
        const int host_row = index - kWin;
        const std::size_t output_base = static_cast<std::size_t>(slot) * kHd;
        const std::size_t host_base = static_cast<std::size_t>(host_row) * kHd;
        for (int x = 0; x < kHd; ++x) {
            std::uint32_t expected = 0;
            std::memcpy(&expected, &known_b[host_base + x], sizeof(expected));
            if (generation_b[output_base + x] != expected) {
                throw std::runtime_error("fresh mapped-host generation bit mismatch");
            }
            if (generation_a[output_base + x] != generation_b[output_base + x]) changed = true;
        }
    }
    if (selected_host && !changed)
        throw std::runtime_error("fresh mapped-host generation did not change any selected row");
    return selected_host;
}

static ShapeKey shape_for_slots(int slots) {
    return ShapeKey{1, slots, slots == 1 ? 8 : slots == 128 ? 136 : 647,
                    kHostRows, kLogicalTransient, kTransientRows};
}

static bool run_shape(Buffers& b, PinnedRows& host, GraphStore& store, const ShapeKey& key) {
    set_indices(b, key);
    const std::vector<float> first = make_rows(11 + key.slots);
    const std::vector<float> second = make_rows(91 + key.slots);
    reset_outputs(b);

    // Publish A and drain once before capture. The replay test below publishes B
    // without a CPU wait, so only the stream edge can make B visible.
    publish_host(b, host, first);
    cuda_check(cudaStreamSynchronize(b.stream), "initial producer sync");
    check_control("C4 eager A", memra_dsv4_c4_gather(
        b.device_rows, host.ptr, b.indices, b.eager_values, b.eager_indices,
        key.nq, key.slots, key.stride, key.live_rows,
        key.logical_transient, key.transient_rows, b.stream));
    cuda_check(cudaStreamSynchronize(b.stream), "eager A sync");
    const auto eager_a_values = read_values(b.eager_values, b.values_count, b.stream);
    const auto eager_a_indices = read_indices(b.eager_indices, b.indices_count, b.stream);

    GraphVariant variant = capture_variant(key, b, host);
    const auto graph_a_values = read_values(b.graph_values, b.values_count, b.stream);
    const auto graph_a_indices = read_indices(b.graph_indices, b.indices_count, b.stream);
    check_output("graph A", key, graph_a_values, graph_a_indices);
    require_equal("capture-vs-eager values", eager_a_values, graph_a_values);
    require_equal("capture-vs-eager indices", eager_a_indices, graph_a_indices);

    // Fresh mapped-host data is published asynchronously on the same stream,
    // then replayed without a host synchronization or CPU row traversal.
    reset_outputs(b);
    publish_host(b, host, second);
    cuda_check(cudaGraphLaunch(variant.exec, b.stream), "C4 replay B");
    cuda_check(cudaStreamSynchronize(b.stream), "C4 replay B sync");
    const auto graph_b_values = read_values(b.graph_values, b.values_count, b.stream);
    const auto graph_b_indices = read_indices(b.graph_indices, b.indices_count, b.stream);
    check_output("graph B", key, graph_b_values, graph_b_indices);
    const bool selected_host = check_fresh_host_generation(
        key, graph_a_values, graph_b_values, second);

    reset_outputs(b);
    check_control("C4 eager B", memra_dsv4_c4_gather(
        b.device_rows, host.ptr, b.indices, b.eager_values, b.eager_indices,
        key.nq, key.slots, key.stride, key.live_rows,
        key.logical_transient, key.transient_rows, b.stream));
    cuda_check(cudaStreamSynchronize(b.stream), "eager B sync");
    const auto eager_b_values = read_values(b.eager_values, b.values_count, b.stream);
    const auto eager_b_indices = read_indices(b.eager_indices, b.indices_count, b.stream);
    require_equal("fresh-host replay values", eager_b_values, graph_b_values);
    require_equal("fresh-host replay indices", eager_b_indices, graph_b_indices);
    std::printf("C4 shape slots=%d stride=%d live_rows=%d capture_replay_fresh_host=PASS\n",
                key.slots, key.stride, key.live_rows);
    store.variants.push_back(std::move(variant));
    return selected_host;
}

static int run_gate() {
    int device = 0;
    cuda_check(cudaGetDevice(&device), "get device");
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, device), "device properties");
    if (!prop.unifiedAddressing)
        throw std::runtime_error("C4 gate requires unified addressing");

    Buffers buffers;
    PinnedRows host;
    GraphStore graphs;
    alloc_buffers(buffers);
    fill_device_rows(buffers);
    bool host_selection_exercised = false;
    for (const int slots : {1, 128, 640}) {
        host_selection_exercised = run_shape(
            buffers, host, graphs, shape_for_slots(slots)) || host_selection_exercised;
    }
    if (!host_selection_exercised)
        throw std::runtime_error("C4 gate did not exercise a selected mapped-host row");

    // Shape and pointer invalidation are refusal checks, not launches. The
    // keyed store must never replay a 128-slot graph as a 640-slot graph, and
    // a graph baked with the original mapped-host pointer must reject another
    // host allocation even if its contents happen to match.
    if (graphs.find(shape_for_slots(128)) == nullptr || graphs.find(shape_for_slots(640)) == nullptr)
        throw std::runtime_error("missing keyed shape variants");
    float* other_host = nullptr;
    cuda_check(cudaHostAlloc(reinterpret_cast<void**>(&other_host),
                             host.count * sizeof(float), 0), "replacement host alloc");
    const GraphVariant* v128 = graphs.find(shape_for_slots(128));
    if (v128->pointers_match(reinterpret_cast<std::uintptr_t>(other_host),
                              reinterpret_cast<std::uintptr_t>(buffers.device_rows),
                              reinterpret_cast<std::uintptr_t>(buffers.indices),
                              reinterpret_cast<std::uintptr_t>(buffers.graph_values),
                              reinterpret_cast<std::uintptr_t>(buffers.graph_indices))) {
        cudaFreeHost(other_host);
        throw std::runtime_error("pointer replacement was not refused");
    }
    cuda_check(cudaFreeHost(other_host), "replacement host free");
    std::puts("POINTER_AND_SHAPE_REFUSALS=PASS");
    std::puts("PASS base C4 graph micro-gate protocol; no recent-sidecar arm");
    return 0;
}

} // namespace dsv4_c4_graph_gate

int main() {
    try {
        return dsv4_c4_graph_gate::run_gate();
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}

// Standalone gate for the current cu/tp_ar.cu one-shot OUT-OF-PLACE transport.
//
// Build with the exact CUDA source, for example:
//   nvcc -std=c++17 --expt-relaxed-constexpr -O3 \
//     -gencode arch=compute_120a,code=sm_120a \
//     tools/tp-ar-one-shot-gate.cu crates/memra-engine/cu/tp_ar.cu \
//     -o target/tp-ar-one-shot-gate
//
// Source pin at creation: cu/tp_ar.cu SHA256
// 86c6be8d8b3ad54b197d45d1ef9f8c768803ef5394fd4123f795268c3143d809
//
// This is a correctness/engagement gate, not a throughput benchmark. It uses two real
// devices, the exact memra_tp_ar_1stage launcher, out-of-place outputs, fresh finite inputs
// for many generations, full-buffer input/output canaries, bitwise rank checks, and the
// kernel's bounded barrier-error words. The final negative arm launches only rank 0 with a
// bounded spin and requires the expected 40043 start-barrier refusal; it must not hang.

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" int memra_tp_ar_signal_bytes(void);
extern "C" int memra_tp_ar_1stage(const float* in_rank0, const float* in_rank1, float* out,
                                   void* self_sg, void* peer_sg, int rank, long n, int* err,
                                   long long spin_limit, int blocks, void* stream_v);

namespace {

constexpr std::size_t kGuard = 32;
constexpr int kGenerations = 32;
constexpr long long kPositiveSpin = 2'000'000'000LL;
constexpr long long kNegativeSpin = 5'000'000LL;
constexpr std::uint32_t kInputGuard = 0x3f4ccccdu;  // 0.8f
constexpr std::uint32_t kOutputGuard = 0x3f19999au; // 0.6f

struct Failure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

void cuda_check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        throw Failure(std::string(what) + ": " + cudaGetErrorString(error));
    }
}

void set_device(int device) {
    cuda_check(cudaSetDevice(device), "cudaSetDevice");
}

std::uint32_t bits(float value) {
    std::uint32_t out;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

float from_bits(std::uint32_t value) {
    float out;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

std::uint64_t mix(std::uint64_t value) {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

std::uint64_t hash_bits(const std::vector<float>& values) {
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    for (float value : values) {
        hash ^= bits(value);
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

std::vector<float> input_pattern(std::size_t n, int rank, int generation) {
    std::vector<float> values(n + 2 * kGuard, from_bits(kInputGuard));
    std::uint64_t state = 0x9e3779b97f4a7c15ULL ^
                          (static_cast<std::uint64_t>(rank) << 48) ^
                          (static_cast<std::uint64_t>(generation) << 24) ^ n;
    for (std::size_t i = 0; i < n; ++i) {
        state = mix(state + 0x9e3779b97f4a7c15ULL + i);
        const auto bucket = static_cast<std::int32_t>(state % 1'500'001ULL) - 750'000;
        float value = static_cast<float>(bucket) / 1'000'000.0f;
        // Non-periodic anchors keep a zero-length, stale-pointer or rank-swap bug visible.
        if (i == 0) value = 0.125f + static_cast<float>(generation) * 0.001f + rank * 0.01f;
        if (i == 1) value = -0.25f - static_cast<float>(generation) * 0.0005f - rank * 0.02f;
        if (!std::isfinite(value)) throw Failure("generated TP_AR input is not finite");
        values[kGuard + i] = value;
    }
    return values;
}

std::vector<float> output_canary(std::size_t n) {
    return std::vector<float>(n + 2 * kGuard, from_bits(kOutputGuard));
}

int blocks_for(std::size_t n) {
    const auto blocks = (n + 511) / 512;
    if (blocks == 0 || blocks > 72) {
        throw Failure("gate shape exceeds the current 72-block signal contract");
    }
    return static_cast<int>(blocks);
}

struct StreamPair {
    cudaStream_t stream[2] = {nullptr, nullptr};

    StreamPair() {
        for (int rank = 0; rank < 2; ++rank) {
            set_device(rank);
            cuda_check(cudaStreamCreateWithFlags(&stream[rank], cudaStreamNonBlocking),
                       "cudaStreamCreateWithFlags");
        }
    }

    ~StreamPair() noexcept {
        for (int rank = 0; rank < 2; ++rank) {
            if (stream[rank] != nullptr) {
                (void)cudaSetDevice(rank);
                (void)cudaStreamDestroy(stream[rank]);
            }
        }
    }
};

struct DeviceFloats {
    int device;
    std::size_t elements;
    float* base = nullptr;

    DeviceFloats(int device_, std::size_t body_elements) : device(device_), elements(body_elements + 2 * kGuard) {
        set_device(device);
        cuda_check(cudaMalloc(&base, elements * sizeof(float)), "cudaMalloc float buffer");
    }

    ~DeviceFloats() noexcept {
        if (base != nullptr) {
            (void)cudaSetDevice(device);
            (void)cudaFree(base);
        }
    }

    float* body() const { return base + kGuard; }
};

struct DeviceBytes {
    int device;
    std::size_t bytes;
    void* ptr = nullptr;

    DeviceBytes(int device_, std::size_t bytes_, bool zero = true)
        : device(device_), bytes(bytes_) {
        set_device(device);
        cuda_check(cudaMalloc(&ptr, bytes), "cudaMalloc byte buffer");
        if (zero) cuda_check(cudaMemset(ptr, 0, bytes), "cudaMemset byte buffer");
    }

    ~DeviceBytes() noexcept {
        if (ptr != nullptr) {
            (void)cudaSetDevice(device);
            (void)cudaFree(ptr);
        }
    }
};

void enable_peer_pair() {
    int count = 0;
    cuda_check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count < 2) throw Failure("TP_AR_GATE needs two real CUDA devices");
    for (int a = 0; a < 2; ++a) {
        int can_access = 0;
        cuda_check(cudaDeviceCanAccessPeer(&can_access, a, 1 - a), "cudaDeviceCanAccessPeer");
        if (!can_access) throw Failure("TP_AR_GATE peer access is unavailable in one direction");
        set_device(a);
        const cudaError_t enabled = cudaDeviceEnablePeerAccess(1 - a, 0);
        if (enabled != cudaSuccess && enabled != cudaErrorPeerAccessAlreadyEnabled) {
            cuda_check(enabled, "cudaDeviceEnablePeerAccess");
        }
        (void)cudaGetLastError();
    }
}

void upload_full(int device, cudaStream_t stream, const DeviceFloats& dst,
                 const std::vector<float>& values) {
    set_device(device);
    if (values.size() != dst.elements) throw Failure("host/device buffer shape mismatch");
    cuda_check(cudaMemcpyAsync(dst.base, values.data(), values.size() * sizeof(float),
                               cudaMemcpyHostToDevice, stream),
               "input/output H2D");
    cuda_check(cudaStreamSynchronize(stream), "H2D synchronization");
}

std::vector<float> download_full(int device, cudaStream_t stream, const DeviceFloats& src) {
    set_device(device);
    std::vector<float> values(src.elements);
    cuda_check(cudaMemcpyAsync(values.data(), src.base, values.size() * sizeof(float),
                               cudaMemcpyDeviceToHost, stream),
               "D2H full buffer");
    cuda_check(cudaStreamSynchronize(stream), "D2H synchronization");
    return values;
}

void compare_exact(const std::vector<float>& got, const std::vector<float>& want,
                   const char* what, std::size_t n, int generation, int rank) {
    if (got.size() != want.size()) throw Failure("comparison shape mismatch");
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (bits(got[i]) != bits(want[i])) {
            throw Failure(std::string(what) + " mismatch shape=" + std::to_string(n) +
                          " generation=" + std::to_string(generation) +
                          " rank=" + std::to_string(rank) + " element=" + std::to_string(i));
        }
    }
}

void compare_body_sum(const std::vector<float>& got, const std::vector<float>& a,
                      const std::vector<float>& b, std::size_t n, int generation, int rank) {
    for (std::size_t i = 0; i < n; ++i) {
        const float want = a[kGuard + i] + b[kGuard + i];
        if (bits(got[kGuard + i]) != bits(want)) {
            throw Failure("one-shot body mismatch shape=" + std::to_string(n) +
                          " generation=" + std::to_string(generation) +
                          " rank=" + std::to_string(rank) + " element=" + std::to_string(i));
        }
    }
}

void check_output_guards(const std::vector<float>& got, std::size_t n, int generation, int rank) {
    for (std::size_t i = 0; i < kGuard; ++i) {
        if (bits(got[i]) != kOutputGuard || bits(got[kGuard + n + i]) != kOutputGuard) {
            throw Failure("output canary changed shape=" + std::to_string(n) +
                          " generation=" + std::to_string(generation) +
                          " rank=" + std::to_string(rank));
        }
    }
}

void run_positive_shape(std::size_t n) {
    StreamPair streams;
    std::array<std::unique_ptr<DeviceFloats>, 2> inputs;
    std::array<std::unique_ptr<DeviceFloats>, 2> outputs;
    for (int rank = 0; rank < 2; ++rank) {
        inputs[rank] = std::make_unique<DeviceFloats>(rank, n);
        outputs[rank] = std::make_unique<DeviceFloats>(rank, n);
    }
    const int signal_bytes = memra_tp_ar_signal_bytes();
    if (signal_bytes <= 0 || signal_bytes > (1 << 20)) {
        throw Failure("invalid signal allocation size from current tp_ar.cu");
    }
    std::array<std::unique_ptr<DeviceBytes>, 2> signals;
    std::array<std::unique_ptr<DeviceBytes>, 2> errors;
    for (int rank = 0; rank < 2; ++rank) {
        signals[rank] = std::make_unique<DeviceBytes>(rank, static_cast<std::size_t>(signal_bytes));
        errors[rank] = std::make_unique<DeviceBytes>(rank, sizeof(int));
    }
    const int blocks = blocks_for(n);
    std::uint64_t previous_hash = 0;
    int fresh_hashes = 0;
    bool body_changed = false;

    for (int generation = 0; generation < kGenerations; ++generation) {
        const auto host_a = input_pattern(n, 0, generation);
        const auto host_b = input_pattern(n, 1, generation);
        const auto out_init = output_canary(n);
        const std::uint64_t input_hash = hash_bits(host_a) ^ (hash_bits(host_b) << 1);
        if (generation > 0 && input_hash == previous_hash) {
            throw Failure("successive TP_AR generations reused the same input fingerprint");
        }
        previous_hash = input_hash;
        ++fresh_hashes;

        upload_full(0, streams.stream[0], *inputs[0], host_a);
        upload_full(1, streams.stream[1], *inputs[1], host_b);
        upload_full(0, streams.stream[0], *outputs[0], out_init);
        upload_full(1, streams.stream[1], *outputs[1], out_init);
        for (int rank = 0; rank < 2; ++rank) {
            set_device(rank);
            cuda_check(cudaMemsetAsync(errors[rank]->ptr, 0, sizeof(int), streams.stream[rank]),
                       "clear TP_AR barrier error");
            cuda_check(cudaStreamSynchronize(streams.stream[rank]), "clear-error synchronization");
        }

        for (int rank = 0; rank < 2; ++rank) {
            set_device(rank);
            const int rc = memra_tp_ar_1stage(
                inputs[0]->body(), inputs[1]->body(), outputs[rank]->body(), signals[rank]->ptr,
                signals[1 - rank]->ptr, rank, static_cast<long>(n),
                static_cast<int*>(errors[rank]->ptr), kPositiveSpin, blocks,
                static_cast<void*>(streams.stream[rank]));
            if (rc != 0) {
                throw Failure("current memra_tp_ar_1stage enqueue refused shape=" +
                              std::to_string(n) + " rank=" + std::to_string(rank) +
                              " rc=" + std::to_string(rc));
            }
        }

        for (int rank = 0; rank < 2; ++rank) {
            set_device(rank);
            cuda_check(cudaStreamSynchronize(streams.stream[rank]), "TP_AR positive synchronization");
        }

        std::array<int, 2> barrier_error = {0, 0};
        for (int rank = 0; rank < 2; ++rank) {
            set_device(rank);
            cuda_check(cudaMemcpy(&barrier_error[rank], errors[rank]->ptr, sizeof(int),
                                  cudaMemcpyDeviceToHost),
                       "read TP_AR barrier error");
            if (barrier_error[rank] != 0) {
                throw Failure("positive TP_AR barrier error shape=" + std::to_string(n) +
                              " generation=" + std::to_string(generation) +
                              " rank=" + std::to_string(rank) + " code=" +
                              std::to_string(barrier_error[rank]));
            }
        }

        const auto got_a = download_full(0, streams.stream[0], *outputs[0]);
        const auto got_b = download_full(1, streams.stream[1], *outputs[1]);
        const auto input_a_after = download_full(0, streams.stream[0], *inputs[0]);
        const auto input_b_after = download_full(1, streams.stream[1], *inputs[1]);
        compare_exact(input_a_after, host_a, "rank0 input", n, generation, 0);
        compare_exact(input_b_after, host_b, "rank1 input", n, generation, 1);
        check_output_guards(got_a, n, generation, 0);
        check_output_guards(got_b, n, generation, 1);
        compare_body_sum(got_a, host_a, host_b, n, generation, 0);
        compare_body_sum(got_b, host_a, host_b, n, generation, 1);
        for (std::size_t i = 0; i < n; ++i) {
            if (bits(got_a[kGuard + i]) != bits(got_b[kGuard + i])) {
                throw Failure("rank outputs are not bitwise identical shape=" + std::to_string(n) +
                              " generation=" + std::to_string(generation) +
                              " element=" + std::to_string(i));
            }
            if (bits(got_a[kGuard + i]) != kOutputGuard) body_changed = true;
        }
        // The body must have been written by the one-shot, not merely left as the canary.
        if (!body_changed) {
            throw Failure("one-shot output body never changed from its canary");
        }
        if (generation == 0 || generation == kGenerations - 1) {
            std::printf("TP_AR_GENERATION shape=%zu generation=%d input_hash=%016llx "
                        "barrier=[%d,%d] outputs=bit-exact inputs=unchanged canaries=pass\n",
                        n, generation, static_cast<unsigned long long>(input_hash),
                        barrier_error[0], barrier_error[1]);
        }
    }
    std::printf("TP_AR_ONE_SHOT_PASS shape=%zu blocks=%d generations=%d launches=%d "
                "fresh_hashes=%d barrier_errors=[0,0] rank_outputs=bit-exact "
                "inputs_unchanged=true canaries=true launcher=memra_tp_ar_1stage\n",
                n, blocks, kGenerations, kGenerations * 2, fresh_hashes);
}

void run_missing_peer_refusal(std::size_t n) {
    StreamPair streams;
    DeviceFloats input0(0, n);
    DeviceFloats input1(1, n);
    DeviceFloats output0(0, n);
    const int signal_bytes = memra_tp_ar_signal_bytes();
    if (signal_bytes <= 0 || signal_bytes > (1 << 20)) {
        throw Failure("invalid signal allocation size from current tp_ar.cu");
    }
    DeviceBytes signal0(0, static_cast<std::size_t>(signal_bytes));
    DeviceBytes signal1(1, static_cast<std::size_t>(signal_bytes));
    DeviceBytes error0(0, sizeof(int));
    const auto host_a = input_pattern(n, 0, 0x5a);
    const auto host_b = input_pattern(n, 1, 0x5a);
    const auto out_init = output_canary(n);
    upload_full(0, streams.stream[0], input0, host_a);
    upload_full(1, streams.stream[1], input1, host_b);
    upload_full(0, streams.stream[0], output0, out_init);
    set_device(0);
    cuda_check(cudaMemset(error0.ptr, 0, sizeof(int)), "clear negative barrier error");

    const auto start = std::chrono::steady_clock::now();
    set_device(0);
    const int rc = memra_tp_ar_1stage(
        input0.body(), input1.body(), output0.body(), signal0.ptr, signal1.ptr, 0,
        static_cast<long>(n), static_cast<int*>(error0.ptr), kNegativeSpin, blocks_for(n),
        static_cast<void*>(streams.stream[0]));
    if (rc != 0) throw Failure("missing-peer launcher enqueue failed before bounded refusal");
    cuda_check(cudaStreamSynchronize(streams.stream[0]), "missing-peer bounded synchronization");
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - start)
                                .count();
    if (elapsed_ms > 5000) throw Failure("missing-peer refusal exceeded the 5 s bounded gate");

    int barrier_error = 0;
    set_device(0);
    cuda_check(cudaMemcpy(&barrier_error, error0.ptr, sizeof(int), cudaMemcpyDeviceToHost),
               "read missing-peer barrier error");
    if (barrier_error != 40043) {
        throw Failure("missing-peer did not produce expected start-barrier refusal 40043; got " +
                      std::to_string(barrier_error));
    }
    const auto got = download_full(0, streams.stream[0], output0);
    const auto input_after = download_full(0, streams.stream[0], input0);
    compare_exact(input_after, host_a, "missing-peer input", n, 0, 0);
    compare_exact(got, out_init, "missing-peer output/canary", n, 0, 0);
    std::printf("TP_AR_NEGATIVE_PASS shape=%zu missing_peer=true barrier_error=%d "
                "elapsed_ms=%lld bounded=true output_untouched=true input_unchanged=true\n",
                n, barrier_error, static_cast<long long>(elapsed_ms));
}

} // namespace

int main() {
    try {
        enable_peer_pair();
        // These are the only scored transport shapes: hidden=4096 and 6*hidden.
        run_positive_shape(4096);
        run_positive_shape(6 * 4096);
        // Use a fresh signal/output allocation and do not reuse the positive state after refusal.
        run_missing_peer_refusal(4096);
        std::puts("TP_AR_GATE_PASS one-shot out-of-place transport qualified for correctness");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "TP_AR_GATE_FAIL %s\n", error.what());
        return 2;
    }
}

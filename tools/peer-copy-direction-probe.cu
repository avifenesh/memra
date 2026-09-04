// Diagnose copy-engine direction on a peer-capable CUDA fabric.
//
// PP historically publishes an activation with a producer-issued peer write:
// `cuMemcpyPeerAsync(..., producer_stream)`, followed by a producer event and a wait on the
// consumer stream. The generic TP transport instead performs a consumer-issued peer read:
// the producer publishes an event, the consumer waits, then calls `cuMemcpyDtoDAsync` on its
// own stream. These are different fabric paths on PCIe systems. A passing generic
// `cudaMemcpyPeerAsync` probe does not identify which one is sound.
//
// This tool retains the same primary contexts Memra uses, enables peer access in every
// supported direction, poisons each destination, and compares four exact copy programs:
//
//   push-peer  cuMemcpyPeerAsync on the producer stream
//   pull-peer  cuMemcpyPeerAsync on the consumer stream
//   push-dtod  cuMemcpyDtoDAsync on the producer stream
//   pull-dtod  cuMemcpyDtoDAsync on the consumer stream (the TP peer-pull shape)
//
// Every asynchronous arm carries an explicit cross-device event dependency and validates
// 16 KiB, 1 MiB, and 64 MiB in every ordered pair. This is a correctness diagnostic, not a
// bandwidth benchmark.
//
// Build: nvcc -O2 -o peer-copy-direction-probe tools/peer-copy-direction-probe.cu -lcuda
// Run:   ./peer-copy-direction-probe [device-a device-b]
// Exit:  0 = all programs preserve bytes; 2 = one or more byte mismatches; 3 = CUDA error;
//        4 = fewer than two devices; 5 = no peer-capable pair.

#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

static int driver_error(CUresult rc, const char *call, int line) {
    const char *name = "unknown";
    const char *message = "unknown";
    cuGetErrorName(rc, &name);
    cuGetErrorString(rc, &message);
    std::fprintf(stderr, "CUDA driver error line %d: %s: %s (%s)\n", line, call, name,
                 message);
    return 3;
}

#define DRV(call)                         \
    do {                                  \
        CUresult _rc = (call);            \
        if (_rc != CUDA_SUCCESS)          \
            return driver_error(_rc, #call, __LINE__); \
    } while (0)

enum class CopyProgram { PushPeer, PullPeer, PushDtod, PullDtod };

static const char *program_name(CopyProgram program) {
    switch (program) {
        case CopyProgram::PushPeer:
            return "push-peer";
        case CopyProgram::PullPeer:
            return "pull-peer";
        case CopyProgram::PushDtod:
            return "push-dtod";
        case CopyProgram::PullDtod:
            return "pull-dtod";
    }
    return "unknown";
}

static unsigned char pattern(size_t i, int src, int dst, size_t bytes) {
    unsigned long long x = 0xd1b54a32d192ed03ULL ^ (unsigned long long)bytes;
    x ^= (unsigned long long)(src + 1) << 17;
    x ^= (unsigned long long)(dst + 1) << 41;
    x += (unsigned long long)i * 0x9e3779b97f4a7c15ULL;
    x ^= x >> 29;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 32;
    return (unsigned char)x;
}

struct DeviceState {
    CUdevice device = 0;
    CUcontext context = nullptr;
    CUstream stream = nullptr;
};

static int bind(const DeviceState &state) {
    CUresult rc = cuCtxSetCurrent(state.context);
    return rc == CUDA_SUCCESS ? 0 : driver_error(rc, "cuCtxSetCurrent", __LINE__);
}

static int enable_peer(const DeviceState &accessor, const DeviceState &owner) {
    int rc = bind(accessor);
    if (rc != 0) return rc;
    CUresult peer = cuCtxEnablePeerAccess(owner.context, 0);
    if (peer != CUDA_SUCCESS && peer != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED)
        return driver_error(peer, "cuCtxEnablePeerAccess", __LINE__);
    return 0;
}

static int run_copy(const DeviceState &src, const DeviceState &dst, int src_ordinal,
                    int dst_ordinal, CopyProgram program, size_t bytes, size_t *mismatches) {
    std::vector<unsigned char> expected(bytes);
    std::vector<unsigned char> poison(bytes);
    std::vector<unsigned char> actual(bytes);
    for (size_t i = 0; i < bytes; ++i) {
        expected[i] = pattern(i, src_ordinal, dst_ordinal, bytes);
        poison[i] = (unsigned char)~expected[i];
    }

    CUdeviceptr src_ptr = 0;
    CUdeviceptr dst_ptr = 0;
    int rc = bind(src);
    if (rc != 0) return rc;
    DRV(cuMemAlloc_v2(&src_ptr, bytes));
    DRV(cuMemcpyHtoD_v2(src_ptr, expected.data(), bytes));

    rc = bind(dst);
    if (rc != 0) return rc;
    DRV(cuMemAlloc_v2(&dst_ptr, bytes));
    DRV(cuMemcpyHtoD_v2(dst_ptr, poison.data(), bytes));

    CUevent publication = nullptr;
    rc = bind(src);
    if (rc != 0) return rc;
    DRV(cuEventCreate(&publication, CU_EVENT_DISABLE_TIMING));

    if (program == CopyProgram::PushPeer || program == CopyProgram::PushDtod) {
        rc = bind(src);
        if (rc != 0) return rc;
        if (program == CopyProgram::PushPeer) {
            DRV(cuMemcpyPeerAsync(dst_ptr, dst.context, src_ptr, src.context, bytes,
                                  src.stream));
        } else {
            DRV(cuMemcpyDtoDAsync_v2(dst_ptr, src_ptr, bytes, src.stream));
        }
        DRV(cuEventRecord(publication, src.stream));
        rc = bind(dst);
        if (rc != 0) return rc;
        DRV(cuStreamWaitEvent(dst.stream, publication, CU_EVENT_WAIT_DEFAULT));
        DRV(cuStreamSynchronize(dst.stream));
    } else {
        rc = bind(src);
        if (rc != 0) return rc;
        DRV(cuEventRecord(publication, src.stream));
        rc = bind(dst);
        if (rc != 0) return rc;
        DRV(cuStreamWaitEvent(dst.stream, publication, CU_EVENT_WAIT_DEFAULT));
        if (program == CopyProgram::PullPeer) {
            DRV(cuMemcpyPeerAsync(dst_ptr, dst.context, src_ptr, src.context, bytes,
                                  dst.stream));
        } else {
            DRV(cuMemcpyDtoDAsync_v2(dst_ptr, src_ptr, bytes, dst.stream));
        }
        DRV(cuStreamSynchronize(dst.stream));
    }

    rc = bind(dst);
    if (rc != 0) return rc;
    DRV(cuMemcpyDtoH_v2(actual.data(), dst_ptr, bytes));
    *mismatches = 0;
    for (size_t i = 0; i < bytes; ++i)
        if (actual[i] != expected[i]) ++*mismatches;

    rc = bind(src);
    if (rc != 0) return rc;
    DRV(cuEventDestroy_v2(publication));
    DRV(cuMemFree_v2(src_ptr));
    rc = bind(dst);
    if (rc != 0) return rc;
    DRV(cuMemFree_v2(dst_ptr));
    return 0;
}

int main(int argc, char **argv) {
    DRV(cuInit(0));
    int device_count = 0;
    DRV(cuDeviceGetCount(&device_count));
    std::printf("peer-copy-direction-probe: %d device(s)\n", device_count);
    if (device_count < 2) {
        std::fprintf(stderr, "fewer than two devices; no fabric to diagnose\n");
        return 4;
    }

    std::vector<DeviceState> devices((size_t)device_count);
    for (int ordinal = 0; ordinal < device_count; ++ordinal) {
        DRV(cuDeviceGet(&devices[(size_t)ordinal].device, ordinal));
        DRV(cuDevicePrimaryCtxRetain(&devices[(size_t)ordinal].context,
                                     devices[(size_t)ordinal].device));
        int rc = bind(devices[(size_t)ordinal]);
        if (rc != 0) return rc;
        DRV(cuStreamCreate(&devices[(size_t)ordinal].stream, CU_STREAM_NON_BLOCKING));
        char name[256] = {};
        DRV(cuDeviceGetName(name, (int)sizeof(name), devices[(size_t)ordinal].device));
        std::printf("  dev%d %s\n", ordinal, name);
    }

    std::vector<std::pair<int, int>> pairs;
    if (argc == 3) {
        int a = std::atoi(argv[1]);
        int b = std::atoi(argv[2]);
        if (a < 0 || b < 0 || a >= device_count || b >= device_count || a == b) {
            std::fprintf(stderr, "invalid device pair %d,%d\n", a, b);
            return 3;
        }
        pairs.push_back({a, b});
        pairs.push_back({b, a});
    } else if (argc == 1) {
        for (int src = 0; src < device_count; ++src)
            for (int dst = 0; dst < device_count; ++dst)
                if (src != dst) pairs.push_back({src, dst});
    } else {
        std::fprintf(stderr, "usage: %s [device-a device-b]\n", argv[0]);
        return 3;
    }

    int peer_pairs = 0;
    for (const auto &pair : pairs) {
        int can = 0;
        DRV(cuDeviceCanAccessPeer(&can, devices[(size_t)pair.second].device,
                                  devices[(size_t)pair.first].device));
        if (can) ++peer_pairs;
    }
    if (peer_pairs == 0) {
        std::fprintf(stderr, "no peer-capable ordered pair\n");
        return 5;
    }

    for (const auto &pair : pairs) {
        int can = 0;
        DRV(cuDeviceCanAccessPeer(&can, devices[(size_t)pair.second].device,
                                  devices[(size_t)pair.first].device));
        if (!can) continue;
        int rc = enable_peer(devices[(size_t)pair.second], devices[(size_t)pair.first]);
        if (rc != 0) return rc;
    }

    static const size_t sizes[] = {16u * 1024, 1024u * 1024, 64u * 1024 * 1024};
    static const CopyProgram programs[] = {
        CopyProgram::PushPeer,
        CopyProgram::PullPeer,
        CopyProgram::PushDtod,
        CopyProgram::PullDtod,
    };
    int failures = 0;
    for (const auto &pair : pairs) {
        int can = 0;
        DRV(cuDeviceCanAccessPeer(&can, devices[(size_t)pair.second].device,
                                  devices[(size_t)pair.first].device));
        if (!can) {
            std::printf("pair dev%d->dev%d: no peer path, skipped\n", pair.first, pair.second);
            continue;
        }
        for (CopyProgram program : programs) {
            for (size_t bytes : sizes) {
                size_t mismatches = 0;
                int rc = run_copy(devices[(size_t)pair.first], devices[(size_t)pair.second],
                                  pair.first, pair.second, program, bytes, &mismatches);
                if (rc != 0) return rc;
                std::printf("pair dev%d->dev%d %-10s %10zu B mismatches=%zu %s\n", pair.first,
                            pair.second, program_name(program), bytes, mismatches,
                            mismatches == 0 ? "PASS" : "FAIL");
                if (mismatches != 0) ++failures;
            }
        }
    }

    for (int ordinal = 0; ordinal < device_count; ++ordinal) {
        int rc = bind(devices[(size_t)ordinal]);
        if (rc != 0) return rc;
        DRV(cuStreamDestroy_v2(devices[(size_t)ordinal].stream));
        DRV(cuDevicePrimaryCtxRelease_v2(devices[(size_t)ordinal].device));
    }
    DRV(cuCtxSetCurrent(nullptr));

    if (failures != 0) {
        std::printf("peer-copy-direction-probe FAIL: %d mismatching cell(s)\n", failures);
        return 2;
    }
    std::printf("peer-copy-direction-probe PASS\n");
    return 0;
}

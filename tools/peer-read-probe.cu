// tools/peer-read-probe.cu — FLEET TOOL (promoted from lane/glm5-tp-transport in
// lane/glm5-extract2; run by tools/box-health.sh section 8, and standalone for a fabric
// bring-up). Nothing in it is memra-specific: it is a KERNEL peer dereference, which is the
// only check that catches the driver staging SM-issued peer access through system memory
// while nvidia-smi topo -p2p and cudaMemcpy both look healthy.
//
// a simpleP2P-class byte-validating KERNEL peer read.
//
// WHY THIS EXISTS. Our TP transport (`MEMRA_GLM5_TP_TRANSPORT=peer-pull`) moves bytes with the
// COPY ENGINE (`cuMemcpyDtoDAsync`), which the driver's SysMem-staging default does not touch.
// This probe measures the OTHER path — a peer pointer dereferenced from inside a kernel —
// because that is what a future fused pull collective would use, and because it is the only
// check that detects two documented failure modes that every cheaper check reports as healthy.
//
// From darklanes research/pro6000-multicard-research-20260901/RESEARCH.md:
//
//   §2.3b  "On direct-attach topologies (NODE ..., no PCIe switch), the nvidia driver defaults
//          to SysMem staging for GPU-to-GPU memory accesses from CUDA kernels. This makes the
//          PCIe oneshot allreduce ~15x slower than NCCL" -- and "`nvidia-smi topo -p2p r`
//          returns OK while `cudaMemcpy` looks healthy, so neither detects it. Only a kernel
//          peer read (simpleP2P) does."
//   §2.4   rtx6kpro #21: "peer access reports Yes, `cudaMemcpyPeer` runs at 26 GB/s, but
//          kernel peer-reads return zeros."
//   §2.10  a rig measured "6.7 GB/s uni with `topo -p2p` OK while `cudaMemcpyPeer` left the
//          destination unchanged". "Always run a byte-validating test."
//   §2.1   `NativeAtomicSupported=0` on every SM120 pair -- so this probe uses plain loads
//          only. No atomics, no peer flag polling, no CAS. `topo -p2p a` returning NS is
//          EXPECTED, not a fault.
//
// WHAT IT DOES. For every ordered device pair with peer capability: the OWNER fills a buffer
// with a checkable pattern; the ACCESSOR launches a kernel that dereferences the OWNER's
// pointer directly (a peer LOAD, not a copy) and reduces a mismatch count against the pattern
// it recomputes locally. Zero mismatches at every size, both directions, or the probe fails.
//
// The size ladder spans 4 B (one word -- the latency-shaped rung) to 64 MiB, matching the
// engine's own `NATIVE_P2P_PROBE_WORDS` upper rungs, because §2.4 records that LL and Simple
// protocol paths "can fail differently" by size.
//
// It ALSO reports an effective peer-read bandwidth per pair at the top rung. Read that number
// against §2.2's budget: ~52-56 GB/s uni means the SM path is direct; a figure ~15x below that
// means the driver is SysMem-staging kernel peer access and the 3-key ForceP2P form is needed
// (`ForceP2P=0x11;GrdmaPciTopoCheckOverride=1;EnableResizableBar=1` -- NEVER the 5-key form,
// which broke real peer copies with "invalid device ordinal" on driver 580.167.08 while
// `cudaDeviceCanAccessPeer` still read 1). The number is a DIAGNOSTIC THRESHOLD, not a
// benchmark: it is measured with one kernel launch per timed iteration and therefore carries
// the ~5 us launch overhead §2.10 warns about, which is why only the 64 MiB rung is reported.
//
// Build:  nvcc -O2 -arch=sm_120 -o peer-read-probe peer-read-probe.cu
// Run:    ./peer-read-probe            (all peer-capable pairs)
//         ./peer-read-probe 0 1        (one ordered pair)
// Exit:   0 = every pair passed every rung. 2 = a byte mismatch (HARD FAIL).
//         3 = a CUDA error. 4 = fewer than two devices. 5 = no peer-capable pair.
// Prints "Test passed" on success, matching the simpleP2P convention §2.10 asks for.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CK(call)                                                                       \
    do {                                                                               \
        cudaError_t _e = (call);                                                        \
        if (_e != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s:%d: %s (%s)\n", __FILE__, __LINE__,           \
                    cudaGetErrorString(_e), #call);                                     \
            return 3;                                                                   \
        }                                                                               \
    } while (0)

// The pattern is recomputed in-kernel rather than read from a second buffer, so a mismatch
// cannot be explained by a second broken transfer. Same generator as the engine's ladder.
__device__ __host__ inline unsigned int pat(unsigned int i, unsigned int tag) {
    return (i * 0x9e3779b9u) + tag;
}

// Peer LOAD kernel: `remote` points into ANOTHER device's memory. `__ldg` is a plain global
// load through the read-only path -- no atomics anywhere (§2.1).
__global__ void peer_read_check(const unsigned int *__restrict__ remote, unsigned int n,
                                unsigned int tag, unsigned long long *__restrict__ bad) {
    unsigned long long local = 0;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        if (__ldg(&remote[i]) != pat(i, tag)) {
            local += 1;
        }
    }
    // Block-level reduction into one global add per block. The add is on the ACCESSOR's OWN
    // memory, never on the peer's -- peer atomics are unsupported on this fabric.
    __shared__ unsigned long long s;
    if (threadIdx.x == 0) s = 0;
    __syncthreads();
    atomicAdd(&s, local);
    __syncthreads();
    if (threadIdx.x == 0 && s != 0) atomicAdd(bad, s);
}

static const size_t LADDER_BYTES[] = {
    4,                  // one word: the latency-shaped rung
    16u * 1024,         // 16 KiB: our real decode hop size
    64u * 1024,         // 64 KiB: our largest real hop (MLA full-width gather)
    1024u * 1024,       // 1 MiB
    64u * 1024 * 1024,  // 64 MiB: the bandwidth-shaped rung
};
static const int LADDER_N = sizeof(LADDER_BYTES) / sizeof(LADDER_BYTES[0]);

static int probe_pair(int owner, int accessor, int *failures) {
    int can = 0;
    CK(cudaDeviceCanAccessPeer(&can, accessor, owner));
    if (!can) {
        printf("pair %d->%d : NO PEER PATH (cudaDeviceCanAccessPeer=0) -- skipped\n", accessor,
               owner);
        return 0;
    }
    CK(cudaSetDevice(accessor));
    cudaError_t pe = cudaDeviceEnablePeerAccess(owner, 0);
    if (pe != cudaSuccess && pe != cudaErrorPeerAccessAlreadyEnabled) {
        fprintf(stderr, "cudaDeviceEnablePeerAccess(%d -> %d) failed: %s\n", accessor, owner,
                cudaGetErrorString(pe));
        return 3;
    }

    const unsigned int tag = ((unsigned int)owner << 16) | (unsigned int)accessor;
    for (int rung = 0; rung < LADDER_N; ++rung) {
        const size_t bytes = LADDER_BYTES[rung];
        const unsigned int n = (unsigned int)(bytes / sizeof(unsigned int));
        if (n == 0) continue;

        // OWNER side: allocate and fill.
        CK(cudaSetDevice(owner));
        unsigned int *remote = nullptr;
        CK(cudaMalloc(&remote, n * sizeof(unsigned int)));
        std::vector<unsigned int> host(n);
        for (unsigned int i = 0; i < n; ++i) host[i] = pat(i, tag);
        CK(cudaMemcpy(remote, host.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CK(cudaDeviceSynchronize());

        // ACCESSOR side: dereference the owner's pointer from a kernel.
        CK(cudaSetDevice(accessor));
        unsigned long long *bad = nullptr;
        CK(cudaMalloc(&bad, sizeof(unsigned long long)));
        CK(cudaMemset(bad, 0, sizeof(unsigned long long)));
        const int threads = 256;
        int blocks = (int)((n + threads - 1) / threads);
        if (blocks > 1024) blocks = 1024;
        if (blocks < 1) blocks = 1;

        peer_read_check<<<blocks, threads>>>(remote, n, tag, bad);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());

        unsigned long long bad_host = 0;
        CK(cudaMemcpy(&bad_host, bad, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

        double gbs = -1.0;
        if (bytes >= 64u * 1024 * 1024) {
            // Bandwidth-shaped rung only. Warm once, then time a few iterations. This carries
            // per-launch overhead by construction (§2.10's trap) -- it is a THRESHOLD reading
            // for "is the SM path direct or SysMem-staged", never a benchmark number.
            cudaEvent_t a, b;
            CK(cudaEventCreate(&a));
            CK(cudaEventCreate(&b));
            peer_read_check<<<blocks, threads>>>(remote, n, tag, bad);
            CK(cudaDeviceSynchronize());
            const int iters = 5;
            CK(cudaEventRecord(a));
            for (int it = 0; it < iters; ++it) {
                peer_read_check<<<blocks, threads>>>(remote, n, tag, bad);
            }
            CK(cudaEventRecord(b));
            CK(cudaEventSynchronize(b));
            float ms = 0.0f;
            CK(cudaEventElapsedTime(&ms, a, b));
            if (ms > 0.0f) gbs = ((double)bytes * iters) / (ms * 1.0e-3) / 1.0e9;
            CK(cudaEventDestroy(a));
            CK(cudaEventDestroy(b));
        }

        if (bad_host != 0) {
            printf("pair %d->%d %10zu B : FAIL  %llu/%u words differ  <-- KERNEL PEER READ IS "
                   "BROKEN (see RESEARCH.md 2.3b/2.4)\n",
                   accessor, owner, bytes, bad_host, n);
            *failures += 1;
        } else if (gbs > 0.0) {
            printf("pair %d->%d %10zu B : ok    mismatches=0  peer-read ~%.1f GB/s%s\n", accessor,
                   owner, bytes, gbs,
                   gbs < 20.0 ? "  <-- WELL BELOW the 52-56 GB/s budget: suspect SysMem staging, "
                                "consider the 3-KEY ForceP2P form"
                              : "");
        } else {
            printf("pair %d->%d %10zu B : ok    mismatches=0\n", accessor, owner, bytes);
        }

        CK(cudaFree(bad));
        CK(cudaSetDevice(owner));
        CK(cudaFree(remote));
    }
    return 0;
}

int main(int argc, char **argv) {
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    printf("peer-read-probe: %d device(s)\n", ndev);
    for (int d = 0; d < ndev; ++d) {
        cudaDeviceProp p;
        CK(cudaGetDeviceProperties(&p, d));
        printf("  dev%d %s sm_%d%d pciBus=%02x:%02x.%d\n", d, p.name, p.major, p.minor,
               p.pciBusID, p.pciDeviceID, p.pciDomainID);
    }
    if (ndev < 2) {
        fprintf(stderr,
                "peer-read-probe: fewer than two devices -- nothing to probe. This is the "
                "expected result on the single-card rig (LAW:rig-exactness-only); the probe is "
                "a BOX-WINDOW item.\n");
        return 4;
    }

    std::vector<std::pair<int, int> > pairs;  // (owner, accessor)
    if (argc >= 3) {
        pairs.push_back(std::make_pair(atoi(argv[1]), atoi(argv[2])));
        pairs.push_back(std::make_pair(atoi(argv[2]), atoi(argv[1])));
    } else {
        for (int a = 0; a < ndev; ++a)
            for (int b = 0; b < ndev; ++b)
                if (a != b) pairs.push_back(std::make_pair(a, b));
    }

    int failures = 0;
    int peer_pairs = 0;
    for (size_t i = 0; i < pairs.size(); ++i) {
        int can = 0;
        CK(cudaDeviceCanAccessPeer(&can, pairs[i].second, pairs[i].first));
        if (can) peer_pairs += 1;
        int rc = probe_pair(pairs[i].first, pairs[i].second, &failures);
        if (rc != 0) return rc;
    }

    if (peer_pairs == 0) {
        fprintf(stderr,
                "peer-read-probe: NO peer-capable pair on this host. Record the topology: this "
                "card class is not uniformly peer-connected, and host classes exist that present "
                "PEER ISLANDS OF TWO with every cross-island cell reading N/A (RESEARCH.md 3.2) "
                "-- a TP group must be placed INSIDE an island.\n");
        return 5;
    }
    if (failures != 0) {
        fprintf(stderr, "peer-read-probe: %d rung(s) FAILED byte validation.\n", failures);
        return 2;
    }
    printf("Test passed\n");
    return 0;
}

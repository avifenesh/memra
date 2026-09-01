// Vast PP-2 P2P receipt: production direction (dev0 -> dev1) at Step3.7 boundary sizes.
// Build: nvcc -O3 -std=c++17 -arch=sm_120 p2p_probe.cu -o p2p_probe
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_OK(call) do {                                                     \
    cudaError_t err_ = (call);                                                  \
    if (err_ != cudaSuccess) {                                                  \
        std::fprintf(stderr, "CUDA_ERR %s at %s:%d: %s\n", #call, __FILE__,   \
                     __LINE__, cudaGetErrorString(err_));                       \
        std::exit(1);                                                           \
    }                                                                           \
} while (0)

static void enable_peer(int src, int dst) {
    CUDA_OK(cudaSetDevice(src));
    cudaError_t err = cudaDeviceEnablePeerAccess(dst, 0);
    if (err == cudaErrorPeerAccessAlreadyEnabled) {
        CUDA_OK(cudaGetLastError());
    } else if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA_ERR cudaDeviceEnablePeerAccess(%d,%d): %s\n",
                     src, dst, cudaGetErrorString(err));
        std::exit(1);
    }
}

int main() {
    int devices = 0;
    CUDA_OK(cudaGetDeviceCount(&devices));
    std::printf("devices=%d\n", devices);
    if (devices < 2) {
        std::fprintf(stderr, "need two CUDA devices\n");
        return 1;
    }

    for (int dev = 0; dev < 2; ++dev) {
        cudaDeviceProp prop{};
        CUDA_OK(cudaGetDeviceProperties(&prop, dev));
        std::printf("dev%d name=\"%s\" cc=%d.%d pci=%04x:%02x:%02x\n",
                    dev, prop.name, prop.major, prop.minor,
                    prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);
    }

    int can01 = 0;
    int can10 = 0;
    CUDA_OK(cudaDeviceCanAccessPeer(&can01, 0, 1));
    CUDA_OK(cudaDeviceCanAccessPeer(&can10, 1, 0));
    std::printf("canAccessPeer 0->1=%d 1->0=%d\n", can01, can10);
    if (!can01 || !can10) {
        std::fprintf(stderr, "peer access unavailable\n");
        return 1;
    }
    enable_peer(0, 1);
    enable_peer(1, 0);
    std::printf("peer_enabled=1\n");

    constexpr size_t max_bytes = 16ull << 20;
    void* src = nullptr;
    void* dst = nullptr;
    std::vector<unsigned char> pattern(max_bytes);
    for (size_t i = 0; i < pattern.size(); ++i) {
        pattern[i] = static_cast<unsigned char>((i * 131 + 17) & 0xff);
    }
    CUDA_OK(cudaSetDevice(0));
    CUDA_OK(cudaMalloc(&src, max_bytes));
    CUDA_OK(cudaMemcpy(src, pattern.data(), max_bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaDeviceSynchronize());
    cudaStream_t stream{};
    cudaEvent_t begin{};
    cudaEvent_t end{};
    CUDA_OK(cudaStreamCreate(&stream));
    CUDA_OK(cudaEventCreate(&begin));
    CUDA_OK(cudaEventCreate(&end));
    CUDA_OK(cudaSetDevice(1));
    CUDA_OK(cudaMalloc(&dst, max_bytes));
    CUDA_OK(cudaMemset(dst, 0, max_bytes));
    CUDA_OK(cudaDeviceSynchronize());

    // Step3.7 n_embd=4096: decode T=1 is 16 KiB. Prime rows below correspond to
    // 128/256/512/1024-token microchunks of the same [T,n_embd] f32 boundary.
    const size_t sizes[] = {
        16ull << 10,
        2ull << 20,
        4ull << 20,
        8ull << 20,
        16ull << 20,
    };
    std::printf("bytes,tokens_at_nembd4096,batch_us,serialized_us,GB_s\n");
    for (size_t bytes : sizes) {
        const int batch_iters = bytes <= (4ull << 20) ? 1000 : 300;
        const int serial_iters = bytes == (16ull << 10) ? 1000 : 100;

        CUDA_OK(cudaSetDevice(0));
        for (int i = 0; i < 32; ++i) {
            CUDA_OK(cudaMemcpyPeerAsync(dst, 1, src, 0, bytes, stream));
        }
        CUDA_OK(cudaStreamSynchronize(stream));

        CUDA_OK(cudaEventRecord(begin, stream));
        for (int i = 0; i < batch_iters; ++i) {
            CUDA_OK(cudaMemcpyPeerAsync(dst, 1, src, 0, bytes, stream));
        }
        CUDA_OK(cudaEventRecord(end, stream));
        CUDA_OK(cudaEventSynchronize(end));
        float elapsed_ms = 0.0f;
        CUDA_OK(cudaEventElapsedTime(&elapsed_ms, begin, end));
        const double batch_us = elapsed_ms * 1000.0 / batch_iters;
        const double gb_s = static_cast<double>(bytes) * batch_iters /
                            (elapsed_ms / 1000.0) / 1.0e9;

        const auto wall_begin = std::chrono::steady_clock::now();
        for (int i = 0; i < serial_iters; ++i) {
            CUDA_OK(cudaMemcpyPeerAsync(dst, 1, src, 0, bytes, stream));
            CUDA_OK(cudaStreamSynchronize(stream));
        }
        const auto wall_end = std::chrono::steady_clock::now();
        const double serial_us =
            std::chrono::duration<double, std::micro>(wall_end - wall_begin).count() /
            serial_iters;
        std::printf("%zu,%zu,%.3f,%.3f,%.3f\n",
                    bytes, bytes / (4096 * sizeof(float)), batch_us, serial_us, gb_s);
    }

    std::vector<unsigned char> check(16ull << 10);
    CUDA_OK(cudaSetDevice(1));
    CUDA_OK(cudaMemcpy(check.data(), dst, check.size(), cudaMemcpyDeviceToHost));
    size_t wrong = 0;
    for (size_t i = 0; i < check.size(); ++i) {
        wrong += check[i] != pattern[i];
    }
    std::printf("correctness bytes=%zu wrong=%zu verdict=%s\n",
                check.size(), wrong, wrong == 0 ? "PASS" : "FAIL");
    return wrong == 0 ? 0 : 1;
}

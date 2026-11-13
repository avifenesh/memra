__global__ void __launch_bounds__(128) v41_gemv_ffma_narrow_kernel(const v41_bf* __restrict__ A, long lda,
                                                                   const v41_bf* __restrict__ B, long ldb,
                                                                   float* __restrict__ C, long ldc, int M, int N,
                                                                   int K) {
    extern __shared__ __align__(16) uint8_t v41_ffma_sm[];
    int pitch = K + 8;  // elements: rows 16 bytes apart in the bank map
    v41_bf* sB = (v41_bf*)v41_ffma_sm;
    v41_bf* sA = sB + V41_SMF_NC * pitch;
    int tid = threadIdx.x, n0 = blockIdx.x * V41_SMF_NC;
    int chunks = K / 8;
    for (int q = tid; q < V41_SMF_NC * chunks; q += 128) {
        int r = q / chunks, c = q % chunks;
        bool ok = n0 + r < N;
        v41_cp16(sB + r * pitch + c * 8, B + (long)(ok ? n0 + r : 0) * ldb + c * 8, ok);
    }
    for (int q = tid; q < M * chunks; q += 128) {
        int r = q / chunks, c = q % chunks;
        v41_cp16(sA + r * pitch + c * 8, A + (long)r * lda + c * 8, true);
    }
    v41_cp_commit();
    v41_cp_wait<0>();
    __syncthreads();
    if (tid >= M * V41_SMF_NC) return;
    int r = tid / V41_SMF_NC, cc = tid % V41_SMF_NC;
    if (n0 + cc >= N) return;
    const v41_bf* ar = sA + r * pitch;
    const v41_bf* br = sB + cc * pitch;
    float acc = 0.f;
    for (int k = 0; k < K; k += 8) {
        uint4 av = *(const uint4*)(ar + k), bv = *(const uint4*)(br + k);
        const v41_bf* ae = (const v41_bf*)&av;
        const v41_bf* be = (const v41_bf*)&bv;
#pragma unroll
        for (int u = 0; u < 8; ++u) acc = __fmaf_rn(v41_bf2f(ae[u]), v41_bf2f(be[u]), acc);
    }
    C[(long)r * ldc + n0 + cc] = acc;
}

// The narrow form with K in parts of Kp elements, for a call whose rows and columns of all of K do
// not fit a block (5..8 rows of K = 5120): a part is landed, every chain runs through it, the next
// part is landed; a chain is still its K fmas in K order.
__global__ void __launch_bounds__(128) v41_gemv_ffma_narrow_parts_kernel(const v41_bf* __restrict__ A, long lda,
                                                                         const v41_bf* __restrict__ B, long ldb,
                                                                         float* __restrict__ C, long ldc, int M, int N,
                                                                         int K, int Kp) {
    extern __shared__ __align__(16) uint8_t v41_ffma_sm[];
    int pitch = Kp + 8;  // elements: rows 16 bytes apart in the bank map
    v41_bf* sB = (v41_bf*)v41_ffma_sm;
    v41_bf* sA = sB + V41_SMF_NC * pitch;
    int tid = threadIdx.x, n0 = blockIdx.x * V41_SMF_NC;
    int r = tid / V41_SMF_NC, cc = tid % V41_SMF_NC;
    bool chain = tid < M * V41_SMF_NC && n0 + cc < N;
    const v41_bf* ar = sA + r * pitch;
    const v41_bf* br = sB + cc * pitch;
    float acc = 0.f;
    for (int k0 = 0; k0 < K; k0 += Kp) {
        int kn = min(Kp, K - k0), chunks = kn / 8;
        for (int q = tid; q < V41_SMF_NC * chunks; q += 128) {
            int rr = q / chunks, c = q % chunks;
            bool ok = n0 + rr < N;
            v41_cp16(sB + rr * pitch + c * 8, B + (long)(ok ? n0 + rr : 0) * ldb + k0 + c * 8, ok);
        }
        for (int q = tid; q < M * chunks; q += 128) {
            int rr = q / chunks, c = q % chunks;
            v41_cp16(sA + rr * pitch + c * 8, A + (long)rr * lda + k0 + c * 8, true);
        }
        v41_cp_commit();
        v41_cp_wait<0>();
        __syncthreads();
        if (chain)
            for (int k = 0; k < kn; k += 8) {
                uint4 av = *(const uint4*)(ar + k), bv = *(const uint4*)(br + k);
                const v41_bf* ae = (const v41_bf*)&av;
                const v41_bf* be = (const v41_bf*)&bv;
#pragma unroll
                for (int u = 0; u < 8; ++u) acc = __fmaf_rn(v41_bf2f(ae[u]), v41_bf2f(be[u]), acc);
            }
        // the next part overwrites what the chains read
        if (k0 + Kp < K) __syncthreads();
    }
    if (chain) C[(long)r * ldc + n0 + cc] = acc;
}

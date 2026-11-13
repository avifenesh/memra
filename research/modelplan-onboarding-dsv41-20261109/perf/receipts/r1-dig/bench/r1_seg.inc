__global__ void __launch_bounds__(256) v41_gemv_ffma_seg_kernel(const v41_bf* __restrict__ A, long lda,
                                                                const v41_bf* __restrict__ B, long ldb,
                                                                float* __restrict__ C, long ldc, int M, int N, int K,
                                                                int Kp) {
    extern __shared__ __align__(16) uint8_t v41_ffma_sm[];
    int sl = K / V41_KSEG;  // elements a segment
    int spp = Kp / sl;      // segments a part
    int sp = sl + 8;        // elements a staged segment
    int rp = spp * sp;      // elements a staged row
    v41_bf* sB = (v41_bf*)v41_ffma_sm;
    v41_bf* sA = sB + V41_SMF_NC * rp;
    float* red = (float*)(sA + (long)M * rp);  // [M][4 columns][8 segments]
    int tid = threadIdx.x, sg = tid & 7, cc = (tid >> 3) & 3, r0 = tid >> 5;
    int n0 = blockIdx.x * V41_SMF_NC;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    int chunks = Kp / 8, cps = sl / 8;  // 16-byte chunks a staged row, and a segment
    for (int k0 = 0; k0 < K; k0 += Kp) {
        for (int q = tid; q < V41_SMF_NC * chunks; q += 256) {
            int rr = q / chunks, c = q % chunks;
            bool ok = n0 + rr < N;
            v41_cp16(sB + rr * rp + (c / cps) * sp + (c % cps) * 8, B + (long)(ok ? n0 + rr : 0) * ldb + k0 + c * 8, ok);
        }
        for (int q = tid; q < M * chunks; q += 256) {
            int rr = q / chunks, c = q % chunks;
            v41_cp16(sA + (long)rr * rp + (c / cps) * sp + (c % cps) * 8, A + (long)rr * lda + k0 + c * 8, true);
        }
        v41_cp_commit();
        v41_cp_wait<0>();
        __syncthreads();
        int ls = sg - k0 / sl;  // this thread's segment within the part
        if (ls >= 0 && ls < spp && n0 + cc < N) {
            const v41_bf* br = sB + cc * rp + ls * sp;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                int r = r0 + 8 * i;
                if (r < M) {
                    const v41_bf* ar = sA + (long)r * rp + ls * sp;
                    float a_ = acc[i];
                    for (int k = 0; k < sl; k += 8) {
                        uint4 av = *(const uint4*)(ar + k), bv = *(const uint4*)(br + k);
                        const v41_bf* ae = (const v41_bf*)&av;
                        const v41_bf* be = (const v41_bf*)&bv;
#pragma unroll
                        for (int u = 0; u < 8; ++u) a_ = __fmaf_rn(v41_bf2f(ae[u]), v41_bf2f(be[u]), a_);
                    }
                    acc[i] = a_;
                }
            }
        }
        // the next part overwrites what the chains read
        if (k0 + Kp < K) __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int r = r0 + 8 * i;
        if (r < M) red[(r * V41_SMF_NC + cc) * V41_KSEG + sg] = acc[i];
    }
    __syncthreads();
    if (sg == 0 && n0 + cc < N) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int r = r0 + 8 * i;
            if (r < M) {
                const float* q = red + (r * V41_SMF_NC + cc) * V41_KSEG;
                C[(long)r * ldc + n0 + cc] = v41_ktree(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7]);
            }
        }
    }
}

// shared bytes of v41_gemv_ffma_seg_kernel for `m` rows and parts of `kp` (segments of `k / V41_KSEG`)
static long v41_ffma_seg_smem(int m, int k, int kp) {
    int sl = k / V41_KSEG;
    return (long)(V41_SMF_NC + m) * (kp / sl) * (sl + 8) * 2 + (long)m * V41_SMF_NC * V41_KSEG * 4;
}

#!/usr/bin/env python3
"""Apply (or revert with --revert) the small-M row limit experiment on the wt-spec tree: the block-scaled and bf16
small-M GEMM bodies and the grouped expert tiles take up to 8 rows (their MMA fragments hold rows 0..7), the fp32 FMA
small-M chain keeps 4 rows a launch and a call of 5..8 rows runs as two launches. Idempotent."""
import sys

root = "$WT/crates/memra-engine/"
revert = "--revert" in sys.argv
edits = {
    "cu/dsv41_gpu.cu": [
        ("#define V41_SM_MAX 4\n", "#define V41_SM_MAX 8\n// the fp32 FMA small-M chain: one thread per (row, column) of a 128-thread block\n#define V41_SMF_MAX 4\n"),
        ("    __shared__ float sA[V41_SM_MAX][V41_SMF_KC];\n", "    __shared__ float sA[V41_SMF_MAX][V41_SMF_KC];\n"),
        ("            if (ar < V41_SM_MAX)\n", "            if (ar < V41_SMF_MAX)\n"),
        ("""    if (m <= V41_SM_MAX && lda % 8 == 0 && ldb % 8 == 0 && n <= 4096) {
        int smem = (V41_SMF_NC + m) * (k + 8) * 2;
        if (smem <= 99 * 1024) {
            if (smem > 48 * 1024)
                cudaFuncSetAttribute(v41_gemv_ffma_narrow_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
            v41_gemv_ffma_narrow_kernel<<<(n + V41_SMF_NC - 1) / V41_SMF_NC, 128, smem, (cudaStream_t)stream>>>(
                (const v41_bf*)a, lda, (const v41_bf*)b, ldb, c, ldc, m, n, k);
            V41_ERR();
        }
    }
    if (m <= V41_SM_MAX && lda % 8 == 0 && ldb % 8 == 0) {
        v41_gemv_ffma_kernel<<<(n + 31) / 32, 128, 0, (cudaStream_t)stream>>>((const v41_bf*)a, lda, (const v41_bf*)b,
                                                                              ldb, c, ldc, m, n, k);
        V41_ERR();
    }
""", """    if (m <= V41_SM_MAX && lda % 8 == 0 && ldb % 8 == 0) {
        // at most V41_SMF_MAX rows a launch: a call of 5..8 rows (a drafted block's verification) is two
        // launches of the same chains, each row's bytes those of its own launch
        for (int m0 = 0; m0 < m; m0 += V41_SMF_MAX) {
            int mm = m - m0 < V41_SMF_MAX ? m - m0 : V41_SMF_MAX;
            const v41_bf* ap = (const v41_bf*)a + (long)m0 * lda;
            float* cp = c + (long)m0 * ldc;
            int smem = (V41_SMF_NC + mm) * (k + 8) * 2;
            if (n <= 4096 && smem <= 99 * 1024) {
                if (smem > 48 * 1024)
                    cudaFuncSetAttribute(v41_gemv_ffma_narrow_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         smem);
                v41_gemv_ffma_narrow_kernel<<<(n + V41_SMF_NC - 1) / V41_SMF_NC, 128, smem, (cudaStream_t)stream>>>(
                    ap, lda, (const v41_bf*)b, ldb, cp, ldc, mm, n, k);
            } else {
                v41_gemv_ffma_kernel<<<(n + 31) / 32, 128, 0, (cudaStream_t)stream>>>(ap, lda, (const v41_bf*)b, ldb,
                                                                                      cp, ldc, mm, n, k);
            }
        }
        V41_ERR();
    }
"""),
    ],
    "src/dsv41_gpu.rs": [
        ("pub const SMALL_M: usize = 4;\n", "pub const SMALL_M: usize = 8;\n"),
    ],
}
for rel, pairs in edits.items():
    p = root + rel
    s = open(p).read()
    for old, new in pairs:
        a, b = (new, old) if revert else (old, new)
        if s.count(b) == 1 and s.count(a) == 0:
            continue  # already in the wanted state
        assert s.count(a) == 1, (rel, a[:60], s.count(a))
        s = s.replace(a, b)
    open(p, "w").write(s)
print("reverted" if revert else "applied")

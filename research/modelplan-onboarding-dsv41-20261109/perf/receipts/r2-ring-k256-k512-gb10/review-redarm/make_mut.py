#!/usr/bin/env python3
"""Review-only build inputs for memra #1312 (never committed anywhere).

Writes two files next to this script's parent:
  mut/dsv41_gpu.cu   the head's kernel source with the ring kernel templated on a mutation id,
                     picked at run time by REVIEW_MUT (0 = the head's kernel text, unchanged), and
                     a REVIEW_TRACE line naming the form each memra_v41_gemm_ffma call takes.
  mut/dsv41_gate.rs  the head's gate with four more ring shapes: K 768, 1024, 1280 (3, 4 and 5
                     stages) at (2, 30) and (8, 30, 5120).
Every replacement must match exactly once.
"""
import os
import sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cu = open(os.path.join(root, "dsv41_gpu.cu")).read()
rs = open(os.path.join(root, "dsv41_gate.rs")).read()


def rep(text, old, new, what):
    n = text.count(old)
    if n != 1:
        sys.exit(f"{what}: {n} matches, want 1")
    return text.replace(old, new)


# --- the kernel: template on the mutation id ---------------------------------------------------
cu = rep(
    cu,
    "__global__ void __launch_bounds__(32) v41_gemv_ffma_ring_kernel(",
    "template <int MUT>\n__global__ void __launch_bounds__(32) v41_gemv_ffma_ring_kernel(",
    "kernel head",
)
# MUT 4: the 8th row of A is never landed (a row-count bug)
cu = rep(
    cu,
    "        for (int row = 0; row < M; ++row)\n"
    "            v41_cp16(base + (V41_SMF_NC + row) * V41_RING_PITCH + lane * 8, A + (long)row * lda + k0, true);\n"
    "    };\n",
    "        for (int row = 0; row < (MUT == 4 && M > 7 ? 7 : M); ++row)\n"
    "            v41_cp16(base + (V41_SMF_NC + row) * V41_RING_PITCH + lane * 8, A + (long)row * lda + k0, true);\n"
    "    };\n",
    "A rows",
)
# MUT 1: the prologue commits a group only for a stage it lands (no empty groups)
# MUT 2: both guards dropped (the PR body's unrun mutation)
cu = rep(
    cu,
    "    for (int t = 0; t < V41_RING_NS - 1; ++t) {\n"
    "        if (t < nk) land(t);\n"
    "        v41_cp_commit();\n"
    "    }\n",
    "    for (int t = 0; t < V41_RING_NS - 1; ++t) {\n"
    "        if (MUT == 1) {\n"
    "            if (t < nk) {\n"
    "                land(t);\n"
    "                v41_cp_commit();\n"
    "            }\n"
    "        } else {\n"
    "            if (MUT == 2 || t < nk) land(t);\n"
    "            v41_cp_commit();\n"
    "        }\n"
    "    }\n",
    "prologue",
)
# MUT 3: the chain's column guard dropped (a partial last block writes past N)
cu = rep(
    cu,
    "    const bool chain = r < M && n0 + cc < N;\n    float acc = 0.f;\n    for (int t = 0; t < nk; ++t) {\n"
    "        v41_cp_wait<V41_RING_NS - 2>();\n",
    "    const bool chain = r < M && (MUT == 3 || n0 + cc < N);\n    float acc = 0.f;\n"
    "    for (int t = 0; t < nk; ++t) {\n"
    "        if (MUT == 6)\n"
    "            v41_cp_wait<V41_RING_NS - 1>();\n"
    "        else\n"
    "            v41_cp_wait<V41_RING_NS - 2>();\n",
    "chain guard and wait",
)
# MUT 5: the refill guard off by one (the last stage is never landed once the ring refills)
cu = rep(
    cu,
    "        if (t + V41_RING_NS - 1 < nk) land(t + V41_RING_NS - 1);\n",
    "        if (MUT == 2 || t + V41_RING_NS - (MUT == 5 ? 0 : 1) < nk) land(t + V41_RING_NS - 1);\n",
    "refill",
)
cu = rep(
    cu,
    "    int smem = V41_RING_NS * (V41_SMF_NC + m) * V41_RING_PITCH * 2;\n"
    "    v41_gemv_ffma_ring_kernel<<<o0.nblk + o1.nblk, 32, smem, (cudaStream_t)stream>>>((const v41_bf*)a, lda, o0, o1, m,\n"
    "                                                                                     k);\n"
    "    V41_ERR();\n",
    "    int smem = V41_RING_NS * (V41_SMF_NC + m) * V41_RING_PITCH * 2;\n"
    "    cudaStream_t rst = (cudaStream_t)stream;\n"
    "    const v41_bf* ra = (const v41_bf*)a;\n"
    "    int grid = o0.nblk + o1.nblk;\n"
    "    switch (v41_review_mut()) {\n"
    "        case 1: v41_gemv_ffma_ring_kernel<1><<<grid, 32, smem, rst>>>(ra, lda, o0, o1, m, k); break;\n"
    "        case 2: v41_gemv_ffma_ring_kernel<2><<<grid, 32, smem, rst>>>(ra, lda, o0, o1, m, k); break;\n"
    "        case 3: v41_gemv_ffma_ring_kernel<3><<<grid, 32, smem, rst>>>(ra, lda, o0, o1, m, k); break;\n"
    "        case 4: v41_gemv_ffma_ring_kernel<4><<<grid, 32, smem, rst>>>(ra, lda, o0, o1, m, k); break;\n"
    "        case 5: v41_gemv_ffma_ring_kernel<5><<<grid, 32, smem, rst>>>(ra, lda, o0, o1, m, k); break;\n"
    "        case 6: v41_gemv_ffma_ring_kernel<6><<<grid, 32, smem, rst>>>(ra, lda, o0, o1, m, k); break;\n"
    "        default: v41_gemv_ffma_ring_kernel<0><<<grid, 32, smem, rst>>>(ra, lda, o0, o1, m, k); break;\n"
    "    }\n"
    "    V41_ERR();\n",
    "launch",
)
cu = rep(
    cu,
    "#define V41_RING_KC 256\n#define V41_RING_NS 4\n",
    "#define V41_RING_KC 256\n#define V41_RING_NS 4\n"
    "#include <stdio.h>\n#include <stdlib.h>\n"
    "static int v41_review_env(const char* name) {\n"
    "    const char* e = getenv(name);\n"
    "    return e ? atoi(e) : 0;\n"
    "}\n"
    "static int v41_review_mut() {\n"
    "    static int m = v41_review_env(\"REVIEW_MUT\");\n"
    "    return m;\n"
    "}\n"
    "static int v41_review_trace() {\n"
    "    static int m = v41_review_env(\"REVIEW_TRACE\");\n"
    "    return m;\n"
    "}\n"
    "#define V41_REVIEW_FORM(form, kp)                                                                \\\n"
    "    do {                                                                                         \\\n"
    "        if (v41_review_trace())                                                                  \\\n"
    "            fprintf(stderr, \"REVIEW ffma m=%d n=%d k=%d form=%s kp=%d\\n\", m, n, k, form, (int)(kp)); \\\n"
    "    } while (0)\n",
    "helpers",
)
# the trace: which form a memra_v41_gemm_ffma call takes
cu = rep(
    cu,
    "        // a narrow output (the router's gate, the compressor) of up to 8 rows: the chain with K streamed\n",
    "        // a narrow output (the router's gate, the compressor) of up to 8 rows: the chain with K streamed\n"
    "        V41_REVIEW_FORM(\"ring\", 256);\n",
    "trace ring",
)
cu = rep(
    cu,
    "                if (smem > 48 * 1024) V41_SMEM_LIMIT(v41_gemv_ffma_narrow_parts_kernel, 99 * 1024);\n"
    "                v41_gemv_ffma_narrow_parts_kernel<<<(n + V41_SMF_NC - 1) / V41_SMF_NC, 128, smem, st>>>(\n"
    "                    (const v41_bf*)a, lda, (const v41_bf*)b, ldb, c, ldc, m, n, k, kp);\n"
    "                V41_ERR();\n",
    "                if (smem > 48 * 1024) V41_SMEM_LIMIT(v41_gemv_ffma_narrow_parts_kernel, 99 * 1024);\n"
    "                V41_REVIEW_FORM(\"narrow_parts_9to32\", kp);\n"
    "                v41_gemv_ffma_narrow_parts_kernel<<<(n + V41_SMF_NC - 1) / V41_SMF_NC, 128, smem, st>>>(\n"
    "                    (const v41_bf*)a, lda, (const v41_bf*)b, ldb, c, ldc, m, n, k, kp);\n"
    "                V41_ERR();\n",
    "trace parts",
)
cu = rep(
    cu,
    "        // the small-row body, a launch a group of rows\n",
    "        // the small-row body, a launch a group of rows\n        V41_REVIEW_FORM(\"groups_of_8\", 0);\n",
    "trace groups",
)
cu = rep(
    cu,
    "    dim3 grid((n + 63) / 64, (m + 63) / 64);\n    if (al)\n        v41_gemm_ffma_kernel<true><<<grid, 256, 0, (cudaStream_t)stream>>>(",
    "    dim3 grid((n + 63) / 64, (m + 63) / 64);\n    V41_REVIEW_FORM(\"tile\", 0);\n    if (al)\n"
    "        v41_gemm_ffma_kernel<true><<<grid, 256, 0, (cudaStream_t)stream>>>(",
    "trace tile",
)

# --- the gate: more stage counts -----------------------------------------------------------------
rs = rep(
    rs,
    "        (2, 32, 512),\n        (8, 30, 512),\n    ] {\n",
    "        (2, 32, 512),\n        (8, 30, 512),\n        (2, 30, 768),\n        (2, 30, 1024),\n"
    "        (2, 30, 1280),\n        (8, 30, 5120),\n    ] {\n",
    "gate shapes",
)

os.makedirs(os.path.join(root, "mut"), exist_ok=True)
open(os.path.join(root, "mut", "dsv41_gpu.cu"), "w").write(cu)
open(os.path.join(root, "mut", "dsv41_gate.rs"), "w").write(rs)
print("wrote mut/dsv41_gpu.cu and mut/dsv41_gate.rs")

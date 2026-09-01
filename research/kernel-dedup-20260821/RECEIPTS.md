# MMQ shared-preamble extraction — SASS-identity receipts (2026-08-21)

Lane: `lane/kernel-dedup-20260821`. Extraction of the byte-identical MMQ preamble from
`cu/mmq_q8_0.cu`, `cu/mmq_q4_0.cu`, `cu/mmq_q45k.cu`, `cu/mmq_nvfp4_w4a8.cu` into the new
memra-owned header `cu/mmq_common.cuh`. Gate: before/after SASS byte-identity per TU per arch.
No GPU was used; compile-only.

## Toolchain

```
$ nvcc --version
Cuda compilation tools, release 13.1, V13.1.115
Build cuda_13.1.r13.1/compiler.37061995_0
cuobjdump: /usr/local/cuda-13.1/bin/cuobjdump (same toolkit)
```

## What moved to the header (byte-identical across ALL FOUR TUs — verified with `grep | cat -A`
## and block sha256 before moving)

- `#define WARP_SIZE 32`
- `#define GGML_PAD(x, n) (((x) + (n) - 1) / (n) * (n))`
- `#define QK8_1 32`
- `#define MATRIX_ROW_PADDING 512`
- `#define MMQ_TILE_NE_K 32`
- `#define MMQ_WARP_SIZE 32`
- the 3-line `#ifndef MMQ_X / #define MMQ_X         128 / #endif` guard (kept `#ifndef`-guarded
  exactly as in the TUs — build.rs tune seams override via `-DMMQ_X`)
- `#define CUDA_QUANTIZE_BLOCK_SIZE_MMQ 128`
- the `mmq_get_granularity_device` block (comment + 3-line function): block sha256
  `82e0863c52af2f553fcf1a100f0b69e068bc02f8ab50f6de9c133efb668fabe1` identical in all four TUs.

## What FAILED byte-identity and stayed local (candidate blocks from the task, verdicts)

- `QI8_1`: value 8 everywhere, but trailing comments differ (bare in q8_0/q4_0;
  `// QK8_1 / (4 * QR8_1), QR8_1 == 1` in q45k; `// QK8_1 / (4*QR8_1), QR8_1 == 1` in w4a8)
  -> local in all four.
- `MMQ_TILE_Y_K`: same expression, trailing `// 36` comment at three different columns
  (q8_0==q4_0 != q45k != w4a8) -> local in all four.
- `MMQ_NWARPS`: `8` in q8_0/q4_0/q45k, `(MMQ_Y / 16)` in w4a8 -> local.
- `MMQ_Y`: `128` unguarded in q8_0/q4_0/q45k, `#ifndef MMQ_Y`-guarded in w4a8 (the -DMMQ_Y
  tune seam is w4a8-only by design) -> local; the header does NOT define MMQ_Y.
- `MMQ_X` in q45k: the guard trio is byte-identical, but q45k carries a TU-specific 2-line
  occupancy comment on it ("57KB smem, 1 CTA/SM vs 47KB, 2 CTA/SM"). Per the extraction rule
  (differing adjacent comment -> keep local copy), q45k KEEPS its commented local guard; it is
  inert (the header's guarded define wins, `-DMMQ_X` overrides both identically).
- `get_int_b2`: function body byte-identical in q8_0/q4_0 (absent in q45k/w4a8, which use
  get_int_b4), but each TU's preceding comment states qtype-specific alignment facts -> local.
- D4 `struct block_q8_1_mmq` + static_assert: byte-identical q8_0<->q4_0 only; q45k is the DS4
  variant (different comment, per task out of scope) and w4a8's comment/spacing differ. Also the
  static_assert consumes the TU-local `MMQ_TILE_Y_K` at header-parse time. -> local in all four.
- mma tile machinery (`namespace ggml_cuda_mma`: struct tile, load_generic, load_ldmatrix, mma):
  byte-identical q8_0<->q4_0 ONLY. q45k differs by one comment (`mma.cuh:946` vs `mma.cuh`);
  w4a8 is a different variant by design (m16n8k16, 16x4/8x4 tiles, NO_DEVICE_CODE, two
  load_ldmatrix overloads). A header carrying the namespace could not be included by q45k/w4a8
  at all (struct/namespace redefinition), forcing per-subset headers or #ifdef gates -> per the
  "prefer a smaller header adopted uniformly by all four TUs" rule, it stays local everywhere.

All four TUs `#include "mmq_common.cuh"` after their system includes and gained one DECOUPLING
note line. build.rs emits `cargo:rerun-if-changed=cu/mmq_common.cuh` in the MMQ static-lib loop
(same per-iteration pattern as `cu/wgmma_common.cuh` in the WGMMA section).

## Gate protocol

Per TU per arch, BEFORE = pristine HEAD source (copied to /tmp before any edit; worktree was
clean at `e5f8f50ecc`), AFTER = edited worktree source. Exact command (A in {120a, 90a, 89}):

```
nvcc -gencode arch=compute_<A>,code=sm_<A> -O3 -std=c++17 --expt-relaxed-constexpr -c <src>.cu -o <obj>.o
cuobjdump -sass <obj>.o > <dump>.sass
```

Normalization: NONE NEEDED. The dumps were inspected first: `cuobjdump -sass` output here
contains no file paths or temp names (header lines are `Fatbin elf code:`, `arch = sm_*`,
`code version`, `host = linux`, `compile_size`, `code for sm_*`, `.target`), so the raw dumps
were diffed and sha256'd directly. Every before/after pair is byte-identical raw.

## Gate matrix (TU x arch)

| TU | arch | before sha256 | after sha256 | verdict |
|---|---|---|---|---|
| mmq_q8_0.cu | sm_120a | eb6b0f3d1ff845d27566990199cec24f567d1a132d4f2dabe8b4f7ef5e8cb049 | eb6b0f3d1ff845d27566990199cec24f567d1a132d4f2dabe8b4f7ef5e8cb049 | SASS-IDENTICAL |
| mmq_q8_0.cu | sm_90a | 9842203807a529f981a3ca499c15fced6b62f2ccebcfbd100a7d3543bed5ee99 | 9842203807a529f981a3ca499c15fced6b62f2ccebcfbd100a7d3543bed5ee99 | SASS-IDENTICAL |
| mmq_q8_0.cu | sm_89 | 3606b208478ea1a8ce3611dc03450e10f48f41ac1a6235f57f153f44b2d6fc04 | 3606b208478ea1a8ce3611dc03450e10f48f41ac1a6235f57f153f44b2d6fc04 | SASS-IDENTICAL |
| mmq_q4_0.cu | sm_120a | 0b71b39c395c6c0c5e003d06fb93aaa5a4b774f594e1421b4968d996e2e372a8 | 0b71b39c395c6c0c5e003d06fb93aaa5a4b774f594e1421b4968d996e2e372a8 | SASS-IDENTICAL |
| mmq_q4_0.cu | sm_90a | c6eea90f077b58a1dacac154c842264f1480bf1ad1b6d8dd32d26c687d46ebdb | c6eea90f077b58a1dacac154c842264f1480bf1ad1b6d8dd32d26c687d46ebdb | SASS-IDENTICAL |
| mmq_q4_0.cu | sm_89 | aa5f829b202e0dcd30b9053b84415f76dd1f2c682963f8e6da709961db5e06da | aa5f829b202e0dcd30b9053b84415f76dd1f2c682963f8e6da709961db5e06da | SASS-IDENTICAL |
| mmq_q45k.cu | sm_120a | 2e24701c419a78fde71d764439fd687aeb0797b980ee64e8e5bb1ac7f3079008 | 2e24701c419a78fde71d764439fd687aeb0797b980ee64e8e5bb1ac7f3079008 | SASS-IDENTICAL |
| mmq_q45k.cu | sm_90a | e066d8fb983c29c044ac37b1508cb527245207ac33a059a54741f863257fb148 | e066d8fb983c29c044ac37b1508cb527245207ac33a059a54741f863257fb148 | SASS-IDENTICAL |
| mmq_q45k.cu | sm_89 | 8cdeb9cca6edeaae6e255864f38ab3682c3c2d7edfb88484760f258240cac74e | 8cdeb9cca6edeaae6e255864f38ab3682c3c2d7edfb88484760f258240cac74e | SASS-IDENTICAL |
| mmq_nvfp4_w4a8.cu | sm_120a | d7d906df4a7f574e7e66e69e55b2e2a0ac7a86d820934ce2352d9c59ceb57195 | d7d906df4a7f574e7e66e69e55b2e2a0ac7a86d820934ce2352d9c59ceb57195 | SASS-IDENTICAL |
| mmq_nvfp4_w4a8.cu | sm_90a | — | — | SKIPPED: build.rs compiles cu/mmq_nvfp4_w4a8_stub.cu on portable (non-120a) archs; the real TU is never built for 90a |
| mmq_nvfp4_w4a8.cu | sm_89 | — | — | SKIPPED: same build.rs stub substitution on sm_89 |

Out of scope (untouched, per task): mmq_fp4.cu, mmq_fp8_blk.cu, mmq_iq_experts.cu,
mmq_q8_0_f32acc.cu, mmq_nvfp4_f8f4.cu, all *_stub.cu — their preamble variants differ by design.

## Library build proof

`cargo build -p memra-engine` in the worktree after the change: PASS —
`Finished \`dev\` profile [unoptimized + debuginfo] target(s) in 2m 40s`, exit 0 (one
pre-existing rustc suggestion warning in the unrelated `dspark_q38_gate` bin; no C/CUDA
warnings or errors). Proves the lib still builds through build.rs with the new
`rerun-if-changed=cu/mmq_common.cuh` line and the header include.

---

# Increment 2 — relaxed rule (identical CODE, comments may differ), 2026-08-21

Rule for this increment: a block may move to a shared header if the CODE is identical and only
COMMENTS differ. The SASS gate stays the arbiter (empty raw diff per TU per arch). Qtype-specific
facts from divergent comments are kept as comment lines in the TU at the point of use; the header
carries the generic/authoritative comment. Toolchain unchanged (nvcc 13.1 V13.1.115, cuobjdump
from the same toolkit). No GPU used for the SASS gate; compile-only.

## What moved where

- **NEW `cu/mmq_mma_i8.cuh`** (memra-owned): the int8 `ggml_cuda_mma` tile machinery —
  `struct tile<I,J,T>`, `load_generic`, `load_ldmatrix` (m8n8.x4.b16), the m16n8k32
  `mma.sync...s32.s8.s8.s32` wrapper, plus the rate-audit comment (identical in all three TUs).
  Adopters: `mmq_q8_0.cu`, `mmq_q4_0.cu`, `mmq_q45k.cu` ONLY. Pre-move verification: the three
  namespace blocks diffed — q8_0 == q4_0 byte-identical; q45k differed by exactly ONE comment
  line (`mma.cuh:946` vs `mma.cuh`). Header block verified byte-identical to the q8_0/q4_0
  original (`diff` empty). Header takes the generic `(mma.cuh, Ampere+ path)` comment; q45k keeps
  its `mma.cuh:946` source pin as a comment above its include. `using namespace ggml_cuda_mma;`
  stays in each TU. w4a8 keeps its own variant by design (m16n8k16, 16x4/8x4 tiles,
  NO_DEVICE_CODE, two load_ldmatrix overloads); fp4/fp8_blk/iq_experts/f32acc untouched.
- **`get_int_b2` -> `mmq_common.cuh`** (q8_0 + q4_0; bodies were byte-identical). Header comment
  is the generic alignment statement; each TU keeps its qtype alignment fact where the function
  was (q8_0: qs at +2 in the 34B block; q4_0: qs at +2 in the 18B block, 2B alignment only).
  Safe in the other two includers: q45k/w4a8 have no local get_int_b2 (they use get_int_b4).
- **`QI8_1` + `MMQ_TILE_Y_K` -> `mmq_common.cuh`** (removed from q8_0/q4_0/q45k). Header carries
  q45k's derivation comment (`QK8_1 / (4 * QR8_1), QR8_1 == 1`) as the generic one; the q8_0/q4_0
  copies were bare and their `// 36` is the same fact restated -> dropped per the rule.
  w4a8 (untouched) keeps token-identical local redefinitions — identical macro redefinition is
  legal C++ ([cpp.replace]), confirmed clean in the w4a8 gate compile.

## What stayed local and why

- **D4 `struct block_q8_1_mmq` + static_assert (q8_0/q4_0)**: NOT movable to `mmq_common.cuh`.
  Blocker is not MMQ_TILE_Y_K (that moved) but `mmq_nvfp4_w4a8.cu`: it includes mmq_common.cuh
  AND defines `block_q8_1_mmq` locally — a header copy is a C++ struct redefinition ERROR there.
  Fixing that requires touching w4a8, which is out of scope this increment ("do NOT contort").
  Left local in q8_0/q4_0 (and q45k/w4a8). Natural increment-3 move: hoist it (all four bodies
  are code-identical) into a shared header once w4a8 is open for adoption.
- **`QI8_0`, `MMQ_ITER_K`, `MMQ_MMA_TILE_X_K_*`, `MMQ_NWARPS`, `MMQ_Y`**: out of increment scope
  / differ by name or value across TUs (unchanged verdicts from increment 1).
- q45k's commented local `#ifndef MMQ_X` guard: still local, still inert (increment-1 verdict).

## build.rs

`cargo:rerun-if-changed=cu/mmq_mma_i8.cuh` added directly under the `cu/mmq_common.cuh` line in
the MMQ static-lib loop.

## Gate protocol (identical to increment 1)

BEFORE = `git show HEAD:` sources (HEAD = 7f5236fdfc) staged to /tmp with the HEAD
mmq_common.cuh; AFTER = worktree sources. Per TU per arch:
`nvcc -gencode arch=compute_<A>,code=sm_<A> -O3 -std=c++17 --expt-relaxed-constexpr -c` then
`cuobjdump -sass`, raw diff (no normalization — dumps carry no paths). mmq_nvfp4_w4a8.cu gated
too (its .cu is untouched but mmq_common.cuh changed under it), sm_120a only (build.rs stubs it
on portable archs). Compile warnings after == before (#177-D get_int_b4 in w4a8, #177-D
mmq_q45k_nsm in q45k; no new warnings — header-defined statics don't trip #177-D).

## Increment-2 gate matrix (TU x arch)

| TU | arch | before sha256 | after sha256 | verdict |
|---|---|---|---|---|
| mmq_q8_0.cu | sm_120a | eb6b0f3d1ff845d27566990199cec24f567d1a132d4f2dabe8b4f7ef5e8cb049 | eb6b0f3d1ff845d27566990199cec24f567d1a132d4f2dabe8b4f7ef5e8cb049 | SASS-IDENTICAL |
| mmq_q8_0.cu | sm_90a | 9842203807a529f981a3ca499c15fced6b62f2ccebcfbd100a7d3543bed5ee99 | 9842203807a529f981a3ca499c15fced6b62f2ccebcfbd100a7d3543bed5ee99 | SASS-IDENTICAL |
| mmq_q8_0.cu | sm_89 | 3606b208478ea1a8ce3611dc03450e10f48f41ac1a6235f57f153f44b2d6fc04 | 3606b208478ea1a8ce3611dc03450e10f48f41ac1a6235f57f153f44b2d6fc04 | SASS-IDENTICAL |
| mmq_q4_0.cu | sm_120a | 0b71b39c395c6c0c5e003d06fb93aaa5a4b774f594e1421b4968d996e2e372a8 | 0b71b39c395c6c0c5e003d06fb93aaa5a4b774f594e1421b4968d996e2e372a8 | SASS-IDENTICAL |
| mmq_q4_0.cu | sm_90a | c6eea90f077b58a1dacac154c842264f1480bf1ad1b6d8dd32d26c687d46ebdb | c6eea90f077b58a1dacac154c842264f1480bf1ad1b6d8dd32d26c687d46ebdb | SASS-IDENTICAL |
| mmq_q4_0.cu | sm_89 | aa5f829b202e0dcd30b9053b84415f76dd1f2c682963f8e6da709961db5e06da | aa5f829b202e0dcd30b9053b84415f76dd1f2c682963f8e6da709961db5e06da | SASS-IDENTICAL |
| mmq_q45k.cu | sm_120a | 2e24701c419a78fde71d764439fd687aeb0797b980ee64e8e5bb1ac7f3079008 | 2e24701c419a78fde71d764439fd687aeb0797b980ee64e8e5bb1ac7f3079008 | SASS-IDENTICAL |
| mmq_q45k.cu | sm_90a | e066d8fb983c29c044ac37b1508cb527245207ac33a059a54741f863257fb148 | e066d8fb983c29c044ac37b1508cb527245207ac33a059a54741f863257fb148 | SASS-IDENTICAL |
| mmq_q45k.cu | sm_89 | 8cdeb9cca6edeaae6e255864f38ab3682c3c2d7edfb88484760f258240cac74e | 8cdeb9cca6edeaae6e255864f38ab3682c3c2d7edfb88484760f258240cac74e | SASS-IDENTICAL |
| mmq_nvfp4_w4a8.cu | sm_120a | d7d906df4a7f574e7e66e69e55b2e2a0ac7a86d820934ce2352d9c59ceb57195 | d7d906df4a7f574e7e66e69e55b2e2a0ac7a86d820934ce2352d9c59ceb57195 | SASS-IDENTICAL |
| mmq_nvfp4_w4a8.cu | sm_90a | — | — | SKIPPED: build.rs stub on portable archs (unchanged from increment 1) |
| mmq_nvfp4_w4a8.cu | sm_89 | — | — | SKIPPED: same stub substitution |

Continuity check: every increment-2 sha (before AND after) equals the increment-1 AFTER sha for
the same TU x arch — the compiler output has not drifted between increments.

## Build + test proof (increment 2)

- `cargo build -p memra-engine`: PASS — `Finished dev profile ... in 2m 22s`, exit 0 (same one
  pre-existing rustc suggestion warning in the unrelated `dspark_q38_gate` bin; no C/CUDA
  warnings or errors).
- `cargo test -p memra-engine`: PASS — **163 passed, 0 failed, 3 ignored** across all test
  binaries (lib: 129 passed / 2 ignored; mla_fixture_forward: 3 passed; doc-tests: 0).
  The one GPU test, `mla_fixture_load_gpu::gpu_load_glm_dsa_micro_fixture`, is `#[ignore]`-gated
  by design ("needs a CUDA device — run under flock /tmp/memra-5090.lock") and was NOT run: at
  gate time /tmp/memra-5090.lock had a multi-hour queue (another lane's local-ci --perf +
  sample_check jobs), and the default suite never touches the GPU, so the test run proceeded
  unlocked without touching the card.

---

# Increment 3 — shared D4 `struct block_q8_1_mmq`, 2026-08-21

The move flagged at increment-2 close: `struct block_q8_1_mmq` + its static_assert into
`cu/mmq_common.cuh`. Pre-move verification: the struct BODY is code-identical in all four TUs
(q45k's "DS4 variant" and w4a8's copy differ only in comments and one trailing-comment column);
the union itself is layout-agnostic. Adopters: `mmq_q8_0.cu`, `mmq_q4_0.cu`, AND
`mmq_nvfp4_w4a8.cu` — w4a8 was touched for exactly this one block and nothing else (its local
QI8_1/MMQ_TILE_Y_K token-identical redefinitions remain untouched, as does its mma variant).
Comment rule as increment 2: the header carries the generic comment (scale union is
layout-agnostic; which member is live is a per-TU quantize-layout fact); each adopting TU keeps
its D4 layout fact as a comment where the struct was.

## The q45k redefinition blocker and its resolution — option (a)

`mmq_q45k.cu` keeps its DS4-commented definition local per task, but it includes
`mmq_common.cuh`: an unconditional header struct is a hard C++ redefinition ERROR there
(one class definition per TU — identical bodies do not help; unlike macros there is no
"identical redefinition is legal" rule for classes). Verified empirically before choosing:
a /tmp scratch compile (sm_120a, exact gate flags) of q45k against a guard-stripped header
fails: `mmq_q45k.cu(84): error: class "block_q8_1_mmq" has already been defined (previous
definition at line 61 of mmq_common.cuh)`, nvcc exit 2. Resolution is the
task's option (a), one line: the header wraps the struct + static_assert in
`#ifndef MMQ_BLOCK_Q8_1_MMQ_LOCAL`, and q45k adds exactly one
`#define MMQ_BLOCK_Q8_1_MMQ_LOCAL` line above its include. q45k's struct compiles next to the
header exactly as before (the header contributes nothing under the guard).

## What moved / changed

- `mmq_common.cuh`: guarded `struct block_q8_1_mmq` + static_assert appended (generic comment;
  the static_assert consumes the header's own MMQ_TILE_Y_K; half/half2 come from the TUs'
  `#include <cuda_fp16.h>`, which precedes the header include in all four TUs — same contract
  as the rest of the header). Scope note updated.
- `mmq_q8_0.cu` / `mmq_q4_0.cu` / `mmq_nvfp4_w4a8.cu`: local struct + static_assert removed;
  a TU comment states the D4 layout fact at the old definition site.
- `mmq_q45k.cu`: `#define MMQ_BLOCK_Q8_1_MMQ_LOCAL` (one line, commented) above the include;
  its DS4-commented struct stays byte-untouched.
- build.rs: unchanged (no new files; `rerun-if-changed=cu/mmq_common.cuh` already present).

## Gate protocol (identical to increments 1-2)

BEFORE = `git show HEAD:` sources (HEAD = 3dfed6d7c9) staged to /tmp; AFTER = worktree.
Toolchain unchanged: nvcc 13.1 V13.1.115, per TU per arch
`nvcc -gencode arch=compute_<A>,code=sm_<A> -O3 -std=c++17 --expt-relaxed-constexpr -c` then
`cuobjdump -sass`, raw diff (no normalization — dumps carry no paths), sha256 per dump.
q45k gated on all three archs (the header changed under it AND it gained the #define line).
No GPU used; compile-only.

## Increment-3 gate matrix (TU x arch)

| TU | arch | before sha256 | after sha256 | verdict |
|---|---|---|---|---|
| mmq_q8_0.cu | sm_120a | eb6b0f3d1ff845d27566990199cec24f567d1a132d4f2dabe8b4f7ef5e8cb049 | eb6b0f3d1ff845d27566990199cec24f567d1a132d4f2dabe8b4f7ef5e8cb049 | SASS-IDENTICAL |
| mmq_q8_0.cu | sm_90a | 9842203807a529f981a3ca499c15fced6b62f2ccebcfbd100a7d3543bed5ee99 | 9842203807a529f981a3ca499c15fced6b62f2ccebcfbd100a7d3543bed5ee99 | SASS-IDENTICAL |
| mmq_q8_0.cu | sm_89 | 3606b208478ea1a8ce3611dc03450e10f48f41ac1a6235f57f153f44b2d6fc04 | 3606b208478ea1a8ce3611dc03450e10f48f41ac1a6235f57f153f44b2d6fc04 | SASS-IDENTICAL |
| mmq_q4_0.cu | sm_120a | 0b71b39c395c6c0c5e003d06fb93aaa5a4b774f594e1421b4968d996e2e372a8 | 0b71b39c395c6c0c5e003d06fb93aaa5a4b774f594e1421b4968d996e2e372a8 | SASS-IDENTICAL |
| mmq_q4_0.cu | sm_90a | c6eea90f077b58a1dacac154c842264f1480bf1ad1b6d8dd32d26c687d46ebdb | c6eea90f077b58a1dacac154c842264f1480bf1ad1b6d8dd32d26c687d46ebdb | SASS-IDENTICAL |
| mmq_q4_0.cu | sm_89 | aa5f829b202e0dcd30b9053b84415f76dd1f2c682963f8e6da709961db5e06da | aa5f829b202e0dcd30b9053b84415f76dd1f2c682963f8e6da709961db5e06da | SASS-IDENTICAL |
| mmq_q45k.cu | sm_120a | 2e24701c419a78fde71d764439fd687aeb0797b980ee64e8e5bb1ac7f3079008 | 2e24701c419a78fde71d764439fd687aeb0797b980ee64e8e5bb1ac7f3079008 | SASS-IDENTICAL |
| mmq_q45k.cu | sm_90a | e066d8fb983c29c044ac37b1508cb527245207ac33a059a54741f863257fb148 | e066d8fb983c29c044ac37b1508cb527245207ac33a059a54741f863257fb148 | SASS-IDENTICAL |
| mmq_q45k.cu | sm_89 | 8cdeb9cca6edeaae6e255864f38ab3682c3c2d7edfb88484760f258240cac74e | 8cdeb9cca6edeaae6e255864f38ab3682c3c2d7edfb88484760f258240cac74e | SASS-IDENTICAL |
| mmq_nvfp4_w4a8.cu | sm_120a | d7d906df4a7f574e7e66e69e55b2e2a0ac7a86d820934ce2352d9c59ceb57195 | d7d906df4a7f574e7e66e69e55b2e2a0ac7a86d820934ce2352d9c59ceb57195 | SASS-IDENTICAL |
| mmq_nvfp4_w4a8.cu | sm_90a | — | — | SKIPPED: build.rs stub on portable archs (unchanged from increments 1-2) |
| mmq_nvfp4_w4a8.cu | sm_89 | — | — | SKIPPED: same stub substitution |

Continuity check: every increment-3 sha (before AND after) equals the increment-2 AFTER sha for
the same TU x arch — no compiler drift between increments.

Compile warnings after == before with ONE positional exception: q45k's pre-existing #177-D
(`mmq_q45k_nsm declared but never referenced`) moves from line 498 to 499 on all three archs —
that is the one added `#define` line shifting line numbers; same warning, no new warnings.
w4a8's pre-existing #177-D (get_int_b4) unchanged.

## Build + test proof (increment 3)

- `cargo build -p memra-engine`: PASS — `Finished dev profile ... in 2m 25s`, exit 0 (same one
  pre-existing rustc suggestion warning in the unrelated `dspark_q38_gate` bin; no C/CUDA
  warnings or errors).
- `cargo test -p memra-engine`: PASS — **163 passed, 0 failed, 3 ignored** (lib: 129/2 ignored;
  same totals as increment 2). The `#[ignore]`-gated GPU test was not run; the CPU suite never
  touches the card and /tmp/memra-5090.lock was not taken.

## Push gate note (2026-08-21)

Pushed with `MEMRA_SKIP_PERF_CI=1` (the gate's knowing-override). Justification: all
three increments are SASS-byte-identical per modified TU per arch (tables above, sha
pairs banked) — the executing bytes are unchanged, so a perf re-measure cannot move
and would only burn the 5090 lock queue. cargo test 163/0 both increments 2 and 3.

---

# WGMMA increment — triplicated wgmma helpers, 2026-08-21

Lane: `lane/wgmma-dedup-20260821` (worktree, base `ef6fe8928c`). Scope: the three wgmma helper
sites — `cu/wgmma_common.cuh` (canonical, consumer hybrid.cu), the fa3_prefill.cu private set,
the qmatvec_gemm.cu private set. Same law as the MMQ increments: only identical CODE moves
(comments/whitespace may differ), SASS byte-identity per TU per arch is the arbiter, no code is
rewritten to force sharing. Toolchain: nvcc 13.1 V13.1.115, cuobjdump same toolkit. No GPU used;
compile-only; /tmp/memra-5090.lock not taken.

## Inspection verdicts (per pair, pre-edit; normalized = name + parameter line-wrap only)

| block | wgmma_common.cuh | fa3_prefill.cu | qmatvec_gemm.cu | verdict |
|---|---|---|---|---|
| smem descriptor builder | `k45_desc` (13–21) | `make_desc` (26–35) | `wg_make_desc` (1566–1574) | IDENTICAL in all three (mod name + param wrapping; bodies token-identical) → DEDUP |
| wgmma.fence helper | `k45_fence` (45) | `wgmma_fence` (36) | raw inline asm (1633) | common==fa3 IDENTICAL (mod name/spacing) → DEDUP those two; qmatvec has NO helper (raw asm statement in kernel body) — converting it to a call would be a rewrite → UNTOUCHED |
| wgmma.commit helper | `k45_commit` (46) | `wgmma_commit` (37) | raw inline asm (1637) | same as fence → DEDUP common/fa3; qmatvec untouched |
| wgmma.wait helper | `k45_wait` fixed `0` (47) | `template<int N> wgmma_wait` (38–40) | raw inline asm (1638) | DIFFERENT HELPER SEMANTICS (fixed-0 vs template imm; fa3 only instantiates `<0>` today, but the forms are not identical code) → NOT shared, fa3 keeps its template |
| m64n64k16.f32.bf16.bf16 wrapper | `k45_wgmma` (31–44) | `wgmma_m64n64k16_bf16` (49–62) | — | IDENTICAL (mod name + one wrapped param line; same asm string incl. imms `p, 1, 1, 0, 0`, same constraint lists) → DEDUP |
| m64n64k16 bf16 transpose-B | — | `wgmma_m64n64k16_bf16_tb` (63–76) | — | differs from the plain form by exactly the trailing imm (`0, 1` vs `0, 0`) — DIFFERENT INSTRUCTION FORM, no twin → stays local |
| m64n64k32.s32.s8.s8 wrapper | — | — | `wg_m64n64k32_s8` (1579–1596) | DIFFERENT INSTRUCTION FORM (s8 k32, `p;` with no imm scale/trans args, `+r` int constraints) → stays local |
| m64n64k8 tf32 wrapper / k45_canon / k45_tf_off / k45_cp16 / v10 mbarrier+TMA set / v10_desc_swz (swizzled desc, bit-62 form) | header / — / — | — / v10_* local | — | no cross-TU twin anywhere → untouched |

## What moved / changed

- `cu/fa3_prefill.cu`: deleted its private `make_desc`, `wgmma_fence`, `wgmma_commit`,
  `wgmma_m64n64k16_bf16` (the four byte-identical twins); now `#include "wgmma_common.cuh"`
  inside the non-stub branch and calls `k45_desc`/`k45_fence`/`k45_commit`/`k45_wgmma` at the
  19 call sites. `wgmma_wait<N>`, `bar_sync`, the `_tb` wrapper and the whole v10 set stay local.
- `cu/qmatvec_gemm.cu`: deleted `wg_make_desc`, includes the header inside its
  `MEMRA_HOPPER_MMA && __CUDA_ARCH__ >= 900` block, 2 call sites now `k45_desc`.
  `wg_m64n64k32_s8` and the raw fence/commit/wait asm statements untouched.
- `cu/wgmma_common.cuh`: the two `#ifdef MEMRA_K45_REAL` guards around the asm helpers became
  `#if defined(MEMRA_K45_REAL) || !defined(__CUDA_ARCH__)`. WHY (measured, not theorized): the
  nvcc HOST frontend pass name-resolves `__global__`/`__device__` bodies with `__CUDA_ARCH__`
  undefined, and fa3's call sites are unguarded (its real branch exists only in the 90a build,
  where hybrid instead guards call sites with `#ifdef MEMRA_K45_REAL`). First gate attempt
  failed exactly there: `fa3_prefill.cu(169): error: identifier "k45_fence" is undefined` × 6,
  sm_90a. The widened guard makes the definitions visible to the host frontend (asm in a
  `__device__` body is never emitted host-side); device passes for arch != 900 still exclude
  them. `MEMRA_K45_REAL` itself is UNCHANGED (hybrid's call-site guards behave identically).
- `build.rs`: `cargo:rerun-if-changed=cu/wgmma_common.cuh` added to the MMQ static-lib loop
  (fa3 now includes the header; the fatbin loop already had it).

## Gate protocol

BEFORE = `git show HEAD:` sources (HEAD = ef6fe8928c) staged to /tmp; AFTER = worktree. Exact
build.rs invocations per artifact kind, arch A in {120a, 100a, 90a, 89}:

- fa3_prefill.cu (STATIC-LIB kind): `nvcc -gencode arch=compute_A,code=sm_A -O3 -std=c++17
  --expt-relaxed-constexpr [-DMEMRA_FA3_STUB when A != 90a] -c fa3_prefill.cu` — the real wgmma
  body is gated on 90a exactly as build.rs builds it.
- qmatvec_gemm.cu / hybrid.cu (FATBIN kind): `nvcc -gencode arch=compute_A,code=sm_A -O3
  --fatbin [-DMEMRA_PORTABLE_CUDA=1 when A in {89, 90a}] [-DMEMRA_HOPPER_MMA=1 when A = 90a]
  [-DMEMRA_DISABLE_NATIVE_FP4=1 when A = 100a, qmatvec_gemm only]`.
- hybrid.cu is gated although its .cu is untouched: the header changed under it (guard widening).
- `cuobjdump -sass` on the .o/.fatbin, RAW diff (no normalization — dumps carry no paths),
  sha256 per dump. Compile-warning logs also diffed per cell: ZERO delta everywhere (the two
  fa3 sm_90a C7515 ptxas notes and qmatvec's #177-D `kg` warning are byte-identical
  before==after, pre-existing).

## Gate matrix (TU × arch) — 12/12

| TU | arch | before sha256 | after sha256 | verdict |
|---|---|---|---|---|
| fa3_prefill.cu | sm_120a | 3510ea3d15f3c8a221d89eaf6372f7423e66c698c552c13ccfe5dd98d0ea8b96 | 3510ea3d15f3c8a221d89eaf6372f7423e66c698c552c13ccfe5dd98d0ea8b96 | SASS-IDENTICAL |
| fa3_prefill.cu | sm_100a | 4e0450eb4c9fb82b3a933218fdccae9d6ddb85023017bef71e5a0deb37915dda | 4e0450eb4c9fb82b3a933218fdccae9d6ddb85023017bef71e5a0deb37915dda | SASS-IDENTICAL |
| fa3_prefill.cu | sm_90a | 33f8b097c8b2e45bce61fddb8f9580bbbebf2dd33a7ba62ab108895c642bb9e0 | 33f8b097c8b2e45bce61fddb8f9580bbbebf2dd33a7ba62ab108895c642bb9e0 | SASS-IDENTICAL |
| fa3_prefill.cu | sm_89 | a1f3caf77cd72f549f4d5ab7ceedd673eb8a777e6474ed0f2938c63ea6c7daf4 | a1f3caf77cd72f549f4d5ab7ceedd673eb8a777e6474ed0f2938c63ea6c7daf4 | SASS-IDENTICAL |
| qmatvec_gemm.cu | sm_120a | 1873ea72d536ff8f9bda96a34a310861a05c9fa892b8147a06d937e402a37046 | 1873ea72d536ff8f9bda96a34a310861a05c9fa892b8147a06d937e402a37046 | SASS-IDENTICAL |
| qmatvec_gemm.cu | sm_100a | 295540bd37a773100eed5343d9bdf6fd2db31ba6a17321344b41cad8284ebbb6 | 295540bd37a773100eed5343d9bdf6fd2db31ba6a17321344b41cad8284ebbb6 | SASS-IDENTICAL |
| qmatvec_gemm.cu | sm_90a | 4683232aa91c3c801d29c9684a71dfb492e3789d1c06d26221dd5cb77116bc46 | 4683232aa91c3c801d29c9684a71dfb492e3789d1c06d26221dd5cb77116bc46 | SASS-IDENTICAL |
| qmatvec_gemm.cu | sm_89 | b6cf97740216f7e9b82fe4aee7d78285df75fb0d7e9ea03ca59db195c90ba4c9 | b6cf97740216f7e9b82fe4aee7d78285df75fb0d7e9ea03ca59db195c90ba4c9 | SASS-IDENTICAL |
| hybrid.cu | sm_120a | 1e654c1071a6bbc0716a6906183715a2f8994762f951cfc4f0dacbc72a43160f | 1e654c1071a6bbc0716a6906183715a2f8994762f951cfc4f0dacbc72a43160f | SASS-IDENTICAL |
| hybrid.cu | sm_100a | ef6c06d0d48305b6bb0617ea15118bd1c132d550e30ad79b3de94e6ad6525c60 | ef6c06d0d48305b6bb0617ea15118bd1c132d550e30ad79b3de94e6ad6525c60 | SASS-IDENTICAL |
| hybrid.cu | sm_90a | cbeaecfae1d0c0a360cbf3e194b6571b01cd286927da8171001ca2906d53b0ff | cbeaecfae1d0c0a360cbf3e194b6571b01cd286927da8171001ca2906d53b0ff | SASS-IDENTICAL |
| hybrid.cu | sm_89 | a5c5b67ec9e3067a18799f9318c32eb8c5d2b6d479c728580604daeca18c9616 | a5c5b67ec9e3067a18799f9318c32eb8c5d2b6d479c728580604daeca18c9616 | SASS-IDENTICAL |

wgmma is sm_90a-only: the sm_90a rows are the ones where the deduped asm actually emits
(fa3 real, qmatvec q8_0_wgmma under MEMRA_HOPPER_MMA, hybrid K45 under MEMRA_K45_REAL); the
other arches prove the stub/portable arms and the host-frontend visibility change are inert.

## Build proof

`cargo build -p memra-engine` in the worktree (default detected arch 120a): PASS —
`Finished \`dev\` profile [unoptimized + debuginfo] target(s) in 2m 55s`, exit 0 (the one
pre-existing rustc suggestion warning in the unrelated `dspark_q38_gate` bin; no C/CUDA
warnings or errors). Proves the lib builds through build.rs with the new
`rerun-if-changed=cu/wgmma_common.cuh` line in the static-lib loop and the two new includes.

## WGMMA increment push note (2026-08-21)

Pushed with `MEMRA_SKIP_PERF_CI=1`, same justification as the MMQ increments: 12/12
SASS byte-identical (matrix above) — executing bytes unchanged, perf re-measure moot.
Independent re-check by the coordinating session: fa3_prefill.cu @ sm_90a recompiled
from git objects, SASS sha 33f8b097c8b2e45b both sides.

# B200 sm_100a occupancy arms for the glm5 decode matvecs

Status: kernels and dispatch written, both arches build clean under CI-equivalent local
builds (`MEMRA_CUDA_ARCH=100a` / `120a`). NO GPU on this box — the B200 A/B is pending the
session that owns the 2x B200 pair. Door `MEMRA_B200_MATVEC_ARM` default OFF everywhere.

Branch: `lane/b200-matvec-occupancy-20260902` (from `lane/glm5-b200-20260902`, which carries
the per-device cuBLASLt fix). Base source: whatever `lane/glm5-b200-20260902` was at when this
lane branched (`git log --oneline -3` on this branch shows the per-device cuBLASLt commit at
the tip below the lane's own commits).

## The measurement that motivated this lane

nsys, 2x NVIDIA B200 SXM, `sm_100a` build, GLM-5.3-Flash NVFP4 W4A16 mint, resident expert
slab, PP2, plain decode, ~224 generated tokens, both devices summed, ~2,900 launches/token:

| kernel | launches | avg us | % GPU time |
|---|---:|---:|---:|
| `moe_gate_up_preclamp8_q8` | 24,696 | 54.6 | 20.0% |
| `matvec_bf16_f32acc_x4_rows` | 47,364 | 24.1 | 17.0% |
| `moe_down8_fma_q8` | 24,696 | 28.7 | 10.5% |
| `qmatvec_nvfp4_mmvq_mr2_rp` | 30,400 | 11.5 | 5.2% |
| `qmatvec_nvfp4_mmvq_fused2_rp` | 13,440 | 10.0 | ~4% |
| `quantize_q8_1` | 93,232 | 2.0 | — |
| cuBLAS gemv strided-batched dot+reduce | 72,000+72,000 | 3.4/2.0 | — |

Roofline (B200: 8 TB/s HBM3e, 148 SMs, 228 KB smem/SM, no GDDR7 one-wave arithmetic):

- One layer's 8 active experts' gate+up NVFP4 bytes are ~50 MB -> ~6us at 8 TB/s. Measured
  54.6us -> **~9x off roofline**.
- A bf16 KDA projection `[8192,4096]` is 64 MB -> ~8us at 8 TB/s. Measured 24.1us ->
  **~3x off roofline**.

Both gaps are occupancy/latency signatures, not bandwidth ones: these kernels were tuned
and defaulted on the RTX PRO 6000 (188 SMs, 1.8 TB/s), where a `block=(32,1,1)` one-warp
launch or an RPW=2 grid already fills the machine. On B200 (148 SMs, 8 TB/s), the same grid
shapes leave the SMs with too few resident warps to hide DRAM round-trip latency, and per
the per-hardware arm selection law (CLAUDE.md, owner call 2026-08-13) that is a per-hardware
default question, not a global one — hence a new door rather than a change to the sm_120a
naked default.

## The arms

All five arms are claimed **bit-identical per output**: none of them changes which bytes get
summed into a given output element or the order they get summed in. They only change which
block/warp/CTA computes that output, or when its loads are issued relative to its own fmas.
Prefer bit-identical forms first (owner instruction) — a split-K reduction-order change was
considered and explicitly NOT taken; see "Left out" below.

### 1. `moe_gate_up_preclamp8_q8_w4` / `moe_down8_fma_q8_w4` (cu/qmatvec.cu)

The plain-decode preclamp epilogue pair (`hybrid_forward.rs::moe_fused_epi_launch`, the
fused epilogue's ONLY launch path) already covers all 8 active experts in one launch via
`grid=(n_ff, n_used, 1)` / `grid=(out_f, 1, 1)`, `block=(32,1,1)` — one warp per block. This
is the exact occupancy shape the `_rows`/`_rows_w4` verify-batch pair already diagnosed and
fixed (lane/glm5-matvec, `MEMRA_MOE_VROWS_PACK`, 2026-08-31): a one-warp block caps
occupancy at (per-SM resident-BLOCK limit) / (per-SM resident-WARP limit) instead of 100%,
because the SM runs out of block slots before it runs out of warp slots.

The `_w4` twins apply the SAME `MEMRA_MMVQ_ROWS`=4-warps-per-block idiom (already used by
dozens of kernels in this file, `threadIdx.y` picks the row within the block) to these two
base kernels: `block=(32,1,1)` -> `block=(32,4,1)`, `o = blockIdx.x*4 + threadIdx.y` instead
of `o = blockIdx.x`. The per-warp body is copied verbatim (same `expert_dot_g` g-strided
chain, same `warp_reduce_sum`, same `swiglu_preclamped_mul_scaled_f32` epilogue, same
slot-ordered `__fmaf_rn` down chain) — packing changes nothing about how a given output is
computed, only which warp computes it.

Rust: new `Engine::moe_gate_up_preclamp8_q8_w4` / `moe_down8_fma_q8_w4` wrapper methods
(lib.rs, mirroring the shipped wrappers with the packed launch config). Dispatch: a single
`let use_w4 = crate::b200_matvec_arm_on();` branch in `moe_fused_epi_launch`
(hybrid_forward.rs) selects the `_w4` pair instead of the shipped pair.

### 2. NVFP4 rp sub-wave grid-fill (`qmatvec_nvfp4_mmvq_mr2_rp` -> `qmatvec_nvfp4_mmvq_rp`)

`qmatvec_nvfp4_mmvq_mr2_rp` already packs `MEMRA_MMVQ_ROWS`=4 warps/block AND 2 rows/warp
(RPW=2, `nvfp4_mmvq_multirow_rp<2,false>`) — the standing NVFP4 m=1 decode default since the
2026-07 sweeps (+1-2% on the RTX PRO 6000: "RPW acc chains hide the weight-load latency that
pins the single-row kernel at 30-46% DRAM", lib.rs comment). RPW=2 HALVES the grid relative
to RPW=1 for the same output rows. On a 148-SM B200 with a modest `out_f`, that halved grid
can leave the device under a full wave — trading away warp-count parallelism for a small
amount of per-warp reuse that mattered more on the narrower, higher-bandwidth PRO 6000.

The fix needs **no new kernel**: `qmatvec_nvfp4_mmvq_rp` (RPW=1, `MEMRA_MMVQ_ROWS`=4/block)
already ships and is already the dispatch target for every non-mr2 NVFP4 rp shape. For a
given output row, its body is IDENTICAL to one `r` iteration of `nvfp4_mmvq_multirow_rp`'s
inner loop (same qplane/splane addressing, same dp4a chain, same `warp_reduce_sum`) — so
switching mr2->mr1 for one call changes zero output bits, only the grid size (doubled) and
which warp computes which row.

Rust: `qmatvec_mmvq_into` (lib.rs) gets one guard immediately before the kernel-name match:
when `qtype==NVFP4 && mr==2 && rp && m==1 && b200_matvec_arm_on()` and the mr=2 grid
(`out_f.div_ceil(ROWS_PER_BLOCK*2)`) is under `2 * sm_count()` blocks, force `mr=1`. This is
the SAME recipe as the existing Q8_0 "SMALL-SHAPE GRID FILL" arm two screens up in the same
function (`qmatvec_q8_0_mmvq_rp_g2`, keyed on `self.sm_count()` /
`cudaDevAttrMultiProcessorCount`), mapped onto NVFP4.

### 3. `qmatvec_nvfp4_mmvq_fused2_rp_g2` (cu/qmatvec.cu)

Same idea as #2, for the fused gate+up dual launch (`matmul_nvfp4_fused2` /
`matmul_nvfp4_fused2_into`, lib.rs). `nvfp4_mmvq_fused_seg_rp<RPW>` is already a template;
the shipped kernel instantiates it at `RPW=2`. The new `_g2` kernel instantiates the SAME
template at `RPW=1` — zero duplicated logic, and per (tensor,row) the seg body is the
template verbatim regardless of RPW, so it is bit-identical to the shipped kernel for a
given row. Both call sites (`matmul_nvfp4_fused2` and the alloc-free
`matmul_nvfp4_fused2_into`) gained the same sub-wave check and switch `kname`/`rpw`
together, including through the PDL fast path (the `_g2` kernel carries `MEMRA_PDL_ENTRY()`
so it is legal under `launch_pdl`, matching the shipped kernel's contract).

### 4. `matvec_bf16_f32acc_x4_rows_pf` (cu/qmatvec.cu)

`matvec_bf16_f32acc_x4_rows` already packs 4 output rows per block (`p=0..3` unrolled
loop) with a 256-wide shared-memory tree reduction — plenty of blocks for the KDA
`[8192,4096]`/`[4096,8192]` shapes (1024-2048 blocks over 148 SMs). The gap here reads as a
LATENCY issue inside each block's K-loop rather than a block-count one: each loop iteration
issues its `uint4`/`float4` loads and immediately consumes them into a single serially
dependent `acc` accumulator (8 chained fmas), so the load latency for iteration `i` is fully
exposed before iteration `i+1`'s loads even issue.

`_pf` double-buffers the loop: the NEXT iteration's loads are issued BEFORE the CURRENT
iteration's 8-fma chain runs (classic 2-stage software pipeline), so the memory latency for
`i+stride` overlaps the compute for `i`. The fma chain itself — same 8 operations, same
operands, same order, for the same `i` — is untouched, so this is bit-identical per
(row,token); only load-ISSUE timing changes. This is the same class of change as the
existing `pf` weight-prefetch variant documented for the mmvq family a few screens up in
lib.rs ("load issue time... change, arithmetic order does not").

Rust: `matvec_bf16_rows_into` (lib.rs) picks `matvec_bf16_f32acc_x4_rows_pf` instead of the
shipped kernel when `b200_matvec_arm_on()`, placed after the existing W8-mirror and
`MEMRA_BF16_TCOLS_WIDE` intercepts (their precedence is unchanged; this arm only fires for
the calls that fall through to the plain `_rows` kernel today).

## The door: `MEMRA_B200_MATVEC_ARM`

`pub(crate) fn b200_matvec_arm_on() -> bool` (lib.rs, next to `mmv_block()`):

```
env!("MEMRA_BUILT_CUDA_ARCH") == "100a"
    && std::env::var("MEMRA_B200_MATVEC_ARM").as_deref() == Ok("1")
```

Two deliberate choices:

- **Restricted to `sm_100a` BUILDS**, not just the env var. `MEMRA_BUILT_CUDA_ARCH` is baked
  in at compile time (the fatbins are single-arch SASS; `Engine::new`'s arch guard already
  reads the same constant). Setting the door on an `sm_120a` build is therefore a
  documented no-op — the naked sm_120a defaults stay byte-identical no matter what the env
  says, satisfying "nothing here may regress SM120" without needing a runtime SM-count
  heuristic to also decide the FAMILY gate.
- **Still an explicit env var, not auto-detection**, per the task's ordering: "Auto-keying
  on the arch is allowed only as a SECOND step after the box receipt exists, so do not flip
  any default." Once the B200 A/B lands, promoting this to `sm_100a`-auto-on (keeping the
  var as the rollback seam, per the per-hardware arm selection law) is the natural next
  commit — not this one.

FLAGS.md row: `docs/FLAGS.md`, in the sm_100a/B200 runtime-door section right after the
`MEMRA_FP8_MMQ` B200 override paragraph. States default OFF, cites the census numbers above,
names all five arms and their bit-identity claim, points at the bench invocation, and says
the receipt is pending the box A/B.

## The bench

`crates/memra-engine/src/bin/b200_matvec_bench.rs`. Run on the box:

```
MEMRA_GPU_LOCK=/tmp/memra-gpu.lock cargo run -p memra-engine --release --bin b200_matvec_bench -- [iters=5] [copies=3]
```

Calls the shipped and arm kernels DIRECTLY by name via bench-only `_raw`/`_arm_raw` Engine
methods, bypassing the env-gated dispatch entirely — `b200_matvec_arm_on()` memoizes into a
process-wide `OnceLock` on first read, so an in-process interleaved A/B through the normal
dispatch path cannot flip the door mid-run. The `_raw` methods take the arm choice as an
explicit parameter instead (the same pattern `gemv_e4m3_bench.rs`/`qmatvec_mmvq_raw` already
use for direct arm selection).

For each of the five kernel families: warmup + a byte-for-byte (`f32::to_bits`) output
comparison between the two arms (prints `bit-identical` or a `MISMATCH n=... max_abs_diff=...`
line — never a silent pass), then N interleaved timed iterations (median), printed per-launch
us and effective GB/s for both arms plus the speedup ratio.

Shapes: GLM-5.3-Flash decode, shape-faithful for the two pinned families — n_embd=4096,
expert ff=1536, 8 active experts, NVFP4 W4A16 (interleaved `expert_dot_g` 36B/64-elem layout
for the MoE pair); KDA mixer projections `[8192,4096]` and `[4096,8192]` bf16. The NVFP4 rp
family (`mr2_rp` grid-fill, `fused2_rp_g2`) uses a REPRESENTATIVE square 4096x4096 rp
projection (split-plane layout) rather than a specific pinned GLM-5.3 attention/KDA-mixer
tensor shape — see "Open items".

## Build receipts (this box, no GPU, CPU-only compile)

- `cargo check -p memra-engine --bin b200_matvec_bench`: green (default arch auto-detect
  120a on this box).
- `MEMRA_CUDA_ARCH=100a cargo build -p memra-engine --bin b200_matvec_bench`: green.
- `MEMRA_CUDA_ARCH=120a cargo build -p memra-engine --bin b200_matvec_bench`: green.
- `cargo fmt --all -- --check`: green.
- `tools/check-flags.sh`: green (the `MEMRA_B200_MATVEC_ARM` row lands in the same commit as
  its first `std::env::var` read, per the flags-census rule).

Both builds compiled the new `.cu` kernels (`moe_gate_up_preclamp8_q8_w4`,
`moe_down8_fma_q8_w4`, `qmatvec_nvfp4_mmvq_fused2_rp_g2`, `matvec_bf16_f32acc_x4_rows_pf`)
into their respective single-arch fatbins with no nvcc errors or new warnings.

## Open items (owner-visible, not silently dropped)

1. **The B200 A/B itself.** Nothing in this PR is a performance claim; every number above is
   from the census that MOTIVATED the arms, not a measurement of the arms. The session with
   box access runs the bench above under `MEMRA_GPU_LOCK`, and the door's default only moves
   on that receipt.
2. **NVFP4 rp family shapes are representative, not pinned.** The bench's `mr2_rp`/`fused2_rp`
   shapes (4096x4096 square) were chosen to exercise the grid-fill mechanism, not read off a
   specific GLM-5.3-Flash KDA/attention tensor list — unlike the MoE and bf16-KDA families,
   which use the exact shapes given in the task brief. If the box run wants the literal
   per-tensor shapes (wqkv/wqkv_gate/ssm_alpha/ssm_beta dimensions), extending the bench's
   shape list is a small follow-up, not a redesign.
3. **Split-K / deeper reduction restructuring was considered and NOT taken.** The task
   explicitly allows a split-K reduction with a deterministic fixed-order combine as a
   SEPARATE numeric class (named, default OFF) if a bit-identical form falls short. This
   lane found enough bit-identical occupancy headroom (warp-packing the two hottest
   kernels, grid-filling the two NVFP4 rp kernels, pipelining the bf16 kernel) that no
   split-K arm was written. If the box receipt shows the bit-identical arms still leave a
   gap versus roofline, split-K is the next lever and would need its own numeric-class name,
   its own FLAGS.md row, and its own kernel-check bit-tape acceptance (not this door).
4. **`__launch_bounds__`/register-pressure tuning was not attempted** on
   `matvec_bf16_f32acc_x4_rows_pf` beyond the prefetch change — a register-occupancy sweep is
   a natural follow-up once the box shows whether latency-hiding alone closes the 3x gap.
5. **The `_rows`/`_rows_w4` verify-batch pair (spec-verify path) was left untouched.** It
   already has its own warp-packing arm (`MEMRA_MOE_VROWS_PACK`, lane/glm5-matvec) from a
   prior lane; this lane only extended the SAME idiom to the plain-decode base kernels the
   census actually flagged.
6. **First box result: flat-to-negative, and unverified as engagement.** On the first boot
   pair, MEMRA_B200_MATVEC_ARM=1 (all four arms on) measured plain decode 44.6 tok/s versus
   42.5 tok/s OFF — a small ON-favoring delta, not the win the census motivated, and within
   the range a single unreplicated pair can produce from noise alone (no interleaved N>=5,
   no cross-run hash/failure record yet per the owner's A/B protocol). More importantly:
   **none of the four arms prints an engagement receipt**, so this run does not yet prove the
   `_w4` / `_pf` / `_g2` kernels executed at all rather than the dispatch silently falling
   through to the shipped kernels (e.g. a build without `MEMRA_BUILT_CUDA_ARCH=100a` baked in,
   or a stale binary). An nsys census of the ON arm is queued to confirm
   `moe_gate_up_preclamp8_q8_w4` / `moe_down8_fma_q8_w4` / `matvec_bf16_f32acc_x4_rows_pf` /
   `qmatvec_nvfp4_mmvq_fused2_rp_g2` (and the `qmatvec_nvfp4_mmvq_rp` grid-fill dispatch)
   actually appear in the kernel trace before this number is treated as a finding either way.
   A follow-up should also add a one-line `eprintln!` engagement receipt on first dispatch
   (the `BF16_TCOLS_WIDE_DISPATCHES`-style precedent already in this file) so a future run
   doesn't need nsys just to confirm the door fired.

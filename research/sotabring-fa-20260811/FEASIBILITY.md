# FA3/FA4 mechanisms on `sm_120a`: source-only feasibility

Date: 2026-08-11

Lane: `lane/cx-sotabring`

Scope: hand-porting selected forward-attention mechanisms with `mma.sync` and `cp.async`, without
`wgmma`, `tcgen05`, TMEM, kernel implementation, GPU work, or new measurements.

## Decision

1. **NO-GO as a decode-SOL lane.** Memra's hot decode path is a `T=1`, split-K,
   quantized-KV program. It performs key scoring with DP4A/scalar work and value accumulation with
   CUDA-core arithmetic; it does not contain the dense QK/PV `mma.sync` pair that FA3/FA4 overlap.
   A prefill mainloop hand-port therefore does not attack the named decode mechanism
   (`crates/memra-engine/cu/flash_attn.cu:4961-4984`,
   `crates/memra-engine/cu/flash_attn.cu:7192-7345`).
2. **CONDITIONAL GO as a separate prefill-only research lane.** The one mechanism worth isolating is
   FA3-inspired *cross-warp* ping-pong: one query-row cohort executes synchronous MMA while another
   cohort executes softmax. This is a reduced-form scheduling experiment, not FA3 and not FA4.
3. **NO-GO for a literal FA3 or FA4 port on `sm_120a`.** FA3's programmable same-consumer overlap
   depends on WGMMA commit/wait semantics; FA4's defining pipeline depends on `tcgen05` asynchronous
   MMA and TMEM. Neither instruction family targets `sm_120a`.
4. **Do not spawn the conditional lane from the decode gap alone.** First complete the higher-ranked
   decode work and profile the residual as prescribed by the existing SOL audit. Spawn it only if
   prefill/TTFT becomes an independently accepted target
   (`research/solgap-20260811/REPORT.md:122-169`).

This feasibility study makes no speedup estimate and reports no measurements.

## Current-source correction to the premise

“FlashAttention does not run on SM120” is no longer literally current. At upstream commit
[`a369df707e1980fb328abcc1733e3457ec10155f`](https://github.com/Dao-AILab/flash-attention/tree/a369df707e1980fb328abcc1733e3457ec10155f)
(checked 2026-08-11), upstream has an SM120 forward class. That class explicitly subclasses the
SM80 implementation, forces the SM80 path, and describes its primitive as
`mma.sync.aligned.m16n8k16`; it changes the shared-memory capacity check for SM120
([`flash_fwd_sm120.py:2-20`](https://github.com/Dao-AILab/flash-attention/blob/a369df707e1980fb328abcc1733e3457ec10155f/flash_attn/cute/flash_fwd_sm120.py#L2-L20)).
The inherited implementation uses warp-level `m16n8k16` MMA
([`flash_fwd.py:579-599`](https://github.com/Dao-AILab/flash-attention/blob/a369df707e1980fb328abcc1733e3457ec10155f/flash_attn/cute/flash_fwd.py#L579-L599))
and orders each block as QK, online softmax/rescale, then PV around a `cp.async` staging pipeline
([`flash_fwd.py:1096-1203`](https://github.com/Dao-AILab/flash-attention/blob/a369df707e1980fb328abcc1733e3457ec10155f/flash_attn/cute/flash_fwd.py#L1096-L1203)).

That is useful prior art for an SM120 FA2-style implementation. It is not the FA4 paper pipeline:
there is no TMEM accumulator, `tcgen05` MMA, correction warpgroup, or fully asynchronous MMA.

The current PTX ISA makes the hardware boundary explicit:

- BF16 `mma.sync.m16n8k16` requires SM80 or newer, so it is a valid SM120 primitive
  ([PTX `mma`](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma)).
- `cp.async` is a non-blocking global-to-shared copy and is available from SM80
  ([PTX `cp.async`](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async)).
- `wgmma.mma_async` and its fence/commit/wait operations require `sm_90a`; architecture-specific
  `a` instructions are not an SM120 facility
  ([PTX WGMMA](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-mma)).
- `tcgen05` TMEM allocation/data movement targets the SM100/SM110 families and does not list
  SM120
  ([PTX `tcgen05`](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensorcore-5th-generation-family-instructions)).
- `setmaxnreg` does list `sm_120a`, but it is collective over a warpgroup. Memra's present
  four-warp CTA is one warpgroup, so it cannot redistribute registers between individual producer
  and consumer warps in the FA3 manner without a larger multi-warpgroup CTA redesign
  ([PTX `setmaxnreg`](https://docs.nvidia.com/cuda/parallel-thread-execution/#miscellaneous-instructions-setmaxnreg)).

## What memra already has

### Prefill

- The SM120 primitive is already BF16 `mma.sync.aligned.m16n8k16` with FP32 accumulation
  (`crates/memra-engine/cu/flash_attn.cu:158-163`).
- The default fresh-prefill body assigns 16 query rows to each of four warps and allocates a
  two-stage K/V shared-memory ring for BF16 K/V
  (`crates/memra-engine/cu/flash_attn.cu:1030-1068`). It waits for the current `cp.async` group and
  issues the next K/V tile before computing the current tile
  (`crates/memra-engine/cu/flash_attn.cu:1077-1136`).
- Within each warp and KV iteration, the actual dependency order remains synchronous QK MMA,
  register softmax/rescale, a P-to-shared-memory layout conversion, then PV MMA
  (`crates/memra-engine/cu/flash_attn.cu:1150-1255`). Therefore the local “FA3 overlap” label in
  the dispatch comment means register-resident scores plus copy/compute overlap; it does not
  implement FA3 Algorithm 2's WGMMA “commit but do not wait” pipeline. This is the remaining
  scheduling delta over the current kernel (`crates/memra-engine/src/lib.rs:8863-8932`).
- The cached/chunk-prefill path first dequantizes K/V once to a BF16 workspace, then defaults to
  `fa_prefill_qw_db`, whose second K/V buffer overlaps next-tile copy with current-tile compute
  (`crates/memra-engine/src/lib.rs:9575-9583`, `crates/memra-engine/src/lib.rs:9627-9658`). Its
  inner loop is still QK, softmax, P restage, PV in that order
  (`crates/memra-engine/cu/flash_attn.cu:4786-4908`).
- Step-3.7 serving uses that dequant-once path for full-attention cached prefill and the hd128
  windowed twin for SWA prefill (`crates/memra-engine/src/hybrid_forward.rs:9157-9191`). This—not
  the cacheless fresh-prefill wrapper—is the relevant first target if a Step-3.7 prefill lane is
  ever approved.

### Decode

- Decode has a separate `T=1` split-K contract and emits partial output/max/sum state for a later
  log-sum-exp combine (`crates/memra-engine/cu/flash_attn.cu:4961-4984`,
  `crates/memra-engine/cu/flash_attn.cu:6619-6644`).
- The currently selected v4 family stages and repacks quantized K/V, computes all 32 key scores in
  parallel with DP4A, performs online softmax, and accumulates V with scalar/vector arithmetic
  (`crates/memra-engine/cu/flash_attn.cu:7096-7123`,
  `crates/memra-engine/cu/flash_attn.cu:7192-7345`). It has no QK/PV tensor-core pair to overlap.
- The host dispatch chooses that v4 family for its eligible hd256/depth cases and otherwise retains
  other vector or scalar paths (`crates/memra-engine/src/lib.rs:9900-10018`). Eligible multi-row
  batches can use one z-batched v4 launch plus one combine
  (`crates/memra-engine/src/lib.rs:10021-10025`,
  `crates/memra-engine/src/lib.rs:10076-10097`); ineligible Step-3.7 rows still walk per-session
  KV views and call the row-view entry one at a time
  (`crates/memra-engine/src/decode_batch.rs:1441-1477`).

### Existing compile-gate precedent

The build selects one CUDA architecture, defines `memra_hopper_mma` only for `90a`, and records an
exact built-architecture guard (`crates/memra-engine/build.rs:188-214`). The H100 FA3 source is
compiled as a fail-closed stub on every non-90a architecture
(`crates/memra-engine/build.rs:436-438`, `crates/memra-engine/cu/fa3_prefill.cu:16-23`), while its
live body contains the WGMMA and TMA operations
(`crates/memra-engine/cu/fa3_prefill.cu:36-40`,
`crates/memra-engine/cu/fa3_prefill.cu:100-106`). That is the right isolation model for any SM120
research sibling: a compile-time gate, not merely an environment-variable branch in the default
fatbin.

## Per-mechanism portability

Legend: **portable** means the mechanism can be expressed with the allowed SM120 primitives;
**reduced** means its idea survives but the paper's overlap contract does not; **blocked** means
the defining operation is unavailable.

| Mechanism | SM120 status with `mma.sync` + `cp.async` | Delta over current memra | Honest landing / verdict |
|---|---|---|---|
| FA3 circular async K/V staging | **Portable; substantially present.** A multi-stage `cp.async` ring is legal. | Both fresh BF16-KV prefill and BF16-workspace prefill already prefetch the next tile (`flash_attn.cu:1077-1136`, `4786-4805`). | Prefill only. No new lane for this mechanism alone. |
| FA3 producer/consumer warp specialization | **Reduced.** Dedicated copy and compute roles plus barriers are expressible, but 16-byte `cp.async` replaces TMA and an extra warpgroup is needed to use `setmaxnreg` asymmetrically. | Current four warps all participate in staging and then all compute (`flash_attn.cu:1040-1049`, `1083-1093`, `1125-1135`). | Prefill experiment after ping-pong, not the first increment; a producer group may cost more compute capacity than it frees. |
| FA3 inter-warpgroup ping-pong | **Reduced but real.** Different SM120 warps can be made eligible for synchronous MMA and softmax work at the same time; named barriers can intentionally stagger two independent query-row cohorts and let the hardware scheduler arbitrate. | Current warps own independent query rows, but there is no explicit cohort/ping-pong schedule (`flash_attn.cu:1040-1051`, `1150-1255`). | Best first prefill-only probe. Benefit is unknown until measured later. |
| FA3 intra-consumer two-stage QK/PV/softmax pipeline | **Blocked.** FA3 explicitly commits WGMMA without waiting, does independent work, then waits. `mma.sync` exposes no equivalent async group/commit/wait contract. | Current score registers are consumed immediately after synchronous QK and before synchronous PV (`flash_attn.cu:1150-1255`). | Do not claim or attempt a literal hand-port. Cross-warp ping-pong is the only reduced substitute. |
| FA3 TMA load/multicast choreography | **Not portable as written; replaceable in reduced form.** Cooperative `cp.async` can move the same bytes but lacks TMA's one-thread tensor transaction and multicast choreography. | Current kernels already use cooperative `cp.async` rather than TMA (`flash_attn.cu:1077-1094`, `1112-1136`). | No independent reason to spawn a lane. |
| FA3 FP8 block quantization and incoherent processing | **Algorithmically expressible, but not this lane.** It changes data representation, arithmetic, transforms, and numeric gates rather than just schedule. | Current prefill contract is BF16 MMA with FP32 accumulation (`flash_attn.cu:158-163`); current decode consumes its own quantized KV formats (`flash_attn.cu:168-179`, `7096-7106`). | Separate accuracy/format study; no decode-SOL claim. |
| FA4 fully asynchronous MMA into TMEM | **Blocked.** `tcgen05` and TMEM are the mechanism, not an implementation detail. | Memra's accumulator fragments are registers written by `mma.sync` (`flash_attn.cu:87-101`, `158-163`). | No-go on SM120. |
| FA4 two softmax warpgroups plus correction warpgroup | **Blocked as designed; reduced to the FA3-inspired cross-warp probe above.** The paper decouples correction through TMEM and fully asynchronous MMA. Emulating that exchange through shared memory would add a different pipeline. | Current warp owns its scores, running statistics, and output accumulator through the entire iteration (`flash_attn.cu:1096-1103`, `1171-1255`). | Do not make this the first increment. |
| FA4 partial polynomial `exp2` emulation | **Portable arithmetic, different numerics.** FMA polynomial evaluation can be written on SM120, but its mix with MUFU is tuned and adds registers. | Current softmax uses `exp2f` for every exponential (`flash_attn.cu:1193-1215`, `7308-7317`). | Only after profiling identifies exponential issue as the target and the owner accepts a new numeric config. Not byte-identical. |
| FA4 conditional online-softmax rescaling | **Portable algorithm, different floating-point program.** The thresholded recurrence can be implemented without FA4 hardware. | Current code rescales the running sum and every output fragment whenever the running max changes (`flash_attn.cu:1193-1234`, `7308-7317`). | Later numeric experiment at most. It is not a pure scheduling port and is not the first increment. |
| FA4 LPT/causal/varlen tile scheduling | **Portable and hardware-independent.** | Current prefill maps `blockIdx.x` directly to increasing query tiles (`flash_attn.cu:1041-1049`); current B>1 decode has z-batched and per-row branches (`decode_batch.rs:1113-1143`). | Separate scheduler study. It cannot help B=1, and it does not remove the per-row fallback launches. |
| FA4 2-CTA backward, DSMEM and deterministic-gradient machinery | **Hardware-coupled and irrelevant.** It depends on `tcgen05`/TMEM and serves backward gradients. | Memra's inspected attention surface is inference forward/decode (`lib.rs:8779-8788`, `9819-9845`). | No-go for this engine lane. |

Primary mechanism definitions are from
[FA3 §§3.1-3.2](https://arxiv.org/html/2407.08608#S3) and
[FA4 §§3.1-3.3](https://arxiv.org/html/2603.05451#S3). In particular, FA3 Algorithm 2 says
“commit but do not wait” for WGMMA before overlapping softmax; FA4 describes asynchronous MMA into
TMEM, two softmax warpgroups plus a correction warpgroup, polynomial exponential, conditional
rescaling, and LPT scheduling. The portability classifications above are narrower than saying
“the algorithms are hardware-independent”: some scheduling ideas are portable, but the defining
same-warp asynchronous pipelines are not.

## Where a win could land

| Surface | Relevant current work | Can FA-style MMA/softmax overlap land? | Conclusion |
|---|---|---|---|
| B=1 token decode | Quantized-KV split-K scoring, online softmax, value accumulation and split combine (`flash_attn.cu:7192-7355`, `6619-6644`) | **No direct landing.** There is one query row per head and no dense QK/PV MMA pair in the selected path. | No-go for the named decode gap. |
| B=2..8 decode | Eligible rows share a z-batched v4 launch; ineligible rows use per-session view calls (`lib.rs:10021-10025`, `10076-10097`; `decode_batch.rs:1441-1477`) | **No direct landing.** More rows increase launch aggregation opportunities, not the missing asynchronous-MMA opportunity. | Expand/repair batched decode eligibility if evidence points there; do not substitute a prefill kernel. |
| Cached/chunk prefill | BF16 workspace plus `fa_prefill_qw_db`, including Step-3.7's hd128 full/windowed call sites (`lib.rs:9627-9658`; `hybrid_forward.rs:9157-9191`) | **Yes, in reduced cross-warp form.** QK and PV are dense MMA and softmax is between them. | Only credible home for `memra_fa3_overlap`. |
| Cacheless prefill | Default BF16-KV `fa_prefill_bf16kv_pp` (`lib.rs:8863-8932`) | **Yes, but secondary for the named serving path.** | Cover after the cached path, not instead of it. |
| Whole decode SOL budget | The prior audit attributes the gap primarily to serial PP stages, a host-synchronous router and B>1 attention launch structure, and asks for a profile before residual kernel rewrites (`research/solgap-20260811/REPORT.md:11-19`, `126-169`). | **No defensible decode uplift can be assigned from this source study.** | Preserve the SOL audit's ordering. |

The same prior audit accounts KV reads separately from the model's active weight stream
(`research/solgap-20260811/REPORT.md:25-31`, `46-50`). That reinforces the Amdahl boundary: even a
future attention-local improvement would not close the serial-PP, router-sync, or expert/GEMV
parts of decode. This document deliberately does not extrapolate a percentage from those receipts.

## Ranked go/no-go recommendation

### 1. NO-GO — `memra_fa3_overlap` as the next decode-SOL lane

The mechanism-to-path match fails before performance is considered. The selected decode kernel has
no MMA/softmax/MMA chain, and the SOL audit already ranks concrete scheduling/launch work ahead of
an unprofiled residual kernel rewrite (`research/solgap-20260811/REPORT.md:122-169`). Calling a
prefill experiment a decode-SOL lane would make its success criterion impossible to interpret.

### 2. CONDITIONAL GO — a narrowly named, prefill-only overlap lane

If prefill/TTFT is separately placed back in scope, `memra_fa3_overlap` is worth one bounded probe,
with the subtitle **“FA3-inspired inter-warp ping-pong; not an FA3 implementation.”** Its purpose is
to answer one question: can explicit cross-warp phase staggering improve the target cached-prefill
kernel when every per-row arithmetic dependency remains unchanged?

Smallest first increment:

1. Clone only the full-attention `fa_prefill_qw_db_body<128>` path used by Step-3.7 cached prefill
   (`crates/memra-engine/cu/flash_attn.cu:4719-4729`,
   `crates/memra-engine/src/hybrid_forward.rs:9184-9191`).
   Do not touch decode, windowed masking, hd256, fresh/cacheless prefill, tile sizes, data formats,
   `exp2f`, or online-softmax recurrence.
2. Keep the current cooperative two-stage `cp.async` ring
   (`crates/memra-engine/cu/flash_attn.cu:4737-4749`,
   `crates/memra-engine/cu/flash_attn.cu:4786-4805`). Split the existing four compute warps
   into two independent query-row cohorts and use scoped/named barriers to stagger QK/PV MMA in one
   cohort against softmax in the other. Preserve each row's QK -> softmax/rescale -> PV order.
3. Do not add a dedicated producer warp or `setmaxnreg` in increment 1. Those introduce a second
   mechanism and require a larger warpgroup/resource redesign.
4. Compile the sibling only under a new default-off Rust/NVCC cfg such as
   `memra_fa3_overlap`; allow runtime selection only inside that compiled research build. The
   ordinary SM120 source path must not contain or dispatch the symbol.

Required gates before the probe can be called viable:

- **Naked-build identity:** with the compile gate absent, the SM120 `flash_attn` fatbin and final
  executable hashes match the pre-lane build, and the new symbol is absent. This proves a stronger
  contract than “default branch not taken.”
- **Isolation:** gated build, explicit opt-in, one kernel sibling, one changed schedule. Baseline
  dispatch remains the default even in the research build.
- **Kernel exactness:** require zero output-bit differences against `fa_prefill_qw_db` for the
  existing continuation, BK-aligned/unaligned-tail, and causal/GQA cases. The existing
  dequant-once cases establish the relevant shapes (`crates/memra-engine/src/bin/kernel_check.rs:4076-4105`).
- **Model correctness:** same-prompt golden output, `kernel-check` all green, `run-gen` argmax
  match, and `run-spec` K=1..8 self-consistency.
- **Future evidence, not supplied here:** only after correctness, inspect generated SASS/profile
  to establish that tensor and softmax work actually overlap, then run the prescribed interleaved
  before/after battery on the 5090 development rig and the 2x PRO 6000 verification target. If the
  exact target cached-prefill path has no repeatable end-to-end improvement, delete the arm and
  retain this report as the result.

Only after that first increment passes should the lane consider, in order: the windowed hd128 twin,
hd256, a separate producer/consumer copy group, then FA4's numeric ideas. Polynomial exponential
and conditional rescaling must never be folded into the scheduling increment because they change
the floating-point program and confound the result.

### 3. NO-GO — literal FA4 branding or a decode rewrite hidden inside this lane

A TMEM/tcgen pipeline cannot be truthfully represented by `mma.sync`; a new dense-MMA decode design
would also have to solve quantized-KV unpacking, GQA broadcast, split partitioning, small-M
utilization and combine semantics (`crates/memra-engine/cu/flash_attn.cu:4961-4984`,
`crates/memra-engine/cu/flash_attn.cu:7096-7345`,
`crates/memra-engine/cu/flash_attn.cu:6619-6644`). That is a new decode algorithm, not an FA3/FA4
hand-port. It requires its own profile-derived problem statement and should not inherit this
lane's conditional approval.

## Evidence boundary

- Repository citations refer to commit `b154c427`'s parent source state (`5735586e`) plus the
  progress-only first commit; no kernel source was changed.
- External sources were refreshed on 2026-08-11: the FA3 paper, FA4 v1 paper, NVIDIA PTX ISA 9.3,
  and FlashAttention upstream commit `a369df707e1980fb328abcc1733e3457ec10155f`.
- No GPU command, benchmark, profiler, generated performance surface, format pass, or measurement
  was run. All “could” statements are feasibility hypotheses guarded by future gates, not
  performance claims.

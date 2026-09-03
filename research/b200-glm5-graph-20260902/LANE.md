# glm5_next whole-token decode CUDA graph, per PP stage (lane/b200-glm5-graph-20260902)

Owner order 2026-09-02: *"hardly improve the decode on these cards"*. Target on a 2x B200
pair: **230 tok/s plain = 4.35 ms/token**. Door: `MEMRA_GLM5_DECODE_GRAPH`, **default OFF**,
FLAGS.md row landed in the same commit.

Base: `lane/glm5-b200-int2-20260902` @ `e5d2b0407`. Worktree `../wt-b200-graph`.
**No GPU evidence in this lane.** The rig 5090 is exactness-only and cannot hold the 190.7 GB
artifact; the 2x B200 pair belongs to the spawning session, which runs the gate and the A/B.
Everything below is code, an inventory taken from the source, and a design with its identity
argument. No number in this document is a measurement made by this lane.

---

## 1. The measurement this lane answers (quoted, not re-derived)

nsys, 2x B200 SXM, GLM-5.3-Flash NVFP4, resident PP2, plain decode t=1:

* **~2,900 kernel launches per token**
* **~6 ms/token of launch/gap at ~2.2 us each, inside a ~24 ms token**
* host-side per-layer synchronization for the sigmoid router (`router=sigmoid-host-oracle`)

~25% of the token is spent not executing. That is issue cost, and the only mechanism that
removes issue cost without touching a kernel is graph replay. This family had no graph arm at
all, because the walk was not capturable: **a capture region admits no `cuStreamSynchronize`
and no pageable HtoD**, and the T=1 walk carried 43 of the former and ~156 of the latter per
token.

---

## 2. Host-sync / host-decision / allocation inventory of the T=1 walk

Entry `HybridModel::decode_step_hyper` (`crates/memra-engine/src/hybrid_forward.rs:1954`), and
its PP twin `decode_step_hyper_ppn` (`:3014`). All paths below are in
`/home/avifenesh/projects/wt-b200-graph`. "/layer" multiplies by the per-layer count; the
trunk is 45 hc layers (42 MoE sites, 90 hc glue sites), of which the KDA-mixer layers are the
majority and the MLA/DSA layers the rest.

### 2.1 The blocking ones (what actually forbids capture)

| file:line | what | class | per token |
|---|---|---|---|
| `crates/memra-engine/src/lib.rs:6558-6560` (`Engine::moe_router_sigmoid_topk_host`), reached from `hybrid_forward.rs:12446` → `:10231` | device top-k, then 2 pinned `memcpy_dtoh` and a full **`cuStreamSynchronize`**, so the HOST can compute `base + ex*stride` | **SYNC + DTOH x2** | **42 device-wide drains** |
| `hybrid_forward.rs:12112` (`moe_shexp_add`) | `None => e.htod(&vec![1.0f32; t])` — glm5 has no `ffn_gate_inp_shexp`, so a constant is re-uploaded per MoE layer | **pageable HTOD** | 42 |
| `hybrid_forward.rs:8258` (`mla_attn_cached_pre_wo`) | `e.i32_mirror_store(&mut layer.len_d, ..)` — a synchronizing 4-byte pageable HtoD of the latent length | **HTOD (sync)** | 11 |
| `hybrid_forward.rs:3126` (`hyper_decode_tail`) | `e.dtoh(&logits)` — the end-of-token drain for host sampling | SYNC + DTOH | 1 |

### 2.2 The host-side decisions that make MLA/DSA layers position-dependent

| file:line | what |
|---|---|
| `hybrid_forward.rs:8184` | `let slot = layer.len;` — the host-tracked latent length becomes the append offset, `t_kv`, `n_pools`, and the kpool visibility mask |
| `hybrid_forward.rs:7565` | `mla_append_latent(..., slot, ..)` — a POINTER OFFSET derived from host state |
| `hybrid_forward.rs:7940-7942` | `n_pools = t_kv/pool`, `select_k`, `width = ig.index_width(n_pools)` — **grid width and output size** of the kpool score/select kernels, all from `slot` |
| `hybrid_forward.rs:7891`, `:7905-7917`, `:7971-8000` | host ring geometry, a pools-ready tripwire, and the drain loop that decides on the host how many rows to append before the ring wraps |
| `hybrid_forward.rs:1975-1976`, `:3116-3127` | `cache.pos` → `pos_d` upload; `cache.pos += 1` at the tail |

**This is the whole reason the capture is per-run rather than whole-token.** A captured kpool
node replays last token's grid width. The DSA selection itself never leaves the device
(`hybrid_forward.rs:8016`, `:8031` — `mla_kpool_score` / `mla_kpool_select`, `idx` is never read
back), so the blocker is geometry, not routing.

### 2.3 The KDA mixer, which is clean

`crates/memra-engine/src/kda.rs`: ~13 launches per layer, **zero host syncs, zero DtoH, zero
HtoD**. The conv ring rolls in-kernel (`kda.rs:445`, `kda_conv_silu_decode`); the delta-rule
state is read and written by pointer. The one host action is
`std::mem::swap(&mut rl.ssm_state, &mut rl.ssm_state_alt)` at **`kda.rs:737`**, whose own doc
block at `kda.rs:693-697` says it plainly:

> Stable pointers, no per-step alloc/free ... **NOT capture-safe: a captured graph bakes
> capture-time pointers and never re-runs the host swap, which is why the capture loops refuse.**

That is the one thing this lane had to solve rather than route around. See §4.

### 2.4 Allocations (the ~2,358 `cuMemAllocAsync`/token class)

Not a capture blocker — cudarc allocations inside a capture become mem nodes and the graph
owns them under `AUTO_FREE_ON_LAUNCH` — but they are launch-floor cost and they are why the
captured bodies take a PRIVATE workspace rather than the engine pools. Principal sites:
`hyper.rs:406-409` and `:668` (hc glue, 5 + 1 per site x 90 sites), `hybrid_forward.rs:2176`
and `:2199` (`h`, `z`), `kda.rs:380/422/462/482/487/502/524/543` (~15-17 per KDA layer),
`hybrid_forward.rs:7543-7562` and `:8013-8031` (~15 per MLA layer),
`hybrid_forward.rs:10843-10906` (~6 per expert x 8 per MoE layer). The
`MEMRA_HC_DECODE_WS` door already removes 12-14 per layer.

---

## 3. What was moved to the device

**One thing, and it is the one that mattered:** the routed-MoE selection at t=1.

`moe_ffn_sigmoid_dev` (`hybrid_forward.rs:12695`) is the engine's existing zero-DtoH MoE
consumer, and it is **unreachable on glm5_next**: `sigmoid_resident_dev_eligible`
(`hybrid_forward.rs:9729`) requires `sliding_gated_moe`, and glm5_next's plan has no
`SlidingWindowAttention`, so `decode_batch_program` returns `Generic`
(`crates/memra-gguf/src/execution_manifest.rs:542-556`). Door D
(`MEMRA_MOE_VROWS_DEV_TABLES`) is the right mechanism — it builds the pointer/scale tables on
device with `moe_vrows_tables_from_sel` from the router's own `sel_idx`/`sel_w` — but its
predicate hard-required `t >= 2` (`hybrid_forward.rs:10181`, mirrored at `:10446`), i.e. the
verify walk only.

This lane adds the **T=1 arm**, `vrows_t1_dev` (`hybrid_forward.rs`, in
`moe_ffn_sequential_zq8`), gated by `MEMRA_GLM5_DECODE_GRAPH`:

* the router runs `moe_router_sigmoid_topk` and the selection **stays on device**;
* `moe_vrows_tables_from_sel` builds the `[3*n_pairs]` plane-major tables at `n_pairs = n_used`;
* the layer runs the verify-rows pair `moe_gate_up_preclamp8_q8_rows` + `moe_down8_fma_q8_rows`.

**Identity argument.** The rows twins are the per-row form of the fused epilogue
(`moe_fused_epi_launch`, `hybrid_forward.rs:14149`), which is the per-token form of the
sequential `qmatvec_expert_q8` + `ffn_act_lim` + `axpy_into` chain (`hybrid_forward.rs:10817`
onward): the same g-strided per-pair dots, the same `swiglu_preclamped_mul_scaled_f32`
expression, the same slot-ordered `__fmaf_rn` down accumulation, and the gate/up macro folds
exactly where `ffn_act_lim` folds them with the down macro folded into the accumulate weight
exactly where `axpy_into` folds it. The TABLE VALUES are term-for-term the host loop's:
`base + ex*stride` is exact integer arithmetic on the same base and stride, the macro scales
are the same f32 planes at the same index, and `selw[p] * macro_down[ex]` is one IEEE-754
single multiply of the same two operands in the same order (no FMA contraction is possible in
a bare product). At t=1 this is the **single-row case** of the claim the vrest lane already
gated at t>=2.

**Fail-closed, and one conjunct is load-bearing.** Door D's comment said `promote_worker_h2d`
"needs no conjunct: it requires t == 1 and this arm requires t >= 2". Lifting the t>=2 bound
invalidates that, so `worker_disk_prefetch`/`promote_worker_h2d` are **hoisted above the
router** and `vrows_t1_dev` carries `!promote_worker_h2d` explicitly. It also carries every
host-visible consumer of the selection door D lists (`moesd::capture_active`, `hidden_trace`,
`MEMRA_MOE_TRACE`/`STATS`/`WEIGHT_TRACE`/`INPUT_TRACE_DIR`) plus `sigmoid_router_enabled()`.

The shared expert's constant HtoD (`hybrid_forward.rs:12112`) was **already** solved by door H
(`MEMRA_HTOD_DIET` → `add_scaled_rows_ones`); the decode-graph door **requires** it rather than
duplicating it, and refuses by name when it is off.

Nothing else moved. In particular the MLA/DSA geometry did not — see §7.

---

## 4. The capture design

`crates/memra-engine/src/glm5_decode_graph.rs`.

**Unit of capture: the maximal CONTIGUOUS run of KDA-mixer layers inside a stage's
`[lo, hi)`.** A run's body is the unmodified `hyper_range_decode_ws_body` over that range: hc
pre/post glue, the KDA mixer, the FFN-input norm (with `MEMRA_GLM5_Q8_FUSE` if armed), the
routed MoE on device tables, the shared expert. One body, two modes — the eager walk and the
capture call the same function, so identity is structural rather than two hand-kept copies.

**Key: `(device ordinal, lo, hi)`, plus the `cache.pos` the pool expects next.** A KDA run
carries no position-derived launch parameter, so a run graph is **position-independent**:
captured once per session per stage, replayed every token, never re-captured. The `pos` field
is the invalidation seam: any rollback, reuse-pool retire or prefix restore moves `pos`, and
the pool is dropped and re-captured rather than replayed against state it no longer describes.
The pool lives on the `Cache` (`crates/memra-kv/src/lib.rs`, as `Box<dyn Any + Send>` because
memra-kv must not depend on the engine's cudarc handles) because a run graph bakes **this**
cache's conv-ring and recurrent-state pointers.

**The ping-pong, solved by parity twins.** `kda_cached` (`kda.rs:697-738`) runs the scan
`ssm_state -> ssm_state_alt` and swaps the two owned buffers on the host. Two options existed:
add an in-graph copy-back (an extra launch per KDA layer per token, and a different program),
or capture each run TWICE, once per phase, and alternate the replays. This lane takes the
second: phase p reads exactly the buffer phase p-1 wrote, which is what the eager step does,
at zero extra work. Capturing twice is free of device side effects because **stream capture
records rather than executes** — which is also why the capture uses
`capture_graph_retained_nowarm` (`lib.rs:29229`), whose warmup-free contract exists precisely
for bodies that carry device side effects. The host `ssm_state`/`ssm_state_alt` fields are
mirrored after every replay, so a fall-back to eager, a prime, or a snapshot sees the pointers
it expects.

**One dispatch the door makes on the caller's behalf.** A captured run always runs the
WORKSPACE form of the walk (`hyper_range_decode_ws_body` against the pool's private
`HyperDecodeWs`), whatever `MEMRA_HC_DECODE_WS` says, because a capture needs stable operand
addresses. The two hc walks are kept call-for-call in step and their byte identity is gated by
`crates/memra-engine/tests/hc_decode_ws_gpu.rs`, so this is not a numeric change — but it is a
choice made for the caller, and it is written here and in the FLAGS.md row rather than left to
be discovered from a counter.

**Buffer ownership.**

* `x_io` — a stable `[streams * n_embd]` stream-state buffer the run graphs read and write.
  The hc walk swaps `x` against `ws.xb` twice per layer, so a run of any length makes an EVEN
  number of swaps and leaves its output in `x_io`. The eager segments copy in and out (two
  D2D of ~64 KB per run per token; noted as the first thing to remove if it prices badly).
* a private `HyperDecodeWs` and a private `F16Scratch`, owned by the pool and swapped resident
  around capture AND around every replay — the `PrimeGraph` precedent (`prime_graph.rs:31-35`):
  the graphs bake the f16 scratch's cvt/Lt pointers and an eager GEMM between replays would
  cross-contaminate them.
* while a capture is open, `Engine::vws_recycle*` **retains** into the capture keeper instead
  of returning the buffer to the verify workspace, so nothing a captured body baked can be
  re-issued to eager work (the draft-graph root cause, `capture_graph_retained_flags`' own
  keeper note).

**Refusals, each named once on stderr as `[glm5-decode-graph] eager: <reason>`:** no hc
topology, no sigmoid router, `MEMRA_HTOD_DIET` off, `MEMRA_SIG_ROUTER=0`, any armed
route/hidden observer, the NVMe worker H2D promotion, cudarc event tracking (`MEMRA_EVT` — a
capture region rejects the per-buffer cross-stream event waits), a glm5-TP-sharded layer, a
layer with no recurrent-state slot, or no KDA layer in the range. Plus the **graph-launch
headroom guard** (`spec::GRAPH_LAUNCH_MIN_FREE`): below the driver-free floor `cuGraphLaunch`
segfaults inside libcuda, so the range early-outs to the eager walk with
`[glm5-decode-graph] graph replay suspended:` — this route HAS a byte-identical eager twin, so
it yields rather than failing the request.

---

## 5. The identity argument, stated as one claim

Under the door the walk executes **one numeric program**, not two:

1. every captured kernel is the kernel the eager walk launches, over the same operands, in the
   same order — the capture records the same function (`hyper_range_decode_ws_body`) the eager
   arm runs;
2. the KDA state buffers a replay reads and writes are the buffers the eager step would read
   and write, because the parity twin selects the phase;
3. the one dispatch change is the MoE arm, whose bit-identity to the sequential chain is the
   structural argument in §3;
4. the hc workspace and f16 scratch are byte-fully-overwritten before any read on every step
   (the standing uninit contract), which is what makes a persistent buffer reuse identical.

Per CLAUDE.md's "one numeric program per request" law, graph-vs-eager is a named pair to keep
honest, and the proof belongs in a serving-shape gate rather than a unit test. That gate is §6.

---

## 6. Gate

`crates/memra-engine/src/bin/glm5_decode_graph_gate.rs`, registered as
`glm5-decode-graph-gate`.

```
GLM5_ARTIFACT=<glm5_next safetensors dir> \
MEMRA_PP_DEVICES=0,1 MEMRA_PP_STAGES=2 \
MEMRA_HTOD_DIET=1 \
cargo run --release -p memra-engine --bin glm5-decode-graph-gate -- --steps 64 --reps 5
```

Two arms from one prompt and one artifact: 64 greedy tokens eager, 64 with the door on. It
asserts

* **byte-identical token ids**, and
* **identical per-(token, layer) expert selections and routing weights**, indices compared
  element-wise and weights compared on their BITS (a last-ulp difference is a different
  program, and `total_cmp` would hide a `-0.0`/`0.0` split the expert accumulation does not).

Then it prints per-token milliseconds for both arms, interleaved A/B, N=5, with medians.

The selection comparison needs an instrument that does not itself break capture: every other
route tap in this engine reads the selection back on the host, which is what the door forbids.
`MEMRA_GLM5_GRAPH_SEL_LEDGER` (`crates/memra-engine/src/glm5_sel_ledger.rs`, gate-harness only)
records the device arm with a `memcpy_dtod` into a PERSISTENT per-(device, layer) slot pre-armed
before the capture opens — a memcpy node at a fixed address, no sync, no DtoH, no in-capture
allocation — and the host arm into a plain host vector. It changes no kernel, no operand and no
launch order on either arm.

SCOPE of the selection half under a pp split: a device ledger slot can only be read back
through an Engine on its own device and the gate binary holds the head engine, so the selection
comparison covers the head stage's captured layers, with both arms filtered to that device so
the comparison is like for like. Token identity is unaffected and covers every stage end to
end, and a device-vs-host selection divergence is a per-layer property that would show on the
head stage too. Widening it needs the gate to hold the per-stage engines, which is a gate
change, not an engine one.

**Non-vacuity is enforced**: the gate FAILS if the door never replayed
(`GLM5_DECODE_GRAPH_REPLAYS == 0` or `_LAYERS == 0`) or if the ledger recorded nothing on an
arm. An eager fall-through would make arm B a copy of arm A and the comparison meaningless —
that is exactly the "a PASS is worthless without a red arm and a non-vacuous set" failure this
repo has paid for before.

**The gate has not been run.** It needs the pair and the artifact.

---

## 7. What remains eager, with the file:line that blocks it

**Every MLA/DSA layer.** Blocking sites, in the order they would have to be fixed:

1. `hybrid_forward.rs:8184` — `let slot = layer.len;`. Needs a device-resident latent length
   (the `kvl.len_d` pattern the generic decode graph already uses: `decode.rs:2023`
   `decode_step_dc_cap_masked`, where the kernel reads the actual `t_kv` from a device counter
   and the KV base pointer is baked over the FULL buffer view).
2. `hybrid_forward.rs:7565` — `mla_append_latent(..., slot, ..)` needs a `_dc` twin that takes
   the append slot as a device pointer, and an in-graph counter increment (`Engine::inc_i32`,
   `lib.rs`, is the existing primitive).
3. `hybrid_forward.rs:7940-7942` — `n_pools`, `select_k`, `width`. This is the hard one: they
   are **grid dimensions and output sizes**, not just scalars. Two shapes are available and
   both exist in this tree: bucket the geometry and retarget the exec per bucket via
   `cuGraphExecKernelNodeSetParams_v2` + `cuGraphExecMemsetNodeSetParams` (the step37
   `TokenGraph::retarget_bucket` pattern, `tp.rs:16556`), or size the grid for the bucket
   ceiling and let the kernel derive its own partition from a device length (the `_dc` "one
   partition" law, `decode.rs:2450`).
4. `hybrid_forward.rs:7971-8000` — the kpool drain loop decides on the host how many rows to
   append before the ring wraps. At t=1 it is exactly one iteration, so this is a shape
   question, not an arithmetic one, but it has to be proven rather than assumed.
5. `hybrid_forward.rs:8258` — `i32_mirror_store` on `len_d`; door H's `i32_set_k` async form
   already exists, so this one is a routing change, not a kernel.

Also eager, deliberately: prefill and prime; the decode tail's logits DtoH (host sampling);
the batched / exact-16 decode tier; and **the spec verify walk, which keeps its own path**
(`MEMRA_SPEC_VERIFY_GRAPH`, default ON for the GDN+MoE family since 2026-08-23). Composition
with `MEMRA_GLM5_TP` is refused by name: a sharded KDA layer is not on this path at all.

---

## 8. The SM120 -20% history, and why B200 is a different question

`docs/FLAGS.md`, `MEMRA_STEP_TP_GRAPH`: the step37 whole-token stitched graph was
**re-measured -20% on the current SM120 stack, 2026-08-25** — ctx 4081, interleaved x3, token
ids identical in every arm, eager 75.99/75.98/75.84 vs graph 59.39/61.27/61.09 tok/s. That row
also states its own reason:

> The remainder is dependency LATENCY (405 small serialized kernels that cannot fill the
> device), not launch overhead, so graph replay cannot recover it.

Two things follow, and both matter for this door:

* **The verdict does not transfer as a NO.** It was measured on a different card, a different
  family, and a different remainder. A 405-child graph whose remainder is latency is not the
  same object as a ~2,900-launch token whose census attributes ~6 ms to gap at ~2.2 us each.
* **It does not transfer as a YES either.** The honest comparable is the LAUNCH-FLOOR SHARE:
  what fraction of the token is per-launch floor rather than work. On the B200 pair that share
  is larger for the same model, because the card is faster (each kernel finishes sooner against
  an unchanged per-launch cost) while the launch count is unchanged. That is a reason to expect
  a different sign, not evidence of one.

So: **this door is priced on the pair, with its own interleaved A/B, and inherits nothing from
that row in either direction.** Flip condition, per the serving laws: interleaved x3
fresh-boot A/B on 2x B200 (x5 on anomaly), greedy identity AND the vendor-default sampled
twin, engagement announce in BOTH arms.

---

## 9. Build and check state

Run on the rig, 2026-09-02, `MEMRA_CUDA_ARCH` as noted:

* `cargo fmt --all -- --check` clean.
* `tools/check-flags.sh`: `runtime literal reads=806`, no uncovered runtime names — every
  `MEMRA_*` read resolves against `docs/FLAGS.md`, including both this lane's names.
* `cargo clippy --release --all-targets -- -D warnings` (120a): **clippy-zero**, which took
  four iterations to reach because the run stops at the first failing crate and each fix
  uncovered the next target. Findings, all fixed here rather than disclaimed:
  1. `hyper_ffn_branch`'s argument count — an allow with its reason, since the list mirrors the
     hc site's dispatch contract (this lane's, from the walk edits);
  2. `contains_key`-then-`insert` in the ledger — rewritten through `Entry::Vacant`, which the
     fallible allocation wanted anyway (this lane's);
  3. a complex return type on the gate bin's arm runner — named as `ArmOut` (this lane's);
  4. a manual `% 32 == 0` in `bin/q8_fuse_gate.rs`, a collapsible `if` in `memra-server`'s
     worker, two manual `% 64 == 0` in `bin/b200_matvec_bench.rs`, and five doc-list-indent
     plus two complex-type findings in `bin/hc_fused_gate.rs` — all already on the lane base
     and never reached, because the earlier failures aborted the run first.

  All code in the repo is ours: a finding is not "pre-existing", it is unfixed. Reaching
  clippy-zero is what makes this branch's CI clippy job green, so it belongs in this lane
  whether or not this lane wrote the line.
* `cargo test -p memra-engine --lib` (120a): **367 passed, 0 failed, 3 ignored**, including this
  lane's two run-splitter tests.
* `cargo check -p memra-engine --bins` green on both `MEMRA_CUDA_ARCH=120a` and `100a`.
* **No new CUDA kernel** — the door reuses `moe_vrows_tables_from_sel`,
  `moe_gate_up_preclamp8_q8_rows`, `moe_down8_fma_q8_rows` and the unchanged KDA/hc kernels, so
  there is no `docs/KERNELS.md` row to add. Stated explicitly rather than silently skipped.

## 9b. First box run, and the two defects it found (2026-09-02)

`glm5-decode-graph-gate` on the pair (int2 head `8c31be2f4`, real artifact, PP2 resident,
`--steps 64 --reps 5 --prompt-len 64`) exited `rc=1` after 57 s. Receipt:
`darklanes research/glm5-b200-20260902/box/gates/gate-glm5-decode-graph.txt`. The engagement
half worked exactly as designed:

```
[moe-vrows-dev-tables] engaged ...
[glm5-vrows] verify MoE batched across rows: pairs=8 (t=1 x 8) ...
[glm5-decode-graph] engaged dev=0 stage=[0, 24) runs=6 captured_layers=18 (2 ping-pong phases each; MLA/DSA layers stay eager)
[glm5-decode-graph] engaged dev=1 stage=[24, 45) runs=6 captured_layers=16 (...)
[glm5-decode-graph] re-capture: a captured layer's recurrent-state buffer moved
Error: DriverError(CUDA_ERROR_INVALID_VALUE, "invalid argument")
```

Both stages captured (34 layers across 12 runs) and the first token replayed. Then two defects
fired in sequence, and the second only existed because of the first.

**Defect 1 — the pointer signature was not phase-invariant, so it fired every token.**
`recur_ptrs` recorded the ORDERED triple `(conv, ssm, alt)`. But `kda::kda_cached` swaps
`ssm_state`/`ssm_state_alt` on the host every step, and the replay MIRRORS that swap by design
(§4, the parity twins) — so the live triple differs from the captured one on every single token,
on a session where nothing moved. The re-capture notice in the log is spurious: it fired on step
2. Fixed by recording the ping-pong pair UNORDERED, `(conv, min(ssm, alt), max(ssm, alt))`: a
phase swap cannot change it, a genuine re-seat still does. This is the root cause; nothing else
in the log had to be wrong for the run to die.

**Defect 2 — the re-capture path destroyed a live exec.** `pool.stages.remove(i)` dropped the
`StageGraphs`, which runs `cuGraphExecDestroy`/`cuGraphDestroy` AND frees every buffer the exec
baked (`x_io`, the private hc workspace and f16 scratch, the capture keeper's transients) — with
the previous replay of that same exec still outstanding on the stream. That is a destroy-in-use,
and `CUDA_ERROR_INVALID_VALUE` at the next graph call is how the driver reports it. Fixed by
draining the stream BEFORE the remove, and again after (the drop's frees are stream-ordered too),
so the new capture cannot race them for the same pool addresses. The drop ORDER inside the
structs was already right — `graphs` before `_keeper`, `runs` before `x_io`/`ws`/`f16` — and now
says so in a comment, because Rust field order is the only thing enforcing it.

**Defect 3, by inspection rather than by receipt — a capture failure failed the request.**
`glm5_capture_stage`'s error propagated out of `decode_step`. This route has a byte-identical
eager twin; a failed capture now degrades to it for the whole range, prints once
(`[glm5-decode-graph] eager from here on stage=[lo, hi) dev=N: capture failed (...)`), and
LATCHES the stage off for the session so the next token does not retry and re-thrash the stream.

**The gate could not have caught any of this, so it now forces the path.** Halfway through the
graph arm it RE-SEATS the first recurrent layer — a fresh buffer holding the same bytes, which is
the shape of a snapshot restore or a reuse-pool rehydrate — so identity must hold ACROSS a
re-capture, and the run FAILS as vacuous if no re-capture happened
(`forced_recapture=` in the verdict line). The re-seated layer is the first with a `recur` slot:
KDA layers carry `recur` and MLA/DSA layers carry `latent`, so it is inside a captured run, and
being first it is on the head stage, which is the device the gate's engine owns. The timing arms
run un-perturbed; a forced re-capture is a correctness arm, never a measured configuration.

## 9c. Second box run: the re-capture itself fails, and what is now instrumented

Run 2 (int2 `18df26ad6`, containing `dd05246b9`) died the same way: both stages engaged, one
`re-capture` line, then `CUDA_ERROR_INVALID_VALUE`, rc=1 at ~70 s
(`gate-glm5-decode-graph-2.txt`). Two readings follow, and the second is the actionable one.

**The phase-invariance fix landed.** Run 1 died at 57 s with the re-capture firing on step 2 (the
spurious per-token one). Run 2 got ~13 s further and the re-capture is the gate's own FORCED
re-seat at step 32 — the arm added in `dd05246b9` doing exactly what it was added to do. The
signature no longer fires on a phase swap.

**The re-capture path itself returns INVALID_VALUE, and drain-before-destroy was not sufficient.**
One deduction narrows this a long way: `cudarc`'s `CudaGraph::drop` runs
`cuGraphExecDestroy`/`cuGraphDestroy` and DISCARDS the result, and `CUDA_ERROR_INVALID_VALUE` is
not a sticky error. So the error we see is returned by a call **this code makes and propagates** —
inside the capture block, or one of the synchronizes, allocations or copies around it. It cannot
be the destroy itself failing silently.

So this lane instruments precisely those calls rather than guessing again. `capture_one` is a
deliberate re-implementation of `Engine::capture_graph_retained_nowarm` (same RELAXED mode, same
`AUTO_FREE_ON_LAUNCH` flag, same event-tracking discipline) for one reason: that helper returns a
bare `DriverError` and cannot distinguish `cuStreamBeginCapture` from
`cuStreamEndCapture`+instantiate from `cuGraphUpload`. Every failable step now routes through
`step()` and prints, before propagating:

```
[glm5-decode-graph] capture-error: call=<which> dev=N stage=[lo, hi) run=i/n layers=[a, b)
    phase=p recapture=true|false stream_capture=none|ACTIVE|INVALIDATED free=X/YMB ledger=bool
    err=...
```

Named calls: `pre-begin-status`, `synchronize(before begin_capture)`,
`cuStreamBeginCapture(RELAXED)`, `capture body` (the walk, including the ledger's `memcpy_dtod`
taps), `cuStreamEndCapture+cuGraphInstantiate`, `cuGraphUpload`, `htod_i32(pos_d)`,
`memcpy_dtod(x -> x_io)`, `cuGraphLaunch`. The re-capture decision also prints its own line with
`stale_pos`, `stale_ptr`, `launched_since_sync`, the stream's capture status and free VRAM, and
the rebuild's `engaged` line is NOT deduplicated (a `note` would hide the second one).

**Candidates, and where each one would now show up.**

| candidate | fits INVALID_VALUE? | how it shows / what was done |
|---|---|---|
| `cuStreamBeginCapture` on a stream still ACTIVE, or INVALIDATED by an earlier failed EndCapture | yes, directly | probed BEFORE the begin and refused by name: `call=pre-begin-status stream_capture=ACTIVE`. A failing capture body now still calls EndCapture, so a body error can no longer leave the stream ACTIVE for the next begin |
| `cuGraphInstantiate` against a mempool churned immediately beforehand | yes | **removed as a variable**: a re-capture now REUSES `x_io`, the private hc workspace and the private f16 scratch, so a rebuild allocates and frees nothing of its own against the stream-ordered pool the graphs' alloc nodes draw from, and the rebuilt graphs bake the same addresses |
| graph-owned memory of 12 just-destroyed `AUTO_FREE_ON_LAUNCH` execs not yet reclaimed when 12 new alloc-bearing graphs instantiate | yes, and this is the leading structural hypothesis | would print `call=cuStreamEndCapture+cuGraphInstantiate ... recapture=true`. Next move if so: instantiate the new graphs BEFORE destroying the old (peak VA doubles for one token), or trim graph memory between |
| ledger `memcpy_dtod` tap with a changed destination | no (slots allocated once, fixed size) | would print `call=capture body` with the inner error; `ledger=` is on every line so a run can be re-taken with it off |
| an already-instantiated graph handed back to capture | no | each capture returns a fresh `CudaGraph`; nothing is re-instantiated |
| `cudaGraphExecUpdate` fed an exec from another run | ruled out | this route has no exec-update path; `graph_update::fa_apply` is not on it |
| the ping-pong twin capturing against the moved buffer while the other phase's exec still references the old one | possible | the re-seat's `cuMemFreeAsync` of the old state buffer is issued before the execs are destroyed; the drain now separates them, and `launched_since_sync` reports whether an exec was live |

**And a capture failure no longer takes the request with it** (already in `dd05246b9`): it falls
through to the byte-identical eager walk for the whole range and latches the stage off, so the
next box run should reach the identity compare and the timing arms even if the rebuild still
fails — with the `capture-error:` line naming the call.

## 9d. Third box run: the trigger is not the ping-pong, and the failure is in the teardown

Run 3 (int2 `0ea1b07f6`, containing `711b929da`; the new `engaged ... recapture=false
free=92235/182631MB` line format confirms the binary) printed both `engaged` lines, then the
short `re-capture: a captured layer's recurrent-state buffer moved` note, then
`CUDA_ERROR_INVALID_VALUE`. It printed **no** `capture-error:` line, **no** `forced_recapture=`,
**no** gate re-seat announcement, and **no** long re-capture decision line
(`gate-glm5-decode-graph-3.txt`). Three readings, all from what did NOT print:

1. **The trigger is the engine, not the gate.** The forced re-seat is at step 32 and never
   announced itself, so the walk died at step 2. Something OTHER than the ssm ping-pong moves
   between the first replay and the second on the real artifact. The signature is invariant under
   the swap by construction (unordered pair) and the unit tests hold, so this lane does not yet
   know what moves — which is why the diff is now NAMED rather than reported as "moved".
2. **The failing call is in the TEARDOWN, not in the capture.** The short note prints
   immediately before `synchronize -> stages.remove -> runs.clear() (exec destroy) -> synchronize`,
   and the long decision line prints immediately after. The long line never appeared and no
   `capture-error:` line appeared, so the error comes from one of those four steps — not from
   `capture_one`, which is fully instrumented and stayed silent.
3. **The gate never got to prove anything**, because step 2 dies.

**Response: re-capture is DISABLED, and an invalidated stage latches to eager.** When the pos or
the signature moves, the door prints one line naming the changed element and both pointers
(`sig_diff=layer 7 ssm pair {0x..., 0x...} -> {0x..., 0x...}`, or `conv_state`, or a layer-count
change), pushes the key into the pool's `failed` list, and runs the byte-identical eager walk for
that stage for the rest of the session. The stale stage is deliberately LEFT IN THE POOL:
destroying its execs is the exact call sequence run 3 died in, and the latch means it is never
consulted again; it is released with the cache. This does two things at once — it removes the
failing teardown from the token path entirely, and it lets the box price the FIRST-capture case,
which is the actual product question.

Every fallible call on the door's per-token path now routes through the same named-error wrapper
as the capture: `alloc(x_io)`, `alloc(HyperDecodeWs)`, `alloc(F16Scratch)`, `htod_i32(pos_d)`,
`memcpy_dtod(x -> x_io)`, `cuGraphLaunch`, `alloc(out)`, `memcpy_dtod(x_io -> out)`, plus the
capture calls from §9c. The gate takes `--trace`, which prints one line per token with the door's
replay and capture counters, so a run that dies mid-walk says which step it reached.

**Still open, and it is the next receipt to buy:** what moves in the signature on the real
artifact between step 1 and step 2. The named diff answers it in one line on the next run.

## 9e. Fourth box run: the door runs the whole walk, and the tape is zeros

Run 4 (int8 `fc8cbf593` = `5fbd67562`, `--steps 64 --reps 5 --prompt-len 64 --trace`) is the
first clean run of the mechanism: both stages engaged, **no `sig_diff`, no `capture-error`,
no eager latch**, and `door: replays=564 captures=2 captured_layers=34 forced_recapture=false` —
12 replays per token (6 runs x 2 stages) across 47 decode steps. The conservative latch from
§9d did its job: the walk no longer dies.

**And every token is 0**: `TOKEN MISMATCH step 1: eager=437 graph=0`, `step 2: 444 vs 0`, and so
on. `argmax` of an all-zero logit vector is 0, so the reading is that the captured range's result
never reaches the eager remainder.

**What run 4 rules out.** The trace prints at the TOP of each step, and it reads
`step 1 ... replays=0 captures=0` then `step 2 ... replays=12 captures=2` — so step 1 captured
AND replayed 12 times. The capture step is not skipping its own launch, which was the first
hypothesis. The graphs launch, on every step, without error.

**The defect: the replay's output contract was an ARGUMENT, and arguments about aliased buffers
do not survive a capture.** The hc walk ping-pongs the stream state between `x_io` and `ws.xb`
(two sites per layer, one `mem::swap` each), and the old contract reasoned that an even number of
swaps leaves the answer back in `x_io`, which the replay then copied out. Whatever the parity
actually is on this trunk, the eager remainder was reading a buffer the graph had not written —
and an untouched `e.zeros(width)` reads exactly as the zero logits observed.

**Fix: record the contract instead of arguing it.** The stage now owns a THIRD buffer, `x_out`,
and the captured body ENDS with `copy_into(x_out, live_x)` — one memcpy node, inside the graph.
`x_out` therefore holds that run's output on every replay whatever the parity did, and the replay
copies from `x_out` into the fresh buffer the eager remainder consumes. The reasoning is gone;
the copy is in the graph.

**And an instrument so the next run names the seam rather than the symptom.**
`MEMRA_GLM5_GRAPH_TRACE=1` (armed by `--trace`) prints one line per captured-run boundary per
token, on BOTH arms and at the SAME layer boundaries:

```
[glm5-graph-trace] pos=P dev=N stage=[lo, hi) seg=[a, b) arm=graph-run sum=0x... nz=K/W absmax=...
```

The eager arm reaches those boundaries by splitting its loop at the same run cuts
(`hyper_range_decode_eager_traced`) — a loop split, not a program change: the same kernels in the
same order. `nz=` is the point: an all-zero hidden and a wrong-but-live hidden are identical in a
token stream and obvious here. The gate also prints step 1's top-1 logit INDEX AND VALUE plus the
nonzero count on both arms, so "all-zero logits" and "wrong but valid token" can never again be
confused for each other.

## 9f. Fifth box run: the trace named the seam, and it is not the capture

Run 5 (int10 `c7cdd0f7f` = `6d8725720`) has the same wrong tape, but the `--trace` output from
§9e does its job. On the graph arm, dev=0 stage `[0, 24)` is LATCHED to eager (all its segments
read `arm=eager-run`/`eager-gap`) while dev=1 stage `[24, 45)` replays. The dev=0 segments:

| seg | arm | absmax |
|---|---|---|
| `[0, 3)` | eager-run | 1.58e-1 (sane) |
| `[3, 4)` | eager-gap | 1.9e-1 (sane) |
| `[4, 7)` | eager-run | **1.356879e19** |
| `[7, 8)` | eager-gap | 1.35e19 |
| `[8, 11)` | eager-run | 1.33e19 |
| `[11, 12)` | eager-gap | 1.31e19 |
| `[12, 15)` onward | eager-run | absmax 0 with `nz=16384/16384` — all NaN, constant checksum |

**Two conclusions, and the first one closes a whole branch of this investigation.**

1. **The corruption starts inside an `eager-run` segment — door ON, capture NOT in use.** So it
   is not capture or replay mechanics at all. It is the door's OTHER enabler, the one that runs
   in both modes: the T=1 device-table MoE arm (`vrows_t1_dev` /
   `moe_vrows_tables_from_sel`, `n_pairs = n_used`) that replaces the host router readback. It
   produces 1e19-class values from the first routed layer and NaN five layers later, which is
   what reading the wrong weight bytes looks like.
2. **The true eager arm (door OFF) is sane** because it uses the host oracle. The two arms are
   therefore not one numeric program, which is precisely the claim the door was making.

`nz=16384/16384` with `absmax=0` is the trace earning its keep twice over: a full nonzero count
with a zero absolute maximum is the signature of an all-NaN buffer, not a zeroed one, and the
constant checksum from `[12, 15)` on says the NaN is saturating rather than drifting.

**Two instruments added, both aimed at this specific question.**

* **`MEMRA_GLM5_GRAPH_HOST_MOE=1`** (gate harness): the door stays ON but the device-table MoE
  arm stands down to the host oracle, and the capture then refuses BY NAME (a host readback and
  its stream drain cannot live inside a capture region). One run with this set separates the two
  enablers instead of confounding them: if the tape goes sane, the MoE arm is the defect and the
  capture is clean; if it stays wrong, both are implicated.
* **`glm5_vrows_dev_tables_gpu`** (rig gate, this lane): compares the DEVICE-built tables against
  the HOST arithmetic they claim to reproduce — pointer for pointer, scale BIT for bit — with no
  model and no weights, because the kernel only computes addresses and the bases are never
  dereferenced. That is what makes it runnable on the exactness-only rig instead of burning a
  pair window. Four cases: fixture-scale strides (the control, which passes even under a 32-bit
  product), **serving-scale strides that put the last expert 6 GiB into the gate slab and 12 GiB
  into the down slab** (the serving posture is `MEMRA_MOE_RESIDENT_GB=130`, so any 32-bit step
  in `base + ex*stride` wraps exactly here), a macro-free bank, and a spread selection.

**The rig gate RAN and PASSED, all four cases** (5090, 2026-09-02, under
`flock /tmp/memra-5090.lock`):

```
[small-strides] OK (24 pointers + 24 scales identical)
[serving-strides-past-4GiB] OK (24 pointers + 24 scales identical)
[no-macros] OK (24 pointers + 24 scales identical)
[spread-selection] OK (24 pointers + 24 scales identical)
```

That is a real elimination, not a null result: the DEVICE table build reproduces the host
arithmetic bit for bit **including at strides that put the last expert 6 GiB into the gate slab
and 12 GiB into the down slab**. So the 32-bit-wrap hypothesis is dead, the plane mix-up is dead,
and the macro fold and its per-(layer, plane) mirror are correct. Reading the source agrees: the
kernel uses `long ex` with `long` strides and the launcher passes `i64`.

**What that leaves.** The consumer kernels and their launcher are t-agnostic by inspection —
`moe_gate_up_preclamp8_q8_rows` derives `tok = pr / n_used`, `moe_down8_fma_q8_rows` takes
`tok = blockIdx.y`, and the launcher derives `let t = n_pairs / n_used` for its grid, all of
which are correct at `t = 1`. But that launcher has SEVERAL dispatch twins — the warp-packed
`_w4` pair (`MEMRA_MOE_VROWS_PACK`, and `MEMRA_B200_MATVEC_ARM` on an sm_100a build), the
token-major down arm, and the plain pair — and **only the plain one has ever executed at t=1**,
because the vrows arm was `t >= 2` by construction until this lane. The box binary is an
int-lane build carrying the B200 matvec occupancy lane, so which twin ran is not knowable from
the log as it stands. The `[glm5-vrows]` engagement line now prints
`pack=/dedup_order=/b200_matvec=/dev_tables=` so the next run says it outright.

**Ranked suspects going in**, from the shape of the corruption: a 32-bit step in the pointer
arithmetic (the kernel itself uses `long ex` and `long` strides, and the launcher passes `i64`,
so if this is it the wrap is somewhere else in the chain); a plane mix-up (gate/up table used for
the down projection); the macro fold being dropped or indexed wrong, which on an NVFP4 bank with
per-expert global scales is worth the documented ~3e4x class of error; and the pre-clamp limit
not reaching the device-table path. The gate above discriminates the first three directly.

## 9g. Sixth box run: bisect settles it, and BOTH rig gates clear the arm

Run 6 ran the bisect from §9f, and it is unambiguous:

| arm | result |
|---|---|
| **6A** `MEMRA_GLM5_GRAPH_HOST_MOE=1` (door on, MoE stands down to the host oracle, capture refuses by name) | **ZERO token mismatches**; eager 30.882 vs "graph" 30.894 ms/token (both eager). Gate fails only as VACUOUS (`replays=0`, forced re-seat vacuous) — exactly as designed |
| **6B** `MEMRA_GLM5_GRAPH_HOST_MOE=0` | identical to takes 4 and 5: `TOKEN MISMATCH step 1: eager=437 graph=0` onward |

So the defect is the T=1 device-table MoE arm and nothing else. The capture is clean: with it
refused and only the eager walk running, the tape is correct on the same binary. That is the
cleanest possible statement of where the remaining work is, and it took a purpose-built bisect to
get it rather than another guess.

**And then both rig gates cleared that arm.**

1. `glm5_vrows_dev_tables_gpu` (§9f): device-built tables match the host arithmetic pointer for
   pointer and scale bit for bit, **including at strides past 4 GiB**. Table build clean.
2. `glm5_verify_batch_gpu::gpu_moe_vrows_pairs_match_sequential_chain_bitwise`, with its loop
   **extended from `2..=8` to `1..=8`** by this lane: `gate 4 PASS t=1: 128 outputs
   bit-identical`, on an NVFP4 slab with a live macro plane and a clamp small enough to bite in
   both signs, with both red arms still biting. Kernel pair clean at t=1.

That loop starting at 2 is worth stating plainly: **the vrows pair was `t >= 2` by construction**
(only the spec-verify walk reached it), so the t=1 form of the program had never executed anywhere
until this door routed decode through it. A coverage hole that wide is exactly where a defect
hides — but the hole is now closed and the program is bit-identical there.

**So the arithmetic is exonerated on both halves, and the divergence is in the INPUTS the arm is
handed on the real artifact.** The fixture differs from the serving shape in every dimension that
is not the arithmetic: 16 experts vs 256, in_f 128 / n_ff 64 vs the real widths, no shared expert,
no `gu_il`, no `MEMRA_MOE_RESIDENT_GB=130` slab, no door-E order plane, and a `z` that is a plain
buffer rather than the hc workspace's `ws.z`.

**Instrument added, and it is the cheapest thing that can answer it.** Under
`MEMRA_GLM5_GRAPH_TRACE` and only OUTSIDE a capture region, the T=1 device arm dumps its inputs
for the first two routed layers:

```
[glm5-vrows-t1] dev=N il=L t=1 n_used=8 n_pairs=8 n_expert=E limit=Some(Pre(x)) gu_il=Some(b)
    qtypes=(qg,qu,qd) row_bytes=(...) strides=(...) macros=bool sel=[..] w=[..]
```

Diffing that against the 6A arm's first routed layer pins which input differs. The grep the box
log needs, for the dispatch-arm question raised in §9f, is `[glm5-vrows]` (the engagement line
carrying `pack= dedup_order= b200_matvec= dev_tables=`), and this new one is `[glm5-vrows-t1]`.

## 9h. Takes 7 and 8: the arm's real inputs, and the macro path cleared

Run 7B's dump, from the instrument added in §9g:

```
[glm5-vrows] ... arm doors: pack=false dedup_order=false b200_matvec=false dev_tables=false
[glm5-vrows-t1] dev=0 il=3 t=1 n_used=8 n_pairs=8 n_expert=288 limit=Some(Pre(10.0))
    gu_il=Some(false) qtypes=(7,7,7) row_bytes=(2304,2304,1152)
    strides=(4718592,4718592,4718592) macros=true
    sel=[135, 59, 264, 287, 2, 259, 193, 176] w=[0.7733, 0.4004, ..., 0.1654]
```

**Every field checks out.** `2304 = 4096/2 + 4096/16` and `1152 = 2048/2 + 2048/16` are exactly
NVFP4 at block 16 for `in_f = n_embd = 4096` (gate/up) and `in_f = n_ff = 2048` (down);
`4718592 = 2048 x 2304 = 4096 x 1152` makes both expert strides consistent with their own row
counts; every `sel` is inside `[0, 288)`; the weights descend like a sigmoid top-k with
`route_norm`. So the arm is handed a well-formed problem.

**`dev_tables=false` was a reporting bug, not a finding.** That field printed
`moe_vrows_dev_tables_on()` — the ENV — while the door was routing through the device build
regardless, because `vrows_t1_dev` does not consult that env at all. Take 8 set
`MEMRA_MOE_VROWS_DEV_TABLES=1`, the field flipped to `true`, and the tape was unchanged. The
field now prints THIS CALL's provenance (`matches!(sel, VrowsSel::Dev(..))`) with the env beside
it, so the two can never be confused again. That mis-report cost a box window; it is the reason
a diagnostic must report the decision it claims to report and not a nearby proxy.

**The macro path is cleared, and cleanly.** `HostExps::macro_scale(e)` is
`self.macros.as_ref().map(|m| m[e]).unwrap_or(1.0)` (`model.rs:2922-2924`) — literally `macros[e]`
— so the rig gate's host reference (`g[ex]`) reproduced the shipped host arm exactly, and the
device kernel's `mac_g[ex]` is the same lookup. Together with §9f and §9g that clears the table
values, the macro lookup, and the kernel pair at t=1.

**What is still missing is the other half of the diff.** Run A prints no `[glm5-vrows-t1]` line at
all, because the host-oracle arm had no dump — so there was nothing to compare 7B's numbers
against. Both arms now print the same line with `arm=device` / `arm=host`, and both carry the
per-expert macro scales for the SELECTED experts rather than a `macros=bool`, so a differing
selection or a differing plane is visible rather than inferred. The first differing field on the
next paired run is the answer.

## 9i. Take 9: the two arms receive IDENTICAL inputs, so the seam is inside the call

Runs 9A (host MoE) and 9B (device MoE) both printed `[glm5-vrows-t1]`, and a diff after stripping
`arm=` printed **nothing**: `sel`, `w`, the per-expert macro scales, `strides`, `row_bytes`,
`qtypes`, `limit` and `gu_il` are identical for the dumped layers. 9B still gives
`TOKEN MISMATCH step 1: eager=437 graph=0`.

**So the device-table consumer is handed exactly the host arm's inputs and computes garbage from
the first routed run.** Combined with §9f-§9h that leaves nothing in the inputs and nothing in the
table build or the kernel pair as the fixture exercises them. The remaining differences are things
the rig fixture does not model: 288 experts against a 130 GB resident slab, the shared expert
merged into the same call, `gu_il`, the `z` vs `ws.z` activation source, and the door-E order
plane.

**The addressing hypothesis is now weak and the ordering one is strong.** Every index inside the
pair is bounded by its own row: `(long)o * rb_g` reaches 2047 x 2304 = 4.7 MB, `(size_t)pr * in_f`
reaches 7 x 2048, `act[(size_t)pr * n_ff + o]` reaches 16 383 — none of them 32-bit-sensitive, and
the only large values (the expert base pointers) are u64 from a table this lane already proved
correct. A genuine out-of-range read against a 130 GB slab would also fault rather than return
1.36e19. What fits a blow-up at the FIRST routed layer far better is the consumer reading an
activation that is not this token's hidden state.

**So the activation is now checksummed on both arms, immediately before the launch that reads it**
(`MEMRA_GLM5_GRAPH_TRACE`, outside a capture region, four lines each):

```
[glm5-vrows-act] arm=device|host il=L t=1 z=<sum/nz/max> zq=<sum/nz> zd=<sum/nz/max>
[glm5-vrows-out] arm=device|host il=L out=<sum/nz/max>
```

`z` is the shared f32 input and `zq`/`zd` are the q8_1 pair the consumer actually reads. The
reading is mechanical: **`z` agreeing while `zq`/`zd` differ puts the seam at the quantize**
(stale, unwritten, or ordered after the launch); **all three agreeing while `out` differs puts it
in the kernels** on a shape the fixture does not reach. `nz` carries the same weight it did in
§9e — a never-written buffer reads `nz0`, an all-NaN one reads full `nz` with `max0`.

## 9j. Take 10: the instrument silenced the arm it existed to observe

Run B printed four identical `arm=host il=3` `[glm5-vrows-act]` lines, `out` lines for il=3..6
with sane maxima (2.0 / 0.22 / 0.42 / 0.37), and **not one `arm=device` line** — the arm the
instrument was added for. The tape was wrong as before.

**Two defects in my own instrumentation, both the same mistake.** The budget was a pair of
process-global counters capped at 4. The gate runs its EAGER arm first, so that arm spent the
whole budget before the device arm ever ran; and because the counter was not keyed by layer, the
four lines it did spend were all the same layer. A budget for a two-arm comparison belongs to the
arm, and a per-layer dump has to be keyed by layer or it reprints the first one.

Both are now one mechanism: `glm5_trace_take_slot(kind, arm, il)` over a `(kind, arm, layer)` set,
capped at 8 distinct layers per `(kind, arm)`, with `glm5_trace_reset()` called by the gate at
every arm switch so the second arm starts with a full budget instead of inheriting the first
arm's exhaustion. `out` for il=3..6 then falls out of the keying rather than needing its own rule.

**And a mislabel fixed before it could cost another window.** `moe_vrows_pairs_q8` is ALSO the
spec-verify walk's launcher with HOST-built tables, so the hard-coded `arm="device"` I put there
would have lied on that path. The label now comes from the provenance
(`VrowsSel::Dev` -> `device`, `VrowsSel::Host` -> `vrows-host`), the same fix the `dev_tables=`
field needed in §9h. Twice in three takes a diagnostic reported a nearby proxy instead of the
decision it named; that is the lesson to carry, not the individual bugs.

The device act/out call sites were already the right ones — inside `moe_vrows_pairs_q8`
immediately after the token quantize and immediately after `moe_down8_fma_q8_rows`. They never
fired; nothing about their placement needed to change.

## 9k. Take 11 and the call-site audit: box slots closed, reproduction moves to the rig

Take 11 still prints no `arm=device` act/out line after the per-arm budget fix. **Eleven box
takes is the limit and the coordinator has closed further slots for this door until the rig
reproduces the token-0 tape. That is the right call and this lane records it as such.**

**The audit that was asked for, done and verified.**

* `moe_vrows_pairs_q8` has **exactly one** call site: `hybrid_forward.rs:10799`, inside
  `if vrows_fires`.
* The rows kernel pair (`moe_gate_up_preclamp8_q8_rows` / `moe_down8_fma_q8_rows`) is launched
  from **exactly two** places, both inside `moe_vrows_pairs_q8` (`:14709`, `:14730`). There is no
  second launcher for the door to reach.
* `vrows_dev` and `vrows_fires` share every conjunct beyond the door term, and both carry
  `vrows_t1_dev`, so **`vrows_dev` cannot be true while `vrows_fires` is false** — and the
  `vrows_dev && !vrows_fires` guard would error by name if they ever disagreed.
* The act print sits immediately before the gate/up launch, inside that one call site, labelled
  from the provenance (`VrowsSel::Dev` -> `device`).

So the instrumented site IS the door's only path, and the shape dump and the act dump are guarded
identically. Which leaves exactly two possibilities, and **one free grep on a file already on
disk decides between them**:

```
grep -c 'glm5-vrows-t1.*arm=device' gate-glm5-decode-graph-11B.txt
```

* **> 0** — the device arm does reach `moe_ffn_sequential_zq8`'s device branch, and the act print
  is being suppressed by its own guard rather than by the path. The only guard that can do that
  is `!glm5_graph_capture_open()`: a layer inside a CAPTURED run executes host-side exactly once
  (during capture, where the print is correctly suppressed as illegal) and never again, because a
  graph REPLAY runs no host code at all. The device arm would then be structurally unobservable
  at that site, and the checksum has to move to a capture-legal form or to the eager-latched
  stage.
* **0** — `vrows_t1_dev` is false on the box and the door's MoE never takes the device arm at
  all, which contradicts take 7B (whose `[glm5-vrows-t1]` line carried real `sel` values from the
  device branch, before the `arm=` field existed). The env gained `MEMRA_HYPER_BATCH=1` at take
  8, which is exactly when the device line disappeared — but `hyper_batch_range_decode` is reached
  from `decode_batch.rs`'s batched serving tick, not from the gate's `decode_step`, so that
  should not reroute a single-session walk. If it does, the door is not on the serving path it
  claims and that is the finding.

**Next, on the rig and not the pair:** a synthetic 288-expert hc trunk in the serving shape,
driven through the same door entry the gate's graph arm takes, with the box's real routing
(`/root/out-coact/sel.bin`, `u8 layer, u8 n_sel, n_sel x (u16 expert, f32 w)`), asserting the
per-layer output checksum against the host arm. The two rig gates this lane already has clear the
table build and the kernel pair in isolation; what is missing is a fixture that runs the whole
door path at the serving scale, which is the only thing that can turn this from box-window
archaeology into a debuggable local failure.

## 9l. The serving-shape reproduction PASSES, and that exposes an error in my own bisect

`glm5_vrows_t1_serving_shape_gpu` ran on the rig at the exact serving shape — 288 experts,
`in_f` 4096 / `n_ff` 2048, `row_bytes` 2304/2304/1152, `expert_stride` 4 718 592 (all asserted
against the box dump), three 1.36 GB banks — driven by the box's OWN routing records from
`box/sel-slice-50mb.bin`. Six real selections, host tables vs `moe_vrows_tables_from_sel`, same
kernel pair, same bytes:

```
rec0 layer=3 sel=[42, 7, 246, 19, 169, 85, 71, 204]   host 0x941b6b4f7efcb312  dev 0x941b6b4f7efcb312  diffs 0/4096
rec1 layer=3 sel=[116, 10, 226, 47, 77, 181, 190, 273] host 0x5aa77b7e3c6d95fa  dev 0x5aa77b7e3c6d95fa  diffs 0/4096
... all six bit-identical, absmax 8.7e4 - 1.35e5, nz 4096/4096
```

**The device-table MoE arm is now cleared end to end**: table values (past 4 GiB strides), the
kernel pair at t=1, and the whole consumer at serving scale with real routing. Three independent
rig gates, no diff anywhere.

**And that elimination exposes an error in the bisect I shipped.** `MEMRA_GLM5_GRAPH_HOST_MOE=1`
stands the MoE arm down **and** makes the capture refuse by name. So box run 6 compared *neither
enabler* (6A, correct tape) against *both enablers* (6B, wrong tape). **That is not a bisect**,
and §9g called it one. It could never attribute the defect to one of the two, and I read it as
having attributed it to the MoE arm — which the rig has now shown is clean. The reasoning that
followed from take 6 onward inherited that mistake.

**The missing cell, added:** `MEMRA_GLM5_GRAPH_NO_CAPTURE=1` — the device-table MoE arm engages
exactly as in serving, the capture never happens, the walk runs eagerly.

**Expected result, written down before the run so it cannot be rationalised afterwards: a CORRECT
tape.** That would pin the defect on the capture/replay and exonerate the arm, consistent with all
three rig gates. A wrong tape would instead mean the arm behaves differently in situ than in the
fixture — the remaining difference being the things the fixture does not model: the shared expert
merged into the same call, `ws.z` as the activation source, and the verify-workspace pool
(`vws_uninit*`) supplying `zq`/`zd`/`ptrs`/`scl`/`aq2`/`ad2` instead of fresh buffers — and it
would say so on the first run.

**One box run answers it**, and it is a single env var on a binary that already exists.

**CORRECTION (same day, before any box slot was spent on it): the capture half is NOT structurally
unreachable on the rig.** I first wrote here that it was, and offered the coordinator a choice
between one box env var and "about a day of fixture work". That was wrong, and the wrong half was
mine: I asserted a structural limit without checking the fixture's own config.

What the door actually requires of a model, checked field by field against the mini hc fixture
`hc_decode_ws_gpu.rs` already builds:

| door conjunct | mini fixture today | gap? |
|---|---|---|
| glm5 config with a sigmoid router | `"model_type": "glm5_next_text"` | no |
| `Pre` clamp, `l > 1e-6` | `"swiglu_limit": 1e30`, and `pre_if_live` is `(limit > 1e-6).then_some(Pre(limit))` (`config.rs:2694`) | no |
| `n_used <= 8` | `num_experts_per_tok: 2` | no |
| uniform expert layout | deterministic fixture, uniform by construction | no |
| `slab_bases.is_some()` (resident `dev_exps`) | `build_dev_exps` uploads when the residency budget covers the expert bytes; this fixture's experts are a few KB, so the budget covers them trivially (`hybrid.rs:520`) | no |
| `moe_q8` (`q8_expert_supported` on all three planes) | fixture expert tensors are f32 | **YES — the only gap** |

So the door is one tensor dtype away from admitting the existing mini fixture: the expert planes
have to be NVFP4 (or another q8-eligible type) instead of f32, and
`memra_gguf::nvfp4_repack::f32_to_nvfp4` — already used by `glm5_verify_batch_gpu`'s slab builder —
is the conversion. That is a bounded change to the fixture's tensor source, not a day of work and
not a structural wall.

**Which means the capture half is reproducible on the rig, and that is now this lane's next task
rather than a request for a box slot.** `MEMRA_GLM5_GRAPH_NO_CAPTURE` remains the cheaper answer
if a slot is going spare, but it is no longer the only way to get one, and no box time should be
spent on this door on my account until the rig has had its turn.

## 9m. The capture half runs on the rig, and take 5's own trace already named the defect

Two things closed on 2026-09-03, one measured and one read back off a box log that had been in
hand since take 5.

**The capture half is no longer unreachable on the rig.** `crates/memra-engine/tests/
glm5_decode_graph_capture_gpu.rs` is a glm5_next fixture whose routed-expert planes are emitted
as NVFP4 (`memra_gguf::nvfp4_repack::f32_to_nvfp4`), which was the single field that made the
door refuse the shared hc fixture by name. Everything else the door needs the fixture already
satisfied. The gate drives 16 real decode steps with the door OFF and then ON and compares the
per-step logits `to_bits`, and it asserts NON-VACUITY on the door's own counters rather than
inferring engagement from a green diff. First green run, single device, 2-layer geometry:

```
[moe-vrows-dev-tables] engaged: ... (MEMRA_MOE_VROWS_DEV_TABLES=1)
[glm5-vrows] ... dev_tables=true (env MEMRA_MOE_VROWS_DEV_TABLES=false)
[glm5-decode-graph] engaged dev=0 stage=[0, 2) runs=1 captured_layers=2 recapture=false
                    free=22118/23983MB
door: captures=1 replays=16 captured_layers=2
graph door: 16 steps bit-identical to eager (16 replays)
test result: ok. 1 passed
```

So a walk that captures and replays, with the T=1 device-table MoE arm live inside the captured
body, is byte-identical to the eager walk. That is the first direct evidence for the capture
half, and it does not reproduce the box tape.

**Take 5's trace already said the graph was not involved.** The line that named the seam was
`seg=[4,7) eager-run absmax=1.356879e19`. The label matters and I did not read it: `eager-run`
is printed by `hyper_range_decode_eager_traced` (`glm5_decode_graph.rs:634`), and
`hyper_range_decode` (`hybrid_forward.rs:2168-2176`) reaches that function ONLY when the graph
arm was not taken — the door tries `glm5_decode_graph_ready` first and returns from
`hyper_range_decode_graphed` when it succeeds. A `graph-run` / `graph-gap` label is what the
captured path prints. So the 1.356879e19 blow-up happened on a walk with no capture and no
replay anywhere in it, and the tape that eleven takes read as a capture/replay defect was the
device-table MoE arm running on its own.

The reason it could run on its own is the defect: `vrows_t1_dev` was keyed on
`crate::glm5_decode_graph_on()`, the door's ENV. That made the door change the program on every
path where it captures nothing — the eager fall-through, a latched stage, a refused stage, and
the `MEMRA_GLM5_GRAPH_TRACE` split walk — which breaks the door's own stated contract that
every refusal falls through byte-identically, and it confounded the door's two enablers so
thoroughly that no box run could attribute a mismatch to either.

**The fix** (`hybrid_forward.rs`, the `vrows_t1_dev` predicate) keys the arm on
`crate::glm5_graph_capture_open()` instead. Inside a capture region is the only place the arm is
required at all: a host sel/w readback issues a `cuStreamSynchronize`, which is illegal there.
Outside one, nothing needs it, and the walk now runs the shipped host-oracle program. The
captured body still gets the device tables. `MEMRA_GLM5_VROWS_T1_DEV=1` (default OFF, gate
harness, FLAGS.md row in this commit) forces the arm on with no capture and no graph anywhere,
which is the bisect cell takes 4 through 11 never had.

What this does NOT settle: whether the device-table arm is exact on the real artifact at
serving scale in situ. The rig has cleared it four ways now (table values past 4 GiB strides,
the kernel pair at t=1, the whole consumer at 288 experts / `in_f` 4096 / stride 4718592 driven
by the box's own routing dump, and now a full capture-and-replay decode), and take 5 says it
blew up on the box. Those disagree, and the one-env-var run that separates them is
`MEMRA_GLM5_VROWS_T1_DEV=1` on a plain decode — no door, no graph.

## 9n. ROOT CAUSE, FOUND AND FIXED: the ping-pong phase was per stage, not per run

The rig fixture of §9m passed because it had ONE captured run. Scaling it to the box's geometry
found it in the first run: 8 layers, `deepseek_sparse_attention` at 3 and 7 (so `kda_runs` splits
into `[0,3)` and `[4,7)`, the box's own shape), 64 routed experts, `num_experts_per_tok` 8. All
16 decode steps diverged from the eager walk, starting at the capture step.

`MEMRA_GLM5_GRAPH_TRACE=1` then put the seam on one screen. At `pos=8`, the capture step:

```
seg=[0, 3) arm=eager-run  sum=0x2a150bb5433fcaf3 absmax=1.296062e0
seg=[3, 4) arm=eager-gap  sum=0x006822bfb749ebce absmax=1.360538e0
seg=[4, 7) arm=eager-run  sum=0x7a60c30243421f21 absmax=1.595194e0
seg=[7, 8) arm=eager-gap  sum=0xab782467d8743d5a absmax=1.810762e0
seg=[0, 3) arm=graph-run  sum=0x2a150bb5433fcaf3 absmax=1.296062e0   <- IDENTICAL
seg=[3, 4) arm=graph-gap  sum=0x006822bfb749ebce absmax=1.360538e0   <- IDENTICAL
seg=[4, 7) arm=graph-run  sum=0x9f3de7b49a145df8 absmax=1.595809e0   <- DIVERGES
seg=[7, 8) arm=graph-gap  sum=0x5e3aaf8096f1a027 absmax=1.815865e0   <- downstream
```

The FIRST captured run replays byte-identically. The SECOND does not. That is not a capture
binding, a stale pointer or a table value; it is an index.

**The defect.** `phase` lived on `StageGraphs` and `glm5_replay_run` flipped it after every
launch. But the ping-pong parity it selects is a property of the TOKEN, not of the run:
`kda::kda_cached` swaps `ssm_state`/`ssm_state_alt` once per KDA layer per token, so every run of
a stage sits at the same parity and each advances by one per token. Flipping a single stage-level
counter once per run made run 0 replay phase 0 (correct), run 1 replay phase 1 while its layers
were still at phase 0, run 2 phase 0 again, and so on. A wrong-phase graph reads the buffer the
eager walk WROTE this token and writes the one it READ: a plausible wrong answer, never a crash,
never an error.

It is invisible at `runs=1`, which is exactly why §9m's 2-layer fixture was green through 16
steps. On the box's `runs=6` stages (`[0,24)` on dev 0, `[24,45)` on dev 1) every other run was
wrong from token 0 on both devices, which compounds to the constant `graph=0` tape takes 4
through 12 all reported.

**The fix.** `phase` moved into `RunGraph`, initialised to 0 at capture (capture's two passes
leave the host fields where they started) and advanced per run in `glm5_replay_run`. The rig
gate, which failed 16/16 steps before it, now reports `graph door: 16 steps bit-identical to
eager (32 replays)`.

### What else landed with it

1. **The device-table MoE arm is keyed on an OPEN CAPTURE**, not on the door env (§9m). It is
   required only inside a capture region and nowhere else, and keying it on the env made the
   door change the program on paths where it captures nothing. `MEMRA_GLM5_VROWS_T1_DEV=1`
   (default OFF, gate harness) forces it on with no graph, which is the bisect cell takes 4-12
   never had. Running the rig gate with it set changed NOTHING on either arm, bit for bit: the
   device-table and host-oracle table provenances are identical in situ, which is the fifth
   independent clearance of that arm and the reason the phase index was the only thing left.

2. **`moe_router_sigmoid_topk_host` now refuses inside a capture region** (`lib.rs`). Under
   `cudaStreamCaptureModeRelaxed` its DtoH is RECORDED, not executed: the call returns success,
   the pinned stage keeps whatever bytes it held, and the caller routes on uninitialised memory
   whose expert ids index the slab out of range, with the garbage pointers baked into the graph.
   Nothing in the twelve box logs could have distinguished that from the phase defect. It is now
   a named capture failure and the door's contract sends the token down the eager walk.

3. **`[glm5-vrows-t1-deny]`** prints, once per layer under the door or the trace flag, every
   conjunct of the T=1 arm's predicate when it stands down. Twelve takes could not tell "the arm
   ran and was wrong" from "the arm never fired", because the only evidence was an `arm=host`
   label that BOTH the sequential loop and the verify-rows host arm can print. (`arm=host` is the
   SEQUENTIAL loop; the verify-rows host arm prints `arm=vrows-host`. Reading the two as one cost
   several takes.)

4. **The selection ledger pre-arms outside capture.** `record_device` skips a layer with no
   persistent slot and the only `prearm` caller was `glm5_capture_stage`, so any cell that ran the
   device arm WITHOUT capturing recorded zero device rows — which is why box take 12's
   `MEMRA_GLM5_GRAPH_NO_CAPTURE=1` run reported `VACUOUS: the selection ledger recorded no rows on
   one of the arms` (21 host rows against 0 device rows). That was the instrument failing, not the
   door. `MEMRA_GLM5_GRAPH_NO_CAPTURE=1` also now forces the arm on, so the cell tests what its
   name says.

### The gate that would have caught it on day one

`crates/memra-engine/tests/glm5_decode_graph_capture_gpu.rs`. The property that matters is not
"a captured walk matches eager" — it is "a captured walk with MORE THAN ONE RUN PER STAGE matches
eager". The fixture's `layer_types` puts `deepseek_sparse_attention` at 3 and 7 for exactly that
reason, and its geometry is a lane invariant, not a detail: a single-run fixture is vacuous
against this defect class and passed 16 steps while the box was producing zeros. The red arm is
recorded and reproducible: revert `RunGraph::phase` to a stage field and the gate fails 16/16
steps with run `[0,3)` still byte-identical.

## 9o. Take 13 prediction, stated BEFORE the A/B, and what to grep

Written ahead of the numbers so it can be wrong on the record.

### What the door removes

The captured span is 34 of the 45 trunk layers (box take 12: `runs=6 captured_layers=18` on
dev 0's `[0, 24)`, `captured_layers=16` on dev 1's `[24, 45)`). Every launch in those layers is
replaced by ONE `cuGraphLaunch` per run, so about 11-12 graph launches per token in total.

What stays EAGER, by construction:

* the 11 MLA/DSA layers (§2.2, §7): their kpool grid width, `select_k` and output size are all
  derived on the host from `layer.len`, so a captured node would replay last token's geometry;
* the hc expand at the trunk entry and the collapse at the tail;
* `pos_d` (one `htod_i32` per stage per token);
* the PP hop between stage 0 and stage 1;
* the tail `e.dtoh(&logits)` and host sampling (`hybrid_forward.rs:3126`), which is one
  device-wide drain per token and is NOT removable by this door.

### The arithmetic

Coordinator's baseline: ~2,400 launches/token at 50.8 tok/s = 19.69 ms/token. At the nsys-
measured ~2.2 us of launch/gap each (§1), issue cost is ~5.28 ms, about 27% of the token.

The captured 34 layers do NOT hold 34/45 of the launches, because an MLA/DSA layer carries the
kpool machinery and a KDA layer's mixer is ~13 launches (§2.3). Bracketing it: if an MLA layer
costs the same as a KDA layer the captured fraction is 76%; if it costs twice as much,
34/(34 + 2x11) = 61%. So 1,460-1,820 launches move inside graphs, worth 3.2-4.0 ms of gap.

Graph replay is not free either. A replayed node still costs driver dispatch, so recovery is
~70-85% of the removed gap: **2.2-3.4 ms off a 19.69 ms token, i.e. 16.3-17.5 ms, i.e. 57-61
tok/s, i.e. +13% to +21%.** Central call: **~59 tok/s.**

NOT priced in, and pure upside: the door also removes 31 of the 42 per-token device-wide
`cuStreamSynchronize` drains (the 42 MoE sites minus the 11 on MLA/DSA layers, which stay
eager), because the device-table MoE arm inside the captured body never reads the selection back.
A drain on a 2-card PP2 pipeline costs more than gap arithmetic charges it. If take 13 lands
above +21% that is where it came from.

**Floor case, and the falsifier:** if B200 kernels at this shape are big enough to hide issue
cost, the gain collapses to +3-5%. So: if `[glm5-decode-graph] engaged` prints on both stages
with ~12 replays/token and tok/s moves less than 3%, the 6 ms gap was not on the critical path
on this silicon, and the door is a concurrency-scaling lever rather than a single-stream one.
That is a real outcome, not a failure, and it should be recorded as one.

### Composition with the best posture, resolved flag by flag

| posture name | resolves to | where it runs | captured? |
|---|---|---|---|
| `KDA_FUSED_PROJ=1` | `MEMRA_KDA_FUSED_PROJ` (`kda.rs:1244`) | INSIDE the captured body | yes, as-is. Pure kernel-group selection, no sync, no HtoD, no host geometry. Declines on TP shards and outside `t in 1..=15`; t=1 decode qualifies |
| `MATVEC_ARM=1` | `MEMRA_B200_MATVEC_ARM` (`lib.rs:778`) | inside, `OnceLock`-cached | yes, as-is. Bit-identical per-output kernel twins, `sm_100a` builds only |
| `Q8_FUSE=1` | `MEMRA_GLM5_Q8_FUSE` (`lib.rs:262`) | inside, `OnceLock`-cached | yes, as-is |
| `HC_FUSED_PRE=2` | `MEMRA_HC_FUSED_PRE` (`hyper.rs:318`) | INSIDE (2 hc sites per layer) | **the reader is `== Ok("1")`, so the value `2` is OFF on this branch.** Either the runner maps it, or the best posture has been running without the fused pre-chain. Worth settling before the A/B is read as a posture number |
| `DSA_DECODE=2` | most likely `MEMRA_B200_MLA_DECODE_ARM` | the MLA/DSA layers, which are the door's EAGER GAPS | composes trivially: the door never captures those layers, so the two are disjoint |
| `PRIME_V2=1` | a prime-path door | prime only | no decode interaction |
| `W8=1` | `MEMRA_STEP_TP_W8` is the only `W8` door in the tree | step37 TP attention projections | names a STEP37 path. Verify it engages at all on glm5_next before crediting it in the posture |
| `GEMV_V2=1`, `BGEMM_PAD_RATIO=1.0` | do not resolve to any `MEMRA_*` reader in this branch | unknown | resolve from the runner before crediting them |

Resolve the last three with
`grep -oE 'MEMRA_[A-Z0-9_]+' /root/lane/graphgate13.sh | sort -u` against
`grep -oE 'MEMRA_[A-Z0-9_]+' crates/memra-engine/src/*.rs | sort -u`.

The general rule, and it is now enforced rather than argued: a flag is capture-compatible iff its
arm adds no host sync, no pageable HtoD, no geometry derived from `cache.pos`, and no buffer it
reallocates per token. Everything in the posture above is kernel selection, which is none of
those. The global hazards (`MEMRA_HTOD_DIET` off, `MEMRA_SIG_ROUTER=0`, the observation modes,
event tracking, TP sharding) are refused BY NAME in `glm5_decode_graph_refusal`, and the two new
guards catch the rest.

### What to grep in the serving log

```sh
grep -c '\[glm5-decode-graph\] engaged'                    # expect 2, one per PP stage
grep    '\[glm5-decode-graph\] eager-latch'                # expect ZERO; a hit names sig_diff
grep    '\[glm5-decode-graph\] eager: '                    # the refusal reasons, by name
grep    'capture failed'                                    # a named capture failure
grep    'glm5-vrows-t1-deny.*capture_open=true'             # MUST be empty
grep    'reached inside an open CUDA graph capture'         # the new router guard
grep    '\[kda-fused-proj\] DECLINED'                      # composition decline
```

Line 4 (`capture_open=true` on a deny) and line 5 (the router guard) are the two that did not
exist for takes 1 through 12 and are the reason a thirteenth take should not need a fourteenth:
either of them names the exact conjunct that failed, on the first line it prints.

## 9p. Box take 13: green, and the split that changes what this door is for

2x B200, 2026-09-03, `c79dc5230`, 64 steps x 5 reps interleaved. **Zero token mismatches on both
runs.** Receipts: `research/glm5-b200-20260902/box/graph13/gate-13{N,B}.{txt,full}` (darklanes).

| run | door | receipt | eager ms/token | graph ms/token | delta |
|---|---|---|---|---|---|
| B | full | `replays=564 captures=2 captured_layers=34 forced_recapture=false` | 30.655 | 21.068 | **-31.28%** |
| N | `NO_CAPTURE=1`, device arm forced on | `replays=0 captures=0 captured_layers=0` | 30.567 | 21.477 | -29.74% |

**The split is the finding, and it is not what this lane was built to prove.** Removing the
per-MoE-layer host router readback and its device-wide drain is ~93% of the prize. The capture and
replay on top of it buys 21.48 to 21.07 ms, about 2%. Twelve takes were spent debugging the 2%.

Two consequences:

1. The FLAGS row now says this plainly. The door's value is the drains it removes; the replay is
   the small tail.
2. The device-table MoE arm is worth considering as its own serving flag on ranges where capture
   refuses. That is a separate decision, it is NOT made by this lane, and it would need its own
   acceptance (the arm is currently reachable only inside a capture region or through
   `MEMRA_GLM5_VROWS_T1_DEV`, which is a gate knob and says so).

My §9o prediction called +13% to +21% with a central ~59 tok/s, from launch-gap arithmetic alone,
and explicitly parked the drain removal as unpriced upside. The measured -31.28% is above that
band, and the run N row says why: the drains WERE the prize and the gap arithmetic was the small
half. The prediction was right about direction and wrong about mechanism, which is the more
useful way to be wrong, and it is recorded rather than quietly restated.

### The two failing gate arms, both instrument defects, both fixed

**1. `VACUOUS RE-CAPTURE ARM`.** The arm asserts that a forced re-seat makes the pool re-capture,
but re-capture had been DISABLED since box run 3 (§9d), so the arm was asserting on a path the
engine deliberately does not take. Fixed on both sides:

* `MEMRA_GLM5_GRAPH_RECAPTURE=1` (default OFF, FLAGS row in the same commit) chooses REBUILD over
  latch on an invalidated stage, with the ordering box run 3 got wrong: drain the stream, THEN
  drop the stage, and refuse to drop at all if the drain fails.
* The rig gate now takes that receipt with no box slot. It replaces a captured layer's `ssm_state`
  with a fresh allocation holding the same bytes, and the engine names what moved:

  ```
  [gate] reseat il=0 ssm 0x304af5000 -> 0x304bafe00 (alt 0x304ae5000)
  [glm5-decode-graph] re-capture dev=0 stage=[0, 8) pos=16 expected_pos=16 stale_pos=false
                      launched_since_sync=true stream_capture=none free=23417/23983MB
                      sig_diff=layer 0 ssm pair {0x304ae5000, 0x304af5000} -> {0x304ae5000, 0x304bafe00}
  re-capture arm: 16 steps bit-identical across 1 forced rebuilds
  ```

  No `CUDA_ERROR_INVALID_VALUE`. The teardown works when the drain precedes the drop.
* `glm5-decode-graph-gate` arms the knob for its own forced-re-seat arm, so a box run takes the
  sm_100a receipt without an env change, and it now counts `GLM5_DECODE_GRAPH_RECAPTURES` rather
  than a capture delta (a latched stage leaves the capture counter alone too, so the delta could
  not tell "rebuilt" from "latched").

The DEFAULT stays latch, and that is deliberate: an invalidated stage falls through to the
byte-identical eager walk, so the only cost of not rebuilding is that one stage of one session
stops being graphed. Nothing is wrong, only slower. The knob exists to price the rebuild, not
because the latch is a workaround.

**2. `step 0: ledger row count 21 (eager) != 6 (graph)`.** Also the instrument. The eager arm
produces only HOST rows: 21 on the head stage, which is 24 layers minus the 3 dense ones. The
graph arm produces BOTH, because the captured KDA runs route through the device-table arm into the
persistent device slots while the eager MLA/DSA gap layers still read their selection back on the
host: 15 device rows plus 6 host rows, which is the same 21. The gate said
`if !host.is_empty() { step_rows = host; }`, which threw the 15 device rows away the moment a
single gap layer pushed a host row. It now MERGES by `(dev, layer)`, and drops any device slot
still carrying its `-1` sentinel so a wiring miss shows as a MISSING row rather than an accidental
match.

Neither arm was ever a token defect. Take 13's token tape was correct on both runs.

## 9q. The serving A/B measured the door's ABSENCE, and why

Take 13's serving A/B ran six boots and every one printed `graph-lines: 0 latch: 0`: zero
`[glm5-decode-graph]` lines of any kind, refusals included, on the graph-on boots as well as the
graph-off ones. Decode was flat across arms (43.7/43.2/42.6 code and 43.2/43.3/43.2 prose
graph-off, against 43.4/43.1/43.8 and 42.5/43.1/43.2 graph-on).

That is not the falsifier of §9o firing. It is the door never running.

**The missing conjunct is the WALK.** `MEMRA_GLM5_DECODE_GRAPH` is wired into
`hybrid_forward::hyper_range_decode` (`hybrid_forward.rs:2168`), the per-session SERIAL hc walk
that `decode_step_hyper` and `decode_step_hyper_ppn` run. Serving with `MEMRA_HYPER_BATCH=1`
routes every session, **including B=1**, through `decode_batch::decode_step_batch_hyper`
(`decode_batch.rs:2702`), whose trunk is `decode_batch_layers`. Grep is the whole proof: every
caller of `hyper_range_decode` is in `hybrid_forward.rs` (lines 2195 and 3636-3695), and
`decode_batch.rs` contains none. The gate binary calls `decode_step`, which is why it armed the
door fine at `replays=564` on the same binary in the same hour.

Zero lines including refusals was the tell, and I should have read it as one earlier: a refusal
prints from `glm5_decode_graph_ready`, which is only reached once `glm5_decode_graph_on()` is
true AND the walk calls `hyper_range_decode`. Silence means the call never happened.

`decode_step_batch_hyper` now says so, once per process, when the door is set on the batched
walk. A door that silently does nothing in serving is the same failure class as the take 12
vacuity, and it gets the same treatment: name it in the log rather than let silence be read as a
refusal.

Extending capture to the batched walk is a separate lane. Its trunk has its own per-row geometry
and the two walks would need their own identity gate against each other.

## 9r. Composition audit, before any default flip

Sibling lane, 2026-09-03: #114's default flip was withdrawn because turning it ON made another
door's branch slice past its buffer and panic the worker, and because a ring was read
cross-stream with event tracking disabled. Neither is catchable by a token tape. This door binds
buffers at capture time across two PP stages, so the same audit is owed here. What follows
separates what was CHECKED from what could not be.

### Refused BY NAME, so composition cannot arise

`MEMRA_GLM5_TP` (a sharded KDA layer is not on this path), `MEMRA_SIG_ROUTER=0`,
`MEMRA_HTOD_DIET` unset, `MEMRA_EVT` (cudarc event tracking: capture refuses cross-stream waits,
which is exactly #114's second failure mode), the NVMe worker H2D promotion, `moesd` capture,
`hidden_trace`, and the observer envs `MEMRA_MOE_STATS` / `MOE_TRACE` / `MOE_WEIGHT_TRACE` /
`MOE_INPUT_TRACE_DIR` / `SIG_ROUTER_LOGIT_TRACE`.

**`MEMRA_MOE_SEL_DUMP` was added to that list in this lane**, and it is the one the rebase found
rather than the audit. Main's #113 landed a device-arm selection dump at the exact site this lane
edits, and it does a DtoH pair per layer-call on the DEVICE arm, which is the arm a captured body
runs. Inside a capture that DtoH is recorded and not executed, so the dump would have written
stale rows and the door would have looked fine. Refused by name now.

### Checked by reading, on the captured path

Every door read inside the captured body (`hyper.rs` and `kda.rs`, which are the glue and the
mixer, plus the MoE dispatch):

| door | verdict |
|---|---|
| `MEMRA_KDA_FUSED_PROJ` | kernel-group selection only. Declines on TP shards and outside `t in 1..=15`. Capture-clean. |
| `MEMRA_HC_FUSED_PRE` | fuses three site kernels into one launch. No sync, no HtoD, no host geometry. Capture-clean. Note the reader is `== Ok("1")`, so the posture's `HC_FUSED_PRE=2` is OFF. |
| `MEMRA_GLM5_Q8_FUSE` | `OnceLock` bool, kernel selection. Capture-clean. |
| `MEMRA_GLM5_W8` + `MEMRA_B200_GEMV_V2` | the fused six-projection q8 twin. Reads pre-existing mirrored weight ranges; **it bounds-checks rather than slices** (`x.len() < t * in_f` returns `Ok(None)`), which is #114's first failure mode guarded at the site. Capture-clean by reading. |
| `MEMRA_HC_DECODE_WS` | the door FORCES the workspace walk inside capture whatever this says, because a capture needs stable operand addresses. Stated in the FLAGS row rather than left to a counter. |
| `MEMRA_MOE_FUSED_EPI` | on the MoE path, but under the door the vrows arm fires instead, so the fused epilogue does not dispatch at t=1. |
| `MEMRA_KDA_CHUNKED` / `MEMRA_GDN_CHUNKED` | t>1 arms; the captured walk is t=1 only. |

### Could NOT be checked, and this is the honest half

* **Every `sm_100a`-gated door is structurally OFF on the rig.** `b200_gemv_v2_level()` and
  `b200_matvec_arm_on()` both return early unless `env!("MEMRA_BUILT_CUDA_ARCH") == "100a"`, which
  a 5090 build never is. So `MEMRA_B200_GEMV_V2`, `MEMRA_B200_MATVEC_ARM`,
  `MEMRA_B200_MLA_DECODE_ARM` and `MEMRA_B200_DSA_DECODE` have been read but never RUN under a
  capture. My verdicts on them are code reading, not receipts.
* **The batched walk** (§9q). The door does not reach it, so their composition is undefined rather
  than verified.
* **The whole serving posture together.** The one A/B that would have exercised it did not engage
  the door, and the hardware is gone: the vast.ai account is out of credit (`billing_creditonly:
  1`, so auto-topup never charges a card) and every instance including the B200 pair was stopped
  on 2026-09-03.

**Therefore: no default flip is proposed, and none should be accepted, on this evidence.** The
door stays OFF. What it has is a green gate on the serial walk and a bit-identity receipt on the
rig; what it does not have is a single serving token that went through it.

## 10. Open items

1. **Run the gate on the pair** (`--steps 64 --reps 5`) and bank the receipt. Until then the
   door is code with an argument, not a measured mechanism.
2. **Price the per-run copy-in/copy-out.** One D2D pair per run per token; if the trunk splits
   into ~11 runs per stage the copies are ~22 x 64 KB. The removal is to have the eager segment
   preceding a run write its output directly into `x_io`.
3. **Price the parity twins' VRAM.** Each run is captured twice, and each capture's body
   allocates its own transients as graph mem nodes, so the reserved transient footprint of a
   stage is doubled relative to one capture. On the 2x192 GB pair that is expected to be noise
   against the expert slabs, but it is a real number and it has not been measured. If it ever
   bites, the alternative is a single capture plus an in-graph `alt -> state` copy-back (one
   extra D2D per KDA layer per token, byte-identical, a different launch count) — priced
   against, not assumed better.
4. **Node census** (`MEMRA_GRAPH_CENSUS` / `graph_update::node_census`) on a captured run: if
   the ALLOC/FREE node count is zero, instantiate with `USE_NODE_PRIORITY` instead of
   `AUTO_FREE_ON_LAUNCH` (the auto-free flag's launch-time mem-pool scan measured ~0.25 us/node
   even with nothing to free — `capture_graph_retained_flags`' own doc).
5. **The MLA/DSA `_dc` arc** (§7). That is what turns per-run capture into whole-token capture
   and is the larger remaining prize; it is a kernel lane, not a wiring lane.
6. **`decode_graph_support`** (`crates/memra-gguf/src/execution_manifest.rs:580`) still omits
   `HyperConnections`/KDA/MLA, so `DECODE_GRAPH.trunk_capabilities(plan).cuda_graph.supported`
   is false for glm5_next and the plan-level rewrite stays unqualified. Widening that table is
   a receipt-backed change and belongs AFTER the gate is green on the pair, not before.
7. **Teardown context.** The pool lives on the `Cache`, and under a pp split it holds graphs
   created in more than one CUDA context. When the `Cache` drops, those destructors run under
   whichever context is current, which is not necessarily each graph's own. Not the error the box
   hit (a re-capture destroys stage s's graphs while stage s's context is current, by
   construction), but it is a real teardown hazard and wants either a per-context drop guard or
   an explicit release before the cache is dropped.
8. **Re-enable re-capture** once the named `sig_diff` says what moves and the teardown's failing
   call is identified. Until then the door prices first-capture only, and any invalidation is an
   eager latch.
9. **If the box points at `cuGraphInstantiate`:** instantiate the rebuilt graphs BEFORE
   destroying the old execs (peak VA doubles for one token, which the 2x192 GB pair can carry),
   or find a trim between. Do not do it speculatively — it doubles peak graph memory and the
   receipt has not asked for it yet.
10. **Serving wiring.** The door is exercised through `decode_step_hyper` / `decode_step_hyper_ppn`
   only. A server-side admission predicate (the `MEMRA_GS_MIN`-style budget gate) is a separate
   decision and is not made here.

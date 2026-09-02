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

# Iteration 3 — GPU DSpark drafter path (2026-08-19)

Queue item 2 of the DSv4-Flash perf program; builds on lane 10 (drafter oracle, ALL
GREEN — wt-dsv4-dspark research/dsv4-dspark-20260818/RECEIPTS.md) and lanes 8/9
(device-resident decode step, 29.8 ms/step → 24.7 single-run under f32x pending
ratification). Branch `lane/dsv4-flash-loader`, lane-10 branch merged at f8ba6845dd.
Box1 (2× RTX PRO 6000 Blackwell 96GB) CPU+GPU. Laws binding: CPU oracle = truth
instrument (realization-stability, lane 6); greedy spec==plain identity; no speed
claims from acceptance observables (q38 box2 correction); interleaved quiet A/B ×5 on
owner corpora at serving temperature for any wall claim.

## Rung 1 — §3.1 ring-hazard DECISION + CPU-oracle gate (queue mandate: FIRST)

**Decision (banked before implementation, DSPARK-SEMANTICS §3.1 cache classes):**
HYBRID, not SGLang-style ring doubling —

1. **Window ring (43 layers)**: TRANSIENT-BATCH KV. The T=k+1 batch's kv rows live in
   a per-round scratch ([T, hd] per layer), never in the ring during the round;
   attention reads redirect in-round positions to the scratch (the reference's own
   [ring | draft] gather shape, M:784). This is REQUIRED for batched attention
   correctness, not just rollback: at a saturated ring, the round's later slots
   (e.g. (pos0+5) % 128) still back LIVE window positions of the first query —
   writing before reading corrupts sequential semantics. Commit = copy accepted rows
   to their ring slots; rollback is free (rejected rows never touch the ring).
   Verdict on SGLang's doubling: unnecessary at depth ≤ 5 — their hazard was writing
   draft rows into live rings before verification; a transient scratch removes the
   write, and 6 < 128 means no in-round slot collision is possible.
2. **Compressor pending + emissions (the real §3.1 hazard, fine [8,1024]×2 pairs +
   fine cur→prev shift, coarse [128,512]×2)**: SNAPSHOT-ROLLBACK with bounded replay.
   Pending state snapshots at round start (~11.5 MB/round across all layers; on GPU a
   D2D copy measured in µs); the batch advances pending IN PLACE sequentially per
   position (later in-round queries must see in-round block emissions — same order as
   sequential decode, so every emission's pooling inputs are bit-identical to the
   sequential twin's); on partial accept: restore snapshot, then replay the committed
   positions' recorded (dst, kv row, post-ape score row) writes + shifts + block
   accounting. NO re-pooling: the store rows the committed prefix emitted during the
   batch are already bit-identical, so replay is pure row writes.
3. **Append-only stores (compressed block store, FP4-QAT indexer store)**: high-water
   mark truncation. Emissions beyond the committed prefix become dead bytes above
   n_blocks; the retry overwrites them in append order (the store's own assert
   enforces ordering).
4. **Drafter main_kv rings (3 blocks)**: written ONLY for committed positions
   (`on_commit` per accepted position — the DSPARK-SEMANTICS §2 mandate: a verifying
   engine must write main_kv for every ACCEPTED position; rejected drafts never touch
   drafter state). At depth ≤ 5 the [128, 576] rings cannot wrap within a round.
5. **Indexer top-k**: recomputed per query from stores — safe once (2)+(3) roll back.

**Why not full shadow (pointer-flip) pending:** acceptance is variable-length (accept
j of 5); a pointer flip only gives all-or-nothing state, while the committed state
must equal plain decode of exactly j+1 positions. Snapshot + bounded replay handles
partial accepts exactly, and the replay payload is a few KB per layer.

**Implementation** (commit 4a9ac814e9 + smoke knob 2e3a0a4547):
- `memra-gguf dsv4_decode.rs`: `CompCkpt` (snapshot + replay payload),
  `CompressorState::{begin_ckpt, decode_ck, rollback_replay, state_views, reset}`,
  `AttnState::{decode_batch, commit_batch, reset, ring_view}` (transient kv + slot
  redirect resolver), `BlockState::forward_batch`, `TrunkState::{verify_batch,
  commit_batch, reset_state}` (all-position logits + taps), `TrunkBatchAdapter`.
- `spec_oracle.rs`: `TrunkOracleBatched` seam (verify_batch/commit) +
  `run_spec_greedy_batched` — one drafter call + ONE batched trunk pass per round,
  accept walk, commit(accepted+1), per-committed-position drafter advance. Round and
  budget accounting reproduces the sequential loop EXACTLY (including the
  budget-truncated final round and the pending-carry no-propose tail), so the
  proposal stream, output tokens and digest stream are comparable item-for-item with
  the lane-10 sequential gate.
- `dsv4_dspark_gate.rs` mode `batched`: pass 1 = sequential twin teacher-forced along
  the banked trajectory, digesting EVERY cache class (window ring; compressor pending
  kv/score + live store; indexer pending + store; 3 drafter rings) per layer after
  EVERY position; full state reset; pass 2 = free-running batched loop whose
  committed state after EVERY round must be BIT-IDENTICAL, class by class, to the
  twin's at the same position. Output identity vs the banked trajectory + printed
  proposal/output digest for cross-mode comparison with the sequential greedy gate
  (an identical sha = behavioral identity through completely different verify
  machinery).
- Unit pins (rig, pure math): `comp_ckpt_rollback_replay_matches_plain_decode`
  (synthetic compressor, n_commit 0..=6 across emission boundaries + overlap shifts —
  bit-equal to a plain-decode twin), `comp_reset_is_construction_state`. 26/26 green.

Arithmetic-identity argument for the twin comparison (why bit-level equality is the
right bar and not a threshold): matmul/rmsnorm/hc/MoE in the oracle are per-row f64
dots — batching rows changes nothing; the batched attention builds the SAME idx list
per position and resolves the SAME float rows in the SAME accumulation order
(redirect only changes WHERE a row is read from, never its value or order); the
compressor advances in the same per-position order. Every difference between the two
passes is therefore exactly the commit/rollback machinery — which is what §3.1 gates.

### Gate runs

Logs on box1 `/home/ubuntu/dsv4-dspark/logs-batched-*.log`, out-dirs `out/batched-{1,2}`.
Binary `target/release/dsv4-dspark-gate`, invocation (cores pinned, detached):

```
taskset -c 0-23  target/release/dsv4-dspark-gate /home/ubuntu/models/dsv4-flash-0731-nvfp4 \
  batched fixtures/dsv4_dspark_fixtures_ref.json out/batched-1     # run1, cores 24-47 for run2
```

- **SMOKE** (`MEMRA_DSPARK_BATCH_SMOKE`, n_new truncated to 24 — banner-marked, never a
  verdict): PASS, 24/24 literal identity, 7/7 round boundaries bit-equal, 0 class fails.
  Machinery feedback only; recorded because it is what justified spending the full run.
- **FULL RUN 1 — §3.1 RING-HAZARD GATE [PASS]** (elapsed 2756s, EXIT=0):
  - 160/160 tokens == banked REF trajectory, **LITERAL IDENTITY, zero corrections**
  - **state classes BIT-equal to the sequential twin at 35/35 round boundaries, 0 class
    fails** (232 classes digested per position across 159 twin positions: window ring,
    compressor pending kv/score, live block store, indexer pending + store, per layer,
    plus the 3 drafter rings)
  - rounds 35, accepted drafts 124 (mean 3.543 / verify round)
  - determinism sha256 `b1011b595deb5d11338bb9e96a28faf98c20445bd822e131c133d88c8e70c87b`
- **FULL RUN 2 — [PASS]** (elapsed 2776s, EXIT=0), independent process on cores 24-47:
  identical verdict line — 160/160 literal identity, 35/35 round boundaries bit-equal, 0
  class fails, 124 accepted drafts, and determinism sha256
  `b1011b595deb5d11338bb9e96a28faf98c20445bd822e131c133d88c8e70c87b` —
  **BYTE-IDENTICAL to run 1**. Determinism ×2 closed.

**Verdict: the banked HYBRID decision is GATED CORRECT on the CPU oracle.** The §3.1
invariant — "cache state after (verify k, reject rest) == cache state after plain decode
of the accepted tokens" — holds bit-for-bit across every cache class at every round
boundary, so transient-batch window kv + compressor snapshot-rollback-with-replay +
append-only high-water marks + accepted-position-only drafter-ring writes is the
ratified mechanism. Shadow/pointer-flip ring doubling stays rejected (it cannot express
a partial accept, and the transient scratch removes the write that was the hazard).

Acceptance note (dspark-q38 law): 3.543 accepted/round is a CORRECTNESS observable of
the free-running loop; it is NOT a speed claim and no wall number is derived from it.

## Rung 2 — drafter VRAM plan (banked arithmetic)

Inputs (all measured, receipts cited): 0731 post-load dev0 82.24 / dev1 78.64 GiB
(0731 re-gate gate table — NO NextN block on 0731, the preview's +3.5 GiB MTP slab
does not exist); usable ≈ 94.5 GiB/card (95.59 capacity − CUDA context ≈ 1.0; lane-4
placement math); 1M-ctx decode caches dev0 6.92 / dev1 7.55 GiB at 7,712 B/token on
dev1 (lane-6 VRAM math, allocator == formula); drafter resident ≈ 10.7 GiB (10.12
stored — prep §1.2 census: 10.117 GiB mtp.0-2 all dtypes — + bf16 promotions of
FP8-blk linears + f32 islands; prep §4.4); step transients < 0.5 GiB (lane-4);
verify-round ckpt (§3.1) ≈ 12 MB + [T,hd] scratch ≈ 0.6 MB — noise.

**Placement: drafter fully resident on dev1.** The tap layers (40/41/42), the shared
embed/head and the hc_head/norm already live on dev1 (PP split at layer 22); the
whole draft path then runs without a single cross-card hop (main_x GEMV → 3 blocks →
shared head → markov chain), and dev0's pipeline-idle window during drafting is free
for nothing else anyway at bs=1.

The binding arithmetic (dev1): 78.64 + 10.7 = 89.34 GiB resident → 94.5 − 89.34 =
5.16 GiB cache budget → at 7,712 B/token ≈ **718K tokens ctx ceiling with the
drafter ON** (≈ 640K with transients + margin — the prep §4.4 "~600K" class,
confirmed). dev0 never binds: 82.24 + 6.92 (1M) = 89.16 < 94.5.

Split across cards does NOT close the gap: free-at-1M is dev1 5.16 − 7.55 < 0 and
dev0 5.34 GiB; total free 10.5 GiB < 10.7 GiB drafter — even a perfect split is
short, and it buys a mid-drafter peer hop. FR-Spec row-trim reduces markov/head READ
traffic (compute), not expert residency — it is the banked perf follow-up, not a
VRAM lever. Remaining VRAM levers if 1M+drafter ever becomes a product bar (owner
call, NOT needed for iteration 3): re-quantize drafter FP8-blk linears to fp4-class,
or accept the ctx cap.

**Decision: drafter ON caps served ctx at ≈600–700K tokens (exact number measured at
load); beyond the cap the engine refuses the drafter (drafter OFF restores the full
1M window). Iteration-3 perf cells run at s ≈ 200–8k — two orders inside the cap.**

## Rung 3 — GPU port (design)

Reuse-first port of the CPU-oracle machinery onto the lane-8/9 device step
(`decode_step_fast` / `block_decode_dev`), CPU oracle as the truth instrument at
every gate:

- **Batched T=k+1 verify**: every `gemm_dev`/`gemv_pre_dev` call takes a row count —
  the heavy mass (wq_a/wq_b/wkv/wo, shared expert, head) batches to m=T in place
  (that is the amortization that makes drafted decode pay: non-routed weight traffic
  is read once per round instead of once per token; routed expert traffic scales with
  T regardless). Position-serial small kernels (rope_at, build_idx, cmp_decode_dev,
  indexer top-k, sink_attn trio) loop t = 0..T−1 in position order.
- **kvc layout extension**: ring region grows to [win + T_max] rows; transient batch
  rows live at win..win+T_max−1; compressed blocks shift to offset win+T_max. Kernels
  are untouched (pure indexing convention — build_idx/topk emit ids under the new
  offsets); commit = T_max-bounded D2D copies to ring slots.
- **§3.1 device rollback**: pend_kv/pend_score/ipend snapshots (D2D, ~12 MB/round),
  n_blocks/i_blocks high-water marks, replay of committed (dst, kv, score) rows
  retained in a device scratch — the CPU-oracle `CompCkpt` program, device-realized,
  gated output-sample + trajectory vs the oracle.
- **Trunk tap**: hc-mean kernel ([4,4096] → [4096]) after layers 40/41/42 into a
  taps buffer [T, 3·4096]; concat feeds mtp.0.main_proj (FP8-blk → bf16 resident,
  one more gemm_dev) + main_norm.
- **Drafter forward**: 3 window-only Blocks at block=5 rows over [ring | draft]
  (the dspark_topk_idxs single shared row set), MXFP4 expert arm (lane-7, gated),
  shared head over 5 rows, sequential markov chain (5× [129280,256] GEMV pairs +
  bias add + device argmax), confidence GEMV. Drafter rings = 3× [128, hd] device
  buffers written per committed position from the taps (free — the verify pass
  computes them).
- **Gates** (CPU oracle = truth): (a) drafter components output-sample vs lane-10
  fixtures at GPU-class derived thresholds (fork rules of the 0731 re-gate);
  (b) greedy spec==plain identity on the GPU loop vs the banked REF trajectory
  (in-band flip adjudication, lane-6 policy); (c) §3.1 device state gate — batched
  round + rollback vs sequential device decode of the committed tokens, bit-level
  per cache class (the device twin of the CPU gate; device==device comparison is
  legitimate HERE because both sides are the same realization and the question is
  state machinery, not numerics); (d) determinism ×2.

## Rung 3 — GPU port: LANDED STATE + gate (b) GREEN

### Mid-edit reconstruction (owner stop was mid-hunk — verdict: COHERENT, one dangling symbol)

The stopped agent's tap-row refactor (base offset replacing a broken per-row shim) was
complete in intent and applied everywhere except ONE call site:
`decode_step_greedy_tap` still called `decode_step_fast_tap_row`, a symbol that does not
exist — the file could not compile. The refactor was otherwise self-consistent
(`decode_step_fast` → `decode_step_fast_tap(.., Option<(&mut CudaSlice<f32>, usize)>)`,
base = `tap_row * n_t * hidden` at both call sites), so the hunk was REPAIRED, not
reverted: the call was pointed at `decode_step_fast_tap`. `cargo check` then went clean
(nvcc included — both new kernels compile), and the f32x default flip was found ALREADY
applied in the same dirty tree.

Two genuine gaps found while gating the reconstruction, both fixed:
- `DsparkCaptureOut` was missing `logits_pre` (pre-markov head logits) and `main_hidden`
  (the tap row itself) — two of the seven arrays the lane-10 CPU components gate compares.
  Added, with `logits_pre` captured BEFORE the chaining loop's in-place bias add.
- No prefill-side tap existed at all, so `dspark_prime_prefill` had no producer and the
  first round would have drafted off a zeroed tap. Added `dspark_prefill_prime`: it
  reuses the existing `GpuCapture::layer_out` hook for the target layers' `[s, hc, hidden]`
  state, runs it through the SAME `memra_dsv4_hc_mean` kernel the decode tap uses (prefill
  and decode taps must not be two numeric realizations of one tap), places each target at
  its concat stride with `place_cols` — reproducing the oracle's
  `main_hidden[(p*n_t + k)*hidden ..]` layout — and seeds taps row 0 with the last prefill
  position's tap, which is what `spec_oracle::run_spec_greedy`'s first
  `propose(t, mh_last, p0-1)` reads.

### Ported (compiles, gated): tap + drafter + markov + accepted-position ring writes

`dsv4_hc_mean` + `dsv4_build_idx_redirect` kernels; `DsparkDev`/`DsparkState` +
`MEMRA_DSV4_DRAFTER=dspark` load on the last stage; `dspark_prefill_prime`,
`dspark_write_rings`, `dspark_block_forward` (5-slot non-causal block attention over
[ring | transient draft rows]), `dspark_forward_spec` (one parallel forward, shared trunk
head, sequential rank-256 markov chaining, fp32 confidence), `decode_step_tap` /
`decode_step_greedy_tap`.

### NOT ported (the remaining rung, explicitly): batched T=k+1 device verify + device rollback

`dsv4_build_idx_redirect` exists but has NO caller — the device batched verify, the
pend_kv/pend_score/ipend D2D snapshots, the high-water marks and the replay scratch are
not written. **Consequence for perf: no drafted wall-clock claim is possible yet**, and
none is made here. The amortization that makes drafted decode pay is batching the
non-routed weight traffic to m=T; with sequential verify the drafted loop does strictly
MORE work per token than plain (it adds a propose and a ring write), so the perf rung is
gated behind the batched port, not behind this gate.

### Gate (b) — greedy spec==plain identity on the device drafter: [PASS], both arms

New bin `dsv4-gpu-dspark-gate` (`crates/memra-engine/src/bin/dsv4_gpu_dspark_gate.rs`)
runs BOTH arms against ONE loaded model: arm P = plain device greedy (drafter resident
but never called), arm D = drafted propose-then-verify with SEQUENTIAL verification (for
greedy, sequential verify is mathematically identical to batched verify and to plain
greedy — DSPARK-SEMANTICS §2 — so this isolates the drafter itself, ahead of the batched
rung). Both arms enter `decode_step_fast_tap`; they differ ONLY in whether the tap
capture is `Some`, so the identity test is precisely "does the drafter perturb trunk
state" — hard pass/fail, no threshold doctrine.

```
MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native \
  target/release/dsv4-gpu-dspark-gate /home/ubuntu/models/dsv4-flash-0731-nvfp4 \
  /home/ubuntu/dsv4-dspark/fixtures/dsv4_dspark_fixtures_ref.json \
  /home/ubuntu/dsv4-it3-out/gpu-dspark-gate 2 0,1
```

| arm | verdict | rounds | accepted | mean/round | accept sha ×2 |
|---|---|---|---|---|---|
| default ⇒ f32x | **PASS** 160/160 LITERAL identity drafted vs plain | 40 | 140 | 3.5000 | `85603e87fadf7876…` identical |
| `f64` control | **PASS** 160/160 LITERAL identity drafted vs plain | 42 | 136 | 3.2381 | `b3765040fa85697b…` identical |

Determinism ×2 in both arms: token streams IDENTICAL, per-round accept counts
BYTE-IDENTICAL. Logs `~/dsv4-it3-out/gpu-dspark-gate{,-f64}.log`.

Acceptance numbers are CORRECTNESS observables (dspark-q38 law) — the two arms differ
because their trajectories differ after a known flip (below), not because one drafts
better. **No wall-clock claim is derived from them.**

### Adjudicated: plain GPU greedy vs the banked torch trajectory diverges at generated index 22

The gate REPORTS (never gates) arm P against the banked torch REF trajectory, and it
agreed only to a common prefix of 22. That was chased rather than assumed, and it is a
pre-existing in-band realization flip, not an f32x regression and not a port bug:

- The **f64 oracle-truth arm reproduces the SAME common prefix 22** — so whatever causes
  it is not the ratified f32x substitution.
- The banked 0731 decode-gate table already records an in-band argmax flip at **s=54**
  (`argmax d/r 666/795 | flip(in-band) | top5 5/5 v0 | verdict PASS`) in ALL THREE arms
  (f64, f32, f32x). Generated index 22 = position 32 + 22 = **s 54** — the same event.
- Free-running greedy amplifies one in-band near-tie into a different tail forever, which
  is exactly why the teacher-forcing gate (not a free-running trajectory compare) is the
  torch-agreement instrument: the banked tf-gate is 158/160 with its disagreements in
  band, and `cpu-verify-dec-f64.log` shows the f64 arm carrying in-band disagreements of
  its own (positions 64/74/89/278).

The "43/160 (f32x) vs 29/160 (f64) positions equal" counts past the flip are incidental
re-convergence and carry no information; the common prefix is the meaningful figure.

### VRAM plan CONFIRMED by measurement

The banked arithmetic predicted dev1 = 78.64 + 10.7 = 89.34 GiB with the drafter
resident. Measured at load with `MEMRA_DSV4_DRAFTER=dspark`:
`[vram post-load] dev0 used 82.24/94.97 GiB | dev1 used 89.30/94.97 GiB (resident 88.36)`
→ **drafter cost 10.66 GiB, dev1 89.30 vs 89.34 predicted (0.04 GiB error).** dev0 is
unchanged, confirming the whole drafter landed on dev1 as planned (no cross-card hop in
the draft path). The ≈600–700K ctx cap with the drafter ON therefore stands as banked,
and iteration-3 cells at s ≈ 200–8k are two orders inside it.

## Rung 3b — f32x RATIFICATION + plain-path interleaved A/B ×5 (MEASURED, bench not serving)

Owner ratified f32x 2026-08-19 (quality-stays condition met). Flip landed in
`Dsv4Gpu::load`: unset/empty `MEMRA_DSV4_DOTS_ARM` ⇒ **f32x** (dots + sink/norm/indexer
f32-accumulation twins); `f64` = the selectable oracle-truth arm; `f32` = the lane-9
intermediate. `hc_sinkhorn` is NOT in f32x (never authorized). Legacy path and prefill
never consult either flag. Banked in darklanes PLAN.md at the f32-dots ruling.

Cell design: strict interleave A,B,A,B,… ×5 (interleaved-A/B protocol law — box clock
drift invalidates cross-run claims), **arm B runs with the env UNSET**, so the same cell
that measures the delta also proves the ratified default resolves to f32x. Driver
`~/it3-ab.sh`, logs `~/dsv4-it3-out/ab-{f64,default}-r{1..5}.log`,
`dsv4-decode-bench <model> <fixtures> <out.json> 1024 0,1`, native expert arm, device
decode path, 0731 NVFP4 mint, SM clocks logged per run (idle 180 MHz between runs).

Arm-B banner every run: `dots arm: f32x (dots + sink/norm/indexer chains)` — **default
flip verified end-to-end ×5.**

Per-run window medians (ms/step), n=21 per window:

| arm | s≈200 | s≈512 | s≈1024 | stream sha256 (×5) |
|---|---|---|---|---|
| A = f64 (old default) | 38.7 / 38.6 / 38.6 / 38.6 / 38.6 | 39.2 ×5 | 40.7 ×5 | `12ec1d210e1431b6…` 1 unique |
| B = unset ⇒ f32x | 24.7 / 24.9 / 24.7 / 24.7 / 24.7 | 25.3 / 25.2 ×4 | 25.8 ×5 | `811fc1c721505993…` 1 unique |

Medians of the 5 runs, and the ratified delta:

| window | f64 | f32x | Δ ms | Δ % | f64 tok/s | f32x tok/s | speedup |
|---|---|---|---|---|---|---|---|
| s≈200 | 38.6 | **24.7** | −13.9 | −36.0% | 25.9 | **40.5** | 1.563× |
| s≈512 | 39.2 | **25.2** | −14.0 | −35.7% | 25.5 | **39.7** | 1.557× |
| s≈1024 | 40.7 | **25.8** | −14.9 | −36.6% | 24.6 | **38.8** | 1.577× |

Each arm is internally byte-deterministic ×5 (one unique stream sha per arm); the two
arms' shas differ, which is the expected f64/f32x numeric fork, not instability. Spreads
are ≤ ±0.5% within an arm, so the interleave has no drift signal to remove.

**This supersedes the regate's informational single-run.** Note the regate's "29.8 →
24.7" was f32 → f32x; against the arm the flip actually replaced (f64), the ratified
default is worth **−13.9 ms/step and +56% tok/s** at s≈200.

**Iteration-3 plain baseline = 24.7 ms/step / 40.5 tok/s bs=1 (s≈200), MEASURED.**
Release bar context: canada-quant 47.5 plain — the plain path is still ~7 tok/s short, as
the lane-9 ranked-items projection said; the drafter is what clears the bar.

**CUDA graphs scope (lane-8 flip condition, mass < 25 ms):** met at s≈200 (24.7) and NOT
met at s≈512 (25.2) or s≈1024 (25.8). Graphs therefore enter scope on the plain path for
short contexts only; the banked lane-8 graphs design applies unchanged, and the gap it
was sized against (1.83 ms) is now the same order as the whole remaining margin — so the
cell is worth running, but it is a short-context win, not a blanket one.

## Rung 4 — perf (design + the plain half MEASURED)

Interleaved quiet A/B ×5, both arms sha-pinned; no number leaves this lane without the
full correctness gate set green. Bar: canada-quant 47.5 plain / ~70 drafted bs=1 on the
same card class.

- **Plain half: DONE** — see Rung 3b. 24.7 ms/step / 40.5 tok/s bs=1 at s≈200, ×5
  interleaved, deterministic per arm. Still ~7 tok/s under the 47.5 plain bar.
- **Drafted half: BLOCKED on the batched device verify**, by design and stated plainly.
  With sequential verify the drafted loop does more work per token than plain, so an A/B
  now would measure a regression that says nothing about the drafter's value. The cell
  runs after the batched T=k+1 port, on SXC owner corpora at serving temperature.

## Next rung (ordered, for whoever picks this up)

1. **Batched T=k+1 device verify + §3.1 device rollback.** Design is banked in Rung 3
   above; `dsv4_build_idx_redirect` is written and unit-shaped but uncalled. Needs: row
   counts through `gemm_dev`/`gemv_pre_dev`, kvc ring region grown to `win + T_max` with
   compressed blocks shifted to `win + T_max`, pend_kv/pend_score/ipend D2D snapshots,
   n_blocks/i_blocks high-water marks, replay scratch. Gate = the device twin of the CPU
   §3.1 gate (device==device is legitimate there: same realization, the question is state
   machinery) + re-run `dsv4-gpu-dspark-gate` (identity must survive batching).
2. **Drafted interleaved A/B ×5** on SXC corpora vs the 40.5 tok/s plain baseline.
3. **CUDA graphs on the plain path**, short-context only (mass 24.7 < 25 at s≈200; 25.2 /
   25.8 at s≈512 / 1024 do NOT meet the lane-8 flip condition).
4. Optional, banked as follow-ups, never correctness requirements: drafter arena/native
   expert perf rungs, FR-Spec row-trim on the markov/head read traffic.

## Rung 3c — plain baseline curve EXTENDED to s≈8k (interleaved A/B ×5, ADOPTED)

The detached cell left running by the previous agent completed at 11:39 UTC and is
adopted here, not restarted. Driver `~/it3-ab8k.sh`, logs
`~/dsv4-it3-out/ab8k/ab8k-{f64,default}-r{1..5}.log`, self-banked
`~/dsv4-it3-out/ab8k/SUMMARY.txt`; `dsv4-decode-bench <model> <fixtures> <out.json> 8200
0,1`, native expert arm, device decode path, 0731 NVFP4 mint, strict interleave
A,B,A,B,… ×5. GPUs verified idle afterwards.

Medians of 5 runs (n=21 per window). One unique stream sha per arm across all 5 runs;
per-window spread across runs ≤ ±0.1 ms — the interleave has no drift signal to remove.

| window | f64 | f32x (default) | Δ ms | Δ % | f64 tok/s | f32x tok/s | speedup |
|---|---|---|---|---|---|---|---|
| s≈200 | 38.6 | **24.7** | −13.9 | −36.0% | 25.9 | **40.5** | 1.563× |
| s≈512 | 39.2 | **25.2** | −14.0 | −35.7% | 25.5 | **39.7** | 1.557× |
| s≈1024 | 40.7 | **25.8** | −14.9 | −36.6% | 24.6 | **38.8** | 1.577× |
| s≈2048 | 43.0 | **26.8** | −16.2 | −37.7% | 23.2 | **37.4** | 1.612× |
| s≈4096 | 43.9 | **27.3** | −16.6 | −37.8% | 22.8 | **36.6** | 1.607× |
| s≈8192 | 45.8 | **28.7** | −17.1 | −37.3% | 21.8 | **34.8** | 1.596× |

Two readings worth banking:

- **The ratified f32x win GROWS with context** (−13.9 ms at s≈200 → −17.1 at s≈8192), as
  expected: the f64 chains it replaced are the sink/indexer scores, whose work scales
  with the index list, so the arm that removed them scales better.
- **The plain curve is shallow**: 40.5 → 34.8 tok/s over a 40× context range (−14%). The
  plain bar problem is therefore not a long-context problem; it is a bs=1 weight-traffic
  problem at every context.

**CUDA-graphs scope is now measured across the whole range, not inferred**: the lane-8
flip condition (kernel mass < ~25 ms) is met at s≈200 (24.7) and at NO other measured
window (25.2 / 25.8 / 26.8 / 27.3 / 28.7, monotonically rising). Graphs are a
short-context-only lever on the plain path. Scoped decision in Rung 4c below.

## Rung 4a — BATCHED T=k+1 DEVICE VERIFY + §3.1 DEVICE ROLLBACK: PORTED and GATED

Branch `lane/dsv4-flash-loader` @ `062962235d` (+ `ea5c87eced`), box1 mirror synced by
bundle. This closes mandate item 1 of the rung-3 "NOT ported" list.

### The design law, banked BEFORE the code was written

**The batched verify must be BIT-EXACT against T sequential single-position steps.** Not
"within tolerance" — bit-exact. The reason is the lane's own verdict instrument: greedy
spec==plain LITERAL identity. If the verify pass computed even slightly different logits
than the plain pass, identity would break at every near-tie and no gate could separate a
port bug from a rounding fork — and the orchestrator's brief carries exactly that warning
("cuBLASLt m-order shifts logits 0.18–3.08").

That is achievable HERE, and only here, because the lane-8 device decode path's dense
projections are **our own kernels** (`dsv4_gemv_bf16`, `dsv4_dots_f32*`), not cuBLASLt: a
batched GEMV that hoists the WEIGHT load across T activation rows keeps every (row,
column) dot's element order and reduction tree exactly as the m=1 kernel had them, so the
value is identical while the weight traffic is paid ONCE per round instead of once per
token. **cuBLASLt is deliberately absent from the batched verify path.** The banked
m-order hazard is not worked around; it is not entered.

### CUDA: 11 new entries, each a twin of a pinned kernel with one added dimension

| entry | added dimension | why it is bit-exact |
|---|---|---|
| `gemv_bf16_m` | m activation rows, x/y strides | templated on M so `part[]` stays in registers; per (t, row) the 8-element chunk ownership and the 128-leaf halving tree are `dsv4_gemv_bf16_kernel`'s verbatim |
| `dots_f32_mrow`, `dots_f32acc_mrow` | s rows, weight row hoisted | per (t, j) element order + reduction tree unchanged; the f64 arm reuses `dsv4_block_sum` sequentially with a sync between rows |
| `hc_sinkhorn_m`, `hc_head_pre_m` | one block per position | body verbatim on the position's own slices |
| `route_m` | one block per position, own token | the hash layers read `tid2eid[tok]`, which is WHY a round needs a token ARRAY and not a token |
| `fp4_gemm_sel_g` | `a_group` maps T×topk slots → T activation rows | only WHICH activation row a slot reads changes; `a_group == 0` is the same expression the pinned launcher always evaluated |
| `combine_rows_m` | one output row per position | per position the topk sum order is the pinned kernel's |
| `sink_attn_dec_mq` (+`_f32acc` twin) | T queries, per-query index list | uniform `slots` = max over the round with −1 pads, which are **bit-inert by the pinned kernels' own pad contract** (score −inf → eval +0.0 → skipped in both the denominator and the output chain) |
| `scatter_rows` | — | verify-round ring commit in one launch |

The pinned kernels are byte-untouched; the only edit to an existing kernel is
`fp4_gemm_sel`'s `a_group` parameter whose 0 case is the identical expression. Bodies are
duplicated rather than refactored into shared `__device__` helpers precisely so the gated
arms' generated code cannot move.

### Rust: the §3.1 machinery, device-realized

- `VerifyWs` — the lane-8 arena widened to `T_max` rows, held **separately** from
  `StepWs` so the gated single-position path's allocations, launches and bytes are
  literally untouched by this rung. ~18.4 MB/dev measured.
- `kvc` grows by `T_max` **TRANSIENT** rows at `[win + cap_blocks, +T_max)`. The window
  ring is READ-ONLY for a whole round; `dsv4_build_idx_redirect` (written last rung,
  uncalled until now) resolves a slot whose backing position falls inside the open round
  to its transient row. **Zero rows are reserved when the drafter is not loaded**, so
  today's allocation is byte-identical.
- `CmpCkptDev` / `LayerCkptDev` — the CPU oracle's `CompCkpt`: full pending snapshot +
  the RAW (kv, score) rows the round wrote + the store high-water mark. `dst` and
  `emitted` are pure functions of the position, so **nothing has to come back to the
  host to replay**. `cmp_rollback_replay_dev` is `CompressorState::rollback_replay`
  verbatim, including the "fully committed ⇒ in-place state is already exact" early out
  and keeping committed emissions' store rows as the round wrote them.
- `block_verify_dev` — batched hc/norm/q/kv/output/MoE with the position-serial pieces
  (compressor state machine, indexer score + top-k, index build) looped `t = 0..T−1` in
  POSITION ORDER, which is what makes in-round block emissions visible to later queries
  exactly as they are sequentially.
- `verify_batch_dev` + `commit_verify_dev`, `spec_greedy_batched_with` — the device twin
  of `spec_oracle::run_spec_greedy_batched` including its budget/carry accounting, with
  **accepted-position-only drafter ring writes** and per-round wall timing (the drafter
  stream is synchronized at the round boundary so no work leaks into the next round's
  measurement).

### Gate (c) — BATCHED == SEQUENTIAL, BIT FOR BIT: [PASS], first run, both gate processes

The decisive port instrument, and the reason (b) means anything on the device at all.
From the SAME warmed cache state, the gate compares:

- the batched round's per-position logits vs T sequential single-position decode steps'
  logits, as RAW BITS (`to_bits()` equality over all 129,280 vocab entries per row) —
  not a tolerance;
- every LIVE trunk cache class after `commit(n)` vs plain sequential decode of exactly
  the n committed tokens, as RAW BITS (the §3.1 invariant, device twin of the CPU-oracle
  gate that ratified the mechanism).

Swept over `warm ∈ {1,2,3,4}` (every fine-compressor phase mod ratio 4) × `commit ∈
{1, T−1, T}` (partial and full accept), plus a minimum-width `T=2` round and a `T=3`
round — **14 cells, 77 logit rows, 3,206 cache-class comparisons, ALL BIT-IDENTICAL**,
reproduced in both independent gate processes.

That result upgrades the identity law from mathematical to byte-level on this device: the
batched verify is not "an approximation of what the plain path would have computed", it
IS what the plain path computes.

### Gate (b) — greedy spec==plain identity, both drafted arms: [PASS] 160/160

### Gate (d) — accepted-position ring writes: [PASS], bit-identical

Arm DB's 3 drafter `main_kv` rings vs a plain greedy arm that wrote a ring row at EVERY
decoded position: 3/3 ring classes bit-identical, 0 mismatching. That is the banked trap
closed by measurement — the reference writes one row per step only because its smoke
drafts every step; a verifying engine owes one row per ACCEPTED position and none for a
rejected draft, and this gate would catch either error.

### Gate (e) — determinism ×2: [PASS]

Invocation (detached, `~/it3-rung4-fullgate.sh`, logs
`~/dsv4-it3-out/rung4-gate-{1,2}/gate.log`, out-dirs banked next to them):

```
MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native \
  target/release/dsv4-gpu-dspark-gate /home/ubuntu/models/dsv4-flash-0731-nvfp4 \
  /home/ubuntu/dsv4-dspark/fixtures/dsv4_dspark_fixtures_ref.json \
  /home/ubuntu/dsv4-it3-out/rung4-gate-1 2 0,1
```

| arm | tokens vs plain | rounds | accepted | mean/round | tokens/round | mean T fwd | accept sha ×2 |
|---|---|---|---|---|---|---|---|
| DS (sequential verify) | **160/160 LITERAL** | 40 | 140 | 3.5000 | 4.000 | 1 | `85603e87fadf7876…` |
| DB (batched verify) | **160/160 LITERAL** | 37 | 122 | 3.2973 | 4.3243 | 5.9189 | `150342bae32b38b5…` |

Both gate processes printed byte-identical verdict lines and identical accept shas.
Determinism ×2: tokens IDENTICAL, per-round accepts BYTE-IDENTICAL, drafter rings
BIT-IDENTICAL. The DS/DB round counts differ because the two loops OPEN rounds
differently (sequential opens one whenever the pending queue empties; batched forwards
T = k+1 per round) — the same difference the CPU oracle showed between
`run_spec_greedy` (37 rounds) and `run_spec_greedy_batched` (35) — while the emitted
token streams are identical. DB's 4.3243 tokens/verify-round reproduces the CPU oracle's
banked 4.32.

Acceptance numbers here are CORRECTNESS observables (dspark-q38 law). **No wall-clock
claim is derived from them**; the wall numbers are in Rung 4b, measured.

### VRAM after drafter + verify

`[vram post-load] dev0 82.24 / dev1 89.30 GiB` (identical to the banked drafter figure —
the verify rung adds nothing at load); `[vram post-bitgate] dev0 82.27 / dev1 89.36` →
the verify arenas + §3.1 checkpoints cost **≈ 60 MB total** (18.4 MB/dev of arena,
reported by the bench as `verify-state bytes/dev [18428380, 18571740]`, plus ~12 MB of
per-layer compressor snapshots and 0.5 MB of transient kvc rows). The ≈600–700K ctx cap
with the drafter ON is unchanged by this rung.

## Rung 4b — DRAFTED vs PLAIN, MEASURED (and the number the lane should quote)

Two cells, deliberately. The first is the lane's standard perf shape; the second exists
because the first one flatters the drafter, and the banked dspark-q38 correction is
precisely about noticing that before quoting a number.

### Cell 1 — interleaved A/B ×5 on the lane's perf fixture (32-token prompt, 8,200 tokens)

Driver `~/it3-rung4-ab.sh`, logs `~/dsv4-it3-out/rung4-ab/ab-{plain,drafted}-r{1..5}.log`,
self-banked `SUMMARY.txt`; strict interleave A,B,A,B,… ×5; ONE thermal window (12:06→13:01
UTC); same binary both arms; greedy serving shape (device argmax, 4-byte D2H) in BOTH
arms; native expert arm, device decode path, f32x default. Arm A = plain WITHOUT the
drafter resident (the real plain serving config and the arm the banked 24.7 ms baseline
was measured on); arm B = drafted (DSpark proposal + batched T=k+1 verify + §3.1
commit/rollback). One unique stream sha per arm across all 5 runs.

The comparable statistic is **mean ms/token in both arms**: a drafted round emits
`1 + accepts` tokens for one round of wall time, so a median of per-token shares is
meaningless when the shares differ 6× by accept count. WIDE bands (total time / total
tokens over generated-token ranges) are quoted rather than the 21-token windows, because
21 tokens is only ~4 drafted rounds:

| generated tokens | plain ms/tok | drafted ms/tok | plain tok/s | drafted tok/s | speedup |
|---|---|---|---|---|---|
| [0, 200) | 24.62 | 17.55 | 40.6 | **57.0** | 1.40× |
| [200, 512) | 25.10 | 17.64 | 39.8 | 56.7 | 1.42× |
| [512, 1024) | 25.37 | 17.43 | 39.4 | 57.4 | 1.46× |
| [1024, 2048) | 26.17 | 15.28 | 38.2 | 65.4 | 1.71× |
| [2048, 4096) | 26.89 | 15.73 | 37.2 | 63.6 | 1.71× |
| [4096, end) | 27.56 | 16.76 | 36.3 | 59.7 | 1.64× |
| whole run (8,200) | 26.95 | 16.40 | 37.1 | **61.0** | 1.64× |

Run-to-run: drafted overall 16.35 / 16.39 / 16.40 / 16.40 / 16.43 ms/token across the 5
interleaved runs, with byte-identical acceptance every run (1,482 rounds, 6,717 accepted,
4.5324/round, 5.5331 tokens/round, mean T forwarded 5.9966). Plain is equally tight.

**Read the trend, because it is the whole story of this cell**: the drafted arm gets
FASTER the further into the generation it goes (1.40× → 1.71×). A long greedy continuation
of a 32-token prompt degenerates into repetition, and a drafter predicts repetition almost
perfectly. This cell's acceptance (4.53/round) is the friendliest possible observable —
the exact shape of the q38 trap.

### Cell 2 — OWNER-BLESSED SXC corpora, bounded generation: **the number to quote**

32 real agentic prompts drawn from the owner-blessed SXC pools (hermes + eigen; the banked
extraction schemas and filters), tokenized with the artifact's own `tokenizer.json`
(mean 38 ids, max 278), 128 generated tokens per prompt, 3 reps = 12,288 tokens per arm.
BOTH arms in ONE process against ONE loaded model with the arm order alternating per
(rep, prompt) — a finer-grained interleave than the cross-process protocol, so load-time
and clock drift are removed rather than averaged out. Bin `dsv4-drafted-corpus`, driver
`~/it3-rung4-corpus.sh`, log `~/dsv4-it3-out/rung4-corpus/corpus.log`.

```
plain   : 24.34 ms/token = 41.1 tok/s bs=1
drafted : 21.25 ms/token = 47.1 tok/s bs=1
SPEEDUP : 1.145x
acceptance: 3,321 rounds | 8,877 accepted | 2.6730/round | 3.7001 tokens/round | mean T 5.9041
per pool  : hermes 1.206x (accept 2.870) | eigen 1.091x (accept 2.495)
greedy spec==plain identity on real prompts: 96 / 96 prompt-runs  [PASS]
```

Rep-to-rep: 1.143× / 1.145× / 1.145×. The plain arm reproduces the drafter-free plain
baseline (24.34 vs 24.62 ms/token at the same context band), which is the check that the
in-process instrument is not flattering either side.

**The honest drafted figure is 47.1 tok/s bs=1 (1.145×), not 61.0 (1.64×).** The
difference is entirely acceptance: 2.67 accepted/round on real prompts vs 4.53 on the
self-generated fixture stream. This is the dspark-q38 law reproducing itself inside this
lane, on this drafter, one rung after the correction was banked — and it is why the cell
exists.

Bar position, stated plainly:

| bar (canada-quant, 2×PRO6000, bs=1) | measured | verdict |
|---|---|---|
| plain 47.5 tok/s | 40.5 (fixture s≈200) / 41.1 (corpora) | **~6.5 short** |
| drafted ~70 tok/s | 47.1 (corpora) / 57.0 (fixture [0,200)) | **~23 short on corpora** |

A 96/96 identity pass on real prompts is a bonus correctness result: the greedy
spec==plain law holds on owner traffic, not only on the fixture.

## Rung 4c — where a drafted round actually goes (nsys), and the two levers it names

`nsys profile -c cudaProfilerApi` with the bracket around steady-state work only: plain
steps [16, 48), drafted rounds [4, 12) (`MEMRA_DSV4_BENCH_PROFILE=1`, which now brackets
rounds as well as steps). Driver `~/it3-rung4-graphs.sh`, artefacts
`~/dsv4-it3-out/rung4-graphs/`.

| | plain (per step) | drafted (per round) | ratio | amortization vs 6× |
|---|---|---|---|---|
| GPU busy (sum of kernels) | **23.2 ms** | **72.2 ms** | 3.11× | — |
| `fp4_gemm_sel` (routed experts) | 4.77 | **23.91** | 5.01× | 1.2× — *does not amortize* |
| dense GEMVs (`gemv_bf16` → `_m<6>`) | 8.83 | 14.30 | 1.62× | **3.7×** |
| island dots (`dots_f32acc` → `_mrow<6>`) | 2.70 | 4.68 | 1.73× | **3.5×** |
| `rmsnorm_f32acc` | 1.59 | 1.69 | 1.06× | **5.6×** |
| `hc_sinkhorn` → `_m` | 1.55 | 1.51 | 0.97× | **6.2× (free)** |
| sink trio (→ `_mq`) | 1.42 | 2.99 | 2.11× | 2.8× |
| **drafter forward** (`dots_f32` f64 + `fp4_gemm` + `sink_attn` + cuBLASLt) | — | **≈19.3** | — | per-round, does not amortize |

The batching worked exactly as designed on everything it could: dense weight traffic
amortizes 3.5–3.7×, the launch-latency-bound kernels amortize 5.6–6.2× (Sinkhorn is
free), and the ONE item that cannot amortize by construction — routed-expert weights,
because each position selects its own 6 of 256 experts — costs 5.0× instead of 6.0×
(a little L2 reuse across positions, no more).

Two levers, both measured, both named by this profile:

1. **The drafter's exit head was 21% of a round.** `dspark_forward_spec` ran the shared
   trunk head with `dsv4_dots_f32` — the **f64** kernel — over block_size rows, weight
   row re-read per row: 16.3 ms/round total with one instance at 13–14.7 ms. The trunk's
   OWN head already runs the ratified f32x arm on the SAME weights, and the drafter's copy
   only picks DRAFTS (verification always emits the trunk's argmax, so the emitted stream
   cannot depend on it). Landed as (i) a bit-exact hoist of the dots across the block's
   rows and (ii) `MEMRA_DSV4_DSPARK_HEAD_ARM=f32x`, an arm fork whose DEFAULT stays f64
   because the lane-10 components gate ran the drafter at f64 and a gated component does
   not change default without its gate. Gate result and wall delta below.
2. **Routed-expert traffic scales with T, and T is 6 while acceptance is 2.67.** On owner
   corpora the round forwards 5.90 positions to accept 2.67 — so roughly half of the
   23.9 ms of expert traffic per round is spent on drafts that get rejected. The banked
   confidence separation (accepted-slot mean 5.41 vs first-rejected 0.75) is exactly the
   signal a truncation policy needs. NOT implemented here; banked with its arithmetic as
   the next rung, because it changes the proposal contract and needs its own gate.

### Rung 4c decision — CUDA GRAPHS: **NO**, and the measurement says why

The lane-8 flip condition was "kernel mass < ~25 ms, launch API becomes binding", met at
s≈200 only (Rung 3c). Graphs can recover at most (wall − GPU busy) minus the graph's own
launch cost. Measured:

- **plain**: 24.6 ms wall vs **23.2 ms busy** → gap **1.4 ms/step (5.7%)**. A perfect
  graph takes s≈200 from 40.6 to at most 43.1 tok/s — still 4.4 short of the 47.5 plain
  bar, and the condition is not met at any other context.
- **drafted** (the path we would actually serve): ~78 ms wall vs **72.2 ms busy** → gap
  **5.8 ms/round = 1.05 ms/token (≈4%)**, because one round replaces 5.53 tokens' worth of
  launches. A perfect graph takes the fixture band [0,200) from 57.0 to at most 59.2 tok/s.

Against that, the complexity is not small and it is *shape-dependent*: the decode step's
launch shape changes whenever the index width changes — `nb`/`kk` grow every ratio-th
position on fine layers and every 128th on coarse, and the batched verify's `slots` is the
max over a round whose T varies with acceptance. A graph would need re-capture on every
width change or a shape-keyed graph cache, i.e. the machinery that the 5.7%/4% is supposed
to pay for.

**Decision: graphs are OFF the iteration-3 path.** The banked lane-8 design stays banked
and unimplemented, with the condition restated in measured terms: revisit only if a rung
drives the DRAFTED round's launch gap above ~10% of its wall, which today's two named
levers (drafter head, proposal truncation) would move in the opposite direction by
shrinking the round. Recorded either way, as instructed.

### Rung 4c lever 1 EXECUTED — drafter exit-head arm: gated, then measured on owner corpora

Both arms re-gated through the full `dsv4-gpu-dspark-gate` (logs
`~/dsv4-it3-out/rung4c-gate-{f64,f32x}/gate.log`), then the owner-corpora cell re-run for
each (logs `~/dsv4-it3-out/rung4c-corpus-{f64,f32x}/corpus.log`). Driver `~/it3-rung4c.sh`.

**Gates: BOTH arms [PASS], and the acceptance digest is IDENTICAL between them.**

| arm | (b) identity | (c) batched==sequential | (d) rings | (e) det ×2 | rounds | accepted | accept sha |
|---|---|---|---|---|---|---|---|
| f64 (default, hoisted) | 160/160 LITERAL | 14 cells / 77 rows / 3,206 classes BIT | BIT-identical | PASS | 37 | 122 | `150342bae32b38b5…` |
| f32x (fork) | 160/160 LITERAL | 14 cells / 77 rows / 3,206 classes BIT | BIT-identical | PASS | 37 | 122 | `150342bae32b38b5…` |

The accept sha is **byte-identical across the arms**: on the gate fixture the f32
accumulation on the drafter's exit head changes not one draft. The owner-corpora cells
then reproduced that at scale — 3,321 rounds and 8,877 accepted drafts, identical to the
token in both arms (2.6730/round, 3.7001 tokens/round, mean T 5.9041). So this fork is
measured to be quality-inert on every observable the lane has, which is a stronger
statement than "in band".

**Wall, on OWNER CORPORA (the honest cell), same plain arm throughout:**

| drafter exit-head arm | drafted ms/token | drafted tok/s | speedup vs plain | acceptance |
|---|---|---|---|---|
| f64, weight re-read per row (rung 4b as first measured) | 21.25 | 47.1 | 1.145× | 2.6730/round |
| f64, **hoisted** (bit-exact — same bytes, less traffic) | 20.54 | 48.7 | 1.182× | 2.6730/round |
| **f32x, hoisted** | **17.53** | **57.0** | **1.383×** | 2.6730/round |

Per pool at f32x: hermes 16.66 ms/token = 60.0 tok/s (1.456×), eigen 18.40 = 54.3 tok/s
(1.317×). Identity 96/96 prompt-runs in every cell. Plain reproduces at 24.24–24.34
ms/token across all three cells, which is the check that only the drafter moved.

**+9.9 tok/s on the honest number from one 21%-of-a-round kernel**, of which +1.6 came
free from the bit-exact hoist and +8.3 from the accumulation class. The bit-exact hoist is
landed as the DEFAULT; the accumulation class stays `MEMRA_DSV4_DSPARK_HEAD_ARM=f32x`,
**default f64**, pending owner ratification — the evidence above is the ratification
packet, and the reason it is not self-ratified is that the drafter is a lane-10-GATED
component and this lane does not change a gated component's default without the owner's
call, however strong the evidence.

## Rung 4 — bar position, honestly

| bar (canada-quant, 2×PRO6000, bs=1) | best MEASURED | gap |
|---|---|---|
| plain 47.5 tok/s | 40.5 (fixture s≈200) / 41.3 (owner corpora) | **−6.2 to −7.0** |
| drafted ~70 tok/s | **57.0 (owner corpora, f32x head arm)** | **−13.0** |
| drafted, fixture-stream figure (do NOT quote) | 61.0 overall / 57.0 at [0,200) | — |

Remaining levers, with the arithmetic the profile supports (projections, not claims):

1. **Proposal truncation on the confidence head** — the round forwards 5.90 positions to
   accept 2.67, and routed-expert traffic is 23.9 ms/round scaling with T. Truncating to
   T≈4 removes ≈7.7 ms/round of expert traffic; if tokens/round falls 3.70 → ≈3.45 the
   per-token cost goes 17.5 → ≈16.6 ms ⇒ **≈60 tok/s**. Needs its own gate (it changes the
   proposal contract) and the banked confidence separation (5.41 vs 0.75) is the signal.
2. **Drafter expert GEMMs on the native indirect dispatch** — the drafter's 3 blocks still
   run the prefill-class `dsv4_fp4_gemm_kernel` (1.91 ms/round) rather than the lane-8
   fused indirect path used by the trunk.
3. **FR-Spec row-trim on the markov chain** (banked follow-up): 5 slots × two
   [129280, 256] f32 reads per round.
4. **D2D API mass**: 1,733 device-to-device copies per round (≈6 ms of API), most of them
   the §3.1 compressor snapshot/replay row moves — fusible into a single per-layer kernel
   without touching the gated arithmetic.

Even summed optimistically these project ≈62–65 tok/s drafted, so **iteration 3 as it
stands does not clear the ~70 drafted bar**, and it should not be reported as if it does.
What it does deliver, measured and gated: **1.383× on owner traffic with byte-exact output
identity**, and a batched-verify path that is bit-identical to sequential decode — which
is the foundation any further drafted rung stands on.

### Rung 4d — post-hygiene regression gate: [PASS], both arms

The clippy pass touched the coarse index-build loop (indexing `nbs[i]` → iterating `nbs`,
behaviour-neutral) and a `checked_div` guard, so the decisive gate was re-run rather than
assumed. `~/it3-rung4d.sh`, logs `~/dsv4-it3-out/rung4d-gate-{f64,f32x}/gate.log`:

- f64 arm: (b) 160/160 LITERAL · (c) 14 cells / 77 logit rows / 3,206 classes ALL
  BIT-IDENTICAL · (d) rings BIT-identical · (e) determinism ×2 · accept sha
  `150342bae32b38b5…`
- f32x arm: identical verdict lines, identical accept sha.

Both shas match the pre-hygiene run at `4dd459dbc`, so the refactor is confirmed
behaviour-neutral on the instrument that would notice.

## Lane hazard hit and recovered (worth banking for the shared checkout)

Mid-session, `refs/heads/lane/dsv4-flash-loader` was **deleted out from under this
worktree** by a concurrent lane (branch sweep): `git log` reported "No commits yet" while
the working tree was intact and three of this rung's commits were already made. The
commits themselves were never lost — the WORKTREE's own reflog
(`.git/worktrees/wt-dsv4-loader/logs/HEAD`) still held the full chain, so recovery was
`git update-ref refs/heads/lane/dsv4-flash-loader <tip from that reflog>`.

Two takeaways for anyone sharing this checkout: the per-worktree `logs/HEAD` is the
recovery instrument when a branch ref disappears (the branch's own reflog under
`.git/logs/refs/heads/...` goes WITH the ref); and unpushed lane work should carry a **tag**
as well as a branch, because a branch sweep will not take a tag. This lane's bank is tagged
`dsv4-it3-rung4-bank`.

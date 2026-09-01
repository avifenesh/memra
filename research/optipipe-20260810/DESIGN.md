# Optimistic Round Pipelining for c=1 Speculative Decoding on PP-2 — DESIGN

Lane: `lane/optipipe` — DESIGN ONLY, no GPU work, no build.
Date: 2026-08-10. Base: `dc77de73` (post-specmech merge).

## Mission

Owner's idea (2026-08-10): "keep generating the current step for the next, instead of
stop." While round N's verify occupies stage1 + head, round N+1 drafts from the
OPTIMISTIC full-accept tip and starts its stage0 verify immediately — stage0 never
idles. On mis-prediction (round N accepts fewer than K+1 tokens), discard round N+1's
in-flight work and re-draft from the true tip.

This is the mid-body successor the specmech merge receipt demands: the missing gain is
in VERIFY ISSUE ITSELF (whole-PP-body serialization), and any future attempt needs
boundary-level release after stage-0 TX. This design prices that attempt for c=1
cross-ROUND overlap (vs specmech's cross-SESSION c=2 overlap).

## Sections

1. [EV arithmetic](#1-ev-arithmetic) — complete
2. [State-fork design](#2-state-fork-design) — complete
3. [Mid-body release](#3-mid-body-release) — complete
4. [Numeric-class analysis](#4-numeric-class-analysis) — complete
5. [Failure modes](#5-failure-modes) — complete
6. [Build bill](#6-build-bill) — complete
7. [Verdict](#7-verdict) — complete

## 1. EV arithmetic

### Honest verdict up front

At 100% overlap (every optimistic window valid, q=1) the pipeline DOES beat plain
c=1 — projected 97.6–117.4 tok/s vs plain 81.188. So the mechanism is not dead on
arrival at the ceiling. But at the MEASURED draft-chain quality the validity
probability of an optimistic window is q ≈ 0.38–0.53, and the projection lands at
**75.4–88.1 tok/s vs plain 81.188** — centered at break-even, negative under the
defensible receipt-derived q and the defensible overhead split. The EV does not
robustly clear plain c=1. Detail below; verdict implications in §7.

### The validity probability is NOT the 73.68% acceptance rate

The critical modeling correction. Task framing said "prob p=0.7368 for the drafted
continuation to be the true tip" — that is wrong, and the receipts show why.

A K=1 round's verify batch is `[pending, d1]` (T=2): col 0 is the carried bonus
token (the target's own argmax from the previous round — committed unconditionally),
col 1 is the draft (`spec.rs:4214-4216`: the carried pending "enters the round loop
as round-0's pending — verify col 0"). Full accept of round N means d1 matched.
The NEW pending for round N+1 — `bonus_N` — is the argmax of round N's HEAD output
at its last accepted column. Round N+1's verify col 0 must be `bonus_N`.

`bonus_N` only exists after round N's stage1+head runs. So an optimistic round N+1
issued while round N's stage1 is still executing cannot use the true bonus — it must
substitute the DRAFTER's own next chain token `d2` for it, and draft `d3` on top.
Round N+1's optimistic window is `[d2, d3]`, and it is valid iff:

1. round N fully accepted (d1 == target argmax), AND
2. d2 == bonus_N (the drafter's 2nd chain token matches the target's argmax after d1).

That is exactly "the draft chain is correct 2 tokens deep" — which is precisely the
**K=2 full-accept rate**, measured directly in the specpp2 K sweep with the same
drafter, same model, same greedy determinism.

### Receipt-derived probabilities

From `research/specpp2-20260810/raw/k-sweep/` (deterministic — identical histograms
across all 5 repeats; measured requests = bursts of 16/18/23/17 rounds at K=1 and
13/17/17/16 at K=2/K=3):

| Quantity | Value | Source |
|---|---:|---|
| p1 = P(1-deep chain correct) | 54/74 = **0.7297** | K=1 `full_accept` sums, `r5-K1-server.log` (RESULTS.md:165 "72.97%"; PROGRESS.md:61 quotes 73.68% incl. warmup) |
| q_low = P(2-deep chain correct) | 24/63 = **0.381** | K=2 `full_accept` sums (7+7+6+4)/(13+17+17+16), `r1-K2-server.log` |
| K=2 slot-1 marginal | 41/63 = 0.651 | same logs |
| cond = P(slot2 \| slot1) at K=2 | 24/41 = 0.585 | same logs |
| q_mid = p1 × cond | 0.7297 × 0.585 = **0.427** | mixes K=1 marginal with K=2 conditional |
| q_high = p1² | **0.5325** | optimistic bound: assumes slot-2 conditional as good as slot-1 marginal — refuted by the measured cond=0.585 < 0.730, kept as the upper bracket |
| P(3-deep chain) | 3/61 = **0.049** | K=3 `full_accept` sums — kills K=2-sized optimistic windows and depth-2 pipelining outright |

### Round-interval model (anatomy level)

From `research/specpp2-20260810/RESULTS.md:76-97` (K=1, c=1 anatomy): draft
0.7045 ms, stage0 verify 8.450 ms, TX 0.013, RX 0.014, stage1+head 8.720 ms,
accept 0.024, commit 0.147, other 0.007 — serial round total **18.1045 ms**.

Optimistic depth-1 pipeline steady state:

- device 1 (head stage) per window: stage1+head 8.720 + accept/commit 0.178 +
  draft-chain extension ~0.70 = **9.60 ms** (same stage-balance bound specmech's
  mechanism bill computed for c=2, `research/specmech-20260810/RESULTS.md` gap
  decomposition / specpp2 RESULTS.md:250-255).
- device 0 per window: stage0 8.450 ms — fits under 9.60. Stage0 never idles while
  the optimism holds.
- Valid-successor window interval: **I_hit = 9.60 ms**.
- Mispredict: the in-flight window's stage0 work is discarded (no critical-path
  cost — it ran concurrently on the otherwise-idle device 0), but the replacement
  window runs fully serial from re-draft: 0.70 + 8.45 + 8.72 + 0.18 =
  **I_miss ≈ 18.05 ms**.

Mean interval: I(q) = q·9.60 + (1−q)·18.05. Tokens per resolved window are
unchanged from serial spec: E[tok] = 1 + p1 = 1.7297 (a valid optimistic window is
byte-for-byte a normal round — §4).

| q | I(q) ms | speedup vs serial 18.10 |
|---:|---:|---:|
| 0.381 | 14.83 | 1.221× |
| 0.427 | 14.44 | 1.254× |
| 0.5325 | 13.55 | 1.336× |
| 1.0 (ceiling) | 9.60 | 1.885× |

### End-to-end projection vs plain

Serial spec c=1 K=1 measured **65.918** tok/s; plain c=1 **81.188**; plain c=2
**115.230** (specpp2 RESULTS.md:163-166, 180-183; N=5, box1). Anatomy rounds
(18.1 ms × 74 rounds = 1.34 s) do not account for the measured per-request wall
(128/65.918×4-request aggregate → ≈1.94 s/request): ≈0.13–0.60 s per request is
setup/prefill/SSE/uninstrumented round overhead. Bound it both ways:

- **Case A** (fixed 0.13 s, rounds 1.81 s — uninstrumented overhead scales with
  round rate): tok/s = 128/(0.13 + 1.81/ratio).
- **Case B** (fixed 0.60 s, rounds 1.34 s — anatomy rounds only, rest is fixed):
  tok/s = 128/(0.60 + 1.34/ratio).

| q | Case A tok/s | Case B tok/s | vs plain 81.188 |
|---:|---:|---:|---|
| 0.381 (measured 2-deep) | 79.4 | 75.4 | **−2.2% / −7.1%** |
| 0.427 | 81.3 | 76.7 | +0.1% / −5.5% |
| 0.5325 (optimistic) | 86.2 | 79.9 | +6.2% / **−1.6%** |
| 1.0 (impossible ceiling) | 117.4 | 97.6 | +44.6% / +20.2% |

Read: the single defensible receipt-derived probability (q=0.381, the K=2
full-accept rate) loses to plain in BOTH overhead splits. Only the optimistic
q_high in the favorable overhead split wins, by +6.2%. The ceiling beats plain
c=1 but never approaches plain c=2's 115.2 — and this is a c=1-only mechanism;
it does nothing for c≥2 where spec already loses 42.8% (specpp2 RESULTS.md:182).

### Depth and K variants — both closed by the receipts

- **Depth-2 pipelining** (two optimistic windows in flight): window N+2 needs a
  4-deep chain; with cond ≈ 0.585 per extra token, P ≈ 0.381 × 0.585² ≈ 0.13.
  Mean interval barely improves over depth-1 while discard traffic explodes. Dead.
- **K=2 optimistic windows**: validity needs a 3-deep chain = 0.049 measured.
  I(0.049) ≈ 17.6 ms for a bigger window but serial K=2 already loses 28.12%
  (specpp2 RESULTS.md:166). Dead.
- Depth-1, K=1 is the only arm worth pricing, and it prices at break-even.

## 2. State-fork design

### What the tip actually is

Round state at the moment round N's stage0 finishes (all `crates/memra-engine/src/spec.rs`):

- **Trunk KV/recur cache**: per-layer full-attn `len`/`len_d` plus in-place GDN
  conv/ssm state. Snapshot taken BEFORE each round (`spec.rs:5262`
  `cache.snapshot_into(e, &mut snap)? // §C: snapshot BEFORE draft+verify`;
  persistent buffers allocated once at `spec.rs:5104`).
- **MTP draft scratch**: `scratch.set_len(e, pos + base0 - 1)` at round start
  (`spec.rs:5274`) IS the draft-side rollback — rows are position-addressed and
  never revisited below the truncation point (`spec.rs:455-467`: "rewinding it is
  also just a `len` reset").
- **Seed chain**: `h_seed_buf` (hidden of `last_token`) and `fill_prev` (hidden of
  the row before it), `spec.rs:4640-4647`. Round N's accept updates them from the
  verify's own hidden stack (`spec_seed_gather`, `spec.rs:5702` region).
- **Pending bonus**: `pending`/`SpecSession::pending_tok` — the uncommitted token
  riding as col 0 of the next verify (`spec.rs:357`, `spec.rs:4214-4216`).
- **Host tails**: `out`, grammar/penalty state, telemetry.

### Fork contents for optimistic round N+1

Round N+1 forks at the OPTIMISTIC tip = "round N fully accepted": committed len
advances by K+1 = 2 rows, pending' = d2 (drafter chain token 2), draft = [d3].

| State | Fork action | Cost |
|---|---|---|
| Full-attn KV per layer | NOTHING — append-only, position-addressed (`memra-kv/src/lib.rs:403-414`). Round N's verify already appended rows at [pos, pos+2); N+1's verify appends at [pos+2, pos+4). Rows below pos+2 are shared by construction | free |
| KV `len`/`len_d` | the hard part — see the two-counter problem below | one device counter |
| GDN conv/ssm | round N's verify already advanced them T=2 in place; the optimistic fork's "state at the tip" is exactly the post-verify state. N+1's snapshot (its own `snap` refresh) copies it — the EXISTING per-round `snapshot_into` D2Ds. NOT free but already paid every round today | already in round cost |
| MTP scratch | draft chain for [d2, d3] appends rows normally; on discard, `scratch.set_len` truncates (`MtpScratch::can_rewind_to`/`set_len`, `spec.rs:3570,3588`). The TRUE-HIDDEN REFRESH (`spec.rs:4244-4247`) rewrites committed positions from verify hiddens each round, so a stale optimistic row is overwritten before any future read | free |
| `h_seed_buf` / `fill_prev` | must be DOUBLE-BUFFERED: N+1's draft needs the optimistic seeds (chain hiddens `h_nextn` from the drafter, `spec.rs:1789-1795`) while N's accept path may still rewrite the real ones. Two [n_embd] f32 buffers | 2 x n_embd x 4B, trivial |
| pending/out/grammar | host-side shadow copies; grammar is the existing per-round clone (`spec.rs:5288` region) | trivial |

### The two-counter problem (the real design center)

Round N's stage1 FA kernels read each layer's `len_d` as the attention bound, and
the devacc path REWRITES `len_d` after accept (`spec_rollback_kv`,
`spec.rs:5705-5711`, `lib.rs:1907-1911`; `commit_verified_prefix` host-arm mirrors
at `spec.rs:2765-2769`). If N+1's stage0 verify enqueues appends that bump the SAME
`len_d` while N's stage1 kernels are still queued behind it on another stream, the
bound races.

But the stage0/stage1 split saves us: KV is STAGE-OWNED (`memra-kv/src/lib.rs`
`new_pp2`/`new_ppn`, :427-461 — each layer's KV lives on the device that runs its
stage). Stage0 layers' `len_d` counters are touched only by stage0 work; stage1
layers' only by stage1+head work. Cross-round ordering per stage is total (one
stream per stage, `pp.rs` stage streams), so within stage0 the sequence is:
N-stage0-appends, then N+1-stage0-appends — no interleaving. The race is only the
ROLLBACK: N's accept writes stage0 layers' `len_d` (rollback to saved+base+n_acc)
from the primary/stage1 side AFTER N+1's stage0 has already appended optimistically.

Resolution, two cases:

- **N fully accepts (the q case)**: rollback writes `len = saved + 2` — exactly the
  value N+1's optimistic append started from. `spec_rollback_kv`'s write is
  idempotent-by-value here; the only requirement is that it not land BETWEEN N+1's
  append and N+1's FA read on stage0's stream. Fix: in optimistic mode, on full
  accept, SKIP the stage0-layer `len_d` writes entirely (host mirror `kvl.len`
  still advances — it is host-only bookkeeping). The devacc kernel already has the
  accept count on device; gating the write on `n_acc == k+1` is a one-branch kernel
  change, and the host arm's `kv_lens_done` flag (`spec.rs:2753,2766`) already
  models "lens were written elsewhere".
- **N mis-predicts**: N+1's in-flight stage0 work is garbage. Discard = the
  existing rollback semantics: `len_d` write to `saved + base + n_acc` (already
  what `spec_rollback_kv` does) plus `scratch.set_len` truncation plus dropping
  N+1's host shadows. The one NEW ordering requirement: the rollback write must be
  ordered AFTER N+1's queued stage0 kernels have drained (else a queued FA reads a
  rolled-back bound mid-flight and the append kernels re-extend it). Cheapest
  correct form: stage0's stream is fenced behind the accept decision before the
  rollback lands — one event wait on the mis-predict path only (I_miss already
  budgets a full serial round, §1).

Is `spec_rollback_kv` + `commit_verified_prefix` sufficient for discard? For
full-attn KV and MTP scratch, YES — len truncation is the whole mechanism, and
optimistic rows above the true tip are dead by construction (next append
overwrites them). For GDN conv/ssm, NO EXTRA MECHANISM NEEDED but one extra
constraint: N+1's stage-resident linear layers advance conv/ssm IN PLACE during
its optimistic verify. On discard, restore from N+1's own `snap` (the per-round
snapshot N+1 took at fork time — which equals N's post-verify state = the state
`commit_verified_prefix` would have produced on full accept... but on MIS-predict
N's own commit rebuilds conv/ssm to `saved + n_acc + 1 < saved + 2`
(`ssm_conv_ring_rebuild`/`gdn_scan_s128`, `spec.rs:2773-2828`), which is NOT what
N+1 snapshotted. So the discard path must roll conv/ssm back to N's PRE-round
snapshot and let N's normal commit arms rebuild the true prefix — i.e. N's `snap`
must stay alive until N+1 validates. Two persistent snapshot buffer sets
(alternating N/N+1), doubling today's one-set cost (`spec.rs:5104` comment: ~50MB
class per set for the linear layers).

### Ring sessions: EXCLUDE

`MEMRA_SWA_RING` Step35 sessions wrap the SWA plane physically
(`memra-kv/src/lib.rs:188-192` `can_rewind_to`: rewind legal only while the
aligned view start hasn't been lapped; `rollback` refuses at :639-641). An
optimistic round writes 2 extra rows ahead of the true tip; on a ring layer near
its lap boundary those writes can evict the exact rows a mis-predict rollback
needs. `can_rollback` (:554-566) would catch it AFTER the fact with a hard error
("full re-prime required") — correct but a serving-latency cliff. The prefix-cache
precedent already refuses ring sessions (`memra-server/src/worker.rs:4756`).
Admission rule: `cache.has_swa_ring()` (`memra-kv/src/lib.rs:551-553`) => serial
spec, no pipelining. Lap-safety proof is possible (bound: optimistic overhang ≤
K+1 rows ≤ alignment slack) but not worth carrying in increment 1.

### Round-stream burst interaction

The ROUND-STREAM burst arm (`spec.rs:5153-5262`) already runs M rounds with zero
readbacks — it is the OTHER answer to host-sync elimination, but it is
single-stage-stream serial: each burst round's verify still walks stage0 then
stage1 in issue order, so it does not overlap stage0 with stage1 across rounds.
Optimistic pipelining and the burst arm are mutually exclusive per burst
(`stream_active` gate at `spec.rs:5153`); increment 1 targets the non-burst path
and must refuse when `stream_active` is on rather than compose with it.

## 3. Mid-body release

### Two boundary slots are sufficient

No third boundary slot is needed for the only live design (depth 1, K=1).
`BoundaryRt` already owns two persistent slots and `tx_pipelined()` returns the
actual slot ticket chosen by its boundary-local atomic (`pp.rs:453-475,
1026-1043`). More importantly, stage1 does NOT consume the persistent slot for
its whole trunk. `rx()` waits `ev_tx`, copies the slot into a fresh
stage1-owned `work` buffer, then records `ev_rx` immediately after that copy
(`pp.rs:1103-1133`). The stage1 layers and head read `work`; they no longer
touch the boundary slot.

Therefore the maximum live boundary ownership is exactly:

```text
slot a: round N     TX published -> stage1 RX-copying/released
slot b: round N+1   optimistic stage0 TX publishing
```

Round N cannot produce its accept decision until stage1 has passed round N's
RX copy and completed the head. By the earliest instant a full accept permits
round N+2 stage0 to start, slot a's `ev_rx` has necessarily completed. Even if
stage1 consumed the slot for its whole body, that causal bound would still make
two slots enough. The actual early copy makes the ownership interval shorter.
A third slot would buy only depth-2 issue (N+2 before N resolves), which §1 has
already closed on acceptance probability.

A miss still consumes a `tx_pipelined()` ticket. Do not derive the slot from
logical round parity; carry the returned ticket with the optimistic round. If
the round is discarded before RX, its old `ev_rx` is already complete and
there is no current reader, while stage0 stream order puts any later overwrite
after the discarded TX. Re-recording the same CUDA event is legal: a stream
wait binds the most recently recorded state visible at the time of the wait and
is unaffected by a later record ([CUDA 13.3.1 Event
API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html)).
No synthetic RX or "cancelled slot" state is required.

### The missing seam is issue release, not storage

The specmech schedule already alternates these slots across sessions, but B's
verify is released only after A's entire `decode_step_t_core_ppn()` has returned
and set `verify_done` (`spec.rs:576-600`; `specmech RESULTS.md:139-141`). The
needed API split is:

1. `verify_stage0_issue(...) -> VerifyBoundaryTicket`: enqueue embed, stage0
   layers, boundary TX, and `ev_tx`; return the slot plus round-local handles as
   soon as the TX record is issued.
2. `verify_stage1_finish(ticket) -> VerifyResult`: enqueue RX, stage1 layers,
   output norm/head, and final publication.

At round N's TX record, the coordinator releases round N+1's optimistic
`verify_stage0_issue` while the first caller continues with round N's
`verify_stage1_finish`. This is the boundary-level release the specmech receipt
requires. It must use the two stage-specific engines concurrently; wrapping
both halves in the existing primary-engine mutex would recreate the measured
whole-body serialization.

### Named fence sequence

Use events for byte ownership and a host decision for control; no device-wide
synchronize belongs on the steady-state hit path:

```text
F0  reverse-ready:
    primary records E_ready after prior caller-stream consumers / fork inputs;
    S0 waits E_ready.  Use the existing fence_stages_behind law once at
    pipeline admission or miss restart, not as a whole-body fence per successor.

F1  boundary publish (existing):
    S0_N waits slot[s].ev_rx -> stage0_N -> TX slot[s]
         -> record slot[s].ev_tx -> MID-BODY RELEASE N+1.S0.

F2  boundary acquire/release (existing):
    S1_N waits slot[s].ev_tx -> RX-copy to work_N
         -> record slot[s].ev_rx -> stage1_N + head -> record E_head_N.

F3  resolve:
    primary/devacc waits E_head_N -> accept_N -> existing 8-byte accept readback.
    FULL: publish E_hit_N, commit N, then release N+1.S1 and N+2.S0.
    MISS: do not issue N+1.S1; order stage0 rollback after N+1.S0/TX,
          restore the alternating snapshot, then record E_restart_N.

F4  successor acquire:
    on FULL, N+1.S1 waits its carried slot ticket's ev_tx; N+2.S0 uses the next
    tx_pipelined ticket.  On MISS, replacement S0 waits E_restart_N.
```

`E_head_N` can be the last-stage-to-caller publication already represented by
`publish_to`; `E_ready`/`E_restart` are targeted forms of the existing reverse
publication law. `E_hit_N` is a logical coordinator edge in increment 1 (the
current devacc readback already supplies the host verdict), not a third
boundary event. The non-negotiable partial order is:

```text
N.S0/TX < N.S1/head < accept(N) < N+1.S1
N.S0/TX < N+1.S0/TX < {hit continuation | miss rollback}
```

This preserves one optimistic stage0 window and prevents invalid stage1 work
from entering the miss critical path assumed by §1's `I_miss`.

## 4. Numeric-class analysis

### Contract: change issue time, never round math

Optimistic pipelining is a scheduling transform, not a new numeric class. For
every COMMITTED round, hold this tuple fixed relative to serial K=1 spec:

```text
(verify tokens [pending, d1], pos0, pre-round cache state,
 stage cut, layer/row traversal, kernel dispatch, launch geometry,
 reduction order, output-norm/head path)
```

The only changed coordinate is the wall-clock time at which the next round's
stage0 launches are enqueued. Do not concatenate two rounds, change T=2, batch
their rows, fuse across the round boundary, or select a different kernel arm.
Those would be numeric changes and are outside this design.

On a full-accept hit, the optimistic `[d2, d3]` window, positions, KV prefix,
and recurrent input state are exactly the serial round N+1 inputs (§1-2).
Stage0 executes the same layer subgraph on those bytes. On a miss, no optimistic
value is observable: its stage1 is never issued, its stage0 state is restored,
and the replacement round executes the ordinary serial subgraph from the true
tip. Discarded arithmetic is not allowed to feed logits, accepted tokens,
router state, grammar state, or the replacement's recurrent/KV state.

### Why reduction order is unchanged

Each PP stage retains one totally ordered stage stream:

```text
S0: N.stage0 -> N+1.stage0 -> (N+2.stage0 or miss rollback/replacement)
S1: N.stage1/head -> N+1.stage1/head -> ...
```

No two rounds execute the same layer concurrently, no reduction spans the PP
cut, and the boundary operation is a straight f32 copy. Cross-device timing may
change which stage is busy first, but cannot reorder a dot product, RMS sum,
router reduction, attention combine, or GDN recurrence inside either round.
The layer/row loops in `step35_verify_batch_layers()` remain layer-major with
rows advanced in the same order (`spec.rs:2128-2172`); splitting issue at the
existing stage boundary does not alter those loops or their launch parameters.

This is stricter than saying greedy argmax probably stays stable: for a hit,
the per-round device results must be bit-identical to serial issue. A cache hit
caused by wrong-path warming may remove I/O, but it may not select a different
math kernel or pointer layout. Any resource whose hit/miss changes arithmetic
dispatch fails admission until the two routes are bit-paired.

### The b1fix one-class rule remains authoritative

The b1fix contract is that PP-N Step3.5/Step3.7 always uses
`step35_decode_batch_layers` at every live serving width, including B=1; it may
never fall back to the eager/fused B=1 class (`b1fix RESULTS.md:24-37`,
`decode_batch.rs:781-806`). Verify already mirrors that rule:
`step35_verify_batch_layers()` calls the same authoritative batched layer walk
for each verify row, and the PP head uses the Step35 `rms_norm + matmul` branch
(`spec.rs:2128-2172, 2100-2111`).

Optipipe must preserve that call graph in BOTH halves of the split API. In
particular:

- stage0 and stage1 continue calling `step35_verify_batch_layers()` on their
  existing `[lo, hi)` ranges;
- the confidence gate changes only serial-vs-overlapped issue, never the
  Step35 dispatch predicate;
- a threshold crossing mid-session is legal because both sides execute the
  same numeric class; and
- fail-closed behavior under `MEMRA_STEP35_BATCH=0` remains unchanged — no
  optipipe fallback may reopen eager B=1.

Thus traffic history may determine WHEN a round is issued, but not WHICH
program computes it. That is the exact distinction b1fix requires: scheduling
history must not recreate multiple completion hashes.

## 5. Failure modes

### THE DECISIVE VARIANT: confidence-gated pipelining

Import DSpark's scheduling principle, not its drafter architecture. DSpark
chooses verification work from predicted prefix survival and a measured engine
throughput curve ([paper, arXiv 2607.05147](https://arxiv.org/abs/2607.05147));
here the only choice is whether to issue the one optimistic successor stage0.

Define the observed validity label precisely:

```text
v_N = 1 iff round N accepts d1 AND d2 == bonus_N
    = the two-deep chain / K=2 full-accept event, not K=1 acceptance.

q_hat = mean(v) over the last W resolved opportunities in the same
        (target hash, draft hash, prompt-template/workload bucket, PP rig) class.
```

The default-off arm remains PLAIN c=1. A low-confidence request must not fall
back to serial K=1 spec (65.918 tok/s), which already loses 18.81% to plain.
The gate admits the combined `K=1 spec + depth-1 optipipe` mechanism only when
its expected value clears plain.

### Economic threshold

From §1, `I(q) = 18.05 - 8.45q` ms and the two end-to-end bounds are:

```text
S_A(q) = 128 / (0.13 + 1.81 * I(q) / 18.10)
S_B(q) = 128 / (0.60 + 1.34 * I(q) / 18.10)
```

Solving `S(q) >= 81.188 * (1 + margin)` gives:

| required margin over plain | q in Case A | q in Case B | robust threshold |
|---:|---:|---:|---:|
| tie (0%) | 0.424 | 0.575 | **0.575** |
| +3% | 0.478 | 0.648 | **0.648** |
| +5% | 0.513 | 0.695 | **0.695** |
| +10% | 0.594 | 0.804 | **0.804** |

Use a 5% guard band because the fixed-overhead split and the new fence/state
costs are projections, not measurements. The design threshold is therefore
**q* = 0.695, operationally 0.70**. At q=0.70, `I=12.135 ms` and the projection
is **95.27 tok/s (Case A) / 85.42 tok/s (Case B)**: +17.35% / +5.22% over
plain. q=0.60 is only +10.41% / +1.00%; that is too close to measurement noise
and omitted implementation overhead to admit.

This threshold is a pre-build price, not a forever constant. Once the real
`I_hit`, `I_miss`, state-reconcile cost, and memory pressure are measured on the
target rig, re-solve the same equation and freeze the resulting threshold in
the receipt. If the measured conservative threshold exceeds the traffic's
upper tail, the feature stays off.

### How much measured traffic qualifies?

The K=2 acceptance receipts expose burst aggregates, and all five repetitions
are deterministic. Excluding each run's warmup, the distribution is:

| burst shape | full accepts | q | repetitions |
|---:|---:|---:|---:|
| 13 rounds | 7 | 0.538 | 20 |
| 17 rounds | 7 | 0.412 | 20 |
| 17 rounds | 6 | 0.353 | 20 |
| 16 rounds | 4 | 0.250 | 20 |
| **total** | **480 / 1,260** | **0.381** | **80 bursts** |

At q*=0.70, **0/80 measured bursts and 0/1,260 round-weighted observations
(0%) belong to an above-threshold burst**. Even the five warmup bursts are
4/6=0.667 and do not qualify. This is the only defensible traffic-fraction
estimate from the receipts: the logs do not retain the ordering of individual
`v_N` bits, so they cannot reconstruct how often a short sliding window would
transiently cross q*. Do not turn that missing sequence into a positive-tail
claim. On the measured prompt corpus the gate stays plain.

### Gate state machine and storm breaker

- Cold start is OFF/plain. Use `W=32`; admission requires at least 32 labeled
  opportunities and `q_hat > 0.70` (integer form: at least 23/32 hits). Labels
  are keyed by exact model/draft hashes and reset when either changes.
- Actual optipipe hit/miss results update the rolling window. A calibrated
  draft-confidence proxy may seed a new request later, DSpark-style, but the
  current head has no confidence head; draft top-prob is only a candidate until
  calibrated against retained `v_N` traces.
- Do not pay serial-spec rounds merely to warm the estimator. Offline/shadow
  calibration may extend a serial draft to d2 and compare it with `bonus_N`,
  but that extra draft step is measured overhead and is not enabled on plain
  production traffic without its own positive EV receipt.
- While ON, one miss takes the normal reconcile path. Three consecutive misses
  OR rolling `q_hat < 0.60` trips a circuit breaker: drain the optimistic
  stage0 ticket, restore state, and exact-demote the session to plain for the
  rest of the request. No in-request re-enable; this prevents threshold
  oscillation and bounds a non-stationary/miscalibrated run to a short loss.
- EOS, stop, budget tail, abort, constraint/sampling, ring cache, round-stream,
  or insufficient VRAM suppresses successor issue. An error after state
  mutation is fail-closed: if restore cannot be proven complete, abort the
  request rather than resume from ambiguous state.

The breaker does not rescue a bad average—q*=0.70 does that. It limits the tail
when a previously good bucket shifts abruptly (tool syntax, code block, language
switch, or head-quality cliff).

### One-extra-round activation memory

Depth 1 does NOT double the whole verify footprint:

- The retained cross-boundary activation is one T=2 f32 residual. For the
  Step-3.7 width (`n_embd=4096`) that is `2 * 4096 * 4 = 32 KiB`, held in the
  opposite persistent boundary slot. The two slots already exist and are
  prewarmed; this design adds no third slot.
- Stage0 N+1 has one transient stage0 working set live while stage1 N runs.
  Stage0 rounds are ordered on one stream, so two stage0 working sets never
  overlap; stage1 N+1 is not allocated before the hit decision. The exact pool
  high-water is implementation-dependent and must be captured, not guessed.
- The optimistic KV overhang is two rows in stage0-owned preallocated caches.
  The material persistent increment is the SECOND recurrent snapshot set from
  §2 (~50 MB class) plus two 4096-f32 seed buffers (~32 KiB), not an extra model
  or expert bank.

Preallocate the second snapshot/seed set and boundary slots before admission.
If the allocation or headroom check fails, remain plain; never discover the
shortfall after optimistic state has been written.

### Device-side validity and discard (devacc)

The existing devacc path already computes `acc_out = [n_acc, bonus]` with
`spec_accept_greedy`, then lets `spec_seed_gather` and `spec_rollback_kv`
consume it before the existing 8-byte D2H (`spec.rs:5674-5714`). Add a tiny
device decision:

```text
valid_N = (acc_out.n_acc == 1) && (acc_out.bonus == optimistic_d2)
```

Record `E_decide_N` after that flag. Stage0's reconcile kernel is queued after
N+1's stage0/TX and waits `E_decide_N`: on hit it leaves the optimistic lengths
alone; on miss it truncates stage0 KV lengths and restores recurrent state from
the correct alternating snapshot. Use stage-local pointer tables/engine so the
head device never races a remote stage's `len_d`.

The host consumes the SAME existing 8-byte readback to decide whether to call
`N+1.verify_stage1_finish(ticket)` or drop the ticket after reconcile. That adds
no acceptance sync. Do not enqueue stage1 speculatively and hope to cancel CUDA
kernels: an event can order work but cannot unschedule it, and wasted stage1
would invalidate §1's miss price. A future fully device-driven continuation
would need conditional graph/dispatch machinery and is a separate increment.

### Failure ledger

| failure | consequence | required containment |
|---|---|---|
| stale or pooled q_hat | admits a losing request | hash/workload-scoped window; OFF on cold start; retain labels |
| miss storm / domain shift | repeated 8.45 ms wasted stage0 + rollback tail | 3-miss / q_hat<0.60 breaker; demote to plain |
| slot ticket dropped without RX | later overwrite races an unknown reader | ticket owns hit-RX or miss-reconcile terminal state; RAII drain on abort |
| stage0 len_d write races optimistic append | wrong FA bound / cache corruption | stage-local decision event and hit-skip/miss-restore kernel (§2-3) |
| snapshot generation overwritten | miss restores N+1 state instead of N | two alternating sets, generation-tagged until decision retires |
| lazy allocation under overlap | hidden sync, OOM, or use-before-init | prewarm all persistent state; capture per-device peak; plain fallback |
| wrong-path cache warming changes dispatch | timing history becomes numeric history | require hit/miss routes to select identical math (§4) |
| request cancellation during in-flight S0 | cache/session freed under GPU work | session teardown waits reconcile/drain event before freeing state |

## 6. Build bill

### Effort class

**Total: L, correctness-risk high.** There is no new model math and only one
tiny control kernel, but this crosses three difficult ownership domains at once:
two CUDA contexts/streams, mutable speculative cache state, and a scheduler
mode transition. Treat it as four reviewable increments plus a promotion block;
do not land it as one diff.

| increment | deliverable | primary files | effort | evidence produced |
|---|---|---|---:|---|
| 0. Mid-body seam | Split PP verify into `verify_stage0_issue -> VerifyBoundaryTicket` and `verify_stage1_finish`; serial wrapper calls both immediately. Rewire the existing two-session experimental coordinator to release peer stage0 at TX | `crates/memra-engine/src/spec.rs`, `crates/memra-engine/src/pp.rs` | M | **Standalone measurable:** true A.S1/B.S0 timeline and pair interval, without same-session state fork |
| 1. Fork + reconcile | Two alternating snapshot/seed generations; stage-local KV-len pointer tables; device validity flag; hit-skip/miss-restore; generation-tagged ticket teardown | `spec.rs`, `round_stream.rs`, `lib.rs`, `cu/spec_sample.cu`; `crates/memra-kv/src/lib.rs` only if a stage-range helper is genuinely missing | L | forced-hit, forced-miss, abort, and alternating-generation state identity; reconcile latency + per-device peak memory |
| 2. Depth-1 controller | One same-session optimistic stage0, no stage1 before resolve; carry actual boundary ticket; forced ON/OFF diagnostic door; retain per-attempt `v_N`, hit/miss, breaker, and phase timings | `spec.rs`, `pp.rs`, `crates/memra-server/src/worker.rs` | M | first end-to-end optipipe throughput, exactness, and observed `I_hit/I_miss`; still not automatically admitted |
| 3. Confidence admission | Hash/workload-scoped W=32 estimator, q*=measured conservative threshold, cold-off/plain policy, storm demotion, metrics/reason strings, unit-tested thresholds | `worker.rs`, `spec.rs`, `docs/FLAGS.md`, `docs/TESTING.md` | M | decisive auto-vs-plain result on low-q and retained high-q traffic |
| 4. Promotion block | Target-rig interleaved runs, raw logs, report, pre-release battery; default only if the auto arm clears plain with the frozen margin | `research/optipipe-<date>/`, gate scripts; generated perf surfaces only if published numbers actually move | M rig time | ship/hold receipt; no promotion from a forced-only win |

### Increment 0 is the one independently measurable piece

Increment 0 needs no optimistic cache semantics because the existing specmech
pair owns two independent sessions. It can answer the prerequisite question:
does a release exactly after A's TX let B stage0 issue early enough that GPU
execution actually overlaps A stage1? Require event timestamps proving
`B.S0.start < A.S1.end`, retain the raw trace, and compare N=5 pair intervals to
the current 48.66 ms sum-shaped receipt.

That result is a seam verdict only. A positive cross-session overlap does not
override §1's q economics; a negative result kills increments 1-4 before state
fork work. Increment 1's rollback latency can be micro-measured, but it cannot
produce a throughput verdict without increment 2. The first complete c=1 EV
measurement is increment 2; the first deployable-policy measurement is
increment 3.

### Required gates by increment

**Increment 0 (pure split / pair localizer):**

- `decode-batch-gate --mode ppspec --ts 2,5,9 --reps >=2`, both device orders;
- existing specmech plain/serial/pair byte identity and c=1 serial fallback;
- `pp-transport-smoke`; and
- an issue trace with TX release, S0/S1 start/end, slot id, and event generation.

**Increment 1 (state mutation):** add a dedicated `optipipe-gate` or equivalent
engine harness with scripted all-hit, miss-at-each-position, alternating hit/miss,
three-miss storm, EOS, cancellation, and allocation-failure cases. After every
case compare against serial: cache `pos`, every host/device KV len, recurrent
state bytes, MTP scratch len, pending/seed bytes, next logits, and emitted ids.
Run a long alternating-generation soak; one green replay is insufficient for
the cross-stream flake class documented in `docs/TESTING.md`.

**Increments 2-3 (observable serving behavior):**

- `kernel-check` ALL GREEN;
- `run-gen` argmax MATCH on Step-3.7;
- `run-spec` K=1..8 self-consistency (shared spec code changed even though
  optipipe admits only K=1);
- `decode-batch-gate --mode ppspec`, including Step35 T=2 and both PP orders;
- `tools/accept-gate.sh --full` with unchanged acceptance counts/text hash;
- `tools/serve-smoke.sh`, `tools/serve-stress-gate.sh`, and the manual
  `research/spec-gate-20260806/exactness.py` phase-switch/demotion matrix;
- the b1fix one-hash matrix extended with optipipe OFF, forced hit/miss, q gate
  crossing, first-late, and breaker demotion; and
- explicit refusals for ring, round-stream, sampled/constrained, c>=2, non-PP2,
  same-device PP, host-bounce, and insufficient-headroom shapes.

### Performance/promotion matrix

All performance points are N=5 interleaved on the designated 2x RTX PRO 6000
verification box, with raw logs, per-run q distribution, thermal regime, and
per-device peak memory retained. The local 5090 remains a correctness/iteration
gate; it cannot establish a cross-device PP-2 performance claim.

1. **Seam A/B:** increment-0 old whole-body release vs TX release, existing
   specmech c=2 shape. This decides whether to continue.
2. **Low-q guard:** the exact specpp2 prompt corpus. Auto must engage 0% at the
   frozen q* and be byte/performance equivalent to plain (no online shadow tax).
3. **q strata:** retained workloads with measured q in `<0.60`, `0.60-0.70`,
   `0.70-0.80`, and `>=0.80`; compare plain, forced optipipe, and auto. Do not
   synthesize a high-q label from aggregate K=1 acceptance.
4. **Miss anatomy:** separately retain hit, single-miss, and storm intervals,
   including rollback/event time and concurrent GPU state. Re-solve q* from
   these measured values.
5. **Memory:** fresh-session and steady-state peaks on both devices, with the
   second snapshot preallocated. Allocation refusal must fall back before any
   state mutation.
6. **Pre-release:** the project battery on Vast 2x PRO 6000—`kernel-check`,
   `run-gen`, `run-spec` K=1..8—and repeat the decisive A/B there before any
   default, merge, tag, or release.

Promotion requires BOTH: conservative measured q* <= a retained traffic
segment's lower confidence bound, and auto throughput >= plain by the frozen
margin on that segment while low-q traffic remains plain. Otherwise keep the
door default off or delete it; a forced high-q demo alone is not a product win.

## 7. Verdict

**CONFIDENCE-GATED-BUILD.**

Do NOT build or promote unconditional optipipe. At the only defensible measured
validity, q=24/63=0.381, it projects **75.4-79.4 tok/s vs plain 81.188**
(-7.1% to -2.2%). The optimistic q=0.5325 bound still loses in the conservative
overhead case. The receipt corpus has **0/80 bursts above q*=0.70**, so this
feature would correctly remain plain on every measured burst.

Do build the confidence-gated mechanism in the staged order of §6, beginning
with the independently measurable mid-body seam. The case for spending that
increment is real:

- the q=1 ceiling is 97.6-117.4 tok/s, +20.2% to +44.6% over plain, so the
  schedule has genuine upside on a sufficiently predictable traffic segment;
- existing double-buffered PP boundary slots are sufficient; the missing
  primitive is release at stage0 TX plus a resolve/restart event discipline;
- hit rounds stay in the exact b1fix numeric class, while miss rounds can be
  made unobservable through the two-generation state fork; and
- increment 0 can falsify real stage overlap before paying the L-class state
  and scheduler bill.

The promotion bar is fixed: remeasure the anatomy, solve the conservative q*
with a **5% margin**, and admit only a retained segment whose q lower bound
clears it. Under today's model that is q*=0.695, rounded to **0.70**, projecting
85.42-95.27 tok/s (+5.22% to +17.35%). Auto must remain plain with zero shadow
tax below the threshold, survive forced miss storms and phase demotion with one
completion hash, and pass the full target-rig battery. If increment 0 does not
show actual S0/S1 overlap, or no real traffic segment clears the remeasured q*,
the verdict collapses to HOLD and increments 1-4 stop.

In short: **build the gate and the measurable seam; do not bet production on
the current average.**

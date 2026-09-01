# mtp11: spec-round host-sync audit + the deferred-readback port plan (2026-08-30)

Owner order: port the mature families' deferred-readback round structure
(crates/memra-engine/src/spec.rs slice 2, `MEMRA_DSPARK_DEFER_READBACK`) onto the
qwen4_exp spec loop. This file banks work items 1 (sync audit, code level; box ms land
when the box frees) and 2 (mature-pattern mapping). All line numbers at branch tip
c6445821f, `crates/memra-engine/src/qwen4exp_gpu.rs` unless said otherwise.

## 1. Host synchronization points per spec round, CURRENT loop

Shape: greedy, non-trace, guard armed (ship policy `adapt k_lo=1 + pmin 0.3`), dev1
placement, `j` = drafts actually chained this round (j <= k_round <= K=5).

| # | site | code | what crosses | bytes | count/round | class |
|---|------|------|--------------|-------|-------------|-------|
| S1 | chain argmax dtoh | `draft_row_argmax`: `clone_dtoh(tok)` L7932; call sites L8126, L8159 | the draft pick (RAW trim-space index) | 4 | j | BLOCKING dtoh. Host waits for the whole draft step tail to drain (exit mixer + lm_head matvec + 2-pass argmax; the head is 4.757 ms/round at K=5, 54% of draft cost, mtp5) before it can stage the next step |
| S2 | chain guard prob dtoh | `draft_row_argmax`: `prob_of_token_device` + `clone_dtoh(pd)` L7927-7928 | p-min confidence | 4 | j | BLOCKING dtoh, fires with S1: 2 host round trips per drafted token |
| S3 | verify argmax drain | `forward` want_argmax fast path L5085-5091 | K+1 target argmaxes | 4(j+1) | 1 | the verify/accept readback is ALREADY small and merged (one `clone_dtoh` of the per-row device argmaxes; the [t,vocab] block never crosses in greedy non-trace). Answer to the audit question: 4-byte argmaxes cross, not rows |
| S4 | zero-draft round logits dtoh | `forward` L5094-5096 via L8182 (tlen==1) and the dynk plain tail L8106 | full [1, vocab] f32 row + host argmax over 248,320 | 993,280 | per zero-draft round (pmin stops at step 0; dynk tail unused in ship config) | the t==1 greedy step never takes the argmax fast path (`exact` requires t>1), so guard-collapsed stretches pay ~1 MB dtoh + a host vocab scan per committed token |
| S5 | chain-step embed h2d | `mtp_draft_forward` fuse L7245-7255 (`embed_host` row copy + `take_f32_h2d`) | 1 x 2560 f32 embed row | 10,240 | j-1 | pageable htod (driver staging, quasi-blocking); exists because the next chain token must round-trip to host (S1) to be embedded from the HOST-resident table |
| S6 | verify + replay embed h2d | `forward` entry.embed L4901-4917; replay via L8325 | (j+1) + (a+1) rows | 10,240/row | 2 | pageable htod, small |
| S7 | dev1 wide-seed crossing | `cross_wide_rows` `stream.synchronize()` L7590; call sites L8069 (prefill), L8322 (replay) | (a+1) x 40,960 B P2P | 40,960/row | 1 (+1 at prefill) | timed sync BY DESIGN (this is the crossing instrument; 0.020-0.037 ms/round measured, mtp10) |
| A1 | per-call allocs | `draft_row_argmax` `alloc::<u32>(1)` L7920; `prob_of_token_device` partial alloc (lib.rs) | - | - | 2j | stream-pool churn, not syncs; noted |

Structural blocking syncs per round: **2j + 1** (S1+S2+S3), plus S4 on zero-draft
rounds. At the ship K=5 fully drafted: **11 blocking host round trips per round**, i.e.
the owner's "device argmax but a blocking 4-byte dtoh EACH, K syncs" is confirmed and,
with the guard armed, it is 2K.

Out-of-round but in-loop: the trunk PREFILL returns the FULL [n, vocab] logits block to
host (L5094 at t=n via L8048) and the loop reads ONE row (x0). At a 940-token prompt
that is a ~934 MB dtoh inside `prefill_ms`. Fixed in the port (last-row-only readback
under the deferred seam).

Out of scope, coexisting (the SEPARATE host-twins lane, per the task): 48 MoE router
dtoh (L5610) + 12 QSA indexer dtoh+mask h2d (L5995-6020) per verify chunk (~12 ms/round
of host-twin bubbles, PROFILE-7 round-cost identity), 1 router + 1 indexer per draft
chain step, and the PLE host n-gram gather (L6963-6992). The port must coexist with
all of these; the PLE one BOUNDS the port (below).

Sampled-mode delta: the verify readback is the full [t, vocab] block (~0.99 MB/row)
because targets are host-sampled per row; inherent to the host sampler, unchanged by
this port (the chain deferral still applies; drafts are argmax picks in sampled mode).

### Found while auditing: verify wide-capture skips under decode graphs (t==1)

`graphs_mode` (L4858) does not exclude an armed verify, and `forward_graphs_tail`
(L5725) has no `stash_wide` copy and no argmax sink. Two consecutive t==1 forwards on
an armed state (= consecutive zero-draft rounds under pmin, the thinkon collapse shape)
route the second through the graphs tail and SKIP the wide capture at that position;
the next replay then seeds the draft from a stale wide row. Byte identity is safe
(verify arbitrates every commit) but acceptance silently degrades exactly where the
guard is working hardest, invisible to the byte-identity gates. The port gates
`graphs_mode` on `state.verify.is_none()` (one line; the dynk plain tail loses decode
graphs, and dynk is unused in the ship config).

## 2. The mature pattern (spec.rs slice 2), mapped onto this round

What spec.rs does (L295-312, `MEMRA_DSPARK_DEFER_READBACK`, default ON there): the
draft-chain DtoH is deferred past verify dispatch and merged with the verify-argmax
readback into ONE host sync (2 blocking DtoH/round -> 1). The verify embeds DEVICE
tokens (`chain_d`) through the resident embed table (`embed_gather_u32_t`,
bit-identical rows by the QT_BF16 bits<<16 contract), so the host dispatches the whole
verify while the draft still executes instead of blocking ~1.7 ms on the chain. The
related `state_copy_batch_on` slice measured 0.67 ms/round of pure dispatch on the q38
route (the K=1-class number the owner cites).

Mapping and the three structural differences here:

1. **PLE bounds the drain count at 2, not 1.** The trunk's layer-1 PLE block host-hashes
   the chunk's ACTUAL token ids into n-gram table rows (host-resident 102 GB table,
   `host_ngram_ids` + `NgramTable::gather_into`). The verify chunk contains the drafted
   ids, so the verify CANNOT be dispatched device-token-blind on this family. The
   honest minimum is: (a) ONE chain drain after the whole chain is dispatched (replaces
   the 2j per-step round trips), then (b) the existing single verify-argmax drain (S3).
   Structural syncs per round: 2j+1 -> **2**.
2. **The embed table is host-resident and untied here** (`embed_host`, f32
   [248320, 2560]; lm_head is a separate tensor, so no free device gather source).
   The port builds a DRAFT-ENGINE-resident bf16 table (1.27 GB on card 1, which holds
   only the ~5 GB draft bank + head copy; card 0's 2.5 GiB headroom is NOT touched,
   honoring the mtp9 OOM lesson). bf16 residency is bit-exact because this artifact's
   embed rows dequant via bits<<16 (checked at arm time value-by-value; a dirty value
   falls back to a raw f32 table, 2.54 GB, always exact). With the FR-Spec trim armed
   the table is gathered in TRIM-RANK order (row i = embed[d2t[i]]), so the RAW device
   argmax index gathers its own next-step row and no d2t crossing exists on device;
   the drain maps raw -> target ids on host through the existing `draft_token`.
   (Trim is OFF in the ship config; the table is then the full embed and the map is
   identity.)
3. **The p-min guard is a sequential chain-stop; deferral makes it post-hoc.** The
   deferred arm computes every step's confidence into device slots
   (`prob_of_token_device_col`, the zero-sync gemma machinery) and applies the
   truncation at the drain: same probs bit-for-bit, same first sub-threshold step, same
   discarded token, so the committed bytes and all counters (guard_stops, drafted,
   zero_draft_rounds) are identical BY CONSTRUCTION; what changes is COST SHAPE (the
   chain always dispatches k_round steps; the sequential arm stops early). Under the
   ship policy the waste is bounded: after a guard stop a=0, so adapt sets the next
   window to 1. Because the guard CAN force a readback, both arms are built and both
   are measured (the task's requirement): `defer_guard_sync` keeps a per-step 4-byte
   prob readback (chain stops exactly like today, argmax still deferred; j+1 syncs),
   default deferred guard reads the window at the drain (2 syncs).

What the port does NOT change: the verify chunk program (rows bit-identical to t==1
decode), the accept walk, rewind, replay, the admission policy arithmetic, the router/
indexer host twins, sampled-mode host sampling, trace mode (trace stays on the host
chain arm; defer + trace refuse loudly).

Expected win class, code level (box numbers will replace this): kill 2j blocking round
trips each of which waits out the draft step tail (head-dominated ~0.95 ms/step at
K=5), kill j-1 pageable h2d stagings, overlap host dispatch of step j+1 with device
execution of step j (this route is host-issue-bound: verify-ISSUE 44-58%, verify-WAIT
0.0% on the sibling route receipt), plus the S4 zero-draft ~1 MB dtoh replaced by a
4-byte argmax readback. The 0.67 ms K=1-class question is answered by the box A/B.

## 3. Seams (flags law: default OFF, both arms, rollback = drop the flag)

- `SpecOpts::defer` / gate binary `--spec-defer`: the deferred round. Requires
  `arm_spec_devchain(de)` (the chain-embed table on the draft engine); refuses
  loudly without it, with trace, or on a trim-state mismatch.
- `SpecOpts::defer_guard_sync` / `--spec-defer-guard-sync`: the sequential-guard
  sub-arm inside defer (the guard-forces-a-readback measurement arm).
- FLAGS.md rows land in the same PR as the seams.

## 4. Gates (every commit)

Rig (tiny, serialized, correctness only): the 15-arm tiny gate + new arms:
defer byte-identity vs plain AND vs the host-arm spec run; pmin twin pinning
host-guard == defer-post-hoc == defer-guard-sync on tokens AND counters
(guard_stops/drafted/accepted/zero_draft_rounds); a reversed-d2t trim arm (the
trim-rank table path); the dir-bf16 model twin (the bf16 table path; the fixture's
random f32 embeds exercise the f32 fallback).

Box (when free; the YaRN ladder owns the GPUs until then): spec-gate 4/4 + long-prompt
6/6 byte identity, verify-bit 24/24, then the measurement battery of item 4
(interleaved x5 per shape, K ladder at ship admission, defer OFF/ON/guard-sync arms).

## 5. Port landed (2026-08-30)

Commits: 03686fb2a (the port + gates + FLAGS rows), c94a3867b (--defer-ab K ladder in
one load), 5fabc0302 (run-battery.sh). Rig tiny gate: 19 arms PASS failures=0
(tiny-fixture-gate-mtp11.tsv beside this file), including:

- mtp-spec-defer (fixture): pmin0 host==defer==plain; pmin0.5 all-stop host==defer==
  defer-gsync==plain with counters equal (guard_stops=23, zero_draft=23);
  trim-rev (reversed full-width d2t) all three arms equal; f32-fallback table path.
- mtp-spec-defer-dirbf16: the bf16 BIT-CLEAN table path, same identities.
- guard-trunc-pin: the deferred guard's truncation walk pinned on 6 handcrafted
  windows including the mid-chain dip and the p == pmin boundary. The deterministic
  fixture cannot produce a mid-chain INTEGRATION stop (intra-round confidence never
  crosses a threshold its first pick passed — swept (k, pmin) over {3,6,2} x 12
  values); stated in the receipt, covered by this pin + the box defer-ab counter
  identity at ship pmin 0.3, where mid-chain stops occur constantly.

Sync structure after the port (defer arm): chain = 0 in-scope syncs while
dispatching, 1 drain; verify = 1 drain (unchanged); zero-draft rounds commit via
device argmax (S4 dies); prefill dtoh = 1 row (~1 MB instead of ~n MB).
defer_guard_sync arm: j+1 syncs (per-step 4-byte prob + verify drain).

## 6. Box measurements (DONE 2026-08-31 — perf/PROFILE-8.md, receipts beside this file)

Measured ms for the table in section 1 (thinkon, per-round chain phase, x5 medians):
S1+S2+S5 together are worth **~0.01-0.02 ms/round at K=1** (chain 0.95 host vs 0.93-0.94
deferred arms) — NOT the 0.67 ms class. Section 1's own out-of-scope row explains it:
the router/indexer HOST TWINS inside each draft step already serialize the host, so the
2-per-step dtoh only cost the gap between adjacent twins. S3 unchanged. S4/prefill
readbacks eliminated as coded (structure receipts; too small to see at these spreads).
The post-hoc guard arm ADDS +0.20 ms/round at K=5 on thinkon (dead drafts past the
stop, 36 stops/256 tok); the gsync arm avoids them and carries a positive-but-inside-
spread sign everywhere. Both seams stay OFF (PROFILE-8 §7).

THE BATTERY'S HEADLINE: the first 256-token spec-gate caught a LATENT mtp10-era
byte-identity defect (short prompt <= k_cap prefilled EXACT instead of FUSED; gen-157
flip on a 0.024-margin row; reproduces at the mtp10-close commit). Fixed 94f1cecc2;
diagnosis chain + new gates receipted in PROFILE-8 §0.

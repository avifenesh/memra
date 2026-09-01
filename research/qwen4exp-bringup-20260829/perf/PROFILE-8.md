# PROFILE-8 — mtp11: the deferred round readback (spec.rs slice-2 port), measured

Round: mtp11 (owner-ordered loop-port lane). Code: 03686fb2a + 8f80bde1f (seams
`SpecOpts::defer` / `defer_guard_sync`, default-OFF), 94f1cecc2 (the gen-157 fix, below).
Audit: spec/mtp11/AUDIT.md. Port facts: spec/MTP-SPEC.md Deliverable 8. Receipts:
spec/mtp11/*.tsv (pulled from box ~/realgate/mtp11). Box: sbox-eval (2x RTX PRO 6000
Blackwell 96 GB), artifact ~/data/q48fn-nvfp4, ship admission everywhere (dev1 + K=5 +
adapt k_lo=1 + pmin 0.3). Protocol: interleaved x3 default + receipted escalation to x5
(2026-08-30 fleet amendment) — every cell escalated on rules a+b (within-arm spreads
1.9-4.0% on this box, verdicts inside 2x pooled spread), so every number below is x5.

## 0. THE HEADLINE FINDING IS A CORRECTNESS ONE (gen-157)

The battery's first 256-token spec-gate broke byte identity at gen 157 on raw prompt 2
(len 6) — and the failure REPRODUCES AT THE mtp10-CLOSE COMMIT. Every previously green
spec-gate ran 64 tokens; verify-bit never rewinds; the tiny fixture's 18-token prompt
never fit inside k_cap. Diagnosis chain (receipts in spec/mtp11/):

1. defer arm vs CONTROL (defer off): identical divergence (prompt 2, token 157, same
   accept stats) — trunk-side, not the port (spec-gate-k5-m11-{defer,CONTROL}-raw.tsv).
2. seam bisect: idxdev=0 and longatt=0 both still fail — the new default-ON seams
   innocent (bisect-verdicts.txt).
3. commit bisect: fails at cb8ef020c AND at 35a0b4c98 (mtp10 close) — LATENT, not a
   regression.
4. trace: the flip commits target 437 on a 0.024-margin row (top1 20.965 vs top2
   20.941) — accumulated small state error flipping the first thin-margin argmax
   (spec-trace-k5-m11-DIAG-trace.tsv).
5. K bracket: K=2/3/4 PASS, K=5 FAILS — k_cap = K+1 crosses the prompt length at 6.
6. rewind-bit sweep (new gate `--rewind-bit-gate`): keep 1..5 over t=6 chunks all
   BIT-clean — rewinds innocent (rewind-bit-gate-m11-rewindbit.tsv).
7. exact-pattern replay (new gate `--rewind-bit-replay`): mismatch from ROW 0 — the
   PREFILL itself (rewind-bit-replay-m11-replay-p2.tsv).

ROOT CAUSE: `forward`'s `exact = t > 1 && t <= k_cap` treated a SHORT PROMPT's prefill
as an exact verify chunk: the armed state prefilled through the per-row DECODE programs
while the plain baseline prefilled FUSED — bit-different trunk state from token 0.
FIX (94f1cecc2): exact requires `base_pos > 0`; the prefill always runs the fused
program. Gates grown: tiny arm `mtp-armed-prefill-bit` (short-prompt armed vs unarmed
prefill + 1 step, bitwise) + the two rewind gates above. Post-fix: raw 4/4 at 256
(spec-gate-k5-m11-FIXCHECK.tsv).

## 1. Gates at the fixed tip (phase 1) — ALL GREEN

verify-bit 24/24; spec-gate byte identity DEFER arm: raw 4/4, thinkon 4/4, long 6/6;
defer-gsync arm thinkon 4/4. Every defer-ab row below also hard-checked chain AND
admission-counter identity per rep: PASS everywhere (39 interleaved A/B reps total).

## 2. Per-shape A/B at K=5 ship admission (256 tok/arm, x5 after escalation)

| shape | host ms/tok | defer | defer-gsync | defer/host | gsync/host |
|---|---|---|---|---|---|
| thinkon | 12.862 | 12.967 | 12.786 | 0.992 | 1.006 |
| thinkoff | 9.418 | 9.441 | 9.345 | 0.998 | 1.008 |
| efflow | 12.507 | 12.543 | 12.316 | 0.997 | 1.016 |
| raw | 8.454 | 8.475 | 8.454 | 0.998 | 1.000 |
| long (724-tok agentic) | 16.037 | 16.161 | 15.974 | 0.992 | 1.004 |

Decision medians; within-arm spreads 1.9-4.0% (receipted per arm) — the per-shape
verdicts sit INSIDE the pooled spread (rule b fired everywhere), so treat the K=5
per-shape signs as noise-bounded. What is NOT noise: the defer arm never wins at K=5,
and gsync's sign is positive on 4/5 shapes.

## 3. K ladder, thinkon + raw (the knee mechanism)

thinkon (defer/host by decision medians): K=1 **1.009**, K=2 1.002, K=3 0.999,
K=4 0.994, K=5 0.993, K=6 0.993, K=8 0.992 — monotone decay. gsync/host is positive at
ALL SEVEN rungs (1.001-1.009). raw: flat at every K (0.997-1.006, no stable sign).

Mechanism, attributed from the per-round phase columns (thinkon):

| K | chain ms/round host | defer | gsync | reading |
|---|---|---|---|---|
| 1 | 0.95 | 0.94 | 0.93 | the 2 blocking dtoh are worth ~0.01-0.02 ms/round here |
| 5 | 2.10 | **2.30** | 2.08 | post-hoc guard DISPATCHES PAST THE STOP: +0.20 ms/round of dead drafts (36 guard stops/256 tok on thinkon); gsync keeps the sync savings without them |

## 4. The 0.67 ms K=1-class question: NOT reproduced on this family, and why

The chain-side saving measures ~0.02 ms/round (S1+S2+S5 of the audit) against the
owner-cited 0.67 ms/round class from the q38/dspark route. The audit's own structural
caveat is the explanation: this family's draft step carries PER-LAYER HOST TWINS
INSIDE it (MoE router dtoh + QSA indexer mask per chain step — the separate lane the
port had to coexist with), so the host was already serializing at those twins; the
2-per-step 4-byte dtoh only added the gap between two adjacent syncs, not a full
drain. spec.rs's dspark route has no such twins, which is where its 1.7 ms blocking
chain and 0.67 ms batching wins lived. The deferred round's structural savings on this
family cannot exceed the inter-twin gaps until the host-twin lane lands device-side
routing/selection.

## 5. Guard arms (the owner's measure-both order)

Post-hoc drain guard (default defer): loses on guard-heavy shapes at K>=3 (dead
drafts, table above). Sequential sub-arm (defer_guard_sync): keeps the deferred
argmax + table embeds with today's chain-stop, positive sign everywhere
(+0.3-1.6% medians) but inside pooled spread per rule (b). Counters
(guard_stops/zero_draft/drafted/accepted) identical across all three arms in every
rep — the truncation semantics receipt.

## 6. Sampled probe (serving law)

Vendor defaults, defer armed, thinkon: SPEC-ENGAGEMENT rounds=150,
rounds_with_accepts=71, accepted=105/199 drafted; 15.36 ms/tok
(spec-sampled-k5-m11-sampled2-thinkon.tsv). Greedy bonus A/B in the same load:
plain 15.83 vs defer-spec 13.43 ms/tok (**1.179x**, rep0 byte-identical) — the mtp10
thinkon 1.174x reproduces through the deferred round.

Chain-embed table on the real artifact: 248,320 x 2560 **bf16 bit-clean** (proven
value-by-value at arm time), 1,212.5 MiB on card 1, armed in 0.9 s.

## 7. Verdict (defaults, flags law)

- `SpecOpts::defer`: **stays OFF.** No shape wins at the ship K; the post-hoc guard
  actively pays dead drafts on guard-heavy content.
- `SpecOpts::defer_guard_sync`: **stays OFF as a default** — positive sign at every
  thinkon rung but the verdict never clears 2x pooled spread on this box (rule b);
  a flip needs a quieter box or a bigger effect. Kept as the measured lever: if the
  host-twin lane later removes the router/indexer syncs from the draft step, the
  deferred chain's ceiling rises and THIS receipt is the baseline to re-measure from.
- The port's durable value shipped anyway: the gen-157 byte-identity fix, the
  graphs-tail wide-capture fix, three new gates (rewind-bit, rewind-bit-replay,
  armed-prefill-bit), the 256-token spec-gate standard, and the prefill last-row
  readback (a ~934 MB dtoh at 940-token prompts, now one row under the seam).

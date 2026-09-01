# DSpark Q38 wiring — build results (lane/dspark-q38-recover, 2026-08-17)

Companion to WIRING-SIZING.md. The full arm is built and gated against the LOCAL
arm-a cum-250 export — weights are placeholders for the owner's in-flight recovery
training; SEMANTICS are the deliverable. Swap procedure at the bottom.

## Gates table

| stage | gate | result |
|---|---|---|
| loader census | 62/62 tensors mapped, refuse-on-unrecognized (dspark-class) | GREEN (both sides: SpecForge strict load + memra census) |
| ctx_features | vs SpecForge reference, fixed-seed taps | PASS rel 1.9e-4 |
| block forward (final hidden) | vs reference, 5 layers, block 7, GQA 40/8 | PASS rel 2.0e-4 |
| markov chain | chained greedy decisions, rank-256 vanilla head | PASS EXACT 6/6 tokens; logits rel 9.9e-8 |
| confidence head | raw AcceptRatePredictor (hidden ⊕ markov emb) | PASS rel 2.8e-6 |
| local-ci --perf-quick | correctness battery + gemma 31B perf cells on the touched trunk | GREEN (cells flat, accept-gate pass) |
| E2E exactness (spec vs plain greedy, real Q38 artifact) | dspark_q38_gate | **ALL EXACT** — 3 prompt classes x 96 tokens, byte-identical |

Oracle: `tools/dspark_q38_oracle.py` — runs the SpecForge `DSparkDraftModel` (the
code that trained arm-a) on the export with fixed-seed synthetic taps/noise/logits,
dumping 14 stage arrays. Gate: `dspark_q38_parity` (MEMRA_DFLASH_PREC=bf16).

## The oracle catch (norm-fold class, as predicted)

**YaRN rope.** The arm-a drafter inherits the Q38 target's `rope_parameters`:
rope_type yarn, factor 32, original_max_position 8192, beta 32/1, theta 1e7.
memra's plain `rope_neox` ran fine and produced rel **0.196** final-hidden error —
fluent, silently wrong, would have shown up only as mysteriously bad acceptance.
Fix: per-dim ff divisors through the existing `rope_neox_ff` kernel
(effective inv_freq = HF-yarn remapped frequency, verified vs `Qwen3RotaryEmbedding`
to 1.6e-7) + post-rope mscale 1.3466 on q/k (cos/sin scaling distributes exactly
onto the rotated vector). After the fix: final rel 2.0e-4 (TF32 class). Every
drafter rope site (block q/k, round q/k, ctx ingest k) rides the same primitive.

## What was built (commits on lane/dspark-q38-recover)

1. **Loader** (`dflash.rs`): census gate (refuse unrecognized on dspark-class
   checkpoints), confidence head (host-resident; the reference serving loop never
   consumes it — memra serving ignores it identically), YaRN rope from config,
   markov w2 bf16 under the parity precision seam, sliding_window optional on
   all-full-attention exports.
2. **Qwen trunk taps** (`hybrid_forward.rs`, `spec.rs`): `dflash_tap` sites on
   `prime_layers` (post-residual, chunk-offset aware via the sink's new `base`
   field) and BOTH qwen35 verify arms (tparallel + rowwise). No-ops when no sink
   is armed — gemma paths untouched, CI cells flat.
3. **Serving-class verify** (`spec.rs::dspark_verify_t_am`): one t-row forward
   through `decode_step_t_core_stream` — the SAME funnel MTP verify rides, so the
   numeric class is the serving class by construction — returning per-row argmaxes.
4. **The round** (`dflash.rs::generate_spec_dspark`): the gemma dspark loop's
   hybrid twin. The one real divergence: GDN conv/ssm state mutates in place and
   cannot roll back by KV truncation, so a partial accept does
   snapshot → verify(t=vt) → restore → prefix-replay(keep), with the replay's
   argmaxes asserted equal to the verify prefix (a free structural gate every
   round). Full accepts skip rollback entirely. Per-phase ns counters
   (draft / snapshot / verify / rollback+replay / ingest) under MEMRA_SPEC_STATS=1
   — the verify-toll dataset the economics question needs.
5. **E2E gate bin** (`dspark_q38_gate`): byte-exactness of the spec stream vs
   serving-class plain greedy on the real artifact + tok/s cells; tokenizer prompt
   dir or fixed-id mode.

## Verify economics (cum-250 weights — COST side only; acceptance is placeholder)

5090, Qwen3.8-27B-NVFP4-Q5K-mtp.gguf, chat prompts, ngen 96, MEMRA_SPEC_STATS=1:

| prompt | plain tok/s | spec tok/s | accept | rounds | draft | snap | verify | rollback+replay | ingest |
|---|---|---|---|---|---|---|---|---|---|
| code | 38.9 | 24.2 | 0.139 | 74 | 316ms | 37ms | 1983ms | 1604ms | 24ms |
| prose | 43.3 | 23.8 | 0.140 | 74 | 318ms | 37ms | 1987ms | 1657ms | 24ms |
| agentic | 43.4 | 21.9 | 0.096 | 80 | 347ms | 40ms | 2128ms | 1842ms | 26ms |

Reading (per round, adaptive vt~3 at this acceptance):
- **verify ≈ 27ms ≈ 1.1x a plain step** (plain = 23.9ms) — the qwen T-parallel verify
  funnel is near-flat in t, the GDN toll is largely paid down. Snapshot is noise
  (0.5ms). Draft 4.3ms, ingest 0.3ms.
- **rollback+replay ≈ 22ms/round is the real toll** — the prefix replay is a second
  t-forward. Named fix, sized hours: thread `VerifyCkpt` (the MTP column-state
  stash) through `dspark_verify_t_am` so partial-accept rollback restores column
  m's stashed GDN state instead of replaying — the mechanism already exists for
  the MTP path.
- **Acceptance receipt:** 96 gen / 74 rounds = 1.30 tokens/round — matches the
  cum-250 checkpoint's banked accept-length 1.31 from the B200 training evals.
  The engine reproduces the trained checkpoint's acceptance at distribution
  level; the semantic chain (taps -> yarn rope -> rounds -> markov) checks out
  end to end, not just on synthetic gates.
- Extrapolation to the owner's target weights (accept-len 2.6-3.0): ~3 tok/round
  at ~32ms (draft+verify(7)+snap, replay skipped on full accepts / ckpt-stash on
  partial) ≈ 2x plain single-stream — competitive with the trimmed MTP head's
  ratio, decided by the real A/B per the sizing doc.

## VerifyCkpt column-stash rollback (funded follow-up, 2026-08-17)

The named fix landed: `dspark_verify_t_am_ckpt` fills the MTP path's `VerifyCkpt`
during verify; a partial accept restores column state via `commit_verified_prefix`
(the exact mechanism MTP ships) instead of snapshot-replay. Replay stays as the
oracle arm (`MEMRA_DSPARK_CKPT=0`), and `MEMRA_DSPARK_CKPT_GATE=1` runs BOTH per
partial round and asserts the resulting cache state is BIT-IDENTICAL (pos, every
kv len, every linear layer's conv/ssm buffer, host-compared).

Gate results (5090, real artifact, 3 classes x 96 tokens):
- ckpt-gate arm: ALL asserts silent across every partial round of all classes —
  stash state == replay state bit-for-bit; replay argmax == verify prefix; streams
  EXACT.
- default stash arm E2E: **ALL EXACT**, economics:

| prompt | plain tok/s | spec tok/s (was) | accept | rollback total (was) |
|---|---|---|---|---|
| code | 42.3 | **39.0** (24.2) | 0.139 | **26ms** (1604ms) |
| prose | 43.5 | **39.0** (23.8) | 0.140 | **27ms** (1657ms) |
| agentic | 43.5 | **36.2** (21.9) | 0.096 | **30ms** (1842ms) |

Rollback toll: 22ms/round -> **0.35ms/round** (~60x). Per-round budget now
draft 4.3 + snap 0.5 + verify 27.5 + commit 0.35 + ingest 0.34 ≈ 33ms.
At the cum-250 placeholder acceptance (1.30 tok/round) spec already reaches
**88% of plain** — break-even sits at the placeholder floor. At the owner's
target accept-len (2.6-3.0): ~11ms/token ≈ **2.2-2.5x plain single-stream**
on this card class, before any drafter-side tuning.

## Swap procedure for the owner's final export

One argument. The arm keys everything off the export directory:

```
MEMRA_DFLASH_PREC=bf16 target/release/dspark_q38_parity <new_export_dir> <oracle_dump_dir>   # semantics still hold
target/release/dspark_q38_gate <q38_artifact> <new_export_dir> 128                            # byte-exactness + tok/s
```

Re-run the oracle first only if the CONFIG changed (taps/block/rank/heads):
`tools/dspark_q38_oracle.py <new_export_dir> <dump>/oracle.npz <SpecForge_dir>`.
Weights-only updates (same config) need no oracle re-run — the parity gate loads
the new weights on both sides... note the oracle dump embeds the reference model's
outputs for ITS weights, so for a weights-only swap regenerate the dump with the
same command; the gate then compares apples to apples. Acceptance/thoughput A/B
vs the trimmed MTP head runs on the serving box per the sizing doc's sequence.

## Not in this arc

- Server route (worker.rs env-gated dspark-vs-MTP selection): deferred with
  reasons — at placeholder acceptance the route cannot flip anything, and the
  economics say land the VerifyCkpt rollback first; the engine surface (this
  arc) was the dependency. Hours-class when the recovered weights arrive:
  route predicate + refuse-on-ambiguity vs MTP (gemma_gate guard pattern) +
  served-stream identity check.
- PP/multi-device: the round asserts single-device; taps ride the serial prime.

---

# BOX2 QUALIFICATION — cum1000 (owner's final export), 2026-08-19

Box: hyperscaler 2× RTX PRO 6000 Blackwell Server 96GB (box-2), CUDA 13.2,
MEMRA_CUDA_ARCH=120a. Branch REBASED onto v0.92.0 (only conflicts:
research/tune-data/perf-ci.jsonl append-append journal rows, keep-both). Trunk =
the PRODUCTION artifact `Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` sha256 `1facf36c2db359dc…`
(HF Avifenesh/Qwen3.8-27B-NVFP4-MTP-GGUF, orbench pin). Export = arm-a-cum1000
(STOP-POINT blessed, 7ff4e284d5); config.json byte-identical to the cum-250 census
this lane was built against — weights-only swap per the procedure above.

## Gates (all on box 2; binaries from this branch tip)

| gate | result |
|---|---|
| oracle (SpecForge 2590f48e3a93, CPU fp32) | census OK 62/62; 14 arrays |
| `dspark_q38_parity` (MEMRA_DFLASH_PREC=bf16) | **ALL PASS** — markov tokens EXACT, markov logits rel 1.1e-7, confidence 8.4e-7 (independently reproduced by a second agent: ALL PASS) |
| `dspark_q38_gate` E2E, 12 own-session chat prompts, ngen 128, ×5 reps | **ALL EXACT ×5**, acceptance rows byte-identical across reps |
| same + `MEMRA_DSPARK_CKPT_GATE=1` | **GREEN** — every stash-vs-replay assert silent, all partial rounds (also: 128-prompt pack × 96 tok, 12,168 tokens, ALL EXACT — adopted sibling receipt) |
| serve route smoke (`tools/dspark-serve-smoke.sh`) | **ALL GREEN** — 3 greedy rows byte-identical spec-on vs spec-off over HTTP (sha over reasoning+content), engagement accepted={27,24,33} + [dspark-acc] in log, sampled request declines to plain, concurrent pair completes |

## The boundary catch (fix f8300340cd)

At real acceptance the FINAL round often accepts a draft and the push-then-check
loop emitted max_new+1 tokens (plain emits exactly max_new). Deterministic on 3 of
12 prompts, every rep, always "DIVERGED at Some(128), got len 129" with the shared
prefix byte-identical — invisible at cum-250's 0.14 placeholder acceptance. Budget
check moved before the push; ALL EXACT after.

## Measured acceptance + A/B (greedy engine terms, single stream, GPU 1)

Quiet-window A/B ×5 (box otherwise idle; gate interleaves plain/spec per prompt):

| cell | plain tok/s | spec tok/s | ratio |
|---|---|---|---|
| own 12-pack, ngen 128, reps 1-5 | 62.6-62.9 | 62.5-62.8 | **1.00x** |
| own 12-pack, ngen 768 | 68.9 | 65.7 | 0.95x |
| gsm8k-class ×4, ngen 256 | 68.2 | 75.7 | 1.11x (max 1.22x) |
| gsm8k-class ×4, ngen 768 | 69.7 | 75.0 | 1.08x |
| 128-prompt own pack, ngen 96 (adopted, ×5 reps) | 63.7 | 62.2 | 0.98x (chat 0.962 mean / agentic 0.994 mean, per-prompt max 1.45x) |

Acceptance (tokens/round = 1 + accepted/rounds): own-pack mean **1.43** @128
(range 1.16-1.60, acc-rate 0.198), ~1.50 @768; math **1.69** @256 (max 1.88),
1.5-1.8 @768. Deterministic across reps.

## Verdict vs the 2.2-2.5x projection

**Not reproduced, and the projection's premise does not hold.** It assumed engine
accept-len 2.6-3.0 mapped from the training bank (own 2.880/2.923, gsm8k 4.605) —
but those sglang numbers were measured at temp 0.6 + thinking + max-new 2048 on
the FP8 trunk (rejection-sampling accept-length), a different observable from
greedy argmax acceptance on the NVFP4 GGUF. The engine's cum-250 anchor (greedy
1.30 == banked 1.31) validated the semantic chain, not this protocol mapping.
Cross-check that the pairing is NOT broken: acceptance rises with cum (1.30 →
1.43-1.69), math > chat (matching the bank's ordering), and per-round economics
match the wired budget (draft 4.3 + verify ~27 + commit 0.35ms class). Lineage:
memra drafts through the SERVED trunk's own embd/lm_head (no cross-pairable
drafter embedding; the export's embedding-source tensor is sglang-only and outside
the 62-tensor census); residual gap = FP8-trained taps vs NVFP4-served trunk,
priced into the numbers above.

**Disposition: the wire + gates + serve route are DONE and green; the drafter at
cum1000 is break-even single-stream on this card class and does not displace the
shipping trimmed-MTP config. Serve route ships DEFAULT-OFF (MEMRA_DSPARK_SPEC=1
+ MEMRA_DSPARK_DRAFT). Next acceptance lever is drafter-side (more training /
thinking-distribution corpus match / NVFP4-trunk-taps finetune), not engine-side.**

Evidence: research/dspark-q38-recover-20260816/box2/ (oracle dump, gate logs ×5 +
ckpt + depth, smoke receipts incl. server logs, quiet A/B logs).

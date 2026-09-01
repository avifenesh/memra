# draftcost-moe lane — spec round cost on the qwen35moe (A3B MoE hybrid) class (2026-08-20)

Owner direction: "how is it possible that the draft is slower than the model" + "the draft
[time] should disappear". Measured on the Ornith-1.5-35B-A3B NVFP4 GGUF (one RTX PRO 6000,
rented pair, contended-by-construction — ratios only). Raw logs on-box `~/models/ornith15/
mtp-train/` (runspec-*.log, pmin-sweep*.out, canary-*.json, decode-batch.log), S3-mirrored.

## Diagnosis chain (each step measured, none assumed)

1. **Draft chain is nearly free**: eager head step = 527 us sync-bounded
   (glue 59 / attn 105 / ffn 82 / head 273 us) — MEMRA_SPEC_ANATOMY probe (first lane
   commit). The "draft slower than model" hypothesis: refuted at the compute level.
2. **Rowwise verify was the wall**: round time scaled ~5.6 ms per drafted token
   (K=1: 19 ms/round -> K=8: 58 ms/round) — `qwen35_verify_batch_layers` sent Qwen35Moe to
   the per-row replay ("T full weight reads per layer"), the exact (K+1)-plain-steps disease
   the DENSE admission fixed 2026-08-15. The t-parallel arm already contained the MoE FFN
   (`moe_ffn_il_zq8` at m=T) and GDN handling; the arch fence was qualification, not mechanism.
3. **Batched trunk decode amortizes fine** (decode-batch-bench): B=1 4.66 ms/step,
   B=4 7.85 ms/tick (2.38x aggregate) — MoE expert reads were NOT the bottleneck.

## Change (this lane, commit 2)

`spec.rs qwen35_verify_batch_layers`: admit `Arch::Qwen35Moe` to `qwen35_verify_tparallel`
(t<=16). Rollback seam unchanged: `MEMRA_SPEC_VERIFY_ROWWISE=1`.

## Qualification (ship bar from the dense admission doc, reproduced)

- run-spec K=1..8 self-consistency: PASS every K, BOTH arms (spec ≡ plain token-identical);
  acceptance counts identical arm-vs-arm at every K (`runspec-tparallel.log`, `runspec-rowwise.log`).
- 8-prompt ON/OFF canary, spec-on serve, 256-tok greedy: **8/8 byte-identical** text AND
  identical usage.spec counters (`canary-{tparallel,rowwise}.json`).
- Perf (run-spec, same probe): K=3 73 -> 102 tok/s (+39%), K=8 41 -> 79 (+92%);
  marginal per-K cost 5.6 -> ~1.4 ms. Serve-level (2-probe, 256-tok): ungated spec
  mean 3.12 -> 2.54 s (+23%).

## Open (increment 2 — the remaining ~10 ms/round)

MEMRA_SPEC_PHASE breakdown post-fix: commit-host bucket ~11-14 ms/round (sync-bounded GPU
tail + host round orchestration). The device-resident ROUND-STREAM machinery (zero-readback
M-round bursts: `spec_accept_greedy_dc`, `commit_verified_prefix_stream`, `spec_rollback_stream`)
exists but (a) the qwen35-family serving-class verify has NO stream arm (explicit Err), and
(b) the graph-draft eligibility gate (`mtp_dense && trunk_dense`) is stale vs its own capture
fn, which already accepts resident-MoE heads (`Ffn::Moe(m) if m.dev_exps.is_some()`).
Increment 2 = stream arm for the serving-class verify + graph-draft admission for
resident-MoE heads. GPU-side, the round is already ~token-neutral vs plain (11 ms / 2.3 tok
at the v2 head's acceptance); killing the host tax is what flips spec-on past plain
(projected ~330 tok/s vs plain 215 at measured acceptance). devacc (`MEMRA_SPEC_DEVACC=1`)
measured no effect on this path. Spec-off remains the serving posture until increment 2.

Merge bar: pre-release battery on a quiet PRO 6000 window (kernel-check, run-gen argmax,
run-spec) before this lane merges; perf-ci skip on lane pushes recorded (probe + gated change,
naked 5090 path untouched — battery due at merge).

## Increment 2a — graph-draft admission for resident-MoE heads (2026-08-20, qualified)

Eligibility rewritten to one shared predicate (`mtp_graph_capturable`: Dense OR resident-MoE
head; trunk class dropped — the graph body is the head forward only). Engagement PROVEN on
the 35B-A3B: the eager-chain anatomy probe prints zero (the eager fn never runs), run-spec
K=1..8 PASS with acceptance counts identical to the eager arm at every K. Timing flat vs
increment 1 (expected — the eager chain was already ~0.5 ms; the value is (a) per-step host
argmax/dtoh removal and (b) the captured K-chain is the round-stream burst's prerequisite).
OFF seam: `MEMRA_SPEC_NOGRAPH=1`.

## Increment 2b/2c — sizing (next)

Round-stream burst (`MEMRA_SPEC_STREAM=1`, zero-readback M-round device loop) today:
greedy-only, NON-SESSION only, and the qwen35-family serving-class verify has no stream arm
(explicit Err). 2b = _dc rewrite of `qwen35_verify_tparallel` (device tokens/positions;
the `_dc` twins — `gdn_scan_s128_dc`, fa decode `_dc`, `ssm_conv_ring_rebuild_dc` — already
exist as the pattern). 2c = session-mode round-stream for serving. Post-fix phase data:
round ≈ 21-24 ms wall = ~11 ms GPU (verify-dominated, already ~token-neutral vs plain) +
~10 ms host-serialized orchestration — 2b/2c is what removes the host tax; projected
spec-on ~330 tok/s vs plain 215 at the v2 head's measured acceptance. Serving posture
stays SPEC-OFF until then.

## Masked-vs-embedded head, same eager conditions (owner question, 2026-08-20)

All lane arms above ran the EMBEDDED full-vocab head. Measured both (eager-forced,
MEMRA_SPEC_ANATOMY): embedded step 516-531 us (head phase ~270 us — half the step is the
248,320-row head read); masked v2 (top-32768 own-gen ranks) step 281-288 us, head phase
~43 us — 6x head-read cut, step halved, self-consistency PASS both. Trade today: mask costs
~4pt serve acceptance (0.393 vs 0.431) and saves only ~0.7 ms in a 21 ms host-dominated
round -> embedded stays the right arm until 2b/2c. Post-2b/2c the halved draft step is a
real slice; ranks are v1-era — regenerate from the v2 head's own generations before
re-measuring that trade.

## Increment 2 (the real one): the Qwen35Moe legacy-replay pin (2026-08-20, qualified)

Round-stream (the planned 2b/2c) turned out NOT to be the fix — its own flag history
records NET NEGATIVE on dense (2026-07-10, always-K draft + fixed-width verify waste).
The actual ~10 ms/round tax: `spec_replay` was HARDWIRED for `Arch::Qwen35Moe` — every
round re-ran the accepted tokens through the full trunk ("legacy rollback + duplicate
trunk replay"). The pin's stated bar ("until its retained verify-state commit is proven
equivalent... its verify already executes the serving batched class" — dense rationale)
was unblocked by increment 1: the VerifyCkpt now comes from the same serving-class
t-parallel verify that qualified dense on 2026-08-15. Pin lifted; `MEMRA_SPEC_REPLAY=1`
stays the rollback/A-B seam. (2b stream-verify code landed too — parked: kept for a
future round-stream re-evaluation, engages nothing by default.)

**A stale server binary masked the win for one measurement round** (cargo `-p a --bin x
-p b` builds only the named bins — memra-server was 40 min old). Lesson re-learned: verify
the binary mtime before attributing a null result.

## Qualification (all on the v2-head NVFP4 GGUF, one PRO 6000, ratios only)

- run-spec K=1..8: PASS every K, replay-free AND legacy arms (both ≡ plain).
- CANARY2, serve, 8 prompts × 256 greedy: **8/8 byte-identical across new default /
  legacy replay / plain decode**.
- LONG-CELL (the pin's named gate, serving shape): 15,562-token prompt, spec vs plain —
  **IDENTICAL**.
- run-spec speed: K=5 56.9 (lane start) -> 92.6 (tparallel) -> **149.8 tok/s** (replay-free);
  K=8 41 -> 79 -> 110.
- Serve sweep (256-tok, 2 probes, single runs, contended): plain 1.28-1.32 s;
  spec pmin0 1.21-1.39; **spec pmin0.3 1.14-1.23 s (+9-14% vs plain)**; pmin0.5 similar;
  pmin0.7 acceptance 0.89-0.94. SPEC-ON NOW BEATS PLAIN on this model — first time.

## Merge bar remaining

Pre-release battery on a quiet PRO 6000 window (kernel-check / run-gen argmax / run-spec on
affected models incl. q38-class dense regression), balanced N>=5 interleaved-both-orders A/B
for the board claim, and the pmin default decision (owner: certainty gating adopted for this
model; naked-vs-launch-recipe scope needs the q38 measurement before a global default).

## Cross-engine cell (owner question, 2026-08-20; research ratios — NOT a README surface)

Protocol: one RTX PRO 6000 (box quiet), Ornith-1.5-35B-A3B, single stream, 256-tok greedy,
2 probes (code/agentic), N=6 per engine (2 cycles, OPPOSITE order, fresh server per visit),
each engine's naked defaults + required GPU flags only. Comparable-byte artifacts: vLLM
0.27.1 = official modelopt NVFP4+FP8 ST; llama.cpp upstream (947fd9b, CUDA sm_120) =
official Q4_K_M GGUF (~20.3 GB); memra v0.97.0 = our NVFP4+Q5K GGUF (20.2 GB, trained head).
Raw: `mtp-train/xengine.jsonl` (S3-mirrored).

| engine | code tok/s (med) | agentic tok/s (med) |
|---|---|---|
| vLLM (official NVFP4 ST) | 249.0 | 249.0 |
| memra spec (serving config) | 234.6 | 208.8 |
| llama.cpp (official Q4_K_M) | 205.6 | 217.6 |
| memra plain | 198.5 | 198.0 |

Read: vLLM leads single-stream on this model (CUDA-graph decode + FP8 attention on the
official mixed checkpoint). memra's spec serving config closes to **0.94x vLLM on code**
(acceptance-rich) and 0.84x on agentic (think-heavy, lower acceptance); memra is the ONLY
engine here that exploits the checkpoint's own MTP head (llama.cpp ignores the NextN
tensors; vLLM has no drafter for it). memra plain sits at the llama.cpp class — this
model has had ZERO per-model decode tuning (q38-class models carry months of it); the
plain gap is the tuning backlog, the spec gap closes with acceptance (v3 head) and the
depth ladder. Scope: single-stream only; batched/aggregate throughput not compared
(vLLM's strong axis; our c8 cell exists only contended). Per the positioning rule these
ratios stay in research/ + PERFORMANCE.md-class surfaces with this protocol attached.

# Ornith-1.5 MTP-head continued training — receipts (lane 2026-08-20)

Box: the rented 2x RTX PRO 6000 pair (whole box granted 2026-08-20). All numbers here are
contended-by-construction unless a cell says the box was quiet — ratios only, absolutes not
bankable. Raw logs live on-box under `~/models/ornith15/mtp-train/` and are pulled into
`raw/` here as stages seal.

## Pipeline state

| stage | rig | state | log |
|---|---|---|---|
| prompt pack (4000, stratified, seed 20260811; 44 agentic .txt as extra shard) | CPU | DONE — pack sha `8bb90bea…f83d1` | `mtp-train-prompts.log` |
| own-gen corpus (spec-off server, T=0.6/0.95/20, per-row seeds) | GPU0 | RUNNING | `gen-corpus.log` |
| hidden capture (BF16 trunk, pre-final-norm h, bf16 storage) | GPU1 | RUNNING (trails corpus) | `capture.log` |
| train mtp.* (frozen trunk, lr 5e-5 cosine, 3 epochs) | GPU0 (chained) | PENDING | `train.log`, `train-out/metrics.jsonl` |
| ST gates on official NVFP4 (load / argmax-vs-oracle / serve-st-gate) | GPU1 (chained) | PENDING | `st-gates/` |
| vendor-vs-trained serve A/B (interleaved both orders, usage.spec acceptance) | GPU0 (chained) | PENDING | `ab-head.jsonl`, `ab-summary.json` |

## Fixed decisions (why)

- **Semantics from the serve program, not the vendor's trainer**: pairing (t_j, h_{j-1}),
  rope j+1, full-history causal context — read out of `spec.rs mtp_kv_fill` call sites
  (`fill_prev` = predecessor hidden) and `hybrid.rs` glue. The head must match what memra
  runs at serve, and acceptance 53.7% on the vendor head already proved that glue correct.
- **h stored bf16, not fp16**: pre-norm trunk hiddens carry outlier channels past fp16 range.
- **Corpus at serving sampling** (T=0.6/top-p 0.95/top-k 20): acceptance is measured against
  the serve distribution; greedy-only corpus would train the wrong target.
- **Smoke (180 seqs, GPU0, 2026-08-20 ~01:30Z)**: trainer end-to-end GREEN (844.6M params,
  785 tensors exported). Vendor-head offline baseline on own-gen heldout: top1 0.8347,
  loss 0.8364 (n=1930) — far above serve acceptance, so the offline top1 is a DIRECTIONAL
  proxy only (think-filler tokens are easy); the serve A/B is the gate. lr 1e-4 degraded
  heldout in one smoke epoch → default dropped to 5e-5.
- **A/B needs no GGUF remint**: the official modelopt NVFP4 ST keeps mtp.* BF16, so the
  trained head patches a hardlinked copy of that dir (`patch_st_mtp.py`, fail-closed on
  name/shape/dtype) and the A/B runs on the ST serve path. GGUF remint (+ masked draft v3,
  + the eh_proj Q8_0-vs-NVFP4 remint test flagged in the artifact lane) happens only on a win.

## Results

(pending — filled from metrics.jsonl / ab-summary.json when the chain completes)

## Backup (box3 is a SPOT rental (owner call 2026-08-20))

Everything irreplaceable syncs to `deadstore:darklanes-artifacts/ornith15/` (retired external object store, Ohio,
created 2026-08-20): `mtp-train/` (corpus, prompts, hidden shards, train-out, scripts, logs),
`gates/`, `st-gates/`. First full sync 2026-08-20 ~04:45Z (corpus + 8.4G hiddens); re-synced
incrementally with every poll via `sync-box3.sh` (fresh 1h STS token per run, passed through
the ssh env — no credentials on box disk). NOT backed up (re-fetchable): `bf16/`,
`nvfp4-official/`, the published GGUF artifacts (already on HF).

## Results (updating)

**Vendor-head offline baseline (full heldout, 200 seqs / 106,509 labels):** top1 0.7997,
loss 0.9564 (think 0.8067 / nothink 0.7838). Offline top1 is a directional proxy only.

**Epoch 0 (lr 5e-5 cosine):** heldout top1 0.8083 (+0.86pt vs vendor), loss 0.8926
(think 0.8140, nothink 0.7954). Both modes improved; epochs 1–2 pending.

**ST gates, official modelopt NVFP4 (`9660379a`), GPU1, 2026-08-20:**
- load + forward: PASS (`st-gates/load.log` — mixed NVFP4 experts + FP8 attn/GDN + BF16 rest,
  experts RESIDENT 18.57GB).
- template parity: memra chat render == transformers `apply_chat_template` prompt ids on all
  3 oracle probes (`argmax-vs-oracle-round2.json`, prompt_ids_match).
- raw-id verify probes: argmax matches oracle token 1 (run-gen raw-id branch is gate-only by
  design — round 1's NO-OUTPUT was a harness misuse, not a model failure).
- 48-tok greedy vs BF16 oracle: DIVERGE@22 / @1 / @18 — quant-arm divergence, near-tie triage
  pending (GGUF NVFP4 arm diverged @14 on one probe with Q8_0 concurring; this arm is a
  different quant program: FP8 attention).
- serve-st-gate: /models PASS, chat PASS, **default-spec == tokenwise serve oracle PASS
  (184/184 chars)**; CLI-vs-server exactness FAIL (diverge tok 8, plain-vs-plain) — OPEN
  engine finding for the qwen35moe ST NVFP4+FP8 arm (`st-gates/serve-st-gate2.log`); q38-class
  ST checkpoints pass this cell, so it is arch/artifact-specific. Not a head-A/B blocker
  (both A/B arms share one serve program).

**v1 (depth-1-only) training complete, 2026-08-20 ~06:4xZ:** heldout top1 by epoch
0.8083 / 0.8127 / 0.8140 (vendor 0.7997), loss 0.9564 -> 0.8505. Exports epoch0-2.
**Owner correction (mid-lane):** depth-1-only CE trains just the first drafted token per
round; serve chains K tokens, depth>=2 seeded by the head's own pre-norm output — the
distribution v1 never saw. Serve chain semantics re-read from `mtp_head_forward_dev`
(op-10 carrier, MEMRA_SPEC_HPOST off) + hqmtp `distill/train.py` chain-slot precedent
(self-recursive carrier WITH grad, chain K/V appended, per-slot CE averaged). Trainer v2:
D=3 unrolled passes, cross-pass K/V via a MiniCache shim, serve-exact masks (verified
against the serve rule by enumeration), per-depth heldout metrics — vendor baseline at
depths 2/3 now measurable (owner hypothesis: that is where the head collapses; serve
K=3 measured 24.3%). v1's A/B still runs as the depth-1 data point + harness shakedown.

**v1 serve A/B (K=3, greedy 256-tok, 3 rounds interleaved both orders, `ab-head.jsonl`):**
vendor mean acc 0.3315 (code 0.3884 / agentic 0.2745); v1 depth-1-trained 0.3378
(code 0.4012 +1.3pt / agentic 0.2745 UNCHANGED); plain decode 176 tok/s vs spec ~62 —
spec stays ~2.7x net-loss at this acceptance. Depth-1 training moves code slightly,
cannot move the chain. Streams deterministic per arm (greedy).

**Vendor head depth collapse, quantified (v2 smoke baseline, heldout subset):**
top1 by depth = 0.8135 / 0.3135 / 0.2212 (loss 1.00 / 4.44 / 5.21). The shipped head is
functional at depth 1 and collapses at chain depths 2-3 — the owner's undertrained-head
diagnosis, now with the mechanism localized to the chain. This is what depth-1-only
training (v1) could never fix and what v2's D=3 unroll trains directly.

**v2 (D=3 chain-rollout) final, 2026-08-20 ~09:30Z.** Heldout (n=106,509):
d1 0.8082 / d2 0.5804 / d3 0.4281 vs vendor 0.7997 / 0.2676 / 0.1345 — chain depths
2.2x / 3.2x the vendor head, d1 +0.9pt (no depth-1 sacrifice). Best epoch 2 by
mean-depth top1.

**3-arm serve A/B (same window, interleaved rotation, K=3 greedy 256-tok, `ab-v2.jsonl`):**
vendor 0.3315 acc / 60.1 tok/s; v1 0.3378 / 60.1; **v2 0.4290 / 69.4** (+9.8pt acc,
+15% spec tok/s vs vendor; code probe 0.5404 vs 0.3884, agentic 0.3176 vs 0.2745);
plain decode 159.4 tok/s. Spec remains ~2.3x net-loss on this model — round cost
(sequential MoE draft chain) dominates; serve posture stays SPEC-OFF. Result class:
method-vs-method win on the head, honest spec economics unchanged. Next cost lever:
masked head (head-read cost) on the v2 head + hqmtp quant order.

## Precision provenance (owner question, 2026-08-20)

What trained on what: head WEIGHTS are BF16 in both artifacts (official NVFP4 ships all
785 mtp.* tensors unquantized) — no weight-precision mismatch. CORPUS = generated by the
NVFP4 GGUF artifact through memra serving (on-policy for the served quant). H SEEDS =
captured from the BF16 trunk under torch — the one train/serve mismatch: serve feeds the
head NVFP4+FP8-trunk hiddens. No torch path is serve-faithful anyway (torch dequants
modelopt NVFP4 to bf16 compute ≠ memra kernels). Win transferred regardless (A/B ran ON
the NVFP4 artifacts). **v3 path: capture h through memra's own `MEMRA_DUMP_HN` tap
(decode.rs, arm-consistent pre-head hidden door) replaying the corpus on the NVFP4
trunk, fine-tune from v2 — head then also learns to compensate trunk quant drift.**

## Shipped (2026-08-20T06:14Z)

GGUF remint pipeline: blk.40 patched in the official BF16 GGUF via `patch_gguf_mtp.py`
(hf_mapping transforms, NormPlusOne on norms, fused-exps restack; fail-closed) -> sealed
NVFP4 mint recipe -> masked draft v3 -> gates (run-spec K=1..8 PASS, chat probes coherent,
GGUF-level same-window A/B v1 0.3523 / v2 0.4309, masked v3 0.393) -> published to
`Avifenesh/Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF` (files replaced in place, card updated with
the trained-head story and serve-level protocol). Head quant settled (owner): NVFP4 head =
zero acceptance cost per the hqmtp receipt — no high-precision head arm needed.
Open follow-ups: v3 serve-exact capture via MEMRA_DUMP_HN on the NVFP4 trunk (fine-tune
from v2); ST head-patch repo (nvfp4-patched-v2 gates green except the pre-existing
arch-level CLI-vs-server cell); CLI-vs-server diverge@8 triage (engine lane).

## Draft-logic + fork research (owner direction, 2026-08-20)

**Verify comparison is CORRECT** (spec.rs greedy accept loop): draft[j] judged against the
target's own choice at the same slot (j=0 vs last_logits, j>=1 vs verify col j-1),
prefix-accept, bonus = target's token at the rejected slot. No off-by-one.

**Draft admission: we draft on EVERY round, not on confidence.** The confidence gate exists
— `MEMRA_SPEC_PMIN` (early chain stop) + `MEMRA_SPEC_PMIN0` (zero-draft rounds, the llama.cpp
35B-win mechanism) — but defaults OFF (p_min=0.0). Verified in data: drafted == K x rounds in
every A/B run. Sweep on the v2-head GGUF (directional, 256-tok greedy, `pmin-sweep.out`):
plain ~215 tok/s; pmin off ~82 (acc 0.33-0.45); pmin 0.85 ~112 tok/s (acc 0.66-0.86, mean
draft len 1.17). Gating = +36% spec throughput, still ~2x behind plain.

**The real spec blocker is ROUND COST, not acceptance**: plain 4.6 ms/step vs 18 ms/round
even at draft len 1.17 — graph draft requires a dense trunk (spec.rs), this SKU is a
256-expert MoE, so rounds run EAGER. Even 100% acceptance cannot beat plain at 4x round
cost. Engine lane if spec-on is wanted here: MoE-capable draft graph / fused round.

**CLI-vs-server diverge@8: NEAR-TIE class, margin-proven** (`fork-triage/`, argmax-margin-
probe on the v2 GGUF over the gate's rendered prompt): flip@8 margin 0.1310 vs config spread
0.4179 -> margin-EXPLAINED; 2 flips vs calibrated budget 2; decision-position argmax agrees
across configs with margin 4.13. The fork sits INSIDE the think block (': what is…' vs
'. The capital…' — reasoning-filler coin). Triage isolation: GGUF forks batched-vs-tokenwise
in the worker, ST forks CLI-vs-server — both at the same near-tie, each arm self-consistent,
spec==tokenwise PASS everywhere -> one-numeric-program law holds. The serve-st-gate identity
cell is miscalibrated for think-dense models (q38 passes only because its 48-tok window hits
no near-tie). Gate fix candidates: margin-aware divergence classification in the gate, or a
nothink probe so the identity cell tests answer-phase tokens.

## Round-cost decomposition (owner question: how can the draft be slower than the model?)

decode-batch-bench on the v2 GGUF (box quiet, `mtp-train/decode-batch.log`): B=1 4.66 ms/step
(214 tok/s); B=4 7.85 ms/tick (509.8 tok/s aggregate, 2.38x) — **MoE batched verify amortizes
(1.68x a plain step at m=4), it is NOT the spec bottleneck.** Measured round = 18 ms, so
**~10 ms/round is draft-side**: 3 eager head steps + per-round host syncs ≈ 3.3 ms per
1-layer head step ≈ 0.7x a FULL 40-layer trunk step — ~25x the head's compute floor. Cause:
the draft chain runs EAGER — graph capture requires a dense block and this head carries a
256-expert MoE FFN, so every draft step pays launch/sync overhead.

Consequence, with the measured numbers: if draft-side cost drops to its compute floor,
round ≈ 9 ms → at the v2 head's ungated acceptance (0.43, K=3, 2.3 tok/round) ≈ 255 tok/s
> plain 214 — **spec-on flips to a win on this model from draft-cost work alone**, before
certainty gating adds its part. Owner directions adopted: (1) draft-side lane = graph/fuse
the MoE head chain + hide residual draft latency by chaining K+1 deeper on its own stream
while the trunk verifies K (single-card overlap; the only true dependency is next-round's
seed on the current verify's accepted hidden — full-accept continuations are pre-draftable);
(2) certainty gating (`MEMRA_SPEC_PMIN(0)`) becomes this model's spec posture when spec is
on. Housekeeping: orphan memra-server (pid 3309435, 20 GB) from the sweep's last arm killed
by PID; sweep script's kill is racy — fixed expectation noted.

# Training plan: a DSpark-class drafter for Step-3.7-Flash

Lane: `lane/dspark-plan` — READ-ONLY research, no GPU, no code. Date: 2026-08-11.
Owner GO: "training dspark is worth it."

Grounding: `research/spec-landscape-20260810/SURVEY.md` (§8b DSpark card, §0 hqmtp
evidence, §1 EAGLE/SpecForge costs), `research/optipipe-20260810/DESIGN.md` (q*=0.70),
`research/specpp2-20260810/RESULTS.md` (K=1 72.97%, K=2 full-accept 0.381), the DSpark
paper (arXiv 2607.05147, full text fetched 2026-08-11), DBLast (arXiv 2608.05448,
coordinator-supplied objective doctrine), `/data/projects/hqmtp/` receipts,
`docs/DRAFT-REGIME.md`, `~/projects/darklanes/sft-pipeline/CORPUS.md` (read-only),
`research/train-loop-pilot-20260805/` + `research/darktrain2-20260810/` receipts.

## Verdict up front

**BUILD, staged, behind one cheap hard gate.** The scheduler half of DSpark is already
designed (optipipe); the drafter half is a training problem we have in-house receipts
for (hqmtp: a 212M head chain-distilled to 85.5% of teacher acceptance). The whole lane
is worthless if a trained head cannot be attached at parity — converter-produced heads
historically collapsed to 35–39% acceptance with no tensor-level cause (DRAFT-REGIME
law 2) — so **Phase 0 is an in-engine attach-parity gate on an already-owned small
head, before any Step-scale corpus or training spend.** First measurable milestone: the
Phase-0 parity receipt on the 9B (days-class, owned rigs, ~zero cost). The prize is
quantified: a drafter whose 2-deep full-accept q clears optipipe's q*=0.70 turns PP-2
c=1 spec from −18.8% into a projected +5.2% to +17.4% over plain
(optipipe DESIGN.md §5), and the same recipe lifts the single-card 2.30x board arm.

---

## 1. DSpark mechanism recap (from the paper)

DSpark (DeepSeek+PKU, arXiv 2607.05147) is two coupled pieces. First, a
**semi-autoregressive drafter**: a parallel DFlash-class backbone — target-layer hidden
states injected into the draft KV, anchor token + γ−1 mask tokens in, γ draft logits
out in ONE forward, so draft latency is nearly independent of block size — plus a
**lightweight sequential head** that restores intra-block dependencies. The sequential
head adds a prefix-dependent transition bias B_k on top of the backbone's base logits
U_k; the default **Markov head** is a low-rank (r=256) factorized V×V transition
matrix B = W1·W2 keyed on the previously *sampled* draft token (an RNN variant exists;
the paper's ablation shows it adds only marginal gains and they ship Markov). This
kills "suffix decay": position-wise conditional acceptance stays high and stable
through the block (their Fig. 2; 2-layer DSpark beats 5-layer DFlash at equal γ), and
accepted length beats EAGLE-3 by +30.9/26.7/30.0% and DFlash by +16.3/18.4/18.3% on
Qwen3-4B/8B/14B. Training: target frozen; drafter shares (frozen) target embedding and
LM head; loss = 0.1·L_ce + 0.9·L_tv + 1.0·L_conf with position weights
w_k = exp(−(k−1)/γ). L_tv is the total-variation distance between draft and target
distributions — an exact proxy for acceptance rate (per-step acceptance
= 1 − TVD/2) — so the training objective IS acceptance, not next-token likelihood.

Second, **confidence-scheduled verification**: a one-linear-layer confidence head
c_k = σ(wᵀ[h_k; W1[x_{k−1}]]) predicts per-position *conditional* survival probability,
supervised by the analytical rate c*_k = 1 − ½‖p_d − p_t‖₁, then post-hoc calibrated by
Sequential Temperature Scaling (ECE 3–8% → ~1%). A hardware-aware scheduler combines
cumulative survival ∏c_i with a profiled throughput curve SPS(B) to pick each request's
verify length per round (production V4: budget expands to 4–6 tokens at light load,
contracts smoothly under saturation). Production receipts: DeepSeek-V4-Flash/Pro
serving, +60–85% and +57–78% per-user speed at matched aggregate throughput vs their
MTP-1 baseline. **Which half we already have:** the scheduler — optipipe's
confidence-gated admission (q*, W=32 estimator, breaker) is exactly this principle
imported for the PP-2 round pipeline, and `choose_spec_k` is the load-aware-K seam
(SURVEY §7). **Which half needs training:** the drafter — the backbone, the Markov
head, and the confidence head. Today's Step head is the pretrain-time NextN chain
reused recursively, and it measurably has no depth signal (K=3 pos-3 acceptance 4.8%,
specpp2 k-sweep).

## 2. Data

**The house method is the paper's method.** DSpark uses only *prompts* from a public
blend (Open-PerfectBlend, 1.3M samples: 17.6% chat / 39.4% math / 38.9% code / 4.1% IF)
and **regenerates all responses with the target model**, 10 epochs. That is
DRAFT-REGIME law 1 ("ranks derive from the exact serving model's own generations") and
the hqmtp finding (own-gen beat a generic corpus by +17 pts at a third of the steps)
stated independently by DeepSeek. The drafter trains on **Step-3.7's own generations
from the exact serving artifact** (IQ4_XS + PP-2 shape) — training against the quant
deployment's hidden states is the hqmtp premise, and it dodges the documented
BF16-drafter-on-quant-target failure (−44.6% acceptance, hqmtp lit-distill).

- **Prompt sources** (prompts only, never the counted distribution): the mixed
  frspec-owngen prompt pack; the SFT-pipeline corpus (`sft-corpus-20260802`,
  1–2k verified agentic traces) as *prompt/task seeds* for the agentic slice — its
  responses are V4-Flash's in Qwen ChatML and MUST NOT be training targets for a Step
  drafter (wrong distribution, wrong template); an Open-PerfectBlend-class public
  blend for domain balance. Serve with the Step chat template ON (law 1's chat-cell
  lesson: raw-derived artifacts left 10.9% unproposable tokens in chat cells).
- **How much.** Reference points: EAGLE-1 heads trained on ShareGPT-class data in 1–2
  days on 8x3090; EAGLE-3 needs ~8x that (~68K+464K conversations, regenerated);
  SpecForge shipped Llama-4 Scout/Maverick heads on 320K samples; DSpark used 1.3M
  prompts x 10 epochs; hqmtp got 85.5%-of-teacher on **5.2M own tokens** (with the
  curve still rising on the data axis). Plan: **milestone corpus = 300K
  prompt-response pairs (SpecForge-class), ~150M generated tokens**, with a 30K-pair
  pilot slice first; scale toward the 1M+ class only if the acceptance curve is still
  climbing at 300K (hqmtp's 4x-data increments each paid +8.6 pts — measure, don't
  presume saturation).
- **What is stored per anchor** (DSpark's anchor-bounded packing): sampled anchor
  positions per sequence; per anchor: the tap hidden state(s), the anchor + γ block
  tokens, and the target's **top-64 logits per block position** (hqmtp `.tt.npz`
  pattern; DSpark's HAI-LLM avoids full-vocab logit transfer the same way —
  hidden-state communication + local head projection). Sparse targets, NOT full
  128,896-vocab dumps.
- **Generation regime:** temp-0.7-class serving-realistic sampling for corpus
  diversity (hqmtp: greedy degenerate transcripts poisoned an eval; DSpark
  regenerates "with recommended sampling parameters"), teacher-forced replay for the
  supervision pass. Non-thinking and thinking modes both sampled — Step serves three
  reasoning levels; the corpus must cover the classes we serve (law 1 coverage rule).

## 3. Architecture

Target facts (step37-bringup PLAN.md, receipts in `raw/`): 196B/A11B, `n_embd=4096`,
45 main layers (3 dense + 42 MoE, 288 experts top-8), vocab **128,896**, deepseek-v3
pretokenizer; the official MTP file is 3 chained NextN layers (3.49B incl re-shipped
embed/head; 96-head SWA-512 attention, dense FFN 11264).

- **Backbone: 2–3 dense transformer layers at d=4096**, SWA-128-class attention,
  target-hidden KV injection, mask-token parallel drafting. The paper's depth ablation
  (2-layer DSpark > 5-layer DFlash) and their production choice (3 MoE layers for V4)
  bound this; dense-not-MoE for us (a drafter MoE buys nothing at our draft batch
  shapes and complicates GGUF export). Parameter class ≈ 0.5–0.8B trainable —
  same order as the survey's "~1–2B class head for a 196B target," below it because
  embedding and LM head are frozen/shared per DSpark.
- **Feature taps: single tap, stage-1-resident.** DSpark/DFlash inject hidden states
  from a set of target layers; EAGLE-3 fuses low/mid/high. On PP-2 a multi-layer tap
  straddles the stage split (SURVEY §1's engine-surgery blocker). Plan of record:
  **one tap = the final trunk hidden on the head stage** (the same carrier the NextN
  chain already receives — the seam exists in `spec.rs`), which keeps drafting
  entirely on stage 1 where the drafter already lives (specpp2 §3: head must stay on
  the head stage). Honest risk: single-tap may cost acceptance vs the paper's
  multi-layer injection; ablate a stage-1-only two-tap variant (mid-stage-1 + final)
  before ever considering a cross-stage tap.
- **MTP-weight reuse vs fresh init.** DeepSpec trains from scratch; the existing NextN
  layers are trained as a *sequential chain* with eh_proj concat input — an
  architecture mismatch with a mask-token parallel backbone. Plan: **fresh backbone
  init; frozen shared embedding + head from the target** (per paper), with one cheap
  ablation arm that warm-starts backbone attention/FFN mats from NextN layer weights
  (hqmtp's hot-row warm start paid; this is the analogous bet, and it is one config
  flag in the harness, not a design fork).
- **Trimmed-vocab head (the owner's masked-vocab line).** The drafter's output
  projection is the target head **trimmed to the top-32,768 own-gen ranks** (the house
  regime; d2t map as in every regime draft). Note the served Step head is untrimmed
  today (the official Q8_0 attaches as-published; no Step ranks receipt exists) — this
  lane introduces the Step trim, and law 1 requires deriving those ranks from Step's
  own generations regardless. Composition details: the Markov head's W1/W2 live on
  the trimmed vocab (32,768 x 256 each, ~8.4M params/side — trivial); the confidence
  head is vocab-free (SURVEY §8d). **Escape handling:** out-of-trim target mass is a
  guaranteed rejection, and L_tv accounts for it honestly IF the loss normalizes over
  the full support — hqmtp's subset-softmax bug (log_softmax over the top-64 subset
  collapsed training) is the exact trap: compute the draft log_softmax over the full
  32,768-row draft vocab with sparse teacher targets, and aggregate all out-of-trim
  teacher mass into one escape bucket that the drafter can never win — the training
  signal then prices trim misses correctly. At serve time the **learned-escape
  adaptive trim** (spare head slots written from verify corrections) closes coverage
  gaps exactly as it does for gemma; the drafter head participates by construction
  since escapes arrive as verify corrections regardless of drafter architecture.
- **Confidence head:** the paper's single linear over [h_k; W1[x_{k−1}]] (4096+256 →
  1, sigmoid), trained end-to-end, STS-calibrated on a held-out split. This is the
  signal optipipe §5 says "the current head has no confidence head" about — it seeds
  the q̂ estimator per request instead of paying the W=32 label warmup.

## 4. Training compute bill

Split the bill into its three phases; generation dominates, training is small.

| Stage | Where | Wall-clock estimate | Evidence base |
|---|---|---|---|
| Corpus generation (150M own-gen tokens, chunked) | box1 pair, serving shape, harvest windows under `/tmp/memra-gpu.lock` — **never co-located with serving** (darktrain2 P0: co-located training broke byte-exactness; dedicated windows only) | 2–6 days of box1 time at plain-batched aggregate rates, resumable chunks (frspec-owngen's `--limit` pattern) | specpp2 plain receipts; DRAFT-REGIME chunking law |
| Supervision extraction (tap hiddens + top-64 logits at anchors) | box1, teacher-forced replay through memra with dump doors (Phase 1 build) — the hqmtp rule: trunk hiddens MUST come from the engine (torch reproduction of the hybrid trunk only reached ~0.5 agreement) | rides the replay pass; same order as generation | hqmtp DISTILL.md gate; `BW24_REPLAY_HDUMP` precedent |
| Drafter training (~0.7B trainable, 150M-token corpus, few epochs) | **one PRO 6000 on box1** (plain torch loop, bf16, FA2-class attention — the pinned cu128 venv exists: torch 2.11.0+cu128, receipts `research/train-loop-pilot-20260805/venv-train-freeze.txt`) | **1–3 days** one card; hqmtp's full 212M pipeline was ~12h ("generate → extract → tt → KD 24k → CE 4k"), this is ~3x params at 2x width | hqmtp system-design.md; train-loop-pilot timings.jsonl (full 9B loop measured stage-by-stage) |
| 5090 laptop (24 GB) | harness dev, unit-scale runs, Phase-0 attach A/B on the 9B — dev-iteration gate only, CPU-quota rule applies | days-class iteration, no production training | rig doctrine |
| Rented 8xH100/H200 (optional accelerator) | compresses drafter training to hours AND enables target-in-torch online mode (BF16 Step ≈ 400 GB fits 8x80 GB) if the engine-dump route stalls | low-thousands $ per SURVEY §1; NOT the plan of record | SURVEY §1(c) |

**Framework:** DeepSpec (DeepSeek's open training repo — ships Eagle3, DFlash, and
DSpark reference implementations) adapted to the memra dump format, with the hqmtp
harness (`distill/extract.py` → `train` → `eval_chain.py`) as the in-house pattern that
already produced the 85.5% receipt. Unsloth/TRL (the train-loop-pilot stack) is the
wrong tool here — this is a custom-architecture torch loop, not a PEFT fine-tune; the
pilot's contribution is the pinned sm_120 environment and the measured
convert→gate→serve loop it feeds into. Training runs are receipts: loss curves,
configs, seeds, corpus + checkpoint hashes committed per evidence discipline.

## 5. Acceptance targets

The numbers to beat (specpp2, Step-3.7 PP-2, N=5 deterministic): K=1 acceptance
**72.97–73.68%**, K=2 full-accept **q = 0.381**, per-position decay
[0.737] / [0.655, 0.388] / [0.676, 0.384, 0.048].

- **Primary target: 2-deep full-accept q ≥ 0.70** on retained serving-shaped traffic —
  optipipe's admission threshold (q*=0.695 with the 5% guard band, operationally
  0.70). That decomposes as slot-1 ≈ 0.85 x conditional slot-2 ≈ 0.83. Plausibility:
  DeepSeek-V3's pretrain-co-trained MTP hits 85–90% second-token acceptance in
  deployment; DSpark's position-wise curves start at 0.93 (math) and stay stable
  through the block; hqmtp's distilled slots 1–3 EXCEEDED its teacher. A drafter
  co-trained on the serving distribution reaching 0.85-class slot-1 is the literature
  norm, not a stretch.
- **What q ≥ 0.70 buys:** optipipe increment-3 admission opens — projected
  **85.4–95.3 tok/s vs plain 81.2 (+5.2% to +17.4%)** at c=1 PP-2, on traffic where
  today **0/80 measured bursts qualify** (optipipe §5). This is the whole point: the
  trained drafter moves the traffic's q distribution across a gate that is already
  designed, priced, and waiting.
- **Single-card target: K=1 slot acceptance ≥ 0.85 and healthy depth** (slot-3
  conditional ≥ 0.45, vs today's 0.048 on Step) lifts the board arms directly — the
  9B's 2.30x at K=3 is acceptance-limited at depth, and the single-card marginal
  column is cheap (0.9–1.9 ms to T=16), so tokens/round gains convert ~linearly.
  Target: **>2.6x on the 9B-class arm** as the first e2e proof (Phase 2), where the
  measured 2.30x baseline and its frozen protocol already exist.
- **The confidence head has standalone value** even at missed q: calibrated per-token
  survival enables verify-length pruning at c≥2 (DSpark's production mechanism —
  budget 4–6 at light load, contracting under pressure), which composes with the
  specmech multi-session pipeline independently of optipipe's c=1 gate.
- **Honest miss case.** If the trained drafter plateaus at q ≈ 0.55–0.65 — above
  today's 0.381, below q* — optipipe stays correctly closed and the PP-2 c=1 story
  does not ship; the lane's value collapses to the single-card lift plus the
  confidence signal, and that must be priced honestly at Phase-2 exit before Step-scale
  spend (Phase 3+ proceeds only if the 9B pilot's q trajectory supports 0.70-class on
  Step). And the standing arithmetic never moves: even a perfect K=1 drafter loses to
  plain PP-2 c=1 without the schedule (75.9 vs 81.2 tok/s) — the drafter is the
  q-raiser FOR optipipe/specmech, never a standalone PP-2 fix.

## 6. Evaluation gates

**Offline (teacher-forced, before any engine integration):**

1. **Chain acceptance protocol** (hqmtp `eval_chain.py` pattern): self-drafting chain
   eval on ≥45 held-out own-gen docs, per-slot conditional acceptance + chain
   acceptance, vs the current NextN-recursive baseline replayed identically. DSpark's
   position-wise conditional metric (denominator = instances where all prior positions
   accepted) is the depth-decay detector.
2. **Serving-realistic sampling** (DBLast doctrine, arXiv 2608.05448): eval at temp>0
   with the serving sampler settings, not greedy-only — the objective is expected
   VERIFIED length under the serving regime; greedy-only evals overstate heads that
   memorize argmax paths. Report temp-0 and temp-0.7 cells separately.
3. **Confidence calibration:** ROC-AUC (paper: 0.81–0.90 class) and post-STS ECE ≤ ~1%
   on held-out; the optipipe gate consumes absolute magnitudes, so calibration is a
   gate, not a nicety.
4. **Torch-vs-engine agreement gate** (hqmtp's mandatory pre-training gate, PASSED
   pattern): before training on extracted data, score the torch-side replay against
   engine replay on shared hiddens — equal-or-better target-hit rate + ~0.90 agreement
   class, disagreements uniform.

**Integration (after offline win, in order):**

5. Attach through the external-drafter route (`+draft` / `MEMRA_MTP_DRAFT`) —
   **in-engine acceptance parity with the offline metric** (within ~2 pts, the
   torch/engine ULP-class disagreement band), the gate Phase 0 de-risks.
6. The standing battery: `kernel-check` ALL GREEN, `run-gen` argmax MATCH, `run-spec`
   K=1..8 self-consistency, spec/plain byte identity — exactness is non-negotiable
   (drafter quality costs speed, never correctness, by verify construction).
7. Board-protocol e2e: N=5 interleaved on box1, decision on **tok/s, never
   acceptance** (law 3); PP-2 claims re-proven on the Vast 2x PRO 6000 verification
   box before any default flip. The optipipe interaction re-solves q* from measured
   I_hit/I_miss with the new head (optipipe §5: "a pre-build price, not a forever
   constant").

## 7. Staged bill

| # | Phase | Deliverable | Rig | Effort | Exit receipt |
|---|---|---|---|---|---|
| 0 | **HARD GATE: attach parity** | Round-trip the 9B's own MTP head through the trained-head export path (safetensors→GGUF converter) and A/B in-engine acceptance vs the byte-verbatim extraction of the SAME head; localize or fix the 35–39% converter collapse (DRAFT-REGIME law 2 open mystery); if the hqmtp StudentSV export (`export_sv.py`) attaches under existing seams, run it as the trained-head arm | 5090 / box1 window | **S–M, ~zero cost** | in-engine acceptance within 2 pts of byte-verbatim, run-spec K=1..8 PASS; **no Phase ≥2 spend until green** |
| 1 | Dump doors | Extend the replay hidden-dump precedent to (a) Step-3.7 PP-2, (b) top-64 target logits at sampled anchors, (c) anchor-block packing format DeepSpec can read; parity-check against the hqmtp torch gate on the 9B | box1 | M | extraction-parity receipt; format doc |
| 2 | **9B DSpark pilot — first measurable milestone** | DeepSpec-adapted harness; small backbone (1–2 layers, d=1024–2048 class) + Markov + confidence heads on Qwen3.5-9B own-gen; offline chain acceptance vs the 2.30x board arm's trimmed MTP; then attach + e2e | 5090 dev, box1 train | M | offline chain ≥ NextN-baseline chain; attach parity; e2e A/B vs 2.30x; **q-trajectory read for the Phase-3 go/no-go** |
| 3 | Step corpus + trim | 30K-pair pilot slice then 300K own-gen pairs on the serving artifact (chunked box1 windows); Step's own-gen ranks → the 32,768-row trimmed head (also, independently, the missing Step trim for the CURRENT head — a free side-deliverable) | box1 | M (rig-time L) | corpus + ranks hashes; trim `--validate` verdict on the existing head |
| 4 | Step drafter training | 2–3 layer d=4096 backbone + heads, naked-KD→CE-reinforce order per hqmtp, DSpark loss weights, fresh-init + NextN-warm-start ablation, STS calibration | box1 one card (rented 8xH100 fallback) | L | loss/config/seed receipts; offline gates 1–4; q measured on held-out serving-shaped traffic |
| 5 | Integration + admission | Attach; battery (gates 5–7); optipipe q* re-solve with measured I_hit/I_miss; confidence-seeded admission; specmech composition cell | box1 + Vast verification box | M | e2e A/B receipts; ship/hold per the frozen optipipe promotion bar |

Effort classes per house convention (S hours, M days, L week+). Total new-training
cash cost on the plan of record: ~zero (owned rigs); the rented-node arm is an
explicitly optional accelerator. Every phase commits raw logs beside its summary
(evidence discipline), and every phase has a kill condition: Phase 0 red stops the
lane; Phase 2's q trajectory gates Phase 3; Phase 5's promotion bar is optipipe's,
already frozen.

## Objective doctrine (binding, from DBLast + the paper)

Train for **expected verified length**, not next-token likelihood: the loss centers on
L_tv (TVD = the analytical acceptance complement) with DSpark's exponential position
weights as the baseline form, and a DBLast-style survival-weighted refinement —
weighting position k by the predicted survival of its prefix — as a priced ablation;
CE is the small anchor term (α_ce=0.1), exactly inverting the usual ratio. Model
cross-position dependence explicitly (the Markov/RNN head exists for this; hqmtp's
chain-rollout training "put the student above its teacher at depth" for the same
reason). Evaluate at serving-realistic temperature, never greedy-only.

**The optipipe tie, in one paragraph:** optipipe's verdict was
CONFIDENCE-GATED-BUILD — the schedule wins (+5.2% to +17.4% projected at q=0.70,
ceiling +44.6% at q=1) but today's head leaves 0/80 measured bursts above the gate.
A DSpark-class drafter is the q-raiser: slot-1 x conditional-slot-2 at
literature-normal co-trained levels (0.85 x 0.83) clears q*=0.70, and its calibrated
confidence head replaces optipipe's cold-start W=32 label warmup with a per-request
seed signal — unlocking increment-3 admission on real traffic rather than a forced
demo. The two lanes are one product: optipipe built the gate; this lane builds the
head that walks through it.

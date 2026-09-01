# Speculative-decoding landscape survey — candidates for memra, ranked for PP-2 serving

Date: 2026-08-10. Lane: `lane/spec-landscape` (from `restructure/public-split` @ 3f8ca2ef).

Scope: what could raise decode throughput beyond the current MTP head, especially on the
PP-2 two-card serving shape, and how to improve what we have. Sources: papers and engine
docs (arXiv, vLLM/SGLang/TensorRT-LLM/llama.cpp), quoted with their own setups; all
memra numbers are committed receipts, cited by path. This is a survey, not a measurement
lane — nothing here is a claim about memra performance until measured under the board
protocol.

---

## 0. The house baseline this survey extends

The house method is the **own-trimmed MTP draft regime** (`docs/DRAFT-REGIME.md`):
byte-verbatim NextN/MTP head extraction from the serving GGUF, a 32,768-row draft head
trimmed by ranks from the model's OWN generations, NVFP4 head + Q4_K_M block, and the
serve-time **adaptive trim** — coverage escapes arrive as verify corrections and are
written into spare head slots (learned-escape loop; receipts 2026-07-19: 31B chat
−17% → +2.5%, trim ≥ untrimmed on every measured cell). This is original work that
predates the public convergence (SpecForge/speculators now ship 128k→32k hot-token draft
heads with d2t/t2d maps — the same mechanism class, without the learned escape loop).
Everything below is evaluated as an **extension of** this regime, not a replacement.

Measured state:

- **Single card WINS**: Qwen3.5-9B K=3 + own-trim 2.30x, 27B 1.27x, 35B-A3B K=2 1.29x
  (docs/PERFORMANCE.md spec board); spec 1.82x at c=1, crossover c=2..4, flat at c≥4
  (`research/spec-scaling-20260806/`).
- **PP-2 LOSES at every c**: verify = 95.13% of a K=1 round (17.222 of 18.1045 ms),
  draft only 0.70 ms; K=1 c=1 −18.81%, c=2 −42.76% vs plain; plain scales +41.93%
  c=1→2, spec flat (`research/specpp2-20260810/RESULTS.md`).
- **The head decays hard with depth** (Step-3.7, serve receipts,
  `research/specpp2-20260810/raw/k-sweep/`): per-position acceptance K=1 [0.737];
  K=2 [0.655, 0.388]; K=3 [0.676, 0.384, **0.048**]. tokens/round 1.74 / 2.04 / 2.11 —
  the single +1-trained head reused recursively has almost no depth-3 signal.
  (run-spec CLI at K=8: 77.8% K=1 falling to 11.0% K=8,
  `research/step-draft-20260807/RESULTS.md`.)
- **In-house drafter-training evidence** (hqmtp, `/data/projects/hqmtp/`): a 212M
  half-width draft block + 32,768-row hot head, chain-distilled on own-generations,
  reaches chain acceptance 0.507 = **85.5% of the co-trained teacher** (0.593) at ~13%
  of the draft FLOPs; slots 1–3 exceed the teacher; own-gen training beat a generic
  corpus by 17 points at a third of the steps. Acceptance-side evidence only (no
  wall-time claim yet), but it proves we can train heads that hold up at depth.

Two receipt-derived numbers used throughout:

- **PP-2 marginal verify column ≈ 7.5–8.7 ms e2e** (K-sweep round deltas: 26.35 →
  35.01 → 42.51 ms/round for K=1/2/3) — 61–71% of a plain step (12.32 ms). On this
  fine-grained MoE (Step-3.7, 288 experts/layer), extra verify columns activate extra
  distinct experts, so column amortization is weak.
- **Single-card marginal verify column ≈ 0.9–1.9 ms** up to T=16 (ms/column falls
  4.01 → 1.68 from T=2→16), then a 2.3x per-column cliff at T=17 — the exact-verify
  width tier (`lib.rs:5846`, `research/spec-scaling-20260806/` §3).

---

## 1. EAGLE-1 / 2 / 3 — feature-level autoregressive draft heads

**Mechanism.** EAGLE-1 (ICML'24, arXiv 2401.15077) trains a one-decoder-layer draft
head that autoregresses at the *feature* level: it predicts the target's next top-layer
feature from (current feature, sampled-token embedding), then reuses the target's LM
head to get the draft token. EAGLE-2 (EMNLP'24, arXiv 2406.16858) keeps the same
weights and adds a context-aware **dynamic draft tree**, using draft-head confidence
(a good acceptance-rate proxy) to decide which branches to grow per round. EAGLE-3
(NeurIPS'25, arXiv 2503.01840) drops the feature-prediction loss for **direct token
prediction** and fuses low/mid/high trunk-layer features as input, with
**training-time test (TTT)**: training unrolls the draft head multiple steps feeding
its own outputs, so deep-position acceptance stops collapsing and the head finally
scales with training data.

**Measured (their setups).** EAGLE-3: "3.0x–6.5x vs vanilla AR, ~1.4x over EAGLE-2,
acceptance length up to 7.5 on HumanEval" (paper, Vicuna-13B / LLaMA-3.x / DSL-8B, temp
0). Spec-Bench A100 leaderboard (Vicuna-13B, bs=1, greedy): EAGLE-3 **3.02x** overall
with 5.71 mean accepted tokens vs EAGLE-2 2.46x/4.43 and EAGLE-1 2.16x/3.64. Batch
behavior: in the paper's vLLM study EAGLE-1's throughput gain peaks at bs=24 while
EAGLE-3's peaks at bs=56 (chain length 2, no tree); SGLang production numbers (E2E
Networks writeup): EAGLE-3 1.81x at bs=2 and **still 1.38x at bs=64**, where EAGLE-2
drops to 0.93x — high acceptance is what lets spec survive batching.

**Draft-training cost.** Cheap-to-moderate, and falling. EAGLE-1: "trainable within 1–2
days on 8x RTX 3090" (official README); heads are 0.24–0.33B for 7–13B targets, ~1B for
70B. EAGLE-3 needs ~8x more data (ShareGPT+UltraChat, ~68K+464K conversations,
responses regenerated by the target) plus the TTT unroll; SpecForge (LMSYS) trained
Llama-4 Scout (2.0x) and Maverick (2.18x) heads on 320K samples, and supports
online mode (target held in GPU during training) or offline mode (precomputed hidden
states; ~12TB disk for that corpus). Reduced 32k draft vocab with d2t/t2d is now a
SpecForge/speculators standard.

**What an EAGLE-3 head for Step-3.7 would take.** (a) Data: own-gen corpus via the
existing `frspec-owngen` machinery + target-generated responses — owned-rig work.
(b) Hidden-state taps: EAGLE-3 input is a fusion of three trunk-layer features; memra's
engine would need to expose low/mid/high per-token features during generation (spec.rs
currently hands the drafter only the final-hidden/NextN carrier) — real engine work,
also needed at serve time, on both PP-2 stages (the tap layers straddle the stage
split). (c) Training: the head for a 196B/A11B MoE is ~1-2B class; online mode needs
the BF16 target resident (~400 GB — a rented 8xH100/H200 node), offline mode needs the
hidden-state dump path built into memra first. Rented-compute cost is low thousands of
dollars; calendar time ~1–2 weeks including data. (d) Serving: the trained head enters
through the external-drafter route — and DRAFT-REGIME law 2's open mystery bites here:
converter-produced (non-byte-verbatim) drafts collapsed to 35–39% acceptance with no
tensor-level cause found. Any trained-head plan must first close that route or
reproduce hqmtp's external-head attach with acceptance parity in-engine.

**Fit to memra.** High on single card: it is the acceptance ceiling-raiser, and the
llama.cpp ecosystem now carries EAGLE-3 GGUF conversion (PR #18039 merged; the on-disk
llama fork already had EAGLE3 support per project memory), so GGUF-side precedent
exists (one-layer draft, target tokenizer inherited, reduced vocab + d2t — structurally
close to our draft files). On PP-2 it does NOT fix the stage bubble (see §9 arithmetic:
even perfect K=1 acceptance loses to plain at c=1).

**Honest blocker.** The feature-tap contract (engine surgery on both stages) plus the
trained-head attach mystery; and TTT training is genuinely new infrastructure for us
(though DeepSpec and SpecForge are open reference implementations).

---

## 2. Medusa / Hydra — multi-head parallel draft

**Mechanism.** Medusa (arXiv 2401.10774) bolts N independent MLP heads onto the frozen
target, head k predicting token t+k+1 from the same final hidden state; candidates
combine into a tree verified with tree attention. Because heads are independent
(position k cannot see the sampled token at k−1), deep-head accuracy is weak. Hydra
(arXiv 2402.05109) makes the heads sequentially dependent — head k takes the sampled
draft token embeddings from earlier heads — recovering much of the lost acceptance.

**Measured.** Medusa: 2.2–3.6x (their repo, Vicuna class, with tree + typical
acceptance — the headline number uses *relaxed* acceptance, not lossless); Spec-Bench
A100 lossless: Medusa 1.80x, Hydra 2.20x (Vicuna-13B). Hydra++: "1.31x over Medusa
decoding, 2.70x over AR" (paper).

**Draft-training cost.** Cheapest of the trained methods: Medusa-1 is ~5 hours on a
single A100 for Vicuna-7B on 60k ShareGPT samples (frozen backbone).

**Fit to memra.** Poor as a direction: it is strictly dominated by EAGLE-class heads on
lossless acceptance (Spec-Bench), its wins depend on tree verification (weak on our
PP-2, §6), and our native MTP head IS already a sequentially-dependent draft head —
adopting Medusa would be a downgrade of the house method, not an extension. The one
transferable idea is already absorbed: multiple cheap projections from one trunk state
is what the hqmtp StudentSV slots do.

**Honest blocker.** Lossless Medusa is mid-pack; the headline numbers require relaxed
acceptance, which violates memra's exactness contract (verify==decode bit identity).

---

## 3. ReDrafter — RNN drafter (Apple)

**Mechanism.** An RNN draft head on top of the target's last hidden state (recurrent
state carries the in-block token history), combined with beam search over draft
candidates and dynamic tree attention over the deduplicated beams; one drafter forward
per draft token, but the recurrence gives it real sequential conditioning.

**Measured.** "Up to 3.5 tokens per generation step" (Apple ML blog, open-source
models); TensorRT-LLM integration: "up to 2.5x throughput on H100" (NVIDIA/Apple,
greedy, production-shape); OpenReview: 2.5x on H100 with TP + continuous batching.

**Draft-training cost.** Cheap (drafter is small, trained on target generations;
same order as Medusa/EAGLE-1).

**Fit to memra.** The interesting part for us is not the RNN (our MTP chain already
conditions sequentially through the NextN block) but the receipt that **beam + tree
verify pays on single-card serving shapes** — and that TensorRT-LLM productized
accept/rollback for it. As a method swap it offers nothing over an EAGLE-3-class or
TTT-tuned MTP head, and its verify widening is the wrong direction for PP-2 (§6/§9).

**Honest blocker.** Beam drafting multiplies verify columns (expensive on the
MoE/PP-2 shape) for acceptance gains our own tree design already bounds at +3–6% e2e
on the dense single-card case (`research/gemma4-bringup/TREE-DRAFT-DESIGN.md`).

---

## 4. Draft-model speculation — the classic separate small model

**Mechanism.** The original SpS form (Leviathan/Chen 2023): a small same-family LM
drafts K tokens autoregressively; target verifies. No target modification, no head
training if a sibling model exists.

**Measured.** Spec-Bench A100: SpS with Vicuna-68M drafting Vicuna-13B = 1.54x overall
(and 1.52–1.57x across 7B/33B) — well below EAGLE-class. vLLM's method table rates
draft-model "high gain low-QPS / medium high-QPS". The classic 7B-drafts-70B pattern
survives mostly where no trained head exists.

**Could a Qwen-small draft Step-3.7?** Not naturally: **Step-3.7 uses the deepseek-v3
tokenizer** (`tokenizer.ggml.pre = "deepseek-v3"`, crates/memra-tokenizer/src/unicode.rs:358)
with a 128K byte-BPE vocab; Qwen models use the Qwen BPE — different vocabularies, so
naive SpS is impossible. The gap is now formally closed in the literature: "Lossless
Speculative Decoding Algorithms for Heterogeneous Vocabularies" (ICML'25 **oral**,
arXiv 2502.05202) gives three lossless cross-vocab algorithms (string-level exact
matching with bidirectional token translation among them), and vLLM ships TLI
(token-level intersection, `use_heterogeneous_vocab: true`) — but greedy-draft-only
today, and acceptance across model families is structurally lower (the drafter models a
different distribution AND a different segmentation). StepFun ships no small dense
sibling on the same tokenizer; DeepSeek smalls (distill line) are LLaMA/Qwen-tokenizer
models, so no free lunch there either.

**Draft-training cost.** None if a sibling exists (it doesn't, for Step); otherwise
you are training a small LM — the most expensive option in the survey.

**Fit to memra.** Poor for Step-3.7 (tokenizer wall + translation layer + a second
model resident in VRAM we bill by the byte). For Qwen targets we already have
better-than-SpS drafters (the trimmed MTP head beats what a 0.5B external drafter
would give at a fraction of the residency). The TLI/hetero-vocab machinery is worth
knowing about as the escape hatch if a future target ships headless with no donor.

**Honest blocker.** Tokenizer mismatch for the pair actually asked about; and SpS
acceptance economics lose to head-based drafting everywhere we serve.

---

## 5. Self-speculative, training-free methods

**5a. Prompt-lookup / PLD.** Draft = longest n-gram match of the recent context found
in the prompt, continued verbatim. Measured: "consistent 2.4x on summarization and
context-QA" (author repo, input-grounded tasks); Spec-Bench lossless overall 1.56x
(13B) but with a strong task skew (2.30–2.66x summarization, ~1.05x translation/QA).
Zero training, near-zero draft cost, ships in vLLM/transformers/llama.cpp. Fit: cheap
to add to the verify path (proposals are just token sequences); composes with the
adaptive-trim loop (matched spans are by construction in-distribution). Blocker: gains
concentrate on input-grounded workloads; does nothing for freeform generation.

**5b. Lookahead decoding (LMSYS, ICML'24, arXiv 2402.02057).** Runs a Jacobi-style
sliding window alongside decoding, harvesting n-grams from the iteration trajectory
into a pool, verified in the same forward. Measured: up to 1.8x on MT-bench (7B,
FlashAttention build), "4x with strong scaling on multiple GPUs in code completion";
Spec-Bench lossless: **1.30x** (13B) — the honest general-workload number. Costs extra
FLOPs per step (trades log(FLOPs) for latency). Fit: poor — memra's decode is
memory-bound but our exact-verify width tier (T≤16) is precious budget, and lookahead
spends it on speculative *window* columns with the worst acceptance-per-column in this
survey. Blocker: per-step FLOP inflation collides with the c≥4 batching wins that
already own high-QPS.

**5c. LayerSkip / self-spec via early exit (Meta, ACL'24, arXiv 2404.16710).** Draft =
the target's own first E layers + early-exit LM head; verify = the full model, reusing
the shared prefix compute/KV. Measured: 1.34–2.16x task-dependent. BUT the good numbers
require *training the target* with early-exit loss + progressive layer dropout —
"you will also only obtain speedups" with the recipe (HF blog). Fit: violates memra's
frozen-artifact posture (we serve published GGUFs byte-exact; we do not retrain
trunks). A no-training early exit on a stock model has weak acceptance. Blocker:
changes the model; dead on arrival for us.

**5d. Jacobi / CLLMs (ICML'24, arXiv 2403.00835).** Fine-tune the target so Jacobi
fixed-point iteration converges in few steps (consistency training); 2.4–3.4x claimed.
Same blocker as LayerSkip, stronger: it *is* a target fine-tune. Not lossless w.r.t.
the published artifact. Out.

**5e. Suffix decoding (Snowflake, NeurIPS'25 spotlight, arXiv 2411.04975).** Draft =
walks a **suffix tree over previous outputs** (global across requests + per-request),
speculating adaptively long continuations where history repeats. Measured: "up to 5.3x
on agentic benchmarks (SWE-Bench, AgenticSQL), 2.8x faster than EAGLE-2/3-class
model-based methods on those workloads" (paper); Arctic/vLLM production: 1.8–4.5x
end-to-end across SWE-Bench subtasks (Snowflake blog); ships in vLLM
(`method=suffix`, dynamic spec depth, max_spec_factor). Zero training. Fit:
**excellent for memra's declared serving workload** (agentic, repeated tool-call
schemas, multi-turn) — and it is the one drafter class whose cost does not grow with
concurrency (vLLM's table: "no extra draft model; no added workload at peak traffic",
consistent with our c≥4-plain policy). Composes with the house regime rather than
replacing it: suffix proposals can feed the same verify path, and misses feed the
adaptive-trim escape loop. Blocker: needs an engine-side proposer seam (external token
proposals into the spec verify path) + suffix-tree state in the server; wins are
workload-dependent and must be measured on OUR traffic shape, not benchmark agents.

---

## 6. Tree / multi-candidate verification

**Mechanism.** SpecInfer (ASPLOS'24, arXiv 2305.09781) introduced token-tree
verification: multiple candidate sequences share a prefix tree, verified in one target
pass with tree attention; accept the best root-path. EAGLE-2's dynamic trees grow
branches by draft confidence. Measured: SpecInfer 1.5–2.8x distributed / 2.6–3.5x
offloading (vs incremental-decoding baselines of the time); EAGLE-2's tree is worth
20–40% over EAGLE-1's chain at equal weights.

**In-house state.** We already have a fixed-topology tree design on the shelf:
`research/gemma4-bringup/TREE-DRAFT-DESIGN.md` — spine + top-2 siblings at shallow fork
depths, path-duplicated rows (v1 = ZERO new kernels, rides the existing b16 verify tier
and rollback machinery), estimated +3–6% e2e on the 26B dense at ~17% first-miss rate.

**The PP-2 question, quantified.** The hypothesis "more tokens verified per round
amortizes the 95.13% verify share" fails on arithmetic, because the verify share is
*per-column stage compute*, not per-round fixed overhead:

- What a tree amortizes is the **non-verify** part of the round — and that is 4.87% of
  a PP-2 K=1 round (draft 0.70 + accept 0.024 + commit 0.147 + other 0.007 ms of
  18.10). Perfectly amortizing ALL of it caps at +5.12% round rate
  (`research/specpp2-20260810/RESULTS.md`), while c=1 needs +23.16% just to tie plain.
- Each added tree column costs real verify time. Measured e2e round deltas on the
  PP-2 K-sweep give **7.5–8.7 ms per added column** (26.35 → 35.01 → 42.51 ms for
  K=1/2/3) — 61–71% of a full plain step. The MoE trunk is why: extra columns route to
  extra experts, so weight-read amortization is weak (contrast the dense single-card
  ksweep: 0.9–1.9 ms marginal per column, T=2→16 ms/column falling 4.01→1.68).
- The acceptance value of a column is bounded: a depth-1 top-2 sibling can only
  recover first-position misses (26.3% of rounds at 0.737 accept), at the EAGLE-cited
  20–40% recovery = **+0.05–0.11 tokens/round**, against a ~5.5–8 ms column cost. On
  PP-2 that LOWERS tokens/ms (0.0659 → ~0.055 e2e class). Tree verify on the PP-2
  Step-3.7 shape is anti-leverage.

**Verdict.** Tree verification is a **single-card, dense-model** lever for us — the
shelf design's +3–6% on 26B stands, worth measuring there — and it is the standard
partner of an EAGLE-3-class head if we train one. It does not make the PP-2 verify
share better; it makes the round longer. The 95.13% share is an idle-stage problem
(one stage always waits), and only schedule-level fixes address it (§7).

**Cost.** v1 tree: no training, no new kernels, moderate engine work (fork drafting,
path accept rule, repack-commit). Tree-masked verify (v2) is a new kernel.

---

## 7. Batched / continuous-batching speculation

**How the big engines reconcile spec with batching.** vLLM integrates spec into
continuous batching with dynamic speculation length ("lookahead scheduling"; disable
spec when batch pressure is high — dynamic speculative decoding adjusts per step);
their table is explicit that n-gram/suffix methods are favored at high QPS because
they add no drafter load at peak. SGLang gates spec behind topk/overlap constraints
and got its EAGLE-3 to hold 1.38x at bs=64 (vs EAGLE-2's 0.93x) — acceptance quality,
not scheduling magic, is what survives large batches. PARD (AMD, ICLR'26,
arXiv 2504.18583) converts an AR draft model to parallel drafting via cheap fine-tune
(their PARD-2 claims avg 1.3x over PARD, up to 6.94x over AR, "highest throughput
under high concurrency"). DSpark (§8b) adds the load-aware piece: verify length
scheduled per request by predicted survival and engine throughput profile.

**memra state.** This is our measured sore spot in both shapes, and the bills are
already written: single card — spec sessions burst solo and are excluded from batched
decode (`worker.rs:1686/1871`); a pooled cross-session verify is capped at 16 total
columns by the exact-verify tier, bounding the fix at 1.27–1.44x on an arm that plain
batching beats 2.1x at c=8 → **the c≥4-stays-plain policy is correct on single card**
(`research/spec-scaling-20260806/`). PP-2 — the c=2 recovery is the **stage-resident
multi-session pipeline + batched spec prefill** (effort L, in development on
`lane/specmech`; ideal round interval 18.10 → 9.60 ms, a 1.89x round-phase upper
bound, and even that needs prefill composition to clear the measured 6.64 s gap).
The batchdraft lane already measured the trap in the naive version: concatenating
sessions into today's contiguous m=16 verify REGRESSES at m=16 (+3.81% vs 4x m=4,
`research/batchdraft-20260808/RESULTS.md`) — a true per-cache B x T verifier (the
missing `fa_decode_vec_q_rows_seqs` kernel) is the real object.

**What the landscape adds to our bills.** Two ideas import cleanly: (1) *dynamic
per-request speculation length under load* (vLLM/DSpark) — our `choose_spec_k` already
has the seam (it returns K=0 with source `pp2-placement`); extending it from a binary
gate to a load-aware K schedule is cheap scheduler work once multi-session spec
exists. (2) *acceptance-quality-first* (SGLang's EAGLE-3-at-bs=64 receipt) — batched
spec only beats batched plain if tokens/round stays high under the width cap, which
for us means better heads (§1/§8) before wider pools.

**Blocker.** All of this is gated on the specmech pipeline landing; nothing in the
public playbook removes our T=16 exactness tier — it is the price of bit-identical
verify, and we keep it.

### SPD — zero-bubble single-request speculative pipelining

[Speculative Pipeline Decoding](https://arxiv.org/abs/2605.30852) (SPD; arXiv
2605.30852v3, cs.CL preprint; no accepted venue stated) targets the **c=1** pipeline
bubble that ordinary microbatch pipelining cannot fill. It partitions the frozen target
over `n` pipeline stages and keeps successive unverified positions from one request at
different target depths. A trained Pipeline Draft Module (PDM), on a dedicated `n+1`th
GPU, predicts one token per cycle from partial multi-depth target features; it never
conditions on its own draft hidden states. Phase-shifted target verification uses the
ordinary greedy or rejection-sampling rule; a rejection flushes younger in-flight
activations, truncates stage KV to the last verified prefix, and restarts from the target-correct
token.

**Reported and exactness posture.** On batch-1 H20 inference, the best `n=8` arm reports
**2.53x** over single-GPU autoregressive decode for a 4B target (versus EAGLE-3's 1.97x)
and **2.67x** for a 9B target (versus EAGLE-3's 2.54x); `n=16` loses wall-clock efficiency
despite a higher theoretical pipeline gain because control, communication, verification,
and flushing overheads stop hiding. These are latency figures, not resource-normalized
throughput: `n=8` uses nine H20 ranks, while the autoregressive and EAGLE-3 baselines use
one. This is **not training-free**: the paper distills the PDM from about 1.2M examples
while freezing the target. It is lossless relative to the target only while the standard
verifier and exact rollback remain authoritative.

**Memra composition and verdict.** SPD is distinct from both current memra seams. The
PP-2 prime overlaps **prompt chunks** — stage 0 chunk N+1 with stage 1 chunk N — and does
not speculate decode tokens (`crates/memra-engine/src/hybrid_forward.rs:737-787`). The
shipped speculative pipeline pairs **two warm sessions** and changes only their phase
issue order (`crates/memra-engine/src/spec.rs:5201-5304`,
`crates/memra-server/src/worker.rs:6748-6780`). The dual-active decode design likewise
requires `c>=2` and explicitly falls back to serial at `c=1`
(`research/dualpp-20260811/DESIGN.md:43-75`). SPD therefore closes a real, otherwise-open
c=1 bubble, but it is **not a graft into the request worker or today's verifier**: it needs
stage-granular token flight state, partial-target feature taps, a trained PDM rank, and
delayed cross-stage KV rollback. Verdict: **defer as a real PP research architecture, not
an implementation lane or runtime default**, until c=1 demand justifies `n+1` GPUs and a
new training artifact; it does not replace PP-2 prefill overlap or the c>=2 multi-session
schedule.

---

## 8. MTP-native improvements — extending the house method

**8a. Multi-token MTP heads (DeepSeek-V3 style, K>1 native).** DeepSeek-V3 trains MTP
modules jointly at pretrain time; in deployment the second-token acceptance is
**85–90%, giving 1.8x TPS** (tech report, arXiv 2412.19437). That is what a head that
*saw depth during training* buys — against our Step-3.7 receipts (K=2 pos-2 38.8%,
K=3 pos-3 4.8%). We cannot retrain Step's pretrain-time head, but the gap defines the
prize for 8c.

**8b. DSpark (DeepSeek+PKU, June 2026, arXiv 2607.05147)** — owner-requested; it is a
**real, distinct method**, not a respelling of anything above (nearest relative:
DFlash, its parallel-drafter predecessor; both are in the open-source DeepSpec repo,
and llama.cpp PR #25173 layers DSpark on its merged DFlash support). Mechanism: a
**semi-autoregressive drafter** — a parallel DFlash-class backbone (target-layer KV
injection, gamma mask-token positions drafted in ONE forward, so drafting latency is
~independent of block size) plus a **lightweight serial output head** (Markov or
small-RNN over the block) that restores intra-block dependencies and kills "suffix
decay"; plus **confidence-scheduled verification** — a survival-probability head per
position, combined with a live engine-throughput profile, sets each request's verify
length so batch capacity isn't wasted on doomed suffixes. Measured: accepted length
+30.9/26.7/30.0% over EAGLE-3 and +16.3/18.4/18.3% over DFlash (Qwen3-4B/8B/14B,
offline); in production DeepSeek-V4 serving vs their MTP-1 baseline: **per-user
+60–85% (V4-Flash) and +57–78% (V4-Pro) at matched aggregate throughput**, and it
holds throughput under strict SLAs where MTP-1 collapses. Training cost: a full
drafter-backbone train (DeepSpec repo; more than a head fine-tune, far less than a
model) . Fit to memra: the *drafting* half needs a new drafter architecture (parallel
mask-token backbone — a bigger departure from the NextN chain than EAGLE); but the
*scheduling* half — *confidence-scheduled verify length per request under load* — is
engine work we can adopt independently, and it points at exactly our measured
failure (verify waste under concurrency: deeper K stretches the PP-2 bubble AND
acceptance-doomed columns burn the T=16 pool on single card). Blocker: the full
method is a big training+architecture bet; the schedule-only import needs a
confidence signal our current head doesn't emit (draft-head top-prob is the cheap
proxy; EAGLE-2 uses exactly that).

**8c. Head fine-tuning on own-gen data + TTT (the hqmtp direction).** In-house
receipts: the 212M StudentSV drafter hit 85.5% of teacher chain acceptance at ~13%
draft FLOPs, own-gen data beat generic by 17 points, and chain distillation (soft CE
over the full draft vocab against teacher logits, then CE reinforce) won every slot
once the subset-KL bug was fixed. The landscape's lesson stacks straight on top:
EAGLE-3's TTT (train the head feeding its own outputs, multi-step) is precisely the
cure for our measured K-depth collapse, and it is architecture-compatible with the
existing NextN chain (train the SAME chain recurrence we serve, unrolled K steps, on
own-gen + teacher logits). This is the highest-fit trained candidate: no feature-tap
surgery (unlike full EAGLE-3), no new drafter architecture (unlike DSpark), GGUF
export through the already-proven external-head route — subject to the same
trained-head attach gate as §1 (the converter-collapse mystery must be closed with an
in-engine acceptance-parity receipt before trusting any trained head).

**8d. Vocab-trim interaction with everything above.** The trim composes with every
head-based method and is orthogonal-to-favorable for the model-free ones:
EAGLE-3/SpecForge already standardize a 32k hot-token head + d2t (convergent with our
regime; our adaptive learned-escape loop remains a differentiator they lack);
Medusa/Hydra/ReDrafter heads all project to vocab and trim identically; a TTT/hqmtp
retrained head keeps the trimmed head by construction (train on the trimmed support,
escapes handled by the serve-time adaptive loop); PLD/suffix/lookahead proposals come
from real prior text so they are in-distribution for the trim and their misses are
exactly the escape events the adaptive trim learns from; DSpark's serial head outputs
through a vocab projection (trimmable) while its confidence head is vocab-free.
One law survives every combination: ranks derive from the exact serving model's own
generations, per requant (DRAFT-REGIME law 1).

---

## 9. Synthesis — ranked for PP-2 serving (one page)

**The PP-2 ground truth that orders everything.** At c=1, spec loses 18.81% with
verify at 95.13% of the round; the receipt-level bound is brutal: even a PERFECT K=1
drafter (acceptance 1.0, tokens/round 2.0) at the measured round rate yields
65.918 x (2.0/1.737) = **75.9 tok/s — still below plain's 81.2**. No drafter, head, or
tree fixes PP-2 by itself. The verify share is an idle-stage problem: while one stage
verifies, the other waits. Therefore:

- Makes the verify-share problem **BETTER**: stage-resident multi-session pipelining
  (fills the idle stage with another session's verify — the only structural fix;
  1.89x round-phase upper bound), batched spec prefill, batched cross-session verify
  (single card, capped 1.27–1.44x by the T=16 exactness tier),
  confidence-scheduled/load-aware verify length (stops spending stage time on doomed
  columns).
- Makes it **WORSE**: deeper sequential K (measured: −28.12% K=2, −38.92% K=3), tree
  verification on this shape (7.5–8.7 ms per added MoE column vs +0.05–0.11
  tokens/round for a depth-1 sibling — §6), beam drafting (ReDrafter), lookahead's
  window columns. Anything that widens or deepens the serial verify without filling
  the idle stage.

**Rank 1 — the specmech pipeline, enriched with confidence-scheduled verify
(landscape import: DSpark scheduling, vLLM dynamic spec length).** Expected gain:
the only path to spec-beats-plain at PP-2 c=2 (upper bound 1.89x round-phase; must
also compose batched spec prefill); on single card it is the prerequisite for any
c=2..4 crossover shift. Build bill: already written in
`research/specpp2-20260810/RESULTS.md` §mechanism-bill (resumable
draft/verify/commit phases, stage-ready queues over double-buffered PP boundary slots,
batched prefill composition; effort L, in development on `lane/specmech`) + a small
new piece: per-request K chosen from draft-confidence x load profile (scheduler-only,
rides the existing `choose_spec_k` seam). Gates: ppspec bit identity, run-spec K=1..8,
spec/plain byte identity, N=5 c=2 A/B, serve-stress. No training data, no GPUs beyond
the measuring rigs.

**Rank 2 — TTT/own-gen fine-tune of the house MTP chain (the hqmtp direction,
EAGLE-3's training recipe on our architecture).** Expected gain: attacks the measured
acceptance cliff (pos-2 38.8% → DeepSeek-V3-native-class 85–90% is the ceiling;
hqmtp already recovered slots 1–3 ABOVE its teacher) — worth ~nothing on PP-2 until
Rank 1 lands (perfect-K=1 bound above), then multiplicative with it; on single-card
boards it is the direct 2.30x-raiser (deeper healthy K + the T-amortization curve we
already measured). Build bill: own-gen corpus (frspec-owngen, owned rig), teacher
logits + chain-unrolled distillation (hqmtp/distill code exists; DeepSpec/SpecForge as
reference for TTT), ~1–2 weeks + low-thousands rented compute for a Step-scale head
(196B target: offline hidden-state route or a rented BF16 node), engine work small
(the chain recurrence already serves). HARD GATE first: close the trained-head attach
mystery (converter-collapse, DRAFT-REGIME law 2) with an in-engine acceptance-parity
receipt on a small model before spending the training budget.

**Rank 3 — suffix decoding (+ PLD floor) as a zero-training proposer for agentic
serving.** Expected gain: workload-dependent but the paper class is large exactly on
our declared serving shape (up to 5.3x SWE-Bench/AgenticSQL; 1.8–4.5x e2e in
Snowflake production); zero draft-training cost, zero drafter VRAM, no added load at
peak traffic (compatible with c≥4-plain, and the one spec class whose economics
*improve* with repeated-schema traffic). Build bill: engine proposer seam (external
token proposals into the existing verify/accept path), per-request + global suffix
tree in the server with eviction, K from match statistics (their max_spec_factor
heuristic), then board-protocol A/B on real dogfood traffic; composes with the
adaptive-trim escape loop. Blocker to state up front: on PP-2 it inherits the same
Rank-1 dependency as everything else; measure it single-card/agentic first.

Not ranked: full EAGLE-3 for Step (feature-tap surgery + trained-head gate — revisit
if Rank 2's TTT-on-NextN underdelivers, since DeepSpec/SpecForge make the training
side commodity); full DSpark drafter (biggest training bet, its scheduling half is
already absorbed into Rank 1); Medusa/Hydra/ReDrafter (dominated); LayerSkip/CLLMs
(retrain the target — violates the frozen-artifact posture); cross-vocab draft-model
for Step (tokenizer wall; TLI exists but greedy-only and acceptance-poor); tree verify
(single-card dense lever only — keep the shelf design for the 26B, do not spend it on
PP-2).

## Append-only addenda (no rank changes)

These notes extend the survey after §9; they do not reorder or replace the existing ranks.

### MoESD — target efficiency (Rank 1/systemic rider)

[MoESD](https://arxiv.org/abs/2505.19645) (NeurIPS 2025) defines target efficiency
`T_T(B,1)/T_T(B,gamma)`: batch-amortized verify can make sparse-MoE speculation attractive,
with up to 2.29x on Qwen2-57B-A14B. **Transfer:** keep this with Rank 1's systemic case
and add the planned Step instrumentation rider at c=1..32 (target time and expert-union
coverage). **Conflict/guardrail:** once plain decode activates most experts, verify columns
may be near-free; measure that on Step rather than importing the result into the PP-2 verdict.

### AcceptMoE — verifier-side expert eligibility

[AcceptMoE](https://arxiv.org/abs/2608.02989) reports 2.06x over EAGLE-3 under physical
offload with H2D reduced 73.6–77.1%, using commitment-weighted, residency-aware eligibility.
**Transfer:** residency-conditioned prefetch ordering only; it is a spill scheduler idea.
**Conflict:** eligibility changes the target distribution, so it violates memra's lossless
doctrine; do not use it to mask experts or alter verifier routing.

### SpecMoE — reduced-set drafting

[SpecMoE](https://arxiv.org/abs/2604.10152) uses zero-training self-assisted speculation
under a reduced expert set; full-bank target verification is the lossless-safe boundary.
**Transfer:** Rank-2/zero-training candidate for a Hy3 A/B with reduced-set drafting and
full-bank verification. **Conflict:** reduced-set verification is not lossless; no scored
arm or runtime default may silently execute it.

### SpecPrefetch — async expert prefetch

[SpecPrefetch](https://arxiv.org/abs/2607.24787) uses a tiny adapter to rank next-layer
experts for async H2D while the frozen native router selects executed experts, so errors
tax bandwidth rather than logits. **Transfer:** spill-design alternative for
residency-conditioned prefetch and window-aware scheduling. **Conflict:** predictor output
is never routing authority; no scored path may change expert selection.

### MoE-Prefill / AsyncEP — whole-layer expert-weight streaming

[MoE-Prefill](https://arxiv.org/abs/2605.02960) (arXiv 2605.02960v2,
CoRR/cs.LG preprint; no accepted venue stated) targets **prefill-only** workloads with
large batches or long contexts and a one-token output. AsyncEP replicates attention while
each of `N` GPUs owns a `1/N` expert shard; during layer L compute, NVLink AllGather
assembles the **complete** layer L+1 expert bank on every GPU. That replaces activation
AllToAll and its routing-imbalance barrier with weight movement that can be hidden under
compute. Its CPU-hybrid variant overlaps immediate-next-layer D2D assembly with PCIe H2D
of farther-layer shards; the governing gate admits a batch only when layer compute covers
the slower transfer channel.

**Reported and exactness posture.** Across its 8-GPU A100/H100/H200 cells on
Qwen3-235B-A22B, the paper reports **1.35-1.37x** throughput over the strongest distributed
baseline in each cell on aggregated real-world prefill-only workloads and up to **1.59x**
on long-context synthetic inputs, at 29.8-36.2% per-GPU MFU. It does not report an
interactive autoregressive-decode win.
AsyncEP moves unchanged target weights and removes no experts, so it is exactness-compatible
in principle; memra would still require its own same-prompt byte gates because a scheduling
equivalence claim is not a local numerical proof.

**Memra composition and verdict.** AsyncEP proper assumes multi-GPU expert parallelism,
NVLink AllGather, and enough prefill FLOPs to hide a **whole next-layer bank**. Memra's
server worker schedules request-level prefill (`crates/memra-server/src/worker.rs:4033-4207`)
but contains no expert-prefetch hook; that live seam is engine-side. It has no activation
AllToAll to remove: `MEMRA_SPILL_IO=worker` uses a bounded pinned-buffer CPU pool, reserves
capacity for demand misses, and leaves H2D/cache publication on the CUDA owner thread
(`crates/memra-engine/src/spill_pread.rs:1-5,39-67,325-384,567-650`).
After native routing, it submits only the selected gate/up/down blocks and may promote them
at the synchronized host-routing boundary (`crates/memra-engine/src/hybrid_forward.rs:3617-3652`);
grouped-prefill lookahead likewise submits disk-backed selected experts only
(`crates/memra-engine/src/hybrid_forward.rs:5676-5692`,
`crates/memra-engine/src/moe_cache.rs:1241-1277`). Streaming every expert would discard
the sparse single-card spill economy. Verdict: **no direct graft and no decode default**.
Import the measured compute-vs-transfer admission rule for long/grouped prefill experiments,
and revisit the full method only for a future explicitly gated multi-GPU EP backend.

### DBLast — accepted-length training objective

[DBLast](https://arxiv.org/abs/2608.05448) trains for expected verified length rather than
perplexity alone and uses cross-position dependence for stochastic decoding; independent
blocks degrade as temperature rises. **Transfer:** Rank-2 training-objective note, already
binding on the DSpark plan (`research/dspark-plan-20260811/PLAN.md`), for own-gen MTP
training. **Conflict:** this is training evidence, not serving proof; preserve cross-position
dependence at temp>0 and the frozen-artifact doctrine.

### SliceMoE — bit-slice residency

[SliceMoE](https://arxiv.org/abs/2512.12990) proposes bit-sliced expert caching, mixed-precision
slices, and predictive cache warmup to increase effective residency under miss-rate limits.
**Transfer:** per-expert-quant note for a post-freeze residency/cache study; slice-level
reads could complement spill without changing router decisions. **Conflict:** post-freeze
experimental only; do not turn slices into a GGUF format pivot or alter expert bytes/logits
in scored artifacts.

### Windowed-MTP — drafter-only attention window

[Windowed-MTP](https://arxiv.org/abs/2607.21535) windows the **DRAFTER's attention only** and
leaves target verification attention untouched. Verify that the target path is untouched: that
is the lossless-safe boundary because the full-attention target still decides every accepted
token. Keep unread draft KV in a ring. This is directly relevant to the DSpark drafter KV cost;
see `research/dspark2-spec-20260811/SPEC.md`.

## 10. New web-research addenda — SOTA fold (2026-08-11)

### ASD — regret-budgeted approximate verify

[ASD](https://arxiv.org/abs/2608.03447) (arXiv 2608.03447, Hermes fingerprint
`9de228dce2ae2faf`) replaces strict first-mismatch truncation with budgeted longest-prefix
selection: a local target-logit regret gate may admit selected mismatches, bounded by a
per-block exception cap and a persistent request-level regret budget, after which the
contiguous target-greedy suffix is reused without another target forward pass. It is
training-free and reduces exactly to strict greedy verification when the regret budget is
zero.

**Reported and exactness posture.** ASD reports **+3.05–15.26%** fixed-workload throughput
over matched strict verification, averaging **+7.78%** across seven Qwen3-14B + DSpark-14B
tasks; DeepSeek-V4-Flash + DSpark reports roughly **+10–16%** verifier-side acceptance on
GSM8K and MATH-500 in an FP4-to-FP8 setting. Any nonzero budget is **approximate** because
selected target mismatches change the decoding trajectory; only budget=0 is lossless strict
verification.

**Memra composition and verdict.** After `lane/cx-dspark2` lands, the natural seam is the
`crates/memra-engine/src/spec.rs` verify policy, with a `MEMRA_*` regret-budget flag family
for the local gate, per-block cap, and request budget. This is a **default-OFF, explicitly
blocked serving door for non-exact traffic**, not a scored arm or runtime default: strict
`run-spec` K=1..8 self-consistency remains the default and the gate.

### DraftExpert — fixed-footprint resident draft expert

[DraftExpert](https://arxiv.org/abs/2607.24434) (arXiv 2607.24434, Hermes fingerprint
`1ecd30b9cb632563`) self-distills one lightweight accelerator-resident draft expert per
layer, then uses a fixed-footprint shared + top-1 + draft-expert drafter with
confidence-expansion truncation and target-expert prefetch. The target still performs the
final verification, so the drafter improves the offload schedule without making the draft
expert a routing or verification authority.

**Reported and exactness posture.** On DeepSeek-V2-Lite and Moonlight-16B-A3B across
CPU-GPU and Flash-NPU offload, it reports **1.45x** average decode throughput, **84–87%**
draft acceptance, and **86–88%** prefetch hit rates. The proposed serving composition is
**lossless relative to the frozen target** when final tokens remain target-verified; the
resident draft expert is an exactness-preserving drafter aid, not an approximate verifier.

**Memra composition and verdict.** After `lane/cx-dspark2`, this maps to one fixed-footprint
resident draft expert per layer plus expansion truncation in the spill/spec path, using
`crates/memra-engine/src/moe_cache.rs` residency metadata and `MEMRA_SPILL_IO=worker`; it
should prefetch the target expert set rather than predict or replace full next-layer banks.
Posture is **default-OFF spill+spec research**: a promising composition with worker
residency, preserving strict target verification, but not a runtime default until measured.

### EcoSpec — marginal expert-cost draft-tree selection

[Less Experts, Faster Decoding: Cost-Aware Speculative Decoding for
Mixture-of-Experts](https://arxiv.org/abs/2607.12696) (EcoSpec; arXiv 2607.12696v1,
cs.CL/cs.AI/cs.DC preprint in ICML 2026 submission format; no acceptance stated) changes
which **draft-tree nodes** reach the target verifier. A small decoder-only expert predictor
estimates the target's per-layer routed experts for each root-to-node trajectory. EcoSpec
then greedily ranks nodes by cumulative draft-path probability divided by the marginal
number of predicted `(layer, expert)` pairs not already covered by its dynamic buffer.
The full target still verifies the selected tree, so selection errors cost opportunity or
weight movement rather than changing the accepted-token distribution.

**Reported and exactness posture.** This paper is **not training-free**. It trains the
predictor offline on frozen-target router traces (100 epochs; reported expert-prediction
accuracy 80%, 82%, and 93% across its three model families), while leaving the target and
existing drafter unchanged. On eight H200s at batch 1 and four selected draft nodes, EcoSpec
moves Qwen+EAGLE-3 from **1.22x to 1.36x** over autoregressive decode while reducing mean
unique experts activated per MoE layer from 23.7 to 20.5; GPT-OSS moves 1.14x to 1.31x and
DeepSeek 1.10x to 1.15x. Predictor overhead is about 4 ms in the Qwen setup. Gains shrink
with batch size; at batch 8 both research implementations are slower than autoregressive
decode. Strict target verification makes the method lossless-compatible, not the trained
predictor itself.

**Memra composition and verdict.** Memra's present MTP paths draft a **linear K-token
chain**, then run one batched target verify and longest-prefix accept
(`crates/memra-engine/src/spec.rs:1-7`, `crates/memra-engine/src/gemma_spec.rs:631-638`),
so the §6 tree mechanism is a prerequisite. After that, EcoSpec composes cleanly with
DraftExpert's confidence expansion and with worker prefetch, provided its output remains a
prefetch/candidate-selection hint only. The paper's unit-cost `(layer, expert)` count is not
authoritative for Hy3 `mix_quant`: memra keys residency by **three projection blocks** per
logical expert (`crates/memra-engine/src/moe_cache.rs:24-39`), and mixed layouts make each
projection's `qtype`, `row_bytes`, and `len` authoritative
(`crates/memra-engine/src/model.rs:1283-1304,2200-2229`). A valid local score must therefore
mask pruned ids, charge actual nonresident gate/up/down bytes and disk extents, and treat
resident or pending blocks as already covered (`crates/memra-engine/src/moe_cache.rs:1241-1277,1308-1325`;
`crates/memra-engine/src/hybrid_forward.rs:5620-5653,5665-5692`). Verdict: **promising
post-tree, default-OFF spill+spec research**, not a current lane and not training-free;
never let the predictor select target routing or bypass strict verification.

### OasisKV — spec-draft lookahead as a KV-prefetch oracle

[OasisKV](https://arxiv.org/abs/2608.08097) (arXiv 2608.08097, Hermes fingerprint
`5401fbbbeb696bce`) keeps only the KV entries judged relevant in HBM, predicts future-important
KV blocks from tokens already proposed by a speculative drafter, and stages those blocks from
host or remote memory before the next decode step. The mechanism turns draft lookahead into a
memory-tier scheduling signal rather than another target-model decision.

**Reported and exactness posture.** The paper reports **1.69x** over dense vLLM at a
2,048-token KV budget, about **2x** dense throughput under prefill-decode disaggregation,
and up to **2.1x** on multi-GPU long-context serving; its sparse-attention configuration is
approximate, staying within **0.7 points** of full-attention accuracy (the 1.69x result is
reported at **0.1 points** of accuracy loss). For memra, the proposed adaptation is
**lossless only if it is prefetch-only**: retain the full exact KV and target attention, use
draft tokens solely to schedule reads, and make no distribution or model-byte change.

**Memra composition and verdict.** `crates/memra-engine/src/spec.rs` already owns draft-token
state, so the concrete arm is a next-step KV prefetch/stage queue keyed by those tokens and
reusing the existing host/remote spill plumbing. Verdict: **lossless prefetch-only research
arm**, with the paper's sparse-KV attention deliberately out of scope; measure overlap and
PCIe cost against the worker baseline before treating the oracle as useful.

### ScoutAttention — layer-ahead residual oracle for CPU/GPU KV co-attention

[ScoutAttention](https://arxiv.org/abs/2603.27138) (arXiv 2603.27138, DAC'26; Hermes
fingerprint `14da7609`) is a zero-training, block-sparse KV-offload design. While the GPU runs
layer `i`, it forms `Q_pred^(i+1) = W_Q^(i+1) X^i` from the prior residual stream, ranks the next
layer's KV-block digests, and launches CPU attention for predicted top-k blocks absent from HBM;
the corresponding GPU and CPU attention partials are FlashAttention-merged on the GPU. Predicted
and real queries have 0.93-0.97 cosine similarity across the five tested model families, and an
asynchronous periodic recall refreshes the important-block set as it drifts.

**Reported and exactness posture.** In decode-only P/D-disaggregated Qwen3-14B experiments with
a 2,048-token sparse budget and 32-token blocks, the paper reports **5.1x** FullKV throughput at
64k and up to **2.1x** over HGCA/InfiniGen; GPU idle falls from HGCA's 57% and InfiniGen's 61% to
6%. On Qwen3-8B LongBench, however, accuracy falls **2.5%** at budget 1,024 and **2.1%** at 2,048:
predicted queries and selected sparse blocks participate in the result. ScoutAttention is therefore
**LOSSY**. Under `CLAUDE.md`'s Correctness discipline (`kernel-check`, `run-gen` argmax, and
`run-spec` K=1..8), it is research-arm-only, default-OFF, and must never enter the shipped serving
path.

**Memra composition and verdict.** Memra has no paged/offloaded KV seam today: `memra-engine`
re-exports `memra-kv` (`crates/memra-engine/src/lib.rs:29-33`), whose `KvLayer` is a GPU-resident
token-linear K/V plane with only an optional Step SWA ring
(`crates/memra-kv/src/lib.rs:213-243,285-362,416-547`). `MEMRA_SPILL_IO`'s bounded pinned-buffer
worker and owner-thread publication stage **expert-weight extents**, not KV blocks
(`crates/memra-engine/src/spill_pread.rs:1-5,39-73,325-402`;
`crates/memra-engine/src/hybrid_forward.rs:3617-3652,3952-3977,5657-5692`), and the server worker
only snapshots those MoE spill/cache counters (`crates/memra-server/src/worker.rs:4698-4714`). The
residual-oracle idea is familiar: memra's expert predictor applies future routers to an earlier
MoE input (`crates/memra-engine/src/cpu_experts.rs:771-776,937-1005`), but its measured 2.6-5 ms
lead could not beat 2-4 MB reads and speculative traffic taxed the bus
(`research/moe/expert-prefetch-prediction-pilot.md:38-78`). Scout is not refuted by that result—its
32-token KV blocks are smaller and near-data CPU attention avoids most H2D—but a build starts with
a new block-addressed host/HBM KV store plus CPU-attention and numerical-merge seams beside
`memra-kv`, not by sending KV through the expert cache.

Relative to OasisKV, both methods predict future-important KV blocks, but OasisKV uses draft-token
lookahead across decode steps as a prefetch oracle while Scout uses residual similarity across
layers and computes approximate CPU attention. Making Scout prefetch-only would preserve full
target attention, but then it mostly overlaps OasisKV/InfiniGen and gives up the headline
PCIe-avoidance mechanism. Speculative KV Coding is distinct and orthogonal: it losslessly encodes
the actual K/V values of verified tokens; Scout neither predicts those values nor compresses them,
so a codec can sit below Scout without making its sparse attention exact.

Capacity does not create a near-term opening. Step-3.7-Flash's 105.0 GB artifact cannot load on one
96 GB PRO 6000 at any context; on the 2x serve pair its flat cache costs 83,520 B/token—10.20 GiB
at 128k and 20.391 GiB at 262k—and the measured first-defer point is two simultaneous 262k
sessions. The exact opt-in SWA ring already cuts that component to 5.702 GiB and raises the measured
point to 12 (`docs/PERFORMANCE.md:524-567`; `research/ringval-20260810/RESULTS.md`). The single-card
Qwen3.5-122B-A10B bring-up is gentler: its 60.2 GB IQ4 artifact and 12 full-attention hybrid layers
cost 10.9 KB/token, or 1.46 GB per 128k session; eight 8k sessions were measured, while the
projected roughly 21x128k envelope still lacks its capacity ladder
(`research/122b-bringup-20260806/VERDICT.md`; `research/model-96gb-20260806/ASSESSMENT.md:148-156`).
The paper's model evidence stops at Qwen3 and Gemma3—not memra's Qwen3.5 hybrid, Step, or Gemma4—so
local oracle precision would be a prerequisite even for research. Thus memra reaches Scout's 64k+
context regime, but not a single-session KV-over-VRAM cliff at its current serving caps; pressure
is high-context concurrency, where exactness-preserving rings/prefetch and byte-gated codecs rank
first. Verdict: **noted for completeness, not a near-term lane**; the original lossy mechanism
remains a default-OFF research door only.

### SparDA follow-up — trained Forecast versus Scout's zero-training oracle

[SparDA](https://arxiv.org/abs/2606.04511) (arXiv 2606.04511v1, June 2026;
[author code](https://github.com/NVlabs/SparDA)) moves the same one-layer-ahead decision into a
new per-layer Forecast projection. `F_l` predicts layer `l+1`'s KV-block set while layer `l`
runs; the runtime keeps compressed-key indices on GPU and uses a persistent UVA prefetch kernel
to stage the selected CPU-resident blocks. Decoupling selection from the real attention query also
shrinks the indexer to one Forecast head per GQA group and removes its selector softmax. Each 8B
implementation adds 33.5M trained parameters (0.41%): only these projections are KL-trained
against the original selector, but freezing the backbone does **not** preserve the artifact bytes.

**Reported and exactness posture.** On two sparse-pretrained 8B backbones, the paper reports up to
**1.25x** prefill and **1.7x** decode speedup over the CPU-offloaded sparse baseline. Its **5.3x**
decode-throughput headline is a capacity result versus non-offload sparse: offload permits a much
larger batch after that baseline OOMs, not a same-batch Forecast-only gain. SparDA matches or
slightly improves the existing sparse baseline on the reported aggregate evaluations, but it
inherits that backbone's sparse-attention accuracy limit and its Forecast decides which blocks
participate. As published, it is neither a frozen-byte backend nor full-attention exact; under the
backend doctrine it is **research-arm-only**, even before memra's output gates.

**Scout comparison and memra verdict.** Both designs expose next-layer block addresses early enough
to overlap work. Scout reuses the frozen next-layer query weights on the prior residual,
`Q_pred^(l+1) = W_Q^(l+1) X^l`, with no trained projection, then lets that approximate query drive
CPU sparse-attention partials. SparDA spends 0.41% new model parameters to predict selection more
directly and cheaply, prefetches the chosen bytes, and leaves sparse attention to the real next-layer
query. Thus Scout preserves model bytes but is lossy through approximate co-attention; SparDA changes
model bytes and is approximate through learned sparse selection. The portable lesson is narrower:
make future memory addresses available as a scheduling signal, then price forecast cost, lead time,
pinned-host transfer, CTA interference, and fallback stalls as one storage-to-compute pipeline. A
memra-compatible path must keep the full exact KV available and use either oracle only to order
prefetch; misses must fall back before exact attention. Verdict: **SparDA is a useful trained
research comparator for zero-training Scout/InfiniGen-style prefetch oracles, not a shippable
artifact and not a reason to promote the present Scout verdict into a lane**. The first comparison
still waits on the block-addressed KV tier described above.

### XShare — batch-joint expert cover (cost-model transfer only)

[XShare](https://arxiv.org/abs/2602.07265) (arXiv 2602.07265, February 2026) replaces
independent per-token top-k selection with a per-layer **batch-joint expert cover**: choose a
bounded set that maximizes summed gating score over the whole live batch, then rerank each token's
top-k inside that set. The proxy is modular, so sorted/greedy selection is optimal for its fixed
per-layer budget; speculative verification first covers correlated tokens per request, then unions
and refines those covers across the batch. It requires no retraining. The paper reports up to 30%
fewer activated experts under standard batching, a 3x peak-GPU-load reduction under expert
parallelism, and up to 14% speculative-decode throughput improvement.

**Memra transfer.** Run the cover objective on recorded native router scores as an analysis-only
optimistic expert-cover curve at a chosen captured gating mass—and therefore an upper bound on
expert-union amortization as verify batch `B` rises. Put that curve beside the MoESD
GO/NO-GO harness question. The same batch-joint accounting can price grouped MoE dispatch and
capacity, charging the live union by actual expert bytes/residency/placement rather than summing
per-request top-k counts. This is a cost model, not serving authority.

**Doctrine fence.** XShare changes **which target experts run**: frozen weights and no retraining do
not make rerouting lossless, and accuracy within a reported tolerance is not byte identity. Under
`CLAUDE.md`'s frozen-artifact/backend rule, XShare itself is **research-arm-only** and MUST NOT alter
the verifier's expert set. Only the batch-joint cost model transfers; strict verification always
executes the native router selection.

### WiSP / MV-WSA — PCIe ceiling and marginal-value residency

[WiSP](https://arxiv.org/abs/2606.21868) (arXiv 2606.21868, Hermes fingerprint
`45248f7c4c694a98`) treats routed experts and the KV cache as competing working sets: a
routing-aware pager keeps reused experts resident, while MV-WSA allocates VRAM by equalizing
marginal latency benefit per byte subject to a KV admission floor. Its counter-evidence is
that predicted-routing prefetch helps little for single-stream decode when PCIe bandwidth,
not prediction accuracy, is the binding constraint.

**Reported and exactness posture.** WiSP reaches up to **1.95x** decode throughput over
static offload at the same memory budget when the model does not fit; fixed splits are about
**20%** worse than a per-workflow oracle in trace-driven simulation, and the online controller
adds a further **1.20x** without changing model outputs. This is **lossless/byte-identical**
serving evidence, unlike an approximate verifier or sparse-attention path.

**Memra composition and verdict.** The concrete memra arm is an expert-vs-KV residency
allocation controller in `crates/memra-engine/src/moe_cache.rs`, coupled to the KV budget;
measure the single-stream PCIe ceiling before adding a learned prefetch oracle. Verdict:
**CAUTION for the Hy3 spill lane**—the cheaper lossless allocation arm should precede learned
prefetch, and any policy ranking must use the pinned replay semantics below.

**Replay-semantics cross-reference.** MoE-cache eval ([arXiv 2608.07911](https://arxiv.org/abs/2608.07911), Hermes fingerprint `9280ff620090bb80`) warns to pin fused-event replay and matched-pair probe diversity before ranking spill policies: inconsistent replay inflates recency policies **27–29%** and can invert rankings.

### Speculative KV Coding — ~4x lossless KV compression via a residual predictor

Signal (zlorg digest, X discussion Aug 2026, canonical_key `x:2084610329468625196`): train
a tiny predictor to forecast KV values and store only the residuals; reported ~4x lossless
KV compression, **2.4–3.9x** effective on Qwen3 by size. Attacks the memory-bandwidth wall
that binds long-context and agentic decode — the same wall Speculative KV Coding names is the
one memra's decode is bandwidth-bound against (`research/q27-decode-bw-20260801`), so a KV
footprint cut is a decode-throughput lever, not just a capacity one.

**Exactness posture.** "Lossless" is a residual-reconstruction claim that MUST clear memra's
byte-identity gate (`run-gen` argmax MATCH + `run-spec` K=1..8) before it is more than a
signpost — a predictor that is lossless in the paper's setup is not automatically bit-exact
under memra's kernels/quant. Distinct from spec decode: this compresses the KV of already-
verified tokens; it composes with, does not replace, the MTP draft regime.

**Memra composition and verdict.** Concrete arm: a KV-block residual codec at the
`crates/memra-engine/src/cache` KV-store boundary, default-OFF behind a flag, measured H2D/
read cost vs the bandwidth saved (the storage-to-compute pipeline rule — a smaller footprint
that costs more per-access is a regression). Verdict: **default-OFF research door, gated on
byte-identity first**; rank against the cheaper KV formats already in tree before adopting a
learned predictor. Not yet scoped to a lane.

### Prefix caching / RadixAttention — memra posture cross-check (already shipped)

Signal (zlorg guide `learning/guides/2026-08-11-prefix-caching-and-radix-attention.md`, in
the learning store, not this repo). Cross-checked against the memra serving surface — this is
a **confirmation of shipped work + two net-new doors**, not a gap:

- **Shipped.** memra's prefix cache is live: `PrefixCache` keyed on `PoolKey = (model,
  cache_salt namespace)`, per-namespace entry pool walked by longest-common-prefix
  (`best_lcp`), with the token-weighted `cache_hit_token_ratio` receipt and the full counter
  set (`prefix_hits/misses/inserts/evictions/hit_tokens`, `lcp_histogram`) in `/metrics`
  (`crates/memra-server/src/worker.rs`). This is the guide's #1 (biggest agentic win) and #5
  (measure hit-rate/reused-tokens/evictions) already in place.
- **Exact-prefix-only (#2)** holds for memra too — LCP reuse only, non-prefix sharing
  unsolved here as everywhere.
- **Structure (#3):** memra is an **LCP walk over a per-namespace entry pool**, closer to
  SGLang's token-level radix match than to vLLM's 16-token block-hash chain — relevant for the
  branching/self-consistency shapes the survey's ASD/OasisKV arms target.
- **Multi-tenant salting (#4):** memra **already shipped** vLLM's `cache_salt` design as
  PC-ISO (`worker.rs:233`) — this is exactly the isolation that blocks the `38bca65c` prefix
  oracle triaged in the sec4 review. Independent external confirmation of a mitigation already
  in tree.
- **Net-new doors to watch:** **HiRadix** (hierarchical KV storage — a tiered spill target
  the Hy3 spill lane's storage-to-compute pipeline could feed) and **CacheBlend** (non-prefix
  KV reuse — the one unsolved item above; approximate, so it would enter as a default-OFF
  door under the same byte-identity gate as ASD). Neither is scoped to a lane yet.

### FASER — per-request depth and intra-verify pruning

[FASER](https://arxiv.org/abs/2604.20503) (arXiv 2604.20503, Hermes fingerprint
`77f86351`) makes speculative length request-specific inside a continuous batch, prunes a
predicted rejected suffix while target verification is still in flight, and splits verification
into frontier chunks that overlap with drafting through spatial multiplexing. Its vLLM prototype
reports up to **53% higher throughput** and **1.92x lower latency** versus SOTA systems. This is a
different cut from the §7/D-cut synthesis: D-cut prunes across requests by confidence, whereas
FASER makes K request-local and also removes rejected-token work inside each target verify.

**Exactness and transfer posture.** Request-local K and frontier scheduling can leave the frozen
target and ordinary accept/recovery rule authoritative. Intra-verify early exit is not assumed
free: the predictor can cut a token that full verification would have accepted, so a memra arm
must resolve any cut through the exact target recovery path and clear `run-spec` self-consistency;
otherwise that component remains research-only. The paper's frontier overlap preserves standard
speculative-decoding semantics, but that does not by itself prove memra byte identity.

**Memra composition and re-gate flag.** This sits directly beside `cx-dualpp2`'s wave-aware dual
scheduler. **FLAG for the dual-scheduler promotion re-gate: evaluate per-request K versus a fixed
wave-wide depth.** Memra's PP-2 speculative path is per-batch today, so this costs ragged
per-request draft/verify lengths, request-local accept and KV commit/rollback state, and scheduler
control; intra-verify compaction and frontier overlap add separate layer-level and resource-
partitioning work. Verdict: **candidate re-gate knob, not a free win or present runtime default**.

### Speculating Experts — activation-based expert prefetch, no reroute

[Speculating Experts](https://arxiv.org/abs/2603.19289) (arXiv 2603.19289, Hermes fingerprint
`02ef6141`) predicts next-layer MoE experts from currently computed internal activations and
prefetches their CPU-offloaded weights so transfer overlaps current-layer compute. The paper
reports up to **14% TPOT reduction** over on-demand expert loading. Its secondary speculative-
execution arm instead executes the predicted experts without re-fetching the native router's
actual selection; that preserves overlap by changing which experts run and is therefore lossy
under memra's frozen-router doctrine.

**SpecPrefetch contrast.** SpecPrefetch uses a tiny trained adapter to rank future experts;
Speculating Experts' base path derives its hint from the live internal representation and an
expert-conditioned activation summary. Both are eligible only as hit-rate-aware prefetch oracles:
the frozen native router remains execution authority, and a miss must demand-fetch its selected
experts. **No predictor or activation surrogate may reroute execution.**

**Memra composition and verdict.** A **lossless-only**, hit-rate-aware prefetch arm may feed the
spill worker/window alongside SpecPrefetch, charging wrong predictions as I/O and cache pressure.
Executing speculated experts without re-fetch is **research-arm-only** because it changes model
output; it is ineligible for a scored arm, serving default, or fallback around native routing.

### STEP / ST-MoE — adaptive spatio-temporal expert prefetch

[STEP](https://ieeexplore.ieee.org/document/11617726) and
[ST-MoE](https://arxiv.org/abs/2606.15453) (arXiv 2606.15453, Hermes fingerprint
`6fa40c8e`) exploit expert-demand correlation across adjacent MoE layers and consecutive decode
steps in offloaded inference. STEP adapts expert allocation layer by layer, while ST-MoE uses a
lightweight runtime predictor to stage offloaded experts ahead of use and overlap loading with
ongoing compute without changing native routing. This is closer to memra's Hy3 spill window plus
multi-token decode residency than either pure router-lookahead in Speculating Experts or
SpecPrefetch's trained ranking adapter: the transferable change is an adaptive prefetch schedule
rather than a different expert set.

**Doctrine fence and verdict.** Only the lossless schedule-only form is eligible: the frozen
router remains authoritative, every miss demand-fetches the selected expert, and no predictor may
reroute or skip execution. Verdict: **default-OFF spill research arm**, contrasting the current
fixed window with adaptive layer-wise allocation. Measure prefetch recall and miss-bytes on the
byte-identical artifact staged to the cloudbox `/scratch` path; the five public-eval arms and their
frozen routing remain unchanged.

### EVICT + JetSpec — verify-budget selection and parallel causal drafting

[EVICT](https://arxiv.org/abs/2605.00342) (arXiv 2605.00342) is a training-free,
hyperparameter-free, lossless controller that truncates a draft tree **before** target
verification, retaining its cost-effective prefix from fine-grained drafter signals plus an
offline profile of verification cost. It reports up to **2.35x** over autoregressive decoding and
an average **1.21x** over EAGLE-3 on MoE backbones, with SGLang compatibility. It is a lossless
cousin of D-cut: both remove low-value verify work, but EVICT chooses a prefix before target
execution. The frozen-router/no-reroute fence still applies to every retained node.

[JetSpec](https://arxiv.org/abs/2606.18394) (arXiv 2606.18394v3, Hermes fingerprint
`9f1f59fc`) instead attacks the drafter-quality ceiling: it trains a causal parallel draft head
over fused hidden states from the frozen target, preserving branch-wise causal conditioning while
producing a candidate tree in one forward pass. It reports up to **9.64x** on MATH-500 and includes
vLLM integration. This is a drafter swap, not a change to the target model or router, so it is
eligible as a **default-OFF drafter research arm** provided the frozen target still verifies every
accepted token. EVICT belongs on the verify-budget axis and JetSpec on the drafter-quality axis;
neither changes the existing survey ranking or authorizes target rerouting.

### Mistletoe — quality-neutral acceptance-collapse construction

[Mistletoe](https://arxiv.org/abs/2605.14005) (arXiv 2605.14005v2, Hermes fingerprint
`7c9cb254`) is the constructive attack for the acceptance-collapse class that memra's ADSD
operations detector watches. Sun et al. jointly optimize a drafter-target agreement-degradation
objective with semantic preservation, using null-space projection to suppress draft acceptance
while minimizing semantic drift. The result collapses acceptance length and throughput while
output quality and perplexity remain normal, so output-only health checks do not expose the attack.

**Threat-model transfer and verdict.** Mistletoe makes the acceptance signal itself the required
operational evidence, but it also sharpens the detector's evasion surface: slow drift or
dominant-tenant masking lines up with the ADSD self-contamination weakness already queued. Verdict:
**detection-only security posture**, with request-level acceptance evidence and manual tenant/lane
response; no model, router, serving-default, or survey-rank change follows from the attack.

### KARAT / PNM — KV-offload taxonomy, not expert spill

[Heterogeneous LLM Serving with General-Purpose Processing-Near-Memory](https://arxiv.org/abs/2608.03555)
(arXiv 2608.03555v1, Kimi receipt `d1bb80e4`, retrieved 2026-08-12) puts the full KV cache and
retrieval index beside programmable LPDDR compute, runs index scan/gather/attention there, and
keeps projections and MoE on GPUs. Its useful taxonomy separates capacity-only host/CXL offload,
where KV-side work still crosses the link, from fixed-function near-memory attention and KARAT's
general-purpose near-data execution. Memra's mmap/pread/worker + staged/SLRU paths are orthogonal:
they move selected **expert-weight** extents through pinned host buffers and H2D, not KV/index data
or attention compute. Transfer only the observability schema—name the object, storage tier,
compute site, link bytes, wait/overlap, and fallback separately—so expert spill counters are never
reported as KV offload. This is a spill-taxonomy addendum, **not a speculative-landscape rank**.

### FluxMoE — evict-immediately paging versus SLRU reuse

[FluxMoE](https://arxiv.org/abs/2604.02715) (arXiv 2604.02715v2, Kimi receipt `08161860`,
retrieved 2026-08-12) treats expert tensors as transient pages: materialize the current layer,
prefetch the next, evict after use, and let a budget controller retain more only when KV pressure
permits. The regime boundary is the result: Flux-style minimal residency wins when high-concurrency,
long-context decode is **KV-starved** and otherwise swaps KV; memra's projection-granular SLRU wins
when KV is not the capacity limiter and hot-expert reuse repays residency by avoiding repeated
reads/H2D. Neither policy is universal. Compare them under one memory budget using KV occupancy,
expert hit/miss bytes, H2D, stalls, and throughput together; preserve native routing and exact bytes.

### SS-MoE — reduced experts belong only in the draft

[SS-MoE](https://doi.org/10.1145/3774904.3792218) (Zheng, Xu, and Wang, WWW '26,
DOI 10.1145/3774904.3792218, Grok receipt `6233b5ef`, retrieved 2026-08-12) uses fewer routed
experts as a self-draft and an on-demand GPU expert cache; its public record distinguishes
accuracy-preserving conservative verification from a faster adaptive mode described as nearly
lossless. Memra's doctrine is narrower: a reduced-expert **DRAFT** is lossless-adjacent only when
the unchanged full-expert target, native router, and standard accept/recovery rule verify every
accepted token. Reduced-expert or confidence-bypassed **VERIFY** remains banned by the AcceptMoE
precedent; preserved benchmark accuracy is not target-distribution identity.

### RotaryQuant — role-based precision and the Hy3 shared-expert cross-check

[RotaryQuant](https://arxiv.org/abs/2608.08081) (arXiv 2608.08081v1, Grok queue receipt,
retrieved 2026-08-12) assigns 4-bit dense weights, 2-bit routed experts, and Q8_0 to the shared
expert, reporting shared-expert activation kurtosis `10.10` versus `0.41` for specialists; its
separate KV path uses rotation before 3-bit compression. This supports Hy3's structural decision
to keep its always-on shared expert outside usage-ranked routed-expert tiers, but it does **not**
transfer the measured kurtosis or bit-width. The frozen Hy3 five-arm study varies routed experts
while holding shared-expert tensors byte-identical across arms. Any follow-up must measure Hy3
shared-versus-routed activation kurtosis on the frozen non-public calibration trace and preregister
a separate arm before public scores; it cannot revise the locked five arms or their rankings.

# SOTA sweep — 2026-08-09: latest discoveries for memra

Scope per tasking: what is NEW or DEEPER than `research/sota-harvest-20260808/HARVEST.md`
(read as diff base from branch `lane/sota-harvest`), weighted to the last ~6 months.
In-flight lanes NOT re-ranked here: cross-request batched draft/verify (HARVEST #1),
dynamic microchunks (#3), in-batch prefix dedup (#4). Raw findings with all links and
per-item transfer notes: `NOTES.md` in this directory. Tags: [shipped] = merged in a
production engine; [reproduced] = independent third-party numbers; [paper-only] = no
independent reproduction found; [issue-receipt] = GitHub issue with a reproducible
protocol. No number below is invented; each is quoted from its cited source.

## Executive summary — ranked top 8 (impact on our floors × feasibility)

1. **Close the ~9k generation-degradation bug with a three-probe program** — the sweep
   found the bug's three candidate mechanisms already documented elsewhere: (a) a
   model-level repetition collapse in the exact hybrid-SWA model generation we serve
   (Gemma-4 dense AND MoE, cross-backend, repeat_penalty-immune — gemma#622 +
   ollama#15502 [issue-receipt] with a 10-seed protocol we can rerun as-is); (b)
   reduced-precision attention accumulation error that GROWS with KV length (vLLM/FA3:
   128k NIAH 91%→13%, fixed by two-level FP32 accumulation [shipped, reproduced]) — a
   length-indexed onset is its exact signature; (c) SWA sink-eviction/window mechanics
   (StreamingLLM law + SWAA recipes [paper-only, code released]). Probes: rerun #622's
   protocol on our stack vs a reference stack; diff fast path vs `MEMRA_FAST=0` oracle at
   depth; audit accumulator widths in the FA twins; verify sinks never leave the ring.
   Days-class, uses the existing repro. Highest rank because it is a live correctness bug
   on a serving SKU — it gates the serve-ready axis, not just a board number.
2. **Session-KV TTL pinning across tool-call pauses** (Continuum, arXiv 2511.02230
   [paper-only]; CacheWise 2606.16824; SMetric 2607.08565; keepalive economics
   2607.19214). Agent sessions pause seconds for tools; engines that release KV repay
   full prefill on re-entry. Pin KV with a TTL computed from measured tool-call-duration
   histogram vs exactly-known reload cost; evict on expiry. Their claim: >8x avg job
   completion time on SWE-Bench/BFCL/OpenHands (their harness). Ours to measure as
   TTFT×hit-rate on re-entry via the live cache-meter. Days-class on existing session
   affinity + admission machinery. This is our exact customer shape (6.1's law: agent
   serving is decode + KV-residency, 84.6–99.5% cache-hit when caching works).
3. **Oilbird semantic re-key as v2 of the in-flight suffix-decoding lane** (arXiv
   2608.03839 [paper-only]). On tool-calling traffic ~half of what exact-suffix matching
   misses is already in the pool but unreachable lexically; re-keying the same pool by
   verifier hidden states lifts accepted length +24–29% at matched budget (4.4x AR on
   API-Bank vs 2.0x EAGLE-3). Rides the suffix lane's draft-slot injection; ~1 week after
   that lane lands. Strongest per-line-of-code spec item for agent traffic.
4. **Reopen fp8-K with per-head + calibrated scales, share the lane with NVFP4-V**
   (vLLM FP8-KV state blog 2026-04-22 [shipped, reproduced]). Our fp8-K flip-block
   (74%→20.5% acceptance) predates the upstream fixes; their uncalibrated-per-tensor
   worst case now recovers 94–99% accuracy to 256k–1M ctx, with per-head scales and
   calibration shipped for the hard cases, plus two laws we inherit free: skip small-SWA
   layers (window 512 is big enough to quantize, 128 is not), and watch head_dim=256
   prefill. ITL slope 54% of BF16 and +14.9% throughput at c=8 is the size of the prize
   on KV-capped SKUs. 1–2 weeks including the acceptance battery (which stays the gate).
5. **DSpark is now shipped serving infrastructure — lift its ragged-verify design into
   the in-flight batched-spec lane** (SGLang v0.5.16, PR#30261 [shipped]; 383.7 tok/s at
   accept ~5, DeepSeek-V4-Pro TP8 B300 bs=1). `SGLANG_RAGGED_VERIFY_MODE=compact` is a
   production answer to the exact ragged-batch problem HARVEST #1 must solve; the
   confidence-scheduled variable verify length is the third shipped confirmation of the
   K-policy design (with SpecDec++/Not-a-Bandit; Nightjar 2512.22420 adds spec-restart
   cost). Read-and-lift, days; de-risks two in-flight lanes rather than opening a new one.
6. **llama.cpp merged EAGLE-3 — drafter supply + GGUF plumbing reference + reset
   same-silicon bar** (PR ggml-org/llama.cpp#18039, ~2026-06 [shipped]; >2x reasoning-on,
   >3x reasoning-off claimed in-thread, Gemma-4 EAGLE heads included; Q4_K_M "still looks
   good"). Three uses: trained EAGLE-3 heads for GGUF-served families now exist upstream
   (drafter-supply problem partially externalized — hyperscaler also published P-EAGLE
   checkpoints); their hidden-state routing through a GGUF graph is the reference for any
   hidden-state-conditioned drafter in our loader (also needed by #3); and our spec cells
   should be re-compared against a llama.cpp that now has EAGLE, not just draft-model
   spec, whenever a floor claim references upstream. Audit-class now, bring-up later.
7. **Two small shipped scheduler/kernel liftables**: (a) SGLang v0.5.13 moved prefill
   input H2D transfer ONTO the forward stream (overlap next chunk's token upload with
   current forward) — one-line-class for our tick; (b) llama.cpp/Ollama shipped packed
   gate/up expert projection (+4–9% decode on M5 Max) plus a fix for
   differently-quantized experts — the packing idea applies to our expert walkers, and
   their mixed-quant fix is a days-class audit against our mixed-layout dispatch contract
   before the five-arm study leans on it. Both days-class, both [shipped] upstream.
8. **Intra-GPU SM-level prefill/decode disaggregation — file, gate on hybridity
   metering** (Bullet, ASPLOS 2026; DuetServe 2511.04791; Semi-PD, Nexus, MuxWise — five
   systems in 12 months). Spatial SM partitioning between concurrent prefill/decode
   kernels, ~1.26x throughput claimed, single-GPU single-node = our shape. Competes with
   our three-phase tick + in-flight microchunk lane; open only when tick metering shows
   decode-tail stalls that budget tuning can't fix (same gating receipt as POD-Attention
   in HARVEST). Weeks-class.

Notable non-ranked: third-party NVFP4 receipts on RTX PRO 6000 (Qwen3-32B ≈2x BF16
throughput, accuracy flat — validation + marketing ammo, zero engineering); upstream
SM120 native-FP4 still patch-and-pray (five open/painful issues — our sm_120a kernels
are a moat, and `compute_120f` on CUDA 13.0 is the gencode the fixes converged on);
WitCert's Lean-proved runtime KV-quant risk meter as a serve-ready-shaped alternative to
offline quant evals; ASD (approximate spec) explicitly REJECTED — not output-exact,
fails our argmax gate.

---

## 1. Speculative decoding frontier

**What moved since the 08-08 harvest.** The fixed-K-dies-under-load consensus is now
shipped code, not just papers: DSpark in SGLang v0.5.16 with confidence-scheduled
variable verify and a compact ragged-verify mode [shipped]; EAGLE-3 in llama.cpp
[shipped]; P-EAGLE checkpoints published by hyperscaler [shipped]. The research edge moved to
(a) draft-source quality for tool-calling traffic (Oilbird: exact-match drafting is an
ADDRESSING failure, not a coverage failure — +24–29% accepted length by semantic
re-keying [paper-only]), (b) trees quietly returning via semi-autoregressive drafters
(PCTree: inference-only parent-conditioned branching over DSpark's Markov head, accept
9.41→11.16 at B=16 on Qwen3-4B GSM8K [paper-only] — but bs=1-flavored; the
trees-die-at-load serving verdict is not overturned), and (c) stochastic-spec drafting
(DBLAST: block drafters' per-position independence breaks at high sampling entropy
[paper-only] — matches our 0.55 short-ctx sampled acceptance receipt).

**Real vs paper-only numbers.** Reproduced/shipped: EAGLE-3 1.25–1.32x in a third-party
SGLang replication ("95% of published speedup", HF community post); SGLang EAGLE-3 1.81x
at bs=2 → 1.38x at bs=64 on H100 (and EAGLE-2 going 0.93x at bs=64 — fixed-cost drafting
dies at load); DSpark's B300 383.7 tok/s release-note receipt. Paper-only: Oilbird's
4.4x (code claimed, no third party yet), PCTree, DBLAST, SparseSpec-L (its
horizon-stop formula — stop when marginal acceptance < relative draft cost — is worth
one line in our K-policy regardless). Self-spec stays capped ~1.4–1.5x (KnapSpec law,
reconfirmed by SparseSpec-L's own baselines) — below our trained-drafter c=1 cell;
still fallback-only.

**Transfer.** Oilbird → #3 above (agent traffic, GGUF-agnostic, CPU-side). DSpark →
lift the ragged design + graph keying into in-flight HARVEST #1 (#5 above). llama.cpp
EAGLE-3 → #6. ASD → rejected on exactness. Bole (tree spec for linear-attention hybrids,
in SGLang, TTFT −67.6% on agent workloads) → model-selection intel only: our hybrids
are SWA+full, not recurrent; becomes relevant only if a KDA/linear-hybrid SKU
(Kimi-K3 class) enters the fleet.

## 2. Long-context serving + the depth-degradation bug

**The live bug has three documented candidate mechanisms** (full detail NOTES §2.1–2.4):

- *Model-level repetition collapse in hybrid-SWA models* [issue-receipt]:
  gemma#622/ollama#15502/gemma#610 — word-doubling → single-token collapse during long
  generation on BOTH dense-31B and MoE-26B Gemma-4; 60–100% failure under
  grammar-constrained JSON with free-text fields; repeat_penalty has zero effect;
  cross-backend (Ollama/LMStudio/Cloudflare) ⇒ argued model-level; gemma3:27b clean on
  the same harness. Their 10-seed protocol is directly rerunnable on our stack, and a
  reference-stack run splits model-level from memra-level in one afternoon.
- *Length-indexed precision accumulation* [shipped fix, reproduced]: vLLM/FA3 found FP8
  attention accumulation error growing with contraction dim = context length (128k NIAH
  91%→13%; two-level FP32 accumulation restores 89%; same HW issue as the DeepSeek-V3
  report). A reproducible ~9k ONSET is precisely what generic greedy degeneration does
  NOT produce (our own protocol survey: path-dependent, no fixed threshold) — so a
  length-indexed mechanism (accumulator width in the FA twins, softmax·V at long KV) is
  the prime engine-side suspect. Oracle-vs-fast-path diff at depth is the one-day probe.
- *SWA sink/window mechanics* [paper-only + heavily-reproduced law]: StreamingLLM — evict
  even the first token's KV and windowed models collapse; SWAA catalogs SWA failure
  causes and recovery recipes (FA-decode, sink preservation, interleaving). For a
  natively-SWA-trained SKU the mismatch factor is absent, but "sinks never leave the
  ring buffer" is a checkable invariant, and an unbounded-window debug arm at decode is
  a clean implicating/exonerating probe.

**256k+ arithmetic and KV compression.** The definitive fp8-KV receipt set landed (vLLM
2026-04-22 [shipped, reproduced] — see #4 above). TurboQuant study: low-bit KV error is
CONCENTRATED at 128k–256k and accumulates with length — long-context quality gates must
be per-depth, echoed theoretically by RoPE's provable loss of locality/consistency at
depth (2605.15514 [theory]). Eviction research keeps confirming our lossy-stays-out
doctrine: QEvict shows token importance drifts during decoding so irreversible eviction
is brittle [paper-only]; NVIDIA's EAI blog names the two production collisions (FA never
materializes scores; paged eviction frees ~zero blocks) and TriAttention's pre-RoPE
scoring + forward-packing compaction as the fixes [vendor blog + code] — the compaction
pattern is the reference IF a lossy tier is ever built for research SKUs. WitCert
[paper + Lean-proved artifacts] offers a runtime per-(layer,head,step) attention-error
bound with gating — restored raw-cast fp8 KV 22.8→79.7 on hard RULER; a runtime meter is
philosophically the serve-ready-shaped way to run quantized KV, worth a design read
before the fp8-K/NVFP4-V lane. Upstream hybrid-KV handling remains weak in the GGUF
world (llama.cpp SWA cache can't shift/reuse across slides; full reprocess reports) —
a competitive note, not a gap.

## 3. MoE inference

Incremental sweep. ReMoE (ICML'26 [paper-only, code]) fine-tunes the router for cache
locality: +26% expert reuse, 1.77–1.99x decode under llama.cpp offload — the strongest
offload-side number found, but it's a model-side change; memra's cheaper analog remains
the layer-position-aware SLRU A/B already ranked in HARVEST (#6 there, unchanged).
llama.cpp/Ollama shipped a fix for differently-quantized experts plus packed gate/up
projection (+4–9% on M5 Max) [shipped] — audit their mixed-quant handling against our
mixed-layout dispatch contract, and evaluate load-packing gate/up in our expert walkers
(#7 above). The 2026 EP-placement wave (Director, ExpertPlex, NanoCP, ELDR) is
datacenter-scale; on a PCIe pair the PP-over-TP guidance is now in vLLM's own docs, and
the dual-workstation-Blackwell EP2/TP2/PP2 head-to-head STILL does not exist anywhere —
HARVEST 3.6's publication opportunity remains open and has, if anything, ripened.
Mixed-precision MoE quant theory (2604.06515: allocate bits by activation frequency +
max intra-neuron variance) adds one candidate ranking feature to the five-arm study's
calibration method; no lane until the study reports.

## 4. Prefill/decode overlap + chunked prefill + disaggregation

The field's new answer to prefill/decode interference on ONE GPU is spatial, not
temporal: five SM-partitioning systems in 12 months (Bullet/ASPLOS'26, DuetServe,
Semi-PD, Nexus, MuxWise), ~1.26x-class claims — single-node-transferable but competing
with our tick + in-flight microchunk lane; gate on hybridity metering (#8 above).
Disaggregation-at-scale results mostly do NOT transfer, but two receipts matter: the
GLM-5/OpenClaw tuning report [prod-class] found optimum chunk 3072 (not 2048) and deeper
admission for a 28–30k-in/500-out agent workload with the explicit law that chunk size
and queue depth are non-monotonic — supporting per-SKU sweeps over global defaults; and
"When Does Disaggregation Pay?" [simulation] gives the cost model for why single-node
co-location is right at our scale. SGLang's v0.5.13 scheduler work (FutureMap async
value passing; prefill input H2D moved onto the forward stream) is the one-tick-ahead
overlap direction HARVEST 1.2 named — the H2D-on-forward-stream piece is liftable alone
(#7 above).

## 5. Quantization

NVFP4 on our exact card class now has third-party receipts: Qwen3-32B on RTX PRO 6000,
NVFP4 ≈1.9–2.1x BF16 throughput at c=8–64 with TTFT halved and GPQA/ARC flat within
stderr (Jarvis Labs [reproduced, single-run evals, English-only caveat]). Meanwhile
upstream native-FP4 kernels for SM120 remain broken-to-painful (flashinfer#2577/#2723,
cutlass#3096, flashinfer#2847, vllm#31085 — zeros, garbage, segfaults, Marlin
fallbacks; fixes converged on compute_120f + CUDA 13.0) — our in-house sm_120a NVFP4
path is a genuine moat and the gencode note is worth pinning. FP8-KV: the vLLM state
blog is the actionable item (#4). W4A8: LiquidGEMM [paper-only] argues the W4A8 sweet
spot with a HW-efficient GEMM; the big format studies (2411.02355 v4, 2508.16712) land
where our prod-8bit doctrine already is (W8A8 dominates async batching; aggressive quant
can cost up to −92% HumanEval on small models — keep code-gen in every quant gate).
Sub-4-bit KV stays research-file (INT2 rotation, 2-bit VQ — paper-only, lossy).

## 6. Agent-workload serving

The strongest thematic convergence of the sweep, and it validates the darklanes bet:
agent serving is a decode + KV-residency problem. Trace studies: 84.6–99.5% input-token
reuse across turns, decode = 91.0–98.6% of LLM time (2605.26297); production agent
traces show KV reuse >80% of request tokens vs 54–62% chat, and agents act on complete
responses so session-level TPS is the metric (SMetric). The engine-actionable core is
KV lifetime across tool-call pauses: Continuum's TTL pinning (>8x JCT claim on real
agent benchmarks [paper-only]) + CacheWise's tool-call-metadata-driven eviction
(evictions −2–2.6x, sessions up to 3.5x [paper-only]) — ranked #2 above; the keepalive
economics paper adds the market argument (rational clients will burn prefill with
keepalive replays if the server doesn't pin — pinning is cheaper for both sides, and
cached-token pricing should reflect it). The workflow-DAG layer (Helium, Pythia,
HexAGenT, SpecBox) needs only engine hooks we already have or have ranked: session pin,
priority admission, prefix pin. Oilbird (#3) is the spec-side face of the same
workload shape.

---

## Cited sources (primary)

Spec: arXiv 2608.03839 (Oilbird) · 2608.02123 (PCTree) · 2608.05448 (DBLAST) ·
2608.03447 (ASD) · 2607.27735 (SparseSpec-L) · 2512.22420 (Nightjar) · 2602.06036
(DFlash) · 2608.01651 (Bole) · 2510.22876 v3 (batch-spec done right) ·
lmsys.org/blog/2026-07-06-dspark-sglang · github.com/sgl-project/sglang/pull/30261 ·
github.com/sgl-project/sglang/releases/tag/v0.5.16 + v0.5.13 ·
github.com/ggml-org/llama.cpp/pull/18039 · vllm.ai/blog/2026-03-13-p-eagle ·
vllm.ai/blog/2026-05-26-eagle-3-1.
Long-context/KV: github.com/google-deepmind/gemma/issues/622 + /610 ·
github.com/ollama/ollama/issues/15502 · vllm.ai/blog/2026-04-22-fp8-kvcache ·
vllm.ai/blog/2026-05-11-turboquant · arXiv 2512.10411 (SWAA) · 2309.17453
(StreamingLLM) · 2605.15514 (RoPE limits) · 2608.05326 (QEvict) · 2607.28699 (WitCert)
· 2607.05399 (KV-opt benchmark) · 2608.04074 · 2608.02691 ·
research.nvidia.com/labs/eai/blogs/kv-cache-compression-and-its-infra-problems/.
MoE: arXiv 2605.27081 (ReMoE) · 2604.06515 · 2607.08782 (Director) · 2607.18002
(ExpertPlex) · 2605.21100 (NanoCP) · 2607.00466 (ELDR) · 2601.17063 (FlashMoE) ·
Ollama Aug-2026 release notes (releasebot.io/updates/ollama) ·
docs.vllm.ai parallelism_scaling.
Prefill/disagg: Bullet (ASPLOS'26, xianweiz.github.io/doc/papers/26asplos_bullet.pdf) ·
arXiv 2511.04791 (DuetServe) · 2504.14489 (Semi-PD) · 2507.06608 (Nexus) · 2408.12757
(NanoFlow) · 2608.03741 (disagg-pay simulation) · 2607.02518 (GLM-5/OpenClaw tuning).
Quant: jarvislabs.ai/blog/nvfp4-rtxpro-6000 · flashinfer-ai/flashinfer#2577/#2723/#2847
· NVIDIA/cutlass#3096 · vllm-project/vllm#31085 · arXiv 2509.01229 (LiquidGEMM) ·
2411.02355 v4 · 2508.16712.
Agents: arXiv 2605.26297 · 2511.02230 (Continuum) · 2606.16824 (CacheWise) · 2607.08565
(SMetric) · 2607.19214 (keepalive) · 2603.16104 (Helium) · 2604.25899 (Pythia) ·
2605.16637 (HexAGenT) · 2607.23933 (SpecBox).

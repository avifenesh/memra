# SOTA sweep 2026-08-09 — raw findings (NOTES)

Method: arXiv API feeds (spec-decoding, KV cache, MoE offload, chunked-prefill/disagg,
agentic-workloads cs.DC), keyless web sweeps (Brave chain), engine release notes
(SGLang v0.5.13–v0.5.17, llama.cpp, vLLM blog), vendor blogs (NVIDIA EAI, Jarvis Labs),
GitHub issue archaeology for the SM120 4-bit kernel state and the Gemma-4 repetition bug.
Diff base: `research/sota-harvest-20260808/HARVEST.md` (branch `lane/sota-harvest`) — items
already covered there are listed only when their STATUS changed. In-flight lanes per the
tasking: cross-request batched draft/verify (H#1), dynamic microchunks (H#3), prefix dedup
(H#4) — not re-reported.

Every claim below carries its source. Tags: [paper-only] = no independent reproduction
found; [shipped] = merged in a production engine; [reproduced] = independent third-party
numbers exist; [issue-receipt] = GitHub issue with reproducible protocol.

---

## 1. Speculative decoding frontier

### 1.1 Oilbird — training-free spec re-keyed by verifier hidden states [paper-only, code claimed]
- arXiv 2608.03839 (2026-08-04). https://arxiv.org/abs/2608.03839
- Claim: exact-suffix-match drafters (suffix/ngram class) miss drafts ALREADY IN the pool —
  on their densest tool-calling benchmark ~half of what the strongest exact-match drafter
  misses is present but unreachable by lexical matching. Fix: a second draft source keyed by
  the hidden state the verifier already computed at each committed token, merged into the
  lexical drafter's tree.
- Numbers: +24–29% accepted length at matched pool/budget across three published drafters;
  4.4x AR speed on API-Bank vs 3.9x best training-free baseline and 2.0x EAGLE-3.
- Transfer: direct extension of the in-flight suffix-decoding lane (HARVEST 2A.3 / top-8 #5).
  CPU-side keying by hidden states requires copying one hidden vector per committed token to
  host — cheap. Tool-calling traffic is exactly the memra customer shape.
- Cost class: ~1 week AFTER the suffix lane lands (it rides the same draft-slot injection).

### 1.2 DSpark — STATUS UPGRADE: shipped in SGLang v0.5.16 [shipped]
- SGLang PR #30261, release v0.5.16; LMSYS blog 2026-07-06
  https://www.lmsys.org/blog/2026-07-06-dspark-sglang/
  https://github.com/sgl-project/sglang/pull/30261
- HARVEST 2A.1 carried this as paper+blog; it is now a merged production path:
  `--speculative-algorithm DSPARK`, `SGLANG_RAGGED_VERIFY_MODE=compact`,
  `--speculative-dspark-block-size`. Release notes: 383.7 tok/s at accept length ~5 on
  DeepSeek-V4-Pro, TP8 on B300, bs=1.
- Offline receipts (ai-infrastructure.net summary of the paper): accepted length over
  EAGLE-3 +30.9/26.7/30.0% on Qwen3-4B/8B/14B; over DFlash +16.3/18.4/18.3%.
- The ragged-verify "compact" mode is a shipped reference for the ragged problem our
  in-flight batched-spec lane must solve.

### 1.3 PCTree — trees revived for semi-AR drafters, inference-only [paper-only]
- arXiv 2608.02123 (2026-08-03). https://arxiv.org/abs/2608.02123
- Claim: DSpark's Markov refinement head already contains parent-conditioned structure;
  score alternative children per concrete parent, allocate fixed verify budget over paths —
  a tree WITHOUT retraining and without extra backbone passes.
- Numbers: at B=7, +3.1–29.5% over matched DSpark across Qwen3-{4,8,14}B × 9 benchmarks;
  Qwen3-4B GSM8K B=16: accept length 9.41→11.16, AR speedup 6.14x→6.60x (3-run means).
- Note the tension with the EAGLE-3.1/Red-Hat "trees are dead for serving" verdict
  (HARVEST 2A.4): PCTree is bs=1-flavored; the serving question (verify cost at load)
  is unaddressed. File as "trees may return via semi-AR drafters," not a lane.

### 1.4 Approximate Speculative Decoding (ASD) [paper-only] — BLOCKED by exactness doctrine
- arXiv 2608.03447 (2026-08-04). https://arxiv.org/abs/2608.03447
- Accepts selected argmax mismatches under a logit-regret budget; +3.05–15.26% throughput.
  NOT output-exact by construction — fails memra's argmax gate. Recorded so nobody
  re-discovers it as a win; the FP4-draft/FP8-target compat finding (+10–16% verifier-side
  acceptance on GSM8K/MATH-500) is the one interesting data point: precision-mismatched
  draft/target pairs lose acceptance, which our fp8-K collapse receipt already implied.

### 1.5 DBLAST — dependent block drafting for stochastic (sampled) spec [paper-only]
- arXiv 2608.05448 (2026-08-05). https://arxiv.org/abs/2608.05448
- Block/diffusion drafters assume per-position independence; accepted length degrades as
  target sampling entropy rises. Low-rank latent mixture over positions + acceptance-
  oriented training objective fixes it. Relevant only if we ever train a block drafter for
  the sampled-spec path (our sampled acceptance 0.55 short-ctx receipt says entropy already
  hurts us); architecture note, no lane.

### 1.6 llama.cpp merged EAGLE-3 [shipped, reproduced]
- PR https://github.com/ggml-org/llama.cpp/pull/18039 (merged ~2026-06-16/19);
  docs https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md
- EAGLE-3 (draft reads target hidden states) now runs in the GGUF ecosystem. PR thread:
  Gemma-4 EAGLE-3 models supported; >2x speedup with reasoning ON, >3x with reasoning OFF;
  "with Q4_K_M quantization the speedup still looks good."
- Why it matters to memra: (a) the drafter SUPPLY problem is solved upstream — trained
  EAGLE-3 heads for GGUF-served models now exist and convert; (b) their hidden-state
  plumbing through a GGUF graph is a reference implementation for wiring any
  hidden-state-conditioned drafter into our loader; (c) it resets the llama.cpp same-silicon
  bar for spec cells (they previously had draft-model-only spec).

### 1.7 Bole — tree speculation for hybrid-attention (linear+full) models [shipped in SGLang fork, paper]
- arXiv 2608.01651 (2026-08-03). https://arxiv.org/abs/2608.01651
- Kernel-runtime co-design: tree-structured closed form of linear-attention recurrence,
  parallel verify of all proposal nodes (3.4–7.7x linear-attn tree verify), transient state
  memory −82–99x. Up to 4.72x offline decode vs AR; TTFT −67.6% / TPOT −49.9% under online
  AGENT workloads vs strongest tree baseline. Integrated into SGLang.
- Transfer: memra's hybrids are SWA+full (not linear/recurrent) — the recurrence trick does
  not apply. The negative from HARVEST 2A.5 (component-aware self-spec collapses on
  sequential hybrids, α=0.038) stands. Relevance = model-selection intel: IF a
  KDA/linear-hybrid SKU (Kimi-K3-class, 2607.24653) ever enters the fleet, spec on it is a
  solved problem upstream.

### 1.8 SparseSpec-L — train-free self-spec for long context [paper-only]
- arXiv 2607.27735 (2026-07-30). https://arxiv.org/abs/2607.27735
- Unified efficiency analysis: extending speculation horizon REDUCES speedup when marginal
  acceptance probability < relative drafting cost — the formal version of our measured
  K-vs-load cliff; compares against MagicDec/LayerSkip/EAGLE-3. Self-spec family still caps
  below our trained-drafter c=1 numbers (KnapSpec ~1.4–1.5x law from HARVEST holds).
  Value = the horizon-stop criterion for the K-policy lane, one formula.

### 1.9 Nightjar — load-adaptive spec on/off [paper-only]
- arXiv 2512.22420 (2025-12, v5 2026-02). https://arxiv.org/abs/2512.22420
- Dynamic spec that adapts length and decides when to STOP speculating under load; also
  quantifies restart cost of spec. Convergent with our K→0-subsumes-the-gate design
  (HARVEST 3.3); cite as third independent confirmation (with SpecDec++/Not-a-Bandit).

### 1.10 P-EAGLE — STATUS: hyperscaler pre-trained checkpoints published [shipped]
- hyperscaler ML blog 2026-03-13 + vLLM blog https://vllm.ai/blog/2026-03-13-p-eagle
- Delta vs HARVEST 2A.4: hyperscaler released pre-trained P-EAGLE checkpoints and the vLLM
  integration recipe (v0.16.0, PR#32887). B200 receipts 1.05–1.69x over vanilla EAGLE-3
  (GPT-OSS-20B, MT-Bench/HumanEval/SpeedBench). Training-cost barrier for one-pass
  drafters is now partially externalized.

### 1.11 Batch-spec correctness — HuggingFace paper page traction [status only]
- 2510.22876 v3 (2026-02-15) — already the evidence base of in-flight H#1; v3 adds the
  cost-structure analysis of re-synchronizing position IDs/masks/KV after each ragged
  round. No new numbers to import; the EXSPEC sliding-pool same-length grouping remains
  the cheap shape.

---

## 2. Long-context serving + the depth-degradation bug

### 2.1 Gemma-4 token repetition collapse — the live-bug twin [issue-receipt, cross-backend]
- google-deepmind/gemma#622 (2026-04-11) https://github.com/google-deepmind/gemma/issues/622
- ollama/ollama#15502 (39-trial isolation) https://github.com/ollama/ollama/issues/15502
- gemma#610 (deterministic loop at 14th list item, Cloudflare Workers AI + LMStudio).
- Facts: BOTH gemma-4-31B dense and 26B-A4B MoE show word-doubling → single-token collapse
  during LONG generation; grammar-constrained JSON amplifies to 60–100% failure;
  repeat_penalty 1.0/1.15/1.5 has NO effect (identical seeds fail identically); gemma3:27b
  clean on identical harness; reproduces across Ollama/LMStudio/Cloudflare ⇒ argued
  model-level, not backend. Trigger for full collapse (ollama#15502): dense 31b + grammar
  constraint + free-text string fields; MoE fails differently (malformed JSON).
- Relevance to OUR ~9k-into-generation degradation on an SWA-512 model: (a) a documented
  generation-depth degradation exists at MODEL level for the exact model family class we
  serve — before hunting an engine bug, run the #622 protocol (10 seeds, their prompts, our
  stack + one reference stack) to split model-level vs memra-level; (b) their negative
  result (repeat_penalty useless) saves us a dead end; (c) grammar/constrained decoding
  as an amplifier matters for our constrained-gen surface.
- Cross-check with in-repo `research/greedy-degeneration-protocol-survey.md`: greedy
  degeneration is path-dependent (no fixed onset threshold), 75–80% of long greedy batch
  runs in production vLLM hit repetition (arXiv 2512.04419). A ~9k reproducible onset is
  therefore NOT explained by generic greedy degeneration alone — a consistent onset depth
  points at something length-indexed (accumulation error, window/sink interaction, or
  model-level as in #622).

### 2.2 FP8 attention accumulation error at long context — candidate mechanism for depth bugs [shipped, reproduced]
- vLLM blog 2026-04-22 https://vllm.ai/blog/2026-04-22-fp8-kvcache
- On Hopper FA3: FP8 Tensor-Core accumulation loses precision when the contraction dim
  (= context length) is large — 128k NIAH fell 91%→13%; fixed by two-level accumulation
  into a true FP32 register (SageAttention2 technique; same hardware issue as DeepSeek-V3
  report Fig 7(b)). B200/FlashInfer does not need the fix (accumulation fixed in HW).
- Diagnostic for our bug: if ANY memra decode-attention path accumulates scores·V at
  reduced precision, the error grows with KV length — a length-indexed degradation onset
  is the exact signature. Audit accumulator widths in the FA twins (esp. any fp16
  accumulation in long-KV softmax·V), and re-run the degradation repro with the oracle
  (MEMRA_FAST=0) path — if the oracle is clean at 9k+ and the fast path is not, this is
  the mechanism. Days-class audit, uses existing repro.

### 2.3 SWAA — SWA long-context collapse: causes and recipes [paper-only, code+weights released]
- arXiv 2512.10411 v5 (2026-03-26). https://arxiv.org/abs/2512.10411
- Diagnosis of SWA collapse: (1) training-inference mismatch (attention distribution
  drift when FA-trained model is run with SWA); (2) structural inability to reach distant
  info when SWA is applied everywhere/always. Recipes: FA-decode (full attention during
  decode only), FA/SWA layer interleaving, sink preservation, light fine-tune; specific
  combinations recover quality at 30–100% speedup.
- Transfer: our SWA-512 SKU is natively SWA-trained (mismatch factor absent), but the
  sink-preservation and FA-decode findings define the experiment grid for the bug: verify
  the first-token sinks never leave the ring buffer, and test a debug arm running SWA
  layers with unbounded window during decode (correctness probe, not a product config) —
  if degradation vanishes, the window mechanics/sink handling are implicated; if not,
  look at accumulation (2.2) or the model (2.1).
- StreamingLLM's law (2309.17453, ICLR'24, heavily reproduced): evicting even just the
  first token's KV collapses windowed models; sinks are positional, not semantic.

### 2.4 RoPE provable long-context limits [paper-only, theory]
- arXiv 2605.15514 (2026-05-15). https://arxiv.org/abs/2605.15514
- As context grows, RoPE-based attention provably loses (a) locality bias and (b)
  consistency of token relevance — content-independent, length-only analysis. For 256k
  serving this is the theory ceiling under which all engineering happens; not actionable
  per se, but it justifies per-depth quality gates on 256k SKUs rather than assuming
  short-context evals transfer.

### 2.5 KV-quant error accumulates with sequence length [shipped receipts]
- TurboQuant study, vLLM blog 2026-05-11 https://vllm.ai/blog/2026-05-11-turboquant —
  aggressive low-bit KV degradation is CONCENTRATED at 128k–256k; errors accumulate with
  sequence length.
- Cross-engine benchmark of KV compression: arXiv 2607.05399 (2026-05) — KIVI, TurboQuant,
  SnapKV, CaM compared on one harness (task quality × system perf). First apples-to-apples
  receipt set for the eviction/quant families.
- QEvict (arXiv 2608.05326, 2026-08-05) [paper-only]: token/window importance DRIFTS as
  generated queries evolve ⇒ irreversible eviction is brittle during long decoding;
  three-tier full/quantized-recoverable/deleted design. Their "Future Missed Mass"
  diagnostic is a nice probe shape for any future eviction work.
- WitCert (arXiv 2607.28699 v2, 2026-07-30) [paper + released artifacts + Lean proofs]:
  runtime per-(layer,head,step) upper bound on TV distance between exact and compressed
  attention; meter-driven gating restored raw-cast fp8 KV from 22.8→79.7 on hard RULER
  with bounded diff; certified subtractively-dithered INT8 KV serves 1.88x more tokens at
  same memory in SGLang. Also: aggressive schemes survive on CROSS-LAYER error
  cancellation, not per-step fidelity (28-layer sweep, 0/28 single-layer losses).
  https://github.com/metask-ai/witcert-kv-certificates
- vLLM FP8-KV accuracy receipts (2026-04-22 blog): reasoning ≤1–2 pts loss (worst 97%
  recovery GPQA-Diamond), MRCR AUC 94–98% recovery to 256k, 1M-token Qwen3.5-27B AUC fully
  recovered; ITL slope 54% of BF16 (llama-8B), break-even ~7k tokens; throughput c=8
  +14.9%. All with UNCALIBRATED per-tensor scales (worst case). Hybrid-model law:
  `--kv-cache-dtype-skip-layers sliding_window` — small windows (gpt-oss, 128) should NOT
  be quantized; gemma-4-E2B's SWA-512 windows are big enough to amortize (quantizing them
  won). head_dim=256 models: FP8 prefill regresses ~1.6x (two-level accumulation register
  pressure).

### 2.6 KV compression infra problems — TriAttention update [vendor blog + shipped code]
- NVIDIA EAI blog 2026-06-12
  https://research.nvidia.com/labs/eai/blogs/kv-cache-compression-and-its-infra-problems/
- The two production collisions named precisely: (1) FlashAttention never materializes
  scores ⇒ H2O/SnapKV-class scoring can't run without falling back to eager attention;
  (2) paged eviction frees almost nothing (survivors scatter — evicting 14,400 of 16,000
  tokens can free ~0 blocks). TriAttention's answers: pre-RoPE geometric scoring (no
  scores needed) + forward-packing compaction every ~128 decoded tokens (order-preserving
  repack frees whole blocks).
- Updates HARVEST 2B.7: still lossy/watch for defaults, but the infra story is now solved
  and documented; if a lossy long-context tier is ever built (research SKUs), the
  compaction pattern is the reference.

### 2.7 Hybrid-KV managers upstream [shipped]
- vLLM hybrid KV cache manager (docs), one memory pool shared across full-attn and SWA
  groups; `--disable-hybrid-kv-cache-manager` escape hatch exists because hybrid-model
  KV logic keeps regressing (community reports). LMCache stores/retrieves multiple KV
  groups for Gemma-3-class hybrids. llama.cpp SWA cache still cannot shift/reuse across
  window slides — "advanced cache operations not possible when using SWA cache" (their
  own docs; community receipts of full reprocessing on LCP<0.5). memra's SWA handling is
  competitive here; the upstream weakness is a marketing point, not a gap.

---

## 3. MoE inference

### 3.1 ReMoE — router fine-tuning for cache locality [paper-only + code, ICML 2026]
- arXiv 2605.27081. https://github.com/BUAA-OSCAR/ReMoE
- Biases router toward recently-selected experts ⇒ +26% expert reuse at maintained task
  scores; llama.cpp offload receipts: TPOT −43.6–49.8% (1.77–1.99x decode) on Jetson Orin
  NX; vLLM GPU-CPU offload +8.4% throughput.
- Transfer: model-side (fine-tunes the router) — collides with memra's
  "runtime over artifact changes" preference and would need the five-arm-study-grade gate
  discipline. File as the spill-SKU endgame option; the correct memra-side analog stays
  the layer-position-aware SLRU A/B (HARVEST top-8 #6, unchanged).

### 3.2 llama.cpp/Ollama: mixed-quantized experts + packed gate/up [shipped]
- Ollama release notes Aug 2026 (releasebot mirror): "Fixed Qwen3 MoE decoding for
  differently-quantized experts, plus faster packed gate/up projection (~4–9% on M5 Max)."
- Two takeaways: (a) mixed-layout expert banks are now a REAL upstream configuration —
  their fix is worth reading against our metadata-aware staged/SLRU/grouped dispatch
  contract (CLAUDE.md), days-class audit; (b) packed gate/up projection (fusing the two
  expert matmuls' loads) is a kernel idea applicable to our expert walkers — 4–9% class
  on their hardware, unknown on ours.

### 3.3 Expert-parallel placement papers — datacenter-scale, low transfer
- Director (arXiv 2607.08782): online proactive expert placement for distributed EP.
- ExpertPlex (2607.18002): disaggregated MoE, adaptive persistent kernels, goodput focus.
- NanoCP (2605.21100): request-level dynamic context parallelism for DP+EP decode.
- ELDR (2607.00466): expert-locality-aware decode routing in PD-disaggregated MoE.
- All assume many-GPU EP; on a 2-GPU PCIe pair the HARVEST 1.6 verdict (PP for PCIe pairs,
  no EP2/TP2 head-to-head exists on our class) is unchanged — and the publication gap
  (HARVEST 3.6) is still open. vLLM's own docs now say it plainly: without NVLink, prefer
  PP over TP (docs.vllm.ai parallelism_scaling).

### 3.4 MoE mixed-precision quantization theory [paper-only]
- arXiv 2604.06515 (2026-04): bit-allocation with generalization guarantees; allocation
  criteria = activation frequency + max intra-neuron variance. MODE (2606.17118),
  MxMoE (2505.05799), MoPEQ (2509.02512) same family.
- Feeds the five-arm study's ranking method (variance criterion is one we don't currently
  use for the Q2_K/Q3_K/NVFP4 tiering); no lane until the study reports.

### 3.5 TIDE (2605.20179): lossless I/O-aware expert offload for DIFFUSION LLMs — off-family, skip.
### 3.6 EdgeXpert (2608.05303): MoE+spec co-design on edge accelerator HW — off-target, skip.

---

## 4. Prefill/decode overlap + chunked prefill + disaggregation → single node

### 4.1 Intra-GPU SM-level prefill/decode disaggregation wave [ASPLOS'26 + preprints]
- Bullet (ASPLOS'26, paper PDF xianweiz.github.io/doc/papers/26asplos_bullet.pdf):
  dynamic SM partitioning between concurrent prefill and decode kernels with feedback
  control; community summary: 1.26x throughput, no new hardware. Semi-PD (2504.14489),
  Nexus (2507.06608), DuetServe (2511.04791 — adaptive GPU multiplexing, attention-aware
  roofline), MuxWise (static split) — five systems in 12 months = the field's answer to
  chunked-prefill interference is becoming SPATIAL (SM partition) rather than temporal
  (token budget).
- Transfer: single-GPU, single-node — directly our shape; sm_120a green contexts / SM
  masking would be the mechanism. But it competes with our three-phase tick + the
  in-flight dynamic-microchunk lane; only worth opening when tick-hybridity metering shows
  decode-tail stalls behind prime chunks that budget-tuning can't fix. Weeks-class.
  Same gating receipt as POD-Attention (HARVEST 2B.6) — measure hybridity first.

### 4.2 When Does Disaggregation Pay? [paper-only, simulation]
- arXiv 2608.03741 (2026-08-04): simulates prefill/decode/attention/FFN specialization
  for AGENTIC inference specifically. Useful as the cost model for why single-node
  co-location is right at our scale; no numbers imported (simulation).

### 4.3 GLM-5/OpenClaw serving-parameter tuning report [reproduced-in-prod class]
- arXiv 2607.02518: long-context agent workload (28–30k in / 500 out); best config was
  chunked-prefill 3072 (not 2048), max-running 24 (not 16): +11.6% req/s, TTFT 8.98→6.69s,
  P90 −18.9%. Law: "the optimum is workload-specific; larger chunk sizes and deeper
  queueing do not monotonically improve performance." Receipt supporting our per-SKU
  tick/chunk sweeps rather than one global default.

### 4.4 SGLang scheduler deltas [shipped]
- v0.5.13: Spec V1 deprecated — EAGLE/MTP unified on V2 worker; topk=1 drafting faster;
  per-step scheduler overhead lowered via FutureMap async value passing + prefill input
  transfer moved ONTO the forward stream. v0.5.16: DSpark (above). v0.5.17 current.
- The "prefill input transfer on the forward stream" trick is a one-line-class idea for
  our tick: overlap H2D of the next prime chunk's tokens with the current forward.

---

## 5. Quantization

### 5.1 NVFP4 on RTX PRO 6000 — third-party receipts on our exact card [reproduced]
- Jarvis Labs blog 2026-05-18 https://jarvislabs.ai/blog/nvfp4-rtxpro-6000
- Qwen3-32B, vLLM, RTX PRO 6000 Server: NVFP4 ≈1.9–2.1x BF16 throughput at c=8–64
  (1823.97 vs 869.45 tok/s at c=64), 1.77x at c=128; best TTFT at every concurrency
  (148 vs 339 ms at c=64); GPQA-Diamond tied with BF16 (0.3939 both), ARC within 1 stderr.
  Single-run evals, English-only checkpoint caveat stated by the authors.
- Value: independent validation of the FP4-first thesis on our owned silicon class, with
  honest caveats. Marketing-usable numbers (someone else's, on our card class).

### 5.2 SM120 4-bit kernel ecosystem state — still half-broken upstream [issue-receipts]
- flashinfer#2577 (2026-02-18): NVFP4 mm_fp4 GEMM broken on SM120, ALL backends (cutlass
  returns zeros, cudnn graph unsupported).
- flashinfer#2723 / cutlass#3096 (2026-03-09): CUTLASS grouped block-scaled NVFP4 MoE GEMM
  produced garbage on SM120; compute_120a gencode SEGFAULTs (card reports 12.0 not 12.0a);
  fixed only via FlashInfer patches + compute_120f on CUDA 13.0 → 39 tok/s native FP4.
- flashinfer#2847 (2026-03-21): CUTLASS MXFP4 MoE kernels gated OFF for SM120 despite the
  infra existing. vllm#31085 (2025-12-20): SM120 unrecognized in MXFP4 backend selection,
  falls back to Marlin.
- NVIDIA forums (2026-04): GPT-OSS-120B MXFP4 on PRO 6000 Max-Q "does work on SM120" after
  a long debug path.
- Verdict: upstream native-FP4 on workstation Blackwell remains a patch-and-pray zone;
  memra's own sm_120a NVFP4 kernels are a real moat. Track compute_120f (CUDA 13.0) as the
  gencode the ecosystem converged on for these fixes.

### 5.3 FP8 KV — the definitive receipt set [shipped, reproduced]
- vLLM blog 2026-04-22 (full detail in §2.5 above). Headline for our arithmetic:
  fp8-KV ITL slope 54% of BF16, break-even ~7k tok, +14.9% throughput c=8, accuracy
  94–99% recovery UNCALIBRATED, per-head scales + calibration shipped for the hard cases,
  skip-SWA-layers law for hybrids, two-level accumulation law for Hopper-class FA3.
- Re-opens our fp8-K flip-block (74%→20.5% acceptance collapse) with a specific hypothesis:
  our collapse is consistent with uncalibrated per-tensor scales + no per-head scales;
  their per-head + calibrated path did not exist when we swept. The acceptance battery
  stays the gate; NVFP4-V (HARVEST top-8 #8) and an fp8-K-recalibrated arm can share one
  lane.

### 5.4 W4A8 serving quality [paper receipts]
- LiquidGEMM (2509.01229): HW-efficient W4A8 GEMM for serving; W4A8 argued the
  accuracy/efficiency sweet spot vs W4A4.
- "Give Me BF16 or Give Me Death" v4 (2411.02355, updated 2026-05): most comprehensive
  format study — W4A16 most cost-efficient for synchronous, W8A8 dominates async
  continuous batching; informs our prod-8bit doctrine (consistent with it).
- Systematic characterization (2508.16712): quality-preserving = W8A16-INT, W8A8-FP,
  W8A8KV8-FP; worst cases up to −92% HumanEval on a 13B under aggressive quant — evals
  must include code-gen, echoing our q27 divergence receipts.
- Our w4a8-prefill mxf8f6f4 lane (1.2153x pp512) already banked the practical piece
  (HARVEST prior-delta) — no new lane.

### 5.5 Sub-4-bit KV — still research-file
- Output-aware rotation for INT2 KV (2608.02691); attention-preserving-transform VQ at
  2 bits (2608.04074). Both paper-only, both fixed-budget lossy. RateQuant's law
  (sub-4-bit uniform KV stays badly lossy) unchallenged. No lane.

---

## 6. Agent-workload serving

### 6.1 Agentic AI Workload Characteristics — the doctrine receipt [paper-only, traces]
- arXiv 2605.26297 (2026-05-25): ReAct-style agents traced end-to-end (Gemma/Qwen, five
  benchmarks): with effective context caching, input-token reuse across turns gives
  84.6–99.5% cache-hit ratios and decode dominates 91.0–98.6% of LLM time; tool use has
  temporal structure (read/explore early → execute/write late).
- Meaning for memra: agent serving = a DECODE + KV-residency problem, not a long-prompt
  problem. Our step35 batched-decode investment and prefix-cache metering are aimed at the
  right wall; TTFT-only benchmarks under-weight what agent customers feel.

### 6.2 Continuum — KV TTL pinning across tool calls [paper-only, Berkeley, vLLM impl]
- arXiv 2511.02230 v6 (2026-05-25). https://arxiv.org/abs/2511.02230
- Problem: engines evict finished requests' KV when new work arrives; agent sessions PAUSE
  for tool calls (seconds), lose the cache, repay full prefill. Fix: pin KV with a TTL set
  from reload cost vs queueing-delay cost; auto-evict on expiry; + program-level FCFS.
  Claims: avg job-completion-time improved >8x on SWE-Bench/BFCL/OpenHands agents with
  Llama-3.1-8B/70B, Gemma-3-12B, GLM-4.5-355B.
- Transfer: memra already has session affinity + LCP cache + admission machinery; the TTL
  pin is a policy on top (pin session KV on response-completed-with-tool-call, TTL from
  measured tool-call duration histogram + reload cost we can compute exactly). The 8x is
  their harness; our win shape = TTFT×hit-rate on re-entry, measurable via cache-meter.
  Days-class.

### 6.3 CacheWise — coding-agent KV management [paper-only, vLLM impl, real traces]
- arXiv 2606.16824 (2026-06-15): real coding-assistant traces; prefix-aware scheduling +
  reuse-aware eviction driven by TOOL-CALL METADATA predictions; evictions −2–2.6x,
  session completion up to 3.5x. The metadata signal (which tool ran → how likely/soon
  the follow-up) is a free feature we already see in the serve surface.

### 6.4 SMetric — session-centric scheduling [paper-only, prod trace]
- arXiv 2607.08565: production BAILIAN agent trace shows KV reuse >80% of request tokens
  (vs 54–62% chat); agents act on COMPLETE responses ⇒ cluster TPS is the metric, per-token
  latency constraints relax. Scheduling should balance sessions, not requests.

### 6.5 Keepalive economics [paper-only]
- arXiv 2607.19214 (2026-07-21): clients replaying prefixes on a timer to keep provider
  caches warm is individually rational across Anthropic/OpenAI/Google/DeepSeek pricing.
- Marketplace implication for darklanes: if we DON'T pin (6.2), rational customers will
  burn our prefill capacity with keepalive traffic anyway — server-side TTL pinning is
  strictly cheaper for both sides; also a pricing-design input (cached-token pricing).

### 6.6 Workflow-aware serving (further out)
- Helium (2603.16104): LLM calls as first-class operators, proactive caching + cache-aware
  scheduling across a workflow DAG. Pythia (2604.25899): exploits multi-agent workflow
  predictability. HexAGenT (2605.16637): DAG-deadline-aware scheduling on heterogeneous
  clusters. SpecBox (2607.23933): speculative SANDBOX preallocation for MCP tool sandboxes.
  All platform-layer; engine hooks they need (session KV pin, priority admission, prefix
  pin) are exactly items 6.2/6.3 + HARVEST #4. No engine lane beyond those.

---

## Cross-cutting status updates vs HARVEST (do-not-re-report ledger)

| HARVEST item | status change this sweep |
|---|---|
| 2A.1 DSpark | paper → SHIPPED (SGLang v0.5.16, PR#30261) with prod flags + B300 receipt |
| 2A.3 suffix decoding (in-flight #5) | Oilbird (2608.03839) defines the v2: semantic re-key, +24–29% accepted len on tool-calling |
| 2A.4 P-EAGLE | hyperscaler pre-trained checkpoints + recipe published (vLLM v0.16.0) |
| 2B.4 NVFP4 KV | unchanged; fp8-KV receipts (vLLM 04-22) sharpen the K8/V-low split argument |
| 2B.7 TriAttention | infra problems solved + documented (NVIDIA EAI blog); still lossy, still watch |
| 1.5 KV quant | WitCert runtime gating (Lean-proved) is a new serve-ready-shaped alternative to offline evals |
| 1.6 MoE 1–2 GPU | llama.cpp now ships mixed-quantized-expert decode fix + packed gate/up (+4–9% M5 Max) |
| 3.6 dual-Blackwell benchmark gap | still open; vLLM docs now explicitly recommend PP over TP for non-NVLink — the head-to-head remains unpublished |

# Prior-art survey 2026-08-09 — raw notes index + memra-side grounding

Deliverable paper: `PAPER.md` (this directory). Raw per-question findings live in four
lane files written by parallel research agents:

- `notes-q1a-vllm-trt-lmcache-mooncake.md` — prefix reuse in vLLM APC, TensorRT-LLM block
  reuse, LMCache, Mooncake
- `notes-q1b-sglang-llamacpp.md` — prefix reuse in SGLang RadixAttention and llama.cpp
  slot/prompt cache
- `notes-q2-swa-depth-degradation.md` — rolling-KV/SWA bugs at generation depth in other
  engines
- `notes-q3-admission-preemption.md` — VRAM admission, preemption, parked-session reclaim
- `notes-q4-batched-specdec.md` — cross-request batched draft/verify speculative decoding

Every external claim in PAPER.md traces to a quote + URL in one of those files.

---

## memra-side grounding (read from this repo, 2026-08-09)

The paper's final "steal vs already-better" table compares against THIS state, not a guess.

### Problem 1 — growing-conversation prefix reuse

memra already ships three tiers (docs/SERVING.md):

1. **Continuation pool** (`MEMRA_KV_REUSE`): retired session parks whole (prompt+generation)
   state; resume requires EXACT extension. Single-use.
2. **Cross-request prefix cache** (`MEMRA_PREFIX_CACHE_MB`, default 256MB): device snapshots
   at token boundaries, keyed by exact token-id prefix per (model, cache_salt namespace),
   LRU under a byte budget. Hit = deep copy into the new session. Learning sequence:
   request 1 seeds, request 2 split-primes at LCP and inserts boundary entry, request 3+ hit.
   Bit-identity gated (16/16 partial + 16/16 full, `research/prompt-cache-20260802/`).
   Spec sessions bypass (trunk-only restore would leave draft state unprimed).
3. **Session affinity** (lane/session-affinity, 2026-08-05, SERVING.md §"Session affinity"):
   the rewritten-history answer. Identity tiers: explicit (`session_id`/`user`/header) and
   implicit (structural fingerprint: hash of first/last tokens of template-delimited
   segments, 3-segment minimum). "Identity nominates, BYTES decide": resume only if the new
   prompt reproduces committed tokens exactly up to the last PROMPT-END checkpoint.
   Checkpoint = state at prompt end, before first generated token (rewriting clients mutate
   what was GENERATED, not the prompt). Full-attn KV truncatable by length; checkpoint copies
   only GDN conv/ssm recurrent state. Declines logged with offsets.
   Measured (research/session-affinity-20260805/RESULTS.md, N=3 interleaved): rewritten-turn
   TTFT 0.53-0.65 s ON vs 11.9-14.0 s OFF (20-24x); sum-of-medians TTFT 12.4x, wall 5.30x.
   The task brief's "TTFT grew 5.85x over 10 turns" is the pre-affinity freeze this lane fixed;
   the hardening question is what the mature engines do that this design still lacks
   (eviction policy for checkpoints, host-tier spill, multi-hundred-k budgets).

Known residual (research/tick-seg-20260807/PROGRESS.md §Residual): step35 prefix-cache
entries are EXTENT-CLASSED — the SWA arm predicate keys on request seq_end, so two requests
sharing a prefix but straddling the window produce different prefix KV bytes; a resume
continues from the creator's class. Inside the documented contract; canonical fix =
one numeric class for all SWA prefill rows. Upstream precedent already used: vLLM #51113
(mamba align-chunking prefix-cache poisoning; grain-aligned publish law + off-grid-resume
second law, both now gated in memra via tickinv35 arms).

### Problem 2 — long-generation degradation at depth

Open bug: ~9k+ generated tokens → cross-lingual token soup on a SWA window=512 model
(interleaved SWA/full-attn). Related shipped work: step35 SWA segmentation fixes
(lane/step35-chunkfix + lane/tick-seg): the SWA arm predicate `seq_end > win` was
call-local, so serve's tick segmentation steered prefill arithmetic (maxdiff 1.813e0,
greedy divergence at step 6); fixed by threading request-level seq_end (`queued_after`),
gated bit-identical across budgets 1024/513/512/256/64 + off-grid resumes sp64/256/512.
That class is PREFILL-side; the 9k soup is DECODE-depth and still undiagnosed.

### Problem 3 — VRAM admission with parked sessions

Shipped state (SERVING.md §"64-client robustness", lane/admit-oom 2026-08-06):
- 2026-08-02 gate (free >= 2x per-session cost) failed at c=64 spec-ON: 0/64, all streams
  died with quoted CUDA_ERROR_OUT_OF_MEMORY. Two errors: parked-session delta understated
  live cost 1.49x, plus ~1.3 GiB capture-arena transient not proportional to session count.
- Fix: flat `SPEC_SHRINK_RESERVE` (1.5 GiB) charged on spec-capable models only; gate reads
  `free + pool_cached` (retired KV returned to pinned pool was invisible to driver free);
  step-OOM parks (rebuild + requeue at FRONT, `MEMRA_STEP_OOM_RETRIES`=3, only for sessions
  that emitted nothing, only on quoted OOM). Result 64/64 x3, peak 23.1/24.5 GB, gated with
  teeth (`tools/serve-stress-gate.sh --teeth` inverts to 11/64).
- `max_tokens` sizing: omitted max_tokens reserves ladder slack to the ctx ceiling — 6.3%
  of a 96GB card stranded at c=16, 12.6% at c=32 (research/serving-density-20260806/).

### Problem 4 — cross-request batched draft/verify

memra ships MTP spec decoding per-request (embedded/extracted draft head, one draft file
per model, docs/DRAFT-REGIME.md). Batched tick decodes sessions in per-model chunks
(exact-16 tier where bit-exact 16-batch kernels exist, else 8), but spec sessions step
through their own path; cross-request VERIFY batching is unbuilt. PP-2 door is
eager-decode only (batch/dc/graph/spec unwired over pipeline pairs — main log,
commit d4c3e3c3 message).

### Constraints any transferred mechanism must respect

- Single model, single node, 2x RTX PRO 6000 (96GB each), PP-2 across the pair.
- GGUF runtime; hybrid models carry GDN conv/ssm state that CANNOT be truncated to a
  shorter prefix (snapshot-at-boundary only, never rollback).
- Exactness discipline: cached hit must be bit-identical to the run that computed the
  prefix; every reuse tier is gated, with teeth.
- Flags doctrine: winners are defaults; no new tuning flags for a shipped mechanism.

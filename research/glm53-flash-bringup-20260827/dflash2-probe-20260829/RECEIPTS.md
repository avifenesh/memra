# DFlash2 acceptance probe: GLM-5.3-Flash, 2026-08-29

The cheap cell that prices the drafter arc before anyone commits days to it. One question:
would GLM-5.3-Flash accept DFlash2 drafts, on our real agent traffic, at a rate that pays
for building T-parallel verify?

**Answer: yes. Tokens per verify cycle 3.06 overall (4.66 on tool-wire turns) with the
real published GLM drafter, measured teacher-forced against the serving binary's own
greedy path. Projected 1.8x to 3.1x over the measured 21.3 tok/s plain decode once
T-parallel verify exists. GO.**

## The drafter (real, published, pinned)

- `incoai/GLM-5.3-Flash-DFlash2`, revision `dc77ff1c99eeb2df044ee3d4f0094eb033fee410`,
  `model.safetensors` sha256 `b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b`
  (matches the HF LFS oid byte for byte; verified on the box after download).
- 1.17B-param block-diffusion drafter, 5-layer qwen3-class backbone, hidden 4096,
  vocab 154880 (identical to the target artifact's text vocab; same tokenizer by
  construction), block_size 8 (one anchor + 7 drafted tokens per verify cycle),
  two-tap grouped dynamic convolutions, candidate selector (rank 256, top_k 16),
  conditioned on target hidden states from layers [5, 14, 24, 33, 42].
- It is NOT a standalone LM: it consumes target features and uses the TARGET's own
  embed_tokens and lm_head (both unquantized in our NVFP4 artifact's ignore list).
- Published reference numbers (their card, SGLang, 4x GB300, temp 1.0 top-p 0.95,
  reasoning effort max, academic benchmarks): acceptance length 4.03 to 5.86, beating
  GLM's native MTP at 3.71 to 5.06, 1.73x to 2.79x throughput at concurrency 1.

**LICENSE, and it binds this lane: CC BY-NC-ND 4.0.** Research and evaluation only, no
derivatives, no redistribution of remints. This probe sits squarely inside the
research/evaluation grant. This drafter CANNOT serve customers and CANNOT be
reminted or republished by us. Production draft-side requires either a commercial
license from Inco AI (contact@inco.ai) or training our own drafter. The q38 precedent
worked because z-lab's Qwen3.8-27B drafter is apache-2.0; this one is not.

## What ran where (external-implementation law observed)

- **Target path: the memra serving binary.** Branch `lane/glm5-dflash2-probe`
  (dd7f1d11d base, grouped prefill default ON), server binary sha256
  `f2b9f782c08bf2ac2d39562f56aaffdf43fab7efa05480dca6fc34fa9ba6b512`, serving shape
  pinned per the lane: PP3 over cards 0-2, MEMRA_PREFIX_CACHE_MB=0, TF32 off,
  MEMRA_CTX=8192. Greedy is the instrument, stated as such; greedy rollouts via raw
  `/v1/completions` with `prompt_ids`.
- **Target features: the memra engine.** New capture seam in this branch
  (`hidden-trace: contracted all-rows layer dump`, commit 81d6601c0):
  `MEMRA_TRACE_LAYER_ROWS` dumps the stream-mean (hc_contract) of the completed layer
  output at layers 5,14,24,33,42 for every position, from `run-safetensors`
  teacher-forced over prompt + serving-binary continuation. This is the exact
  aux-hidden definition the SGLang glm5_next DFlash2 integration pins in its unit
  test (PR 36708: mean over the hc_mult stream blocks of hidden+residual captured at
  layer k+1 = completed output of layer k). Capture-on forward reproduces the
  capture-off argmax.
- **Drafter: the z-lab reference implementation** (github.com/z-lab/dflash @ 07ebd93db9,
  `dflash/model.py`), run in PyTorch bf16 on one otherwise-idle card. This is
  probe-only evidence creation outside serving, per the house law, and it is the same
  reference memra's own q38 DFlash2 port was parity-gated against
  (`dflash2_parity.rs`). Nothing external touched a serving path.

## Method: teacher-forced production-shape cycles

The scoring loop mirrors `dflash_generate` cycle for cycle (block 8, incremental
drafter KV cache with crop-to-start, position ids over [new_lo, start+8), anchor
embedding + 7 mask embeddings, candidate-selector walk at temperature 0), with one
substitution: committed tokens come from the serving binary's greedy continuation
instead of a co-evolving local target. Acceptance per cycle is the DFlash2 greedy
rule: longest drafted prefix equal to the target's greedy tokens; the cycle advances
produced = accepted + 1 (the verify bonus token), exactly as production verify would.
For accepted positions this is arithmetic-identical to the production incremental
scheme, because causal attention makes accepted-prefix features independent of the
rejected block tail.

Prompts: the banked gpf-ab pool (real multiturn agent transcripts; box-local at
`~/gpf-ab/prompts.json`, sha256 `de57a7a471f9b1632ac49924430aba5cd1737465479cb59081b8fc0074b53e46`),
cut at blank-line boundaries near 35/60/85/100% so each transcript yields several
decode starting points, 13 scoring prompts of 427 to 6467 tokens, rendered through the
artifact's own chat_template.jinja with reasoning_effort pinned low, greedy rollouts
of up to 256 tokens. 2911 continuation tokens scored, 951 DFlash2 verify cycles.

Gates that had to be green before the numbers count, and were:

- **Render/tokenize parity:** local template render + tokenize of the full A4630
  prompt = 4626 ids = the serving binary's chat-path prompt_tokens. PASS.
- **Retokenization idempotence:** decode(encode(text)) == text on 13/13 continuations
  (the engine has no token-id echo on the raw surface, so scoring runs in canonical
  local-tokenizer space). The only generated-vs-retokenized length drift is exactly
  one token on every finish=stop rollout: the terminal EOS, counted by usage but not
  emitted as text.
- **Capture-vs-serving path agreement:** on all five finish=stop rollouts the capture
  forward's last-position argmax is an EOS-class id (154827/154829), i.e. the exact
  token the serving binary stopped on. 5/5 green. (Capture runs single-GPU with paged
  experts, same NVFP4 serving-numeric class, TF32 off; serving ran PP3.)
- **Greedy-law loop gate:** tail-cycle and repeated-line detectors clean on 13/13
  rollouts; nothing excluded.
- **Drafter pipeline sanity:** feature magnitudes grow monotonically with depth
  (rms 0.07 to 1.99), and acceptance on tool-wire turns reaches the published band,
  which a wrong layer set, wrong stream contraction, or wrong feature order could not do.

## Results

Decode rate (streamed chat, A4630 full prompt, effort low): greedy 21.32 tok/s,
vendor-default sampled (no sampling params sent) 21.28 tok/s. TTFD 7.4 s at 4.6k
(grouped prefill engaged, flag=on execute receipts in the serve log).

**DFlash2 drafter, 951 cycles.** acc@k = fraction of cycles whose first k drafts all
match (the drafter proposes 7 tokens per cycle: block 8 = anchor + 7, so k runs 1..7;
there is no k=8 for this drafter):

| class | cycles | acc@1 | acc@2 | acc@3 | acc@4 | acc@5 | acc@6 | acc@7 | mean accepted | tokens/cycle |
|---|---|---|---|---|---|---|---|---|---|---|
| all   | 951 | 0.731 | 0.486 | 0.327 | 0.209 | 0.137 | 0.100 | 0.072 | 2.06 | 3.06 |
| tool  |  89 | 0.843 | 0.742 | 0.607 | 0.438 | 0.393 | 0.360 | 0.281 | 3.66 | 4.66 |
| prose | 862 | 0.719 | 0.459 | 0.298 | 0.186 | 0.110 | 0.073 | 0.050 | 1.90 | 2.90 |

**n-gram / prompt-lookup floor** (free copy-drafting, same teacher-forced paths, 2371
cycles): mean accepted 0.22 all, 1.54 tool, 0.12 prose. The free floor pays on
repeated tool schemas only and is worthless on think-prose; it does not justify the
verify arc on its own (1.22x ideal, 0.72x to 1.01x realistic).

Versus the published numbers: their acceptance length 4.03 to 5.86 was measured at
temperature 1.0, reasoning effort max, on academic benchmarks. Our comparable figure
(tokens per verify cycle) is 3.06 overall and 4.66 on tool-wire turns, greedy, effort
low, on real agent transcripts. The tool-turn class lands inside their band; the
overall mix sits below it because at effort low these rollouts spend 91% of cycles in
think-prose, where acceptance is 1.9. The published agent-traffic-accepts-higher
intuition holds exactly on the tool-wire class.

## The arithmetic

Measured plain decode t = 1/21.32 s = 46.9 ms/token.

**Sequential verify (today): the acceptance buys nothing.** Every spec entry point
refuse_hypers on this topology, and even if drafts flowed, sequential verify runs
each drafted token through the decode program one at a time: a cycle of L accepted
+ 1 bonus costs (L+1) decode steps plus draft overhead. Throughput <= plain decode
regardless of acceptance. Speedup: none, by construction.

**T-parallel verify (the arc under decision), tokens/cycle = 3.06:**

| bracket | projected tok/s | speedup |
|---|---|---|
| verify = 1.0t, draft = 0.0t (ideal)            | 65.3 | 3.06x |
| verify = 1.0t, draft = 0.2t (expected)         | 54.4 | 2.55x |
| verify = 1.5t, draft = 0.2t (conservative)     | 38.4 | 1.80x |

Verify bracket basis: a 8-to-15-token parallel step is launch-bound, decode-step
class (the lane's grouped-prefill receipts show 616-639 tok/s bulk prefill, so the
arithmetic cost of 8 extra tokens is negligible; 1.5t covers KDA/DSA/mHC step
overheads honestly). Draft bracket basis: one 1.17B bf16 block pass over 8 rows plus
the selector and an 8-row lm_head GEMM, a few ms on this card class; 0.2t is generous.
Tool-heavy sessions run above these numbers (tokens/cycle 4.66 projects to 2.7x to
4.7x on those stretches).

For calibration: q38 serves 119 plain and 250 with DFlash2 (2.1x). The GLM projection
is the same class of win.

## GO/NO-GO

**GO on the T-parallel verify arc.** Decision threshold: at the ~1-2 agent-day build
cost with rollback already scoped, the arc pays if projected speedup at conservative
brackets clears 1.5x, which needs tokens/cycle >= 2.25 at verify 1.5t + draft 0.2t.
Measured 3.06 clears it with margin; even the prose-only class (2.90) clears it.
NO-GO would have required tokens/cycle under ~2.3 overall.

Two decisions deliberately NOT bundled into this GO:

1. **Draft-side licensing.** The verify arc is drafter-agnostic engine work and this
   probe prices it with a real drafter, but serving THIS drafter needs a commercial
   license from Inco AI, or we train our own (the DFlash2 recipe is published; the
   q38 path shows the port surface memra already has: loader, conv-wrapped forward,
   windowed attention, selector walk, all parity-gated in `dflash2_parity.rs`).
2. **Native MTP as the license-clean alternative.** GLM-5.3-Flash upstream ships an
   MTP head (published acceptance 3.71 to 5.06, below DFlash2's). Our NVFP4 artifact
   carries NO mtp/nextn tensors (checked `model.safetensors.index.json`), so that
   route needs a re-mint with the MTP head before it can even be probed. T-parallel
   verify unlocks either route; that is the arc's real value.

Integration scope the arc must cover on the draft side (measured here, not guessed):
the feature tap for the hyper trunk is the stream-mean of completed layer outputs at
5 layers (this branch's `MEMRA_TRACE_LAYER_ROWS` seam is the probe-grade version of
it); the drafter shares the target's embed_tokens and lm_head (no own head); drafter
class facts: vocab 154880, hidden 4096, 5 layers, sliding window 2048, conv kernel 2
group 16, selector rank 256 top_k 16, mask token 154856, block 8.

## Caveats, stated not hidden

- Scoring runs in canonical retokenized space (no token-id echo on the raw surface);
  idempotence was green 13/13, residual BPE segmentation noise is possible within
  continuations but cannot manufacture acceptance, only lose it at span edges.
- Drafter features come from the single-GPU capture forward, not the PP3 serving
  forward; both are the NVFP4 serving-numeric class and the greedy-path agreement
  receipt is 5/5 at the checkable positions. Production DFlash2 would compute
  features in-process on the serving path.
- Greedy is the instrument. Sampled acceptance (rejection sampling at the vendor
  defaults we actually serve) is a different number and MUST be measured in the
  verify-arc gate battery before any perf or pricing claim ships, per the sampled-
  verification law.
- The tool/prose split is span-based on continuation text (tool_call blocks, JSON
  objects, code fences); 89 tool cycles is a thin class, read the class table as
  direction, not a third significant digit.
- reasoning_effort pinned low throughout (the claim shape); effort max traffic would
  shift the think/tool mix and likely the mean.

## Files

- `build_prompts.py`, `rollout.py`, `decode_rate.py`: phase 1 on the serving binary.
- `capture_worker.sh`: per-GPU feature capture via the patched `run-safetensors`.
- `score_dflash2.py`: teacher-forced drafter scoring (the probe core).
- `ngram_baseline.py`, `aggregate.py`, `loop_check.py`, `make_public.py`.
- `summary.json`: the tables above, machine-readable.
- `decode_rate.json`, `loop_check.json`: gate rows.
- `cycles_dflash2.json`, `cycles_ngram.json`: per-cycle rows (position, class,
  accepted, per-k hits; no transcript text).
- `scoring-prompts.public.json`, `rollouts.public.json`: sanitized run manifests
  (shas and 60-char heads only; the pool itself stays box-local, sha pinned above).
- `captures-manifest.txt`: capture receipts incl. the EOS cross-check lines.
- Raw full-text rollouts, token ids, and the ~30 GB of feature dumps are NOT banked:
  regenerable on a box holding the artifact from the pinned pool + this branch, with
  the commands in the scripts. Feature dumps were deleted at lane close (tmp hygiene).

Box state at close: server stopped (PID-verified), all four cards 0 MiB, drafter
checkpoint left at `/root/models/glm53-dflash2` (pinned revision, license-bound,
probe-only) and the z-lab clone at `/root/dflash` for the follow-on arc, both noted
here deliberately.

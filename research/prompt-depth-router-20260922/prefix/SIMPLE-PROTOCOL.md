# Simple-helper prompt-prefix comparison

This is a versioned continuation of [PROTOCOL.md](PROTOCOL.md). The first
qualification source and all of its records remain frozen. Its Qwen code
requests produced no final answer at either the 8,192- or 12,288-token cap, so
that corpus cannot answer the owner's prose/code question. No request from it
entered scored comparison.

The new [generator](workloads_simple.py) uses six synthetic families of four
straightforward Python helpers, with separate prose and code requests. Its
arithmetic qualification task is separate from the six scored families. A
qualification diagnostic on the current model and GPU returned prose and
parseable fenced code at each of 256, 1,024, 4,096 and 16,384 user tokens.
Every code block cleared the unchanged 80-byte threshold. That diagnostic
tests coverage feasibility; its timings are not a K-policy result.

## Frozen comparison

- Use the same pinned Qwen3.8-27B and Gemma 4 12B artifacts and the same
  native binary/source archive as the first attempt. Heads remain full.
- Each cell is an independent sampled request with warm weights and a fresh
  native cache. Temperature is 0.7, top-k is 20 and top-p is 0.95.
- There are six scenarios per family, each containing prose and code at 256,
  1,024, 4,096 and 16,384 user tokens. Actual user-token counts and small
  preparation tolerances are retained.
- Compare fixed K=3 with K chosen from only the first 64, 128 or 256 user
  tokens. Prose selects K=2, code selects K=4, and uncertainty selects K=3.
  No classifier may read the remainder of the user prompt. The prefix
  budgets are separate arms even when they produce the same decision.
- Match prompt, seed, output budget and sampler within each four-arm group.
  The six arm orders are frozen by `pipeline.py` before generation. All three
  fixed depths receive the same warmup.
- Before scoring, greedy fixed-depth outputs must agree, a sampled prefix
  schedule must reproduce the explicit K schedule, and each family must
  cover all eight prose/code qualification cells without an exact-repeat
  loop. Try an 8,192-token cap first, then 12,288 only if needed. If neither
  cap qualifies, stop without a scored result.
- Preserve every scored output. An exact-repeat flag excludes its whole
  matched request from throughput calculations; format misses and capped
  outputs remain visible in the requested-format table. Report the separate
  subset for which all four arms cover the requested final format.
- Use pooled returned tokens / complete native request seconds, per-cell
  paired gains and whole-scenario bootstrap intervals. Include tokenization,
  prefix extraction, forecast, prefill, generation and detokenization in the
  timed request. Model load, warmup and receipt copying are outside it.

The existing format checks are unchanged: code requires a closed fenced
Python block with at least 80 parseable bytes, and prose requires at least
200 final-answer bytes without a source-code fence. These checks establish
format coverage, not whether a program solves its task.

This corpus tests inexpensive routing on simple synthetic helper tasks.
It cannot establish a general code/prose policy or a serving default.
Hosted CPU checks and a non-production research GPU provide qualification;
the local rig runs no gates or benches.

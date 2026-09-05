# Exact prefill work elision

2026-09-05; base `4afb4624f`. Both switches default to `all` (optimization OFF).

Two source-proven redundancies are separated from the experimental grouped
expert math. The public verifier still returns its complete `[T,vocab]` or
`[T]` argmax result. Only the chunked-prefill caller selects the private output
policy: no vocabulary head for intermediate chunks, existing single-row head
for the final hidden row. Every trunk layer, cache write and commit remains.

DSpark is SWA-only. Its prime computes KV directly from trunk taps, without
reading previous drafter KV. For a known suffix, only its last 128 positions
can survive in the persistent rings. The tail policy skips older tap captures
and prime calls but retains the same m=1 projections for every kept row.
Short suffixes keep all rows and therefore preserve live restored slots.
The final tap is always updated. The one-token canonical prime is unchanged.

Current primary reference:
https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/main/inference/model.py
(read 2026-09-05): `ParallelHead.forward` selects the last row unless full logits
are requested; `DSparkAttention` primes only its sliding window; the DSpark
prefill path computes KV without running draft attention.

## Gate scope

`dsv4_prefill_work_gate` runs all four head/draft-policy combinations on real
source prompts, including a one-token canonical case, widths 1/32/64/512,
restored suffixes of 33/128/129/257 tokens, and ring wraps. It compares initial
and final logits, every live trunk class, DSpark rings, sampled tokens, round
bookkeeping and confidence. Explicit counters distinguish the executed head
and prime paths, and a public-verifier test checks that Last does not truncate
an explicitly requested full-row or argmax result.

Reported gate wall includes snapshot/hash checks and is not HTTP TTFT or a
serving-performance claim. The canonical first-token head/prime is excluded
from counters in both arms. 397 engine library tests, the work-counter control,
and clippy including both gate binaries/server pass. The complete correctness
`tools/local-ci.sh` run for base 4afb4624f also finished successfully before these
new edits; it is not a new-feature or performance receipt. Target-card model
gates and serving/cache/concurrency qualification remain pending.

## Current DSpark sampling observation

The same 0731 source's `forward_head` calls its temperature-controlled sampler
for draft tokens; the sampler uses an exponential/Gumbel race when T>0.
Memra currently drafts greedily, while its target acceptance walk is a pure
position-keyed sampled draw. A shared-randomness sampled proposal is a candidate
for improving agreement without changing the target distribution or seeded
plain/spec identity. The `coupled` experiment is now implemented behind
`MEMRA_DSV4_DSPARK_PROPOSAL` (default `greedy`). It uses Memra's existing sorted
categorical sampler, not the upstream Gumbel algorithm, and samples draft slot i
at tap_position + i + 2. The target draw and accept logic are unchanged. The
public greedy proposal remains greedy; only the sampled driver passes a sampler
to the private proposal implementation. Proposal penalties are not applied,
which may lower agreement but cannot bypass the target's penalties.

`dsv4_coupled_proposal_gate` compares plain/greedy-proposal/coupled-proposal
tokens, persistent trunk caches and DSpark rings across two prompt lengths,
three seeds and a penalty case. It also reverses the arm and checks draw-count
engagement. CPU key-alignment and deliberately shifted-key controls pass;
target-card acceptance, CPU sampling cost and serving benefit remain unmeasured.

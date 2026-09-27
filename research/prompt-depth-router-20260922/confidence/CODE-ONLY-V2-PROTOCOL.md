# Versioned code-only held-out qualification

The first held-out protocol stopped before development-C selection. Its
separate boolean-helper qualifier covered seven of eight requested formats:
the 4K prose request exhausted 8,192 returned tokens in reasoning and had
no final answer. All four requested-code qualifier cells returned fenced,
parseable Python; none of the eight outputs triggered the exact loop screen.
That failed eight-cell attempt remains unscored and is retained in private
operator custody.

This v2 protocol is a new, narrower qualification of the owner's **Qwen
code K=3** question. It reuses the six preregistered task-family scenario
files from `HELDOUT-PROTOCOL.md`; none has been run or inspected. Every
scenario still contains its original prose and code requests, but only
requested-code cells gate format coverage and enter the primary throughput
table. Prose requests, capped reasoning and format misses remain visible
diagnostics. No request or seed from the failed boolean qualifier enters
v2 scoring.

The new disjoint qualification family is **tuple utilities**, with four
simple Python functions:

- `swap_pair(pair: tuple[int, int]) -> tuple[int, int]` swaps its components.
- `first_or_none(values: tuple[int, ...]) -> int | None` returns the first
  element or `None`.
- `last_or_none(values: tuple[int, ...]) -> int | None` returns the last
  element or `None`.
- `pair_sum(pair: tuple[int, int]) -> int` adds its two components.

Use the unchanged length-sizing/tokenizer procedure to prepare a separate
eight-request prose/code qualifier at 256, 1,024, 4,096 and 16,384 user
tokens, with seed `20760020` and the same 8,192-token output cap. Freeze
its instruction text, generated token counts, prompt hashes and source
script before reading the development C rates. All four **code** requests
must reach a final fenced Python block of at least 80 parseable bytes and
pass the exact loop screen. A failure closes v2 with no throughput score;
do not change the seed, task, cap or gate again in this lane.

After that qualification, use the unchanged positive-C selection rule from
`HELDOUT-PROTOCOL.md`: choose the single confidence arm with the highest
positive pooled code-request rate on the development grid, or stop if none
beats K=3/C=0. Compare the selected K=3/C arm on the six untouched
scenarios with K=3/C=0 and K=2/C=0, matching prompt, seed, source, sampler
and GPU. Keep the earlier balanced arm order and matched loop exclusions.
Report six paired code requests per length, output tokens / complete native
request seconds, output-length and latency ratios, confidence-shortened
rounds, final-code coverage and the prose diagnostics separately.

This versioning does not turn a prose failure into a passed all-format
gate. A positive v2 result would still be synthetic native evidence, not
live threshold learning, warm-KV reuse, vendor-default HTTP or code task
correctness.

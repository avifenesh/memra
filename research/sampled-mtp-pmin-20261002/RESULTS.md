# Positive-PMIN sampled MTP on Qwen3.5-9B

The bounded local experiment passed. Positive PMIN=0.5 exercised the actual
single-head sampled graph and eager routes. Graph/eager token IDs agreed,
acceptance/residual/bonus paths ran, and forced-zero rounds worked. A separate
3072-request experiment detected no difference from plain sampling at the eight
registered token marginals. The registered serving default remains PMIN=0.

This is evidence for one artifact, source, binary and GPU tuple. It is not a proof
of the full autoregressive distribution, checkpoint fidelity, another family,
multi-head MTP, another topology or a serving default change. Issue #673 stays open.

## Identity and design

- Runtime source: `2873dd4ca37faa15cd4261e7b398926cced30106`; no runtime input changes.
- Artifact: `Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`, SHA256
  `52c9cceb190055e0591a9a30c21f7200572eaf3ff1c59f6e9a1eda838a8f39de`.
- Server SHA256: `eaa0afa71439143ae6c05455d85dd42097053b91dc5fb4e520e639b91d143529`.
- `run-spec` SHA256: `310e00e4ada94dc196223025ebf881b8faa32922c2bb9f92899857b12d088619`.
- GPU: one RTX 5090 Laptop, 24463 MiB, driver 595.91.07; CUDA 13.1, sm_120a.
- Statistical request: fixed 27-token color-sequence prompt, eight output tokens,
  temperature 0.8, top-k 20, top-p 0.95, min-p 0, neutral penalties, context 2048.
  Six fresh server processes, prefix cache disabled, 512 disjoint fixed seeds per arm.
- Statistical analysis: Python 3.14.4, NumPy 2.3.5, categorical total variation at
  each token position, 1999 label permutations. Familywise alpha 0.05 across
  40 comparisons, Bonferroni threshold 0.00125. No thresholds or sample counts
  changed after execution.

See the [preregistration](PREREGISTRATION.md), exact
[frozen collector snapshots](receipts/source-v2/manifest.json), and raw
[statistical manifest](receipts/native-v2-statistics/manifest.json).

## Native path and stopping evidence

The existing `run-spec` probes generated 64 tokens with seed 7 and K=3. Their
same-seed rerun check is only a reproducibility check; the separate HTTP
distribution experiment below addresses a different property.

| Arm | Rounds | Drafted / accepted | Residual rounds | Nonempty full-accept rounds | Zero-draft rounds |
|---|---:|---:|---:|---:|---:|
| PMIN=0 graph | 18 | 54 / 45 | 5 | 13 | 0 |
| PMIN=0.5 graph | 25 | 56 / 41 | 9 | 16 | 0 |
| PMIN=0.5 eager | 25 | 56 / 41 | 9 | 16 | 0 |
| PMIN=1.1, PMIN0 enabled, graph | 62 | 1 / 1 | 0 | 1 | 61 |
| PMIN=1.1, PMIN0 enabled, eager | 62 | 1 / 1 | 0 | 1 | 61 |

These counters describe the first complete internal walk, including any final
round overshoot before the runner truncates the returned tape to 64 tokens.
Both graph/eager pairs returned identical 64-token IDs. All five probes exited 0.

The first matched context provides direct cutoff evidence. With the same emitted
prefix `[5983]`, position 28 and output length 1, the PMIN=0 control drafts
`[11, 2438, 11]`; PMIN=0.5 drafts `[11]`. Both accept one proposal and emit bonus
13358. The collector compares the full emitted prefix up to this point, so a
coincidental matching last token cannot count as a matched context.

PMIN=1.1 is a deliberate edge instrument above every valid probability, not a
recommended setting. The first pending-less proposal survives the documented
slot-zero exemption; 61 later rounds have an empty draft and use the target bonus.

Path engagement is in the actual `[skey] chain=graph_s` / `chain=eager`, `[R...]`,
`[spec-stats]` and `[spec-phase]` lines. No `EXACTNESS q=0` event was observed.
The [diagnostic summary](receipts/diagnostic-summary.json) links the measurements;
the complete [CLI logs](receipts/native-v1/cli-probes/) preserve both seeded walks.

The device primitive `sample_check` reports
`=== sample-check ALL GREEN ===`. The current-source exact rational `spec_stop`
suite reports six passed tests, including the old-rule counterexamples. These
complement the real-model results; neither substitutes for them.

## Finite sampled distribution check

All six arms completed 512 requests, all HTTP 200, each returning eight native
token IDs. No survivor filtering or early-stop padding was needed in this run.
There were 91 to 109 distinct eight-token tapes per arm, so the experiment did
exercise stochastic variation.

| Arm versus plain sampling | Minimum permutation p-value across eight positions | Maximum empirical TV |
|---|---:|---:|
| PMIN=0 | 0.0885 | 0.0703125 |
| PMIN=0.5 graph | 0.365 | 0.07421875 |
| PMIN=0.5 eager | 0.066 | 0.087890625 |
| Forced-zero graph | 0.504 | 0.0546875 |
| Forced-zero eager | 0.395 | 0.060546875 |

No comparison crossed 0.00125. This is **no detected marginal difference at this
finite sample size**, not acceptance of an exact full-sequence equality claim.
The separate CPU red control approximating the old 0.8-to-0.96 bias was rejected
at p=0.0005, with empirical TV 0.16015625.

Every response and seed is retained in the six `responses.jsonl` files under
[native-v2-statistics](receipts/native-v2-statistics/). The complete test statistics
are in [distribution.json](receipts/native-v2-statistics/distribution.json).
The six bulky server traces are gzip-compressed without changing their raw bytes;
[compressed-traces.json](receipts/compressed-traces.json) records both compressed
and original hashes. [SHA256SUMS.json](receipts/SHA256SUMS.json) seals every receipt.

## Vendor-default HTTP, cache and rollback

Fresh default, PMIN=0.5 and rollback-to-default server processes each passed cold,
warm and four-concurrent-request checks on `/v1/chat/completions`. These requests
omitted sampling and thinking overrides, including the seed. Model metadata
installed the pinned [Qwen general-task thinking recipe](https://huggingface.co/Qwen/Qwen3.5-9B/blob/c202236235762e1c871ad0ccb60c8ee5ba337b9a/README.md#best-practices):
temperature 1, top-p 0.95, top-k 20, min-p 0, presence penalty 1.5 and repetition
penalty 1. The profile SHA256 is
`f8363f31c18a7227af37d51ed801c7c5da3237c35351f50c05512027e732de71`.

Every post-listener sampled burst matched the expected temperature, filters and
penalty state. The server reported the same loaded metadata hash in all three
boots. The penalty-bearing recipe uses the eager sampled route; startup canary
traces are excluded from this assertion. Each cold/warm request engaged MTP, and
the warm request restored 224 of 250 chat-template prompt tokens. The concurrent
wave exercised admission. All 18 responses reached their 64-token budget with
valid usage; the cold/warm outputs contained reasoning and no final answer.

The raw request bodies, responses, environment and engagement logs are under
[vendor-default](receipts/native-v3-vendor-profile/vendor-default/),
[vendor-positive](receipts/native-v3-vendor-profile/vendor-positive/) and
[vendor-rollback](receipts/native-v3-vendor-profile/vendor-rollback/). These bounded
responses test serving mechanics, not completed-answer quality. Independent
random seeds mean cold/warm text equality is not an assertion in these cells.

## Timing and resource observations

The statistical arms ran sequentially with diagnostics and synchronized phase
timing enabled. Their full-request p50/p95/p99, in milliseconds, were:

| Arm | p50 | p95 | p99 |
|---|---:|---:|---:|
| Plain | 119.34 | 126.24 | 129.25 |
| PMIN=0 | 129.08 | 151.79 | 156.17 |
| PMIN=0.5 graph | 125.14 | 148.64 | 163.14 |
| PMIN=0.5 eager | 124.56 | 149.70 | 161.61 |
| Forced-zero graph | 151.78 | 159.68 | 163.11 |
| Forced-zero eager | 149.86 | 157.63 | 161.82 |

These are diagnostic observations, not an interleaved performance winner or a
deployment recommendation. Existing draft/verify/commit phase clocks are retained
for every CLI arm. Confidence is the raw head-row maximum followed by a four-byte
read; its time is included in the draft phase. A separate confidence-read clock
was not measured.

The statistical job took 452.059 seconds. Its 250 ms telemetry has 1767 samples
and a peak of 9821 MiB GPU memory. Owned server processes exited before the lease
was returned; cleanup recorded no remaining compute process.

## Retained failures and remaining scope

The first helper build used the wrong Cargo target spelling, `sample-check`.
The actual target is `sample_check`; the corrected build passed. The successful
six-test CPU proof from the first build was retained and not repeated.

The first native cell passed all CLI probes, device primitive tests and generic
default HTTP cells, but failed its statistical collector. API authentication selected the
OpenAI response shape, which lacks native token IDs. One response from each arm
was retained and the distribution verdict remained incomplete. The retry set
`MEMRA_COMPAT=native` explicitly and repeated only the six incomplete statistical
arms. Both attempts and the protocol amendment are retained.

A later trace audit found that the first no-override HTTP cells used generic
defaults (temperature 1, top-p 1, top-k 0), because an inline model launch does not
install Qwen's vendor recipe. Their original `vendor-*` filenames are preserved
as historical labels, but those receipts only establish generic-default controls.
The missing vendor-profile cell ran separately with pinned metadata and explicit
post-listener checks. The same checker rejected all three original generic traces
in [red replays](receipts/vendor-profile-red-replay.json). Statistical and CLI
successes were retained without rerunning them.

This artifact contains one MTP block, so sampled multi-head chain graphs are
unreachable here. They need an admitted multi-head artifact and its own native
receipt. Other families and topologies, full-sequence statistical equivalence,
checkpoint parity, strict runtime qualification and broader quality/performance
decisions remain outside this result. No live serving configuration was inspected
or changed. The PMIN=0 default and support declarations are unchanged.

## Collector checks and reproduction

Install the pinned analysis dependency from
`tools/sampled-mtp-requirements.txt` in a private Python environment. The seven CPU
collector tests pass, including the explicit native-response dialect under API
auth, the statistical red control, early-stop accounting, matched-prefix
validation and refusal of generic/default or startup-only vendor traces. Earlier
test outputs and source snapshots are retained. All 3072 statistical responses
and 18 vendor chat responses also pass the final, stricter response validators.

`tools/collect-sampled-mtp.py` takes the model, prebuilt server and `run-spec`
binaries, an output directory, a private CUDA cache, an unused loopback port and
the caller's inherited canonical lock via `--external-lock`. The full invocation
runs the composite protocol; `--statistics-only` reports only statistical marginals,
and `--vendor-only --vendor-metadata <toml>` reports only vendor-profile HTTP cells.
No mode creates a support promotion or starts compilation.

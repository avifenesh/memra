# request-conditioned speculative K policy

Lane: `lane/cx-cache-k-policy`, train `d43f9e27`. Target rig: box1
(`ubuntu@<rented-box-ip>`), single-card placement, all GPU work under
`flock /tmp/memra-gpu.lock`.

Status: implementation and target-rig evidence complete. No origin push, merge, or tag.

`~/.lanectl/inbox/cx-kpolicy.md` was absent at lane start. The lane registry contains
`cx-kpolicy` on this worktree with status `running`; the mission text is therefore the
lane contract.

## Goal and boundary

Replace the serve path's process-global fixed K plus separate placement/concurrency
gate with one request-owned decision:

```
observable request signals -> K in {0, 1, ...}
```

`K=0` means plain batched decode. Positive K means speculative serving at that depth.
The first policy is a measured table/threshold rule, not a learned controller.

The policy changes scheduling only. It does not change draft or verify math, sampling,
cache bytes, or the engine's exactness contract. Model/drafter availability, constrained
decoding, and sampler eligibility remain authoritative: a positive policy K cannot make
an otherwise-ineligible request speculative.

Operator precedence is deliberate:

- if `MEMRA_SPEC_K` is set to an integer, it is the request-independent operator pin;
- `MEMRA_SPEC_K=0` pins plain serving;
- a positive pin bypasses the automatic placement/concurrency/prompt table, but not the
  correctness eligibility checks above;
- with `MEMRA_SPEC_K` unset, the policy is the default.

The existing `MEMRA_SPEC_GATE=0` rollback remains "ignore placement/concurrency" for
un-pinned requests. Explicit `MEMRA_SPEC_GATE_LOW` / `_HIGH` thresholds remain available.

## Signals at the decision seam

| signal | worker source | v1 use |
|---|---|---|
| placement | cross-device sharded PP-2 predicate | `K=0` at the measured PP-2 placement |
| projected concurrency | `active.len() + 1` at admission | new arrivals use `K=0` above LOW; live greedy spec sessions still demote at HIGH |
| prompt length | rendered/tokenized `prompt.len()` | distinguishes the measured cold and cached-long classes |
| cached prefix length | actual resumed continuation/spec-session tokens (`n_cached`) | long prompts with >=1024 resumed tokens select `K=2` |
| best LCP | prefix-cache `best_lcp` for the request's own namespace | logged at the seam; retained as the next table axis, not used without a winning receipt |
| tenant | `auth::meter_key(cache_ns)` | logged and available for a future tenant table; v1 is tenant-neutral |

Using actual `n_cached`, rather than global cache-hit rate, keeps the rule request-local.
The LCP and tenant inputs are intentionally observational in v1: no receipt yet justifies
different K for equal prompt/cache shapes across tenants, or treating an unserved LCP as
equivalent to resumed bytes.

## Inherited receipts

### K=0 placement/concurrency rows

| policy cell | receipt | decision |
|---|---|---|
| PP-2 q9, c=1/2/4 | spec/plain = 0.50x / 0.33x / 0.19x, N=3+ | `K=0` |
| PP-2 step35, c=1/2/4 | spec/plain = 0.42x / 0.36x / 0.30x, N=3 | `K=0` |
| single-card q9, c=1/2/4 | spec/plain = 1.67x / 1.08x / 0.61x, N=3 | keep LOW=2, HIGH=4 |

Source: `research/specplace-20260808/PROGRESS.md`. These rows are large-margin and
already re-swept on the batched core.

### Prompt-shape rows

The production-drafter acceptance gate supplies deterministic K=3 rows:

| model | p1 cold-short (28 tok) | p2 cold-medium (1845 tok) | p3 cold-long (5411 tok) |
|---|---:|---:|---:|
| q9 | 72.36% | 53.74% | 58.70% |
| q27 | 67.46% | 67.46% | 77.78% |

Source: `tools/fast-gate/accept-refs/*.ref`. This proves prompt shape is a policy input;
it does not by itself select K.

The q9 single-card serve sweep used a 226-token prompt and found K=2 and K=3 tied near
the top (232.9 / 234.6 tok/s), while K=1 and K=5 lost (187.0 / 214.4 tok/s).
Source: `research/spec-scaling-20260806/logs/ksweep/points.jsonl`.

The older PRO-6000 q27 study found deeper-context acceptance above short-context
acceptance (roughly 0.63-0.66 vs 0.51-0.59) and deeper serve optima, but its tree,
burst, and sampled configuration differ from this lane. It motivates measuring K=5;
it does not authorize a default by itself.
Source: `research/q27-deepdive-p2-20260805/RESULTS.jsonl`.

## Measured policy table

The box1 matrix replaced the provisional positive-K rows:

| priority | condition, with `MEMRA_SPEC_K` unset | K | receipt |
|---:|---|---:|---|
| 1 | cross-device sharded PP-2 | 0 | inherited PP-2 ratios 0.19x-0.50x |
| 2 | projected active count `> LOW` | 0 | inherited single-card c=4 ratio 0.61x; LOW=2 |
| 3 | prompt >= 1024 and cached >= 1024 | 2 | q9 282.66 tok/s; q27 124.47 tok/s |
| 4 | otherwise | 3 | exact short winner on q9/q27; within 0.70%/0.45% of cold-long winners |

`1024` remains the predeclared class boundary and existing LCP histogram edge. Both the
prompt and actual resumed-prefix conditions are required, so a small cache hit does not
change K. The cold-short and cold-long rows collapse to one `K=3` rule: their exact
per-model cold-long winners disagree (`K=2` on q9, `K=5` on q27), while `K=3` is only
0.70% and 0.45% behind respectively. Recovering those sub-1% deltas does not justify a
model-family branch in the first policy.

The original direction hypothesis was wrong: deeper K did not pay on cached-long. K=2
beat K=3 by 1.49% on q9 and 7.11% on q27; K=5 was 15.19% and 19.62% behind K=2. The
cached-prefix row is therefore shallower, not deeper.

## Box1 matrix

Primary artifact: q9 NVFP4+MTP plus its production own-trim drafter. Greedy serving,
single card, `max_tokens=128`, one server boot per pinned K, rep-major interleaving,
N=3 medians in one lock hold.

K arms: `{0,1,2,3,5}`. `K=0` uses the new `MEMRA_SPEC_K=0` operator pin so every arm
exercises the same decision seam.

Prompt classes:

| class | workload | required receipt |
|---|---|---|
| cold-short | `p1-code-short.txt`, fresh session/namespace | prompt tokens, acceptance, aggregate tok/s |
| cold-long | `p3-agentic-long-v3.txt`, fresh session/namespace | prompt tokens, acceptance, aggregate tok/s |
| cached-long | exact-extension turn 2 from the long prompt; turn 1 response is appended verbatim | `cached_tokens >= 1024`, acceptance, aggregate tok/s |

Every row records raw client JSON, server log, startup policy line, per-request decision
line, GPU state, errors/sheds, N, and thermal regime. A row without the expected policy K
or cached-token class is invalid, not a measurement.

The q27 spot-check measured `K={0,2,3,5}` for every class, N=3, using the production
drafter. The first q27 cached-long attempt was invalid: its 317.6 MB snapshot exceeded
the default 268 MB prefix-cache budget, so the harness correctly rejected
`cached_tokens=0`. The committed rerun used a 512 MB measurement budget and produced
all 36 valid rows.

## Matrix receipts

All rates below are N=3 medians from independent server boots, rep-major interleaved
inside one exclusive GPU-lock window.

| model | class | selected K | selected tok/s | exact winner | winner tok/s | acceptance at selected K |
|---|---|---:|---:|---:|---:|---:|
| q9 | cold-short | 3 | 349.77 | 3 | 349.77 | 72.36% |
| q9 | cold-long | 3 | 142.08 | 2 | 143.08 | 58.70% |
| q9 | cached-long | 2 | 282.66 | 2 | 282.66 | 56.67% |
| q27 | cold-short | 3 | 143.57 | 3 | 143.57 | 67.46% |
| q27 | cold-long | 3 | 55.46 | 5 | 55.71 | 77.78% |
| q27 | cached-long | 2 | 124.47 | 2 | 124.47 | 66.36% |

Thermal regime: q9 GPU0 post-arm samples stayed at 34-35 C. The separate q27 lock
window started at 26 C and post-arm samples stabilized at 33-37 C. The other GPU was
idle throughout both campaigns. Full ranges and every arm are in
`MATRIX-SUMMARY.md`.

Raw index:

- `raw/47da3098-q9-matrix/`: q9 45-row matrix, plus the preserved invalid q27 cache-budget attempt;
- `raw/4d94e948-q27-spotcheck/`: corrected q27 36-row spot-check;
- `MATRIX-POINTS.jsonl`: validated q9 rows plus corrected q27 rows;
- `MATRIX-SUMMARY.md`: all 27 cells, ranges, acceptance, prompt tokens, and cached tokens.

## Gates and final receipt

The final code candidate was `d1cea0757c171fb9af1bfd0f0440b59ca316152c`.
Its box1 `memra-server` SHA-256 was
`a065347997225842c55b13ec34c92846c5a3355c82e13a44e74e13aa3ed1f95c`.

| gate | result | receipt |
|---|---|---|
| local `memra-server` tests | PASS, 130/130 | final measured table, pin, placement, concurrency, and replay tests |
| `run-spec` | PASS, K=1..8 (8/8) | `gates/d1cea075/run-spec.log` |
| automatic prompt table | PASS | short `K=3`; cold-long `K=3`; cached-long `K=2` with 5478 cached tokens |
| PP-2 placement | PASS | automatic request selected `K=0` and emitted no spec usage |
| operator precedence | PASS | explicit `MEMRA_SPEC_K=3` overrode automatic PP-2 `K=0` |
| #89 single-card crossover | PASS | c=1/2 selected `K=3`; arrivals 3/4 selected `K=0`; live demotion at HIGH preserved |
| repository `accept-gate` | PASS, 1/1 | default q27-p1 smoke cell, acceptance 67.46%, text SHA identical |
| `serve-smoke` | PASS, 0 failures | `gates/d1cea075/serve-smoke.log` |
| generated perf surfaces | PASS | `python3 tools/update-perf-board.py --check` |

The gate window started with both GPUs idle at 27 C. GPU0 ended at 37 C after the
battery; GPU1 remained idle. Artifact and binary hashes are committed beside the logs.

### Full acceptance diagnostic

The optional six-cell `accept-gate --full` diagnostic was 4/6 on the candidate:
q27-p2 and q9-p2 differed from the pinned references. This is inherited train debt, not
a K-policy regression. The exact assigned base `d43f9e27` reproduced both failures with
identical current summaries, including text hashes and spec counts:

| cell | pinned -> base and candidate |
|---|---|
| q27-p2 | 42/126/85 -> 41/123/87; text `fddd52f5...` -> `906ccda8...`; 128 -> 129 completion tokens |
| q9-p2 | 49/147/79 -> 50/150/78; acceptance 53.74% -> 52.00% |

No references were re-pinned. Candidate evidence is under
`diagnostics/b6b1dbff-full-accept/`; the exact-base reproduction is under
`diagnostics/d43f9e27-full-accept/`.

## Mixed workload

The before/after run compared the exact train binary at `d43f9e27` with the exact
candidate binary at `d1cea075`, N=3 independent server boots per arm, rep-major
alternating order. Each rep counted cold-short, cold-long, cached-long setup and
continuation, plus a staggered c=4 wave. Cache setup traffic was included.

| arm | aggregate tok/s median (range) | c=4 wave tok/s median (range) | wall median |
|---|---:|---:|---:|
| before | 294.06 (293.90-295.51) | 390.51 (388.80-391.15) | 5.033 s |
| after | 293.41 (293.38-294.00) | 390.36 (389.90-391.16) | 5.037 s |

- aggregate throughput: `-0.22%`;
- c=4 wave throughput: `-0.04%`;
- workload wall time: `+0.09%`.

This is a neutral served-throughput result within the observed run range. Both arms
used spec for the four sequential/setup requests and plain decode for all four c=4 wave
requests. The policy therefore preserves the measured crossover while replacing the
separate fixed-K/binary-gate controls with one request-owned decision.

The interleaved window ran from 29-36 C on GPU0; GPU1 stayed idle at 27 C. Raw requests,
server logs, hashes, six point rows, and the generated summary are under
`mixed/d1cea075/`.

## Commit sequence

| commit | increment |
|---|---|
| `2879cd41` | freeze the policy design and measurement protocol |
| `da5c8a48` | add the request-owned K seam and operator pin |
| `f4db90d2` | add the box1 K-matrix harness |
| `47da3098` | use the OpenAI usage/cached-token receipt surface |
| `a25c00ed` | add the target-rig policy battery |
| `6013a7ef` | add the mixed-workload before/after harness |
| `4d94e948` | size the q27 measurement cache for its 317.6 MB snapshot |
| `85283a9d` | commit the complete prompt-class matrix |
| `b6b1dbff` | apply the measured `K=3` cold / `K=2` cached table |
| `d1cea075` | align the required battery with the repository acceptance gate |

Final gate, diagnostic, and mixed-workload artifacts are committed with this report.

## Log

- 2026-08-08: read `CLAUDE.md`; branch clean at `d43f9e27`; lane inbox absent but registry
  entry present.
- 2026-08-08: traced fixed K to `worker.rs::step_session`; traced placement/concurrency
  gate and request-local prompt/cache/tenant/LCP signals at admission.
- 2026-08-08: froze the provisional policy and measurement protocol above before new
  public performance measurements.
- 2026-08-08: completed the q9 five-arm matrix and q27 four-arm spot-check, N=3. The
  matrix refuted cached-long `K=5` and selected the shared `K=3` cold / `K=2`
  cached-long table above.
- 2026-08-08: target-rig battery passed run-spec 8/8, all live policy assertions,
  repository accept-gate, and serve-smoke. The optional full acceptance matrix exposed
  two failures already present at the assigned train base; no golden was re-pinned.
- 2026-08-08: completed the interleaved N=3 mixed workload. Aggregate throughput moved
  -0.22%, c=4 throughput -0.04%, and wall time +0.09%; recorded as neutral.

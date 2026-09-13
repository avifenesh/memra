# Frozen-head row oracle and first utility table — 2026-09-13

The measurements below isolate row selection. The active head stays frozen,
draft depth stays at four, and both confidence cuts stay at zero. No joint
controller or production serving result is claimed.

**Verdict:** repairable omissions exist, but this first learned table does not
outperform frequency on held-out one-step correctness. Both repair two states
without observed harm; the table proposes nine swaps versus frequency's sixteen.
That lower action count is a possible cost advantage to measure, not a demonstrated
runtime benefit. Always choosing the highest-scoring outsider has higher net
one-step gain here (eight repairs minus three harms). No learned-policy win or
novelty claim is supported yet.

## Evidence and scope

- Native source: `b8ea3da20d4e81faf7813644f3dc34521779a977`. Build, formatting and flag checks passed.
- Locked Gemma 4 12B target and matching Q8_0 assistant, native Memra; RTX 5090, 500 W power limit. Qwen passed the correctness pilot but has no row-policy measurement here.
- 21/21 two-model pilot checks passed. The row campaign has 192 successful full-output comparisons and one expected refusal of an unfrozen probe.
- 48 prompts: eight code and eight prose per train/calibration/held-out split; 128 generated tokens per successful run. Related prose task formats cross splits, so this is a synthetic within-family diagnostic.
- All 4,608 head IDs are distinct: 4,096 fixed core plus 512 spare rows seeded from training input tokens. Head ID files, prompt bytes and output identities are audited; head rows come from the hash-locked original assistant weights.
- Probe rounds 1,17,33,...; labels stop at the first rejection and at the output budget. Abandoned draft suffixes receive no labels.
- 391 sampled collection states; 391 pass overlapping-row argmax agreement. Maximum observed overlapping logit error: 0.
- Initial compile failure from an undeclared JSON dependency is retained under `row-oracle-receipts/prior-compile-failure`; the measured build uses dependency-free numeric JSON.

## Frozen-state repair headroom

| Split / domain | States | Wrong | Missing targets | Potential one-swap repairs | Correct states exposed to outsider harm |
| --- | ---: | ---: | ---: | ---: | ---: |
| train-code | 59 | 30 | 4 | 1 | 1 |
| train-prose | 74 | 34 | 25 | 13 | 1 |
| calibration-code | 58 | 26 | 2 | 1 | 0 |
| calibration-prose | 61 | 36 | 23 | 17 | 2 |
| heldout-code | 60 | 29 | 2 | 2 | 0 |
| heldout-prose | 79 | 32 | 21 | 7 | 3 |

Across the three splits, only 41 of 77 missing targets could be repaired by
inserting their row at these sampled states. The other 36 would still lose the
draft argmax: coverage alone overstates useful headroom. There were no observed
removal-only repairs. Most opportunity occurred on prose (37 potential repairs
versus four on code), consistent with this head being populated from code.
This aggregate is descriptive; train and calibration observations are not extra
held-out evidence. The held-out headroom alone is nine potential repairs.

Repair counts are the union of insertion, removal and complete-swap cases.
They use the verifier target as oracle information and numerical margins.
They are estimates until actual same-width replacement replay is qualified.
Sparse reached states are not an unbiased sample of all possible states;
these counts cannot be converted into whole-block acceptance or speedup.

## Offline admission table

The training-only table covers 1472 candidate IDs, with four zero-utility pseudo-exposures per row. Calibration selected threshold **0.1** from the registered grid. Held-out labels do not enter fitting or threshold selection.

The victim is fixed at mutable slot 4096. Candidate discovery uses the expensive
full-vocabulary top-16 probe for every selector. All policies share numerical
eligibility. This isolates admission and does not test learned eviction.

| Held-out policy | States | Actions | Beneficial | Harmful | Net repairs | Prompt-mean correctness change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| no_change | 139 | 0 | 0 | 0 | 0 | +0.0000 percentage points |
| highest_outside | 139 | 139 | 8 | 3 | 5 | +4.0646 percentage points |
| frequency | 139 | 16 | 2 | 0 | 2 | +1.4757 percentage points |
| learned | 139 | 9 | 2 | 0 | 2 | +1.4757 percentage points |

## Diagnostic overhead

48 ON/OFF pairs across eight calibration-code prompts, six balanced repetitions each. The geometric request-time ratio is **1.078269**: **+7.83%** time, with descriptive prompt-cluster bootstrap 95% interval **[1.070066, 1.086752]**. This includes full-head allocation/upload, projection, readback and logging.

Arithmetic mean request time was 575.41 ms with the probe off and 620.25 ms
with it on, an added 44.84 ms per 128-token request in this workload.

The timing is one-shot native greedy request time after model loading. It
does not establish serving throughput, latency tails or concurrent capacity.
The cost of this full probe must not be assigned to a future bounded probe;
that implementation needs its own measured cost.

## Reproduction and next gates

Run `python audit_row_oracle.py row-oracle-receipts --out row-oracle-audit.json`
from this directory. The auditor checks source/binary references, initial head
hashes, prompt hashes, raw logs, complete token arrays, trace hashes, reached
prefixes and fixed settings. `row-oracle-receipt-hashes.json` covers every
banked file; `row-oracle-audit.json` contains the utility table, calibration
grid, per-prompt results and separate held-out code/prose metrics.

Follow [ROW-POLICY-GATES.md](ROW-POLICY-GATES.md): physical replacement, bounded
candidate discovery, admission, eviction, then actual online head-only benefit.
Confidence/depth learning and combinations retain their separate gates.

The immediate implementation priority is a bounded candidate probe and physical
replacement replay, followed by a fresh, family-separated admission test. The
current per-token table does not establish an advantage from learned utility;
do not tune it further on this held-out set or promote it to runtime. The full
probe remains an explicit default-OFF research diagnostic.

All 399 evidence files were banked locally. The rented instance was confirmed
destroyed at 2026-09-13 14:57:19 UTC after a manual takeover from an SSH-timed-out
cleanup guard. No serving deployment, upstream push or combined-policy run occurred.

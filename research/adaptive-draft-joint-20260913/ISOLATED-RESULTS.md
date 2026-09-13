# Independent component measurements

2026-09-13. RTX 5090, 32 GB, host power limit 500 W. Native Memra,
Gemma 4 12B IT QAT Q4_0 plus the locked matching Q8_0 assistant. Greedy,
single-request code review, 128 emitted tokens. This is a mechanism experiment,
not sampled serving performance. No combined policy was measured.

Protocol: `ISOLATED.md`. Eight previously untouched held-out prompt groups,
six balanced AB/BA repetitions each, 48 pairs per component. Both arms start with
the same 4096 ranked rows and 512 duplicate spare rows (4608 physical rows).
Each request gets fresh initial state. Target and draft neural weights are frozen.
Primary result is the geometric paired request-time ratio B/A; bootstrap intervals
resample eight prompt groups after averaging repeated log ratios within each group.

| Isolated change B versus A | Request-time change | Descriptive 95% interval | Prompt groups faster | Status |
|---|---:|---:|---:|---|
| Existing append-only head learning versus frozen rows | -4.41% | -6.15% to -2.58% | 7/8 | Complete, raw audit passed |
| Confidence stopping at 0.7 versus disabled | -4.59% | -6.25% to -2.92% | 8/8 | Complete, raw audit passed |
| Accepted-prefix+1 depth versus fixed K=4 | -4.14% | -5.77% to -2.14% | 7/8 | Complete, raw audit passed |

## Head learning alone

All 96 measured runs, two diagnostics and two warmups passed full output identity.
K stayed exactly 4 and both confidence cuts stayed disabled. The frozen arm learned
zero rows; the learning arm added 72–99 rows/request without changing capacity.
Total emitted tokens per arm: 6144. Total measured request time: 29.007219 seconds
frozen versus 27.758909 seconds learning. Aggregated within-request throughput:
211.81 versus 221.33 tokens/s. This excludes model loading and the plain correctness
reference; it includes prompt learning, generation setup, prime, decode, updates
and end-of-request sidecar persistence within the native generation call.

Accepted/drafted fraction rose from 0.304604 to 0.332207. It is explanatory only.
The six repeated workload ratios were 0.955802, 0.955892, 0.955848, 0.955759,
0.956215 and 0.956005. Within-prompt, within-arm timing ranges were below 0.37%
of the median. Campaign telemetry, including process loading/plain reference,
spanned 45–64 C (median 53 C), SM clocks 2752–2902 MHz (median 2895 MHz).
Those samples are hardware context, not energy attribution to the timed section.

This supports the existing append-only learner on this narrow workload. It does
not establish learned row replacement, neural draft training, cross-request
adaptation, confidence calibration, joint-control novelty or production speedup.
See `NEXT-ISOLATED.md` for separate gates on the proposed learned policies.

## Confidence stopping alone

All 100 runs passed, including 96 measured runs. Both heads stayed frozen at zero
learned rows. Maximum K stayed 4; accepted-length adaptation and next-round p-min
stayed off. Enabling the existing in-round 0.7 threshold reduced mean drafted length
from 4.0000 to 2.4685 and raised accepted/drafted fraction from 0.304604 to 0.420255.
Request time totaled 29.017795 seconds without the cut versus 27.692893 seconds
with it, for 6144 output tokens per arm (211.73 versus 221.86 tokens/s within the
request). All confidence calculation/readback cost is included.

Within-prompt timing ranges were below 0.29% of the median. Campaign temperature
was 46–65 C (median 53 C), SM clock 2677–2902 MHz (median 2887 MHz). This is a
fixed-threshold mechanism comparison, not a trained or adaptive threshold policy.
Neither the head gain nor this gain predicts their combination; gains are not added.

## Adaptive depth alone

All 100 runs passed. The head stayed frozen and both confidence gates stayed zero.
Accepted-prefix+1 adaptation with floor 1 and cap 4 reduced mean drafted length
from 4.0000 to 1.9153 and increased accepted/drafted fraction from 0.304604 to
0.468269. Request time totaled 29.017581 seconds fixed versus 27.814231 seconds
adaptive for 6144 tokens per arm (211.73 versus 220.89 within-request tokens/s).
Within-prompt timing ranges were below 0.28%. Campaign temperature was 45–65 C
(median 53 C), SM clock 2685–2902 MHz (median 2895 MHz).

This existing heuristic is an isolated baseline, not a learned marginal-cost
controller. It is not compared causally with the confidence arm across campaigns;
each effect uses its own adjacent fixed-control measurements.

## Execution checks

Both locked models passed the new-host correctness pilot: 21/21 subprocess checks,
including Qwen K=1..8 and Gemma K=1,2,4,6,8 on three public smoke prompts, plus
Gemma recorder ON/OFF identity. Formatting and the runtime flag census passed.
The first formatting attempt exposed a missing `rustfmt.toml` in the export;
restoring the tracked configuration resolved it without changing engine code.
Native source is unchanged between the initial isolation snapshot `19d4b2c4`
and the pre-run metadata/auditor commit `5c5dd290`. Archive and amendment hashes
are retained with the raw receipts. Infrastructure startup failures are retained
in the private operations record and do not count as model failures.

The three component stages total 300 passing runs: 288 measurements, six traced
diagnostics and six unscored warmups. That is 144 measured A/B pairs. The auditor
checks raw and trace hashes, unchanged binaries, the registered prompt hashes,
rank artifact bytes/order, unique per-request state paths, intended isolated
configuration, full output hashes across components/repetitions, and independently
recomputes each summary. Raw outputs, telemetry, head sidecars and manifests are
under `isolated-receipts/`; recomputed statistics are in `isolated-audit.json`.

The final native source captures match the pinned commit after line-ending
normalization; their exact on-host bytes and SHA256 receipts are preserved too.
Toolchain: rustc/cargo 1.97.1, CUDA compiler 13.1.115. Source, manifests and raw
evidence are committed locally; no serving defaults, production deployment or
public performance announcement followed. Rental destruction was confirmed at
2026-09-13 11:32:53 UTC with zero campaign instances remaining.

Recompute from this directory:

```sh
python audit_isolated.py isolated-receipts --out isolated-audit.json
```

The next phase must still measure each proposed learned policy alone before
pairwise or joint tests. These component gains are neither additive nor evidence
of novelty. None of the current results is a Qwen component or sampled-serving result.

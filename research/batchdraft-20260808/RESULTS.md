# Cross-request batchdraft: box1 measurements

Date: 2026-08-09

Verdict: **large target-verify headroom exists, but concatenating into today's contiguous m=16
verify is a measured regression. Build a true per-cache B x T verifier; do not wire the current
single-sequence T=16 path into serving.**

## Headline

At c=4, target verification accounts for a median **93.72%** of the measured steady speculative
round phase in both the synchronized and divergent request arms. The scheduler is not leaving a
large host-side hole between requests: the median serial handoff is **0.011 ms**, while the median
unaccounted steady call time is 0.156 ms (sync) / 0.132 ms (divergent).

That makes verify batching the right target, but not an automatic win. On the same model and PP2
placement, at a frozen depth of 256, the existing contiguous target path scales as follows:

| Verify width | N | Median | p10-p90 | Cost vs serial m=4 calls | Result |
|---:|---:|---:|---:|---:|---|
| m=4 | 25 | 35,200.1 us | 35,124.3-35,265.7 | 1.000 | Baseline K=3 steady row |
| m=8 | 25 | 66,425.1 us | 66,316.2-66,479.0 | 0.944 | 5.65% less than 2 x m=4 |
| m=12 | 25 | 97,723.1 us | 97,622.4-97,913.3 | 0.925 | 7.46% less than 3 x m=4 |
| m=16 | 25 | 146,162.5 us | 146,083.5-146,408.9 | 1.038 | **3.81% more than 4 x m=4** |

The m=16 p10 is also above four times the m=4 p90 (146,083.5 us vs 141,062.8 us), so this negative
result is not median noise. This probe is a contiguous-single-sequence width proxy, not a B x T
implementation: it measures the current projection/kernel envelope while deliberately not claiming
correct multi-cache attention semantics.

## Serving protocol

- Host: box1 (`<private-host-redacted>`), 2 x NVIDIA RTX PRO 6000 Blackwell Server Edition, 97,887 MiB
  each; PP stages on devices `0,1`.
- Runtime source: `3248e4f91ab8dbe892d7b27c7f0fd30abd2c2009`; release server SHA-256
  `37226dc0c903031d38c0e0a94b169d1e4785c70c60b8be1fa5c9942641d18aff`.
- Model: Step-3.7-Flash IQ4_XS trunk SHA-256
  `b940497a9cec2f801f07e3a9783f2115fd8bf79cbd453225b4f73d86bcd11259`; MTP Q8_0 SHA-256
  `469a81667a6cd6d87a85d501d57155fd90cee5af7010fd289c5169881763fd57`.
- Runtime: `MEMRA_CTX=4096`, `MEMRA_PP_STAGES=2`, `MEMRA_PP_DEVICES=0,1`,
  `MEMRA_SPEC_GATE=0`, fixed `MEMRA_SPEC_K=3`, `MEMRA_SPEC_BURST=32`,
  `MEMRA_TICK_TRACE=1`, `MEMRA_SPEC_PHASE=1`.
- Load: c=4 greedy requests, 96 requested output tokens, isolated cache salts. `sync` uses four
  identical prompts; `divergent` uses four distinct suffixes. One unscored c=4 warmup preceded five
  scored point replicates per arm. Arm order alternated by repetition; cooldown was one second.
- N: **5 independent c=4 point replicates per arm**. The per-request breakdown below describes 20
  nested requests per arm and is not presented as N=20 independent experiments.
- Thermal regime: warming, not fixed-clock. Serving moved from 26/27 C before load to 35/36 C after
  teardown. During the sampled window, median SM clocks were 2362/2317 MHz. The width sweep was
  also warming (27/28 C to 33/34 C), and every repetition alternated forward/reverse width order.
- Storage: box1 has no `/scratch`; the pinned source artifacts were on `/dev/root`. The release
  target was on local NVMe at `/opt/scratch/nvme`. The model was fully resident before scored decode,
  so source storage affects startup, not these decode windows; this is not a spill benchmark.
- Lock: serving held `/tmp/memra-gpu.lock` from 21:01:01Z to 21:03:43Z; m-scale held a separate
  block from 21:04:56Z to 21:05:43Z. Both release receipts and zero-memory post states are retained.

The server completed all 40 scored requests without a request error. Fixed speculative bursts can
commit past the client cap, so totals were 392 tokens/point in `sync` and 387 in `divergent`; all
throughput and projections use the actual returned counts. The valid serving wrapper exited after
the scored block because an obsolete post-run grep expected an automatic gate-policy message that
an explicit K pin suppresses. Every measured `[tick-spec]` receipt says `k=3`; the harness now checks
that receipt directly.

## What one request spends

These are sums across each request's three speculative calls, then medians across the 20 requests
nested in the five point replicates. `MEMRA_SPEC_PHASE` uses existing synchronization points; the
categories are wall attribution around GPU work, not isolated kernel-duration events.

| Arm | Rounds/request | Draft | Verify issue + wait | Commit/other | Verify share |
|---|---:|---:|---:|---:|---:|
| sync | 46 (45-46) | 99.1 ms (96.8-100.2) | 1600.1 ms (1564.2-1618.6) | 8.2 ms (7.9-9.6) | 93.72% |
| divergent | 42 (40-45) | 90.7 ms (86.1-97.2) | 1467.3 ms (1387.5-1574.2) | 7.5 ms (6.8-9.0) | 93.72% |

The request-phase total excludes prompt/session setup; the client projection below holds that setup
and every non-verify cost at its measured value.

## GPU activity and gaps

| Arm | GPU0 mean activity | GPU1 mean activity | Joint-zero samples | Longest joint-zero run | Scheduler handoff |
|---|---:|---:|---:|---:|---:|
| sync | 40.83% | 49.71% | 1.90% | 200 ms median, 300 ms max | 0.011 ms median |
| divergent | 41.27% | 50.18% | 1.96% | 200 ms median/max | 0.011 ms median |

Values are medians across N=5 points. `nvidia-smi utilization.gpu` is the percentage of its
vendor-defined sample period in which one or more kernels executed; it is **not SM occupancy**.
NVIDIA also documents a product-dependent underlying period, so the requested 100 ms polling does
not make this a sub-millisecond gap detector
([NVIDIA nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/index.html)). The
joint-zero result is therefore a coarse activity receipt. The trace's monotonic-clock handoff is the
stronger evidence that the worker is not pausing between serial session calls.

Interpretation: the headline opportunity is shared weight work / better matrix geometry, not the
removal of a millisecond-scale scheduler sleep. The two PP stages remain only partly active under
single-request verify streams even though requests are handed off immediately.

## Are four rows available together?

Grouping the observed rounds by scheduler tick gives the live-row width that a round rendezvous
could have used:

| Arm | B=4 waves | B=3 | B=2 | B=1 |
|---|---:|---:|---:|---:|
| sync | 225 | 4 | 1 | 0 |
| divergent | 198 | 2 | 18 | 10 |

The synchronized arm is nearly rectangular. Divergent requests still offer B=4 on 198/228 observed
waves (86.8%); shrinking tails need grouping or a smaller-width fallback, not global padding or a
global minimum-accept rollback.

## Headroom calculation

For each observed live wave, the analysis replaces only its measured target-verify wall. Draft,
setup, host commit, scheduling and output are held constant. Let `f` be verify's fraction of the
steady phase and `r` the candidate fused/serial verify-cost ratio:

```text
steady-phase speedup = 1 / ((1 - f) + f * r)
projected client wall = measured client wall - verify_wall * (1 - r)
```

| Arm | Measured client tok/s | Current flat-width proxy | Ideal one-m=4-per-live-wave ceiling | Zero-verify ceiling |
|---|---:|---:|---:|---:|
| sync | 37.26 | 36.45 tok/s, 0.967x phase (**-2.16%**) | 68.45 tok/s, 3.351x phase (**1.84x client**) | 95.00 tok/s, 15.92x phase |
| divergent | 38.01 | 37.28 tok/s, 0.970x phase (**-1.93%**) | 65.70 tok/s, 3.147x phase (**1.73x client**) | 90.29 tok/s, 15.92x phase |

The ideal column is the answer to “what if four requests' verifies cost one request's verify?” for
the actually observed live widths. It is an upper envelope, not a forecast. The measured current
flat-width column answers a different but essential question: “what happens if we merely concatenate
the tokens and reuse today's width tier?” It regresses.

Therefore stage 2 needs a true multi-sequence verifier and an exact, performant M=16 numeric tier.
It should be rejected if its measured cost resembles the current contiguous m=16 proxy; the large
ideal ceiling is not permission to ship a slower implementation.

## Reproduction and raw evidence

Analysis:

```bash
python3 research/batchdraft-20260808/analyze.py \
  research/batchdraft-20260808/raw/box1/client-20260808T210100Z.jsonl \
  research/batchdraft-20260808/raw/box1/server-20260808T210100Z.log \
  research/batchdraft-20260808/raw/box1/gpu-serving-20260808T210100Z.csv \
  research/batchdraft-20260808/raw/box1/verify-mscale-20260808T210500Z.log
```

Authoritative scored receipts are the `210100Z` serving files and `210500Z` m-scale files under
`raw/box1/`. `205600Z` was a setup-only attempt stopped before load. `205800Z` is retained but
discarded: requests ran, while the first client parser expected an OpenAI response and recorded the
quoted `KeyError: 'choices'`; it has no valid scored client token receipts. The subsequent native
response parser produced the valid block. No OOM, CUDA error, panic, illegal-address or Xid event is
present in the valid logs.

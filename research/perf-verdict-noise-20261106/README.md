# Perf verdict noise floor (2026-11-06)

**Verdict:** the cell-evidence perf verdict (`tools/cell_evidence.py judge_perf`) now needs
complete separation (every candidate run slower than every base run) **and** a median drop of
at least 2% (`PERF_MIN_REGRESSION_PCT`). Rank alone flags identical code on about 1 model in 20
and about 1 two-model head in 10. With the 2% floor that falls to 0.6% of heads on the RTX 5090
laptop and none on GB10. A true 3% regression is still caught 83% of the time on the 5090 and
always on GB10.

## Why

The first GB10 daily regression run showed that the rank rule has no minimum effect. With three
runs per arm, complete separation happens with one-sided exact probability 1/20 when the two arms
run the same program. With two roster models, about 1 head in 10 fails. The daily run keeps its
last passing head as the perf base (`tools/daily-regression.py`), and a head already tested is
skipped. So a false alarm keeps the device failing and its issue open until main moves.

## Data

`runs.jsonl` holds the per-run decode rates (`run-gen`, 128 tokens) extracted from every banked
record with perf runs:

- the 6 records under `research/release-qualification/cells/` (4 on the RTX 5090 laptop, 2 on
  GB10; 5 of them are interleaved A B B A A B against their push base);
- the first gb10 daily record, `research/daily-regression/gb10/2026-11-06-516ec03e946c/` on the
  `regression/gb10` results branch (commit `bc7dd64023`), three reference runs per model.

Reproduce:

```sh
git archive origin/regression/gb10 research/daily-regression/gb10/2026-11-06-516ec03e946c | tar -x -C "$DIR"
python3 noise.py extract ../release-qualification/cells/*/ "$DIR"/research/daily-regression/gb10/2026-11-06-516ec03e946c > runs.jsonl
python3 noise.py analyze runs.jsonl > noise.out
```

## Method

Each within-arm triple is three runs of one binary in one interleaved window. Its residuals
`x/mean - 1`, scaled by `sqrt(3/2)` to undo the shrinkage of centering three values, are draws of
that device's per-run noise. Resampling six of them as the A and B arms of one program gives the
false-alarm rate on identical code. Shifting the B draws down by R% gives the power for a true R%
regression. Seeded; 400k resampled A/Bs per false-alarm cell, 100k per power cell (`noise.out`).

## Results

| | GB10 | RTX 5090 laptop |
|---|---|---|
| triples | 8 | 16 |
| per-run relative sd | 0.23% | 0.99% |
| largest single-run deviation from its arm mean | 0.50% | 2.48% |
| false alarm per model, rank alone | 4.1% | 4.7% |
| false alarm per two-model head, rank alone | 8.0% | 9.2% |
| false alarm per two-model head, rank + 1.5% | 0 | 1.5% |
| false alarm per two-model head, rank + 2% | 0 | 0.64% |
| false alarm per two-model head, rank + 3% | 0 | 0.13% |
| power at 2% for a true 2% / 3% / 5% regression | 53% / 100% / 100% | 42% / 83% / 99% |
| power at 3% for a true 3% / 5% regression | 54% / 100% | 48% / 98% |

The 10 recorded base vs candidate pairs (all perf-neutral changes) have median deltas from
-1.14% to +1.34%. None is separated.

Chosen: 2%. Rejected: rank alone (the bug); 1.5% (1.5% of 5090 heads still fail on noise); 3%
(misses about half of true 3% regressions on both devices). One floor serves every device: GB10's
noise would allow 1%, but per-device floors would add a device-row field for a gain only on the
quieter box.

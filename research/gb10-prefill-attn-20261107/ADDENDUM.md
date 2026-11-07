# Addendum (after the landing of `9805016bb1`)

The per-cell evidence pins the exact source inputs of the landed head, so the follow-up review's
corrections and the second-unit confirmation come in this research/docs-only commit after it.

## Corrections

- The unit-1 final battery (`gates-unit1-final/`) ran `harness/lane_seq-unit1.sh`: its prime-path
  logits step ran the base binary, the lane binary and the lane binary with
  `MEMRA_PRIME_KV_H2P=0` (B / C / Coff). The base binary predates `MEMRA_PP_PRIME`, so its rows are
  `forward_last` and the "B vs C vs Coff: DIFFER" line compares two different passes; the valid
  compare is C against Coff (byte-identical at both lengths). `harness/lane_seq.sh` is the later
  version used on unit 2: it compares h2p, h2 and db inside the lane binary (C / Coff / Cdb) and
  checks the lane out by its resolved SHA. Unit 2 needed two restarts first
  (`gates-unit2-final/failed-starts/`): the first logged "bundle fetch failed", the second built the
  base and then stopped at `git checkout --detach attn-lane`, where git answered "--detach does not
  take a path argument 'attn-lane'" (seen in that start's console output, which was not kept), so the
  ref was gone by then; why it went is not known.
- README, round 4: the per-warp QK pipelining (pqk) loses 33.6% at depth 131,072 and 24.9% at
  4,096 against the shipped h2p; the README said 26% and is corrected here (the decision record's
  -34% was right).
- `docs/KERNELS.md`, `fa_prefill_qw_h2` row: the door column should still say the h2 route is on
  by default only on a unified-memory 120a device (GB10), where it now runs h2p. `docs/FLAGS.md`,
  `MEMRA_PRIME_KV_H2P` row: its gates also cover the fresh 1,000 x 1,000 and partial-CTA
  1,000 x 1,031 / 4,075 x 8,171 ARC B cells, `tests/prime_kv_h2_gpu.rs` in both modes and the nsys
  kernel names. The KERNELS row is corrected here; `docs/FLAGS.md` is compiled into the binaries'
  environment registry, so an edit to it reaches the numeric and perf cells and needs fresh cell
  evidence: the fuller gates list rides the next change that touches that file.

## Confirmation on unit 2 (`served-unit2-final/`, `gates-unit2-final/`)

Same cells as unit 1 (B H H B B H, one fresh boot per cell, main `6f6d816757` against the lane
binary `6f59cac641`, the same code as the landed head), after the same battery on that unit:

| prompt tokens | main (h2) TTFT | h2p TTFT | median change |
|---:|---|---|---:|
| 32,784 | 32.461, 32.503, 32.493 s (median 32.493) | 31.902, 31.906, 31.917 s (median 31.906) | -1.81% |
| 131,051 | 166.639, 167.679, 167.666 s (median 167.666) | 157.720, 157.377, 157.962 s (median 157.720) | -5.93% |
| 259,951 | 433.586, 430.714, 430.495 s (median 430.714) | 393.340, 392.050, 393.173 s (median 393.173) | -8.72% |

Every h2p boot is below every main boot at every length; answers identical in all six boots and to
unit 1's; no request error. MemAvailable floor 66.16-66.34 GiB except cell 03-H at 63.74 GiB (the
two other h2p cells sit at 66.34 and 66.16, so the dip is not the arm; the box's host memory is
shared and the cause is not known). Battery on unit 2: kernel-check ALL GREEN in h2p, h2 and db
modes with all nine ARC B cells bitdiff=0, argmax-margin PASS, run-spec K=1..8 PASS,
`tests/prime_kv_h2_gpu.rs` passing in both modes, nsys naming `fa_prefill_qw_h2p` / `fa_prefill_qw_h2`,
and the prime-path logits of h2p, h2 and db byte-identical at 16,219 and 33,890 tokens (the db arm
the unit-1 run lacked). Probe round 4 replicates on unit 2 (`probe-runs/probe-r4-unit2.log`: h2p
84.60 / 84.75 against h2 67.68 / 67.69 TFLOPS at depth 131,072) with the ragged shapes bit-identical
(`probe-runs/edge-h2p-unit2.log`).

## Raw evidence kept off the repo

The ncu reports (`*.ncu-rep`, raw CSV exports) and the full per-unit run, gate and log directories
from both units are in the lab's evidence backup as snapshot `5e4f0ee7` (87 MiB); their summaries,
logs and per-cell files are the receipts in this directory.


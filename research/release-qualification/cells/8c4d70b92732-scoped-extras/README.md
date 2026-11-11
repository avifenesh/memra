# Scoped extras on the landing build of 8c4d70b92732

Receipts beside the records of 8c4d70b92732 (the landing of memra#1229, memra#1228, memra#1230 and memra#1224, PR memra#1231). They are not cell records: no gate reads them. The landing's final commit is b8ba35eb3aef: it adds only a test fixture re-pin (`research/spill-c-20260919/fixtures/slru-synthetic.json`, its `source_sha256` line) on top of 8c4d70b92732, so these extras were not repeated there; the cell records were. They cover the one place two PRs of this landing meet: memra#1230 closes a pooled cohort member at the round its stream ended, and memra#1224 lets a sampled burst carry its pending token, so a sampled member that closes early now has a carried token, as greedy members already did.

Both ran on the landing's own release binaries, built by `tools/cell_evidence.py` for the records (`memra-server` sha256 prefix `aaabfb3be48a7ca3` for sm_120a, `4a9689d045f79ea0` for sm_121a). GB10 ran them in its record's lease, after the record; the RTX 5090 in a lease of their own, right after its records. Model: Qwen3.8-27B NVFP4 + Q5_K with its MTP head. `extras.sh` is the driver. The gate's answer texts (`cohort/texts/*.txt`, some ending in a space where a stop string cut them) are kept byte-exact as one JSON object per device, `cohort/texts.json`, keyed by file name.

## memra#1230's cohort gate

`tools/spec-stop-string-gate.py --cohort 2,4`: a plain reference boot (`ref`), then the landing binary on the MTP route (`land`, `--settle`), then `judge --case land`. Every boot runs greedy and temperature 1 (seeded).

| device | verdict | member 0 close ms, p50 / max, cohort of 2 | cohort of 4 | alone | member 0 finished inside its cohort |
|---|---|---|---|---|---|
| RTX 5090 Laptop, t1 | PASS | 1 / 90 | 2 / 146 | 0 | 18 |
| RTX 5090 Laptop, greedy | PASS | 1 / 170 | 2 / 283 | 27 | 17 |
| GB10, t1 | PASS | 2 / 118 | 6 / 167 | 0 | 18 |
| GB10, greedy | PASS | 0 / 208 | 2 / 203 | 0 | 13 |

Every stopped member's text, finish reason and usage equal plain decode, every companion wrote one text equal to plain decode in every cohort, and the gate printed `STOP-STRING COHORT GATE: PASS` on both devices (`*/cohort-judge.log`).

## memra#1224's end-of-request cells

`research/spec-sampled-pending-carry-20261110/tools/close_cell.py` at temperature 1, `MODE=close` and `MODE=demote` (`MEMRA_SPEC_DEMOTE_AT=40`), with that lane's stop strings. `close_summary.py` judges each row against the lane's candidate rows of the same device.

| device | close | demote |
|---|---|---|
| RTX 5090 Laptop | rows: 40, with a judged difference: 0 | rows: 16, with a judged difference: 0; 12 demotion handoffs, 0 abort lines |
| GB10 | rows: 40, with a judged difference: 0 | rows: 16, with a judged difference: 0; 12 demotion handoffs, 0 abort lines |

Each row's text, finish reason, token counts and rounds equal the lane's candidate row; the per-row close times are in `*/close-close-judge.log` (sampled rows 0.03 to 53.7 ms on the RTX 5090 and 0.09 to 102.1 ms on GB10, the larger ones on rows where the lane's candidate shows the same time). The close log's 8 matches of `abort` are the cell's cancelled rows, the same count as in the lane's candidate log.

# Exclusions and retained diagnostics

The selection ledgers are authoritative; do not glob every archived run into a score.

- `qwen-mtp-short-v1`: failed greedy gate using conditional cache reuse. Turn 3
  differed when one policy reused state and another primed cold. No scored rows.
- `qwen-mtp-sampled-v2`: GPU interference during the first set; no selected rows.
- `qwen-mtp-sampled-v3`: cycles 0..2 are selected. Cycle 3 is excluded in full,
  including its completed arms, after interference during the measured-native arm.
- `qwen-mtp-cycle-7-attempt-1-v4`: the whole set is excluded after GPU interference
  during native adaptation. Attempt 2 repeats the original seed and order and is selected.

These exclusions are based on failed gates or recorded GPU interference, never on
speed. Raw logs, telemetry and `.contamination.txt` files remain in the archives.
All other selected cycle directories are named by `qwen-selected-sets.json` and
`gemma-selected-sets.json`. The phase replay and report calculations use only those.

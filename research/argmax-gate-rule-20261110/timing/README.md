# Timing cells

The 27B eager cell on the RTX 5090 Laptop GPU (95 W limit) at two window sizes, to size the
whole-prompt calibration cells. Main's unmodified probe (031f146690). Not floors: a floor is
measured over the whole prompt.

| window | wall | source |
|---|---|---|
| 128 rows | 342 s | `rtx5090-laptop-q38-eager-128/result.txt` |
| 512 rows | 1,051 s | file times: the cell started when the 128-row cell's `result.txt` was written (16:21:46.6) and its `probe.log` was complete at 16:39:18.0 |

GB10, the same cell at 128 rows: 418 s (`gb10-q38-eager-128/result.txt`, the recorded sm_121a
build of 031f146690).

Both RTX 5090 tables come from one binary: the 128-row table and the last 128 rows of the
512-row table are byte-identical (`0dcbb5bbd14536ba`).

The 512-row cell's `result.txt` has no exit line. Its runner script was edited while it ran and
the shell lost the script's last lines after the probe had finished. The table is whole (512
rows, the probe's closing lines present).

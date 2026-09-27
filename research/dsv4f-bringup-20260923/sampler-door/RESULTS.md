# DSv4 plain sampler door: device against host (MEMRA_DSV4_SAMPLER)

Scope: one model, one hardware shape. `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`
on 2x RTX PRO 6000 Blackwell Workstation Edition, 2026-09-24. Served plain route (PP-2, matrix
program, one serving lane), main `25bbb91f5`, one binary for both arms. Order h d d h h d d h h d
(N=5 each), one boot per row under `/tmp/memra-gpu.lock`. Cells: greedy c1 x8, sampled c1 x8,
greedy ignore-eos x4. Raw rows: `raw/sampler/`, script `raw/q-pair6.sh`.

| cell | host (default) | device | delta |
|---|---|---|---|
| greedy c1 decode p50 | 50.92 | 50.93 | +0.02% |
| sampled c1 decode p50 | 46.19 | 50.73 | **+9.85%** |
| sampled c1 TPOT p50 | 21.65 ms | 19.71 ms | -1.94 ms |

Text: every request's hash is identical across all ten rows of both arms, greedy and sampled.
The device class (`device-f64-exp-tree-cdf-v1`) can diverge from the host at rounding
boundaries, but no draw in this corpus fell on one. The host sampler costs about 1.9 ms per
sampled token: the full logits readback plus the CPU softmax, filter and draw.

## Decision

The device sampler wins at c1 and matches the host text here. It does not become the default
yet. Under the two-lane pipelined route of #699, the device sampler holds the launch turn for a
whole request, while a host-sampled plain step gives the turn up during its readbacks. Sampled
c2 there serves 72.48..72.75 tok/s aggregate on host sampling (arm A, three rows, of the B-row lane's WS campaign on the #699 lane binary, banked here as `raw/pipelined-host/`), and a turn-holding
route would fall back to serial speed. The flip waits for the device sampler to join the split
step: enqueue the sampling kernel with the step and land one u32 per stage event. That work is
tracked with #699. The door's decide-by moves to 2026-10-08, with this receipt.

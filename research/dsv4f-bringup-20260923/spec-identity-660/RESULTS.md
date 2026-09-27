# DSv4 spec == plain on the served program (memra #660)

Scope: `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`, 2x RTX PRO 6000
Blackwell Workstation Edition, PP-2 served program (matrix experts, EP off, chunked prefill and
prime), DSpark drafter. One gate at a time under `/tmp/memra-gpu.lock`.

## Cause

`MEMRA_DSV4_HC_DOT_SPLIT` (S16, default ON, declared `AllRoutedShapes`) engaged in
`hc_pre_batch_dev` only when `t == 1`. Decode rows took the HC24 split class; a T=k+1 verify row
and every prefill row took sequential `dots_f32acc_mrow`. The two classes differ by association,
so a verified token did not carry the bits of the same token decoded alone, and served DSpark
greedy diverged from plain greedy on every prompt.

## Bisect

`dsv4-gpu-dspark-gate` with `MEMRA_DSV4_GATE_BITONLY=1` (verdict (c): batched T=k+1 verify logits
and every live cache class after commit, raw bits against sequential decode, 14 cells over every
compressor phase and accept count). Binary `b29a19e1...`, source main `5f1b0eda4` plus the gate
port.

| arm | program | verdict (c) |
|---|---|---|
| `dspark-gate-served-r1` | served defaults | FAIL, 2772 findings; verdict (d) 3/3 ring classes mismatch; DB identity fails at generated index 19 (9194 vs 9000) |
| `bis1-hist` | historical pins (HC split 0, dense-fast 0) | ALL BIT-IDENTICAL |
| `bis1-hc0` | served defaults, `MEMRA_DSV4_HC_DOT_SPLIT=0` | ALL BIT-IDENTICAL |
| `bis1-hc0dt0` | served defaults, HC split 0, `MEMRA_DSV4_DENSE_EXACT_TAIL=0` | ALL BIT-IDENTICAL |

Turning off only the HC split restores bit identity with dense-fast and the dense exact tail
still ON: those M1 twins already share the multi-row leaf order and tree.

## Fix

The split kernels take a row count: token rows ride `blockIdx.y` (partial) and `blockIdx.x`
(reducer) through the unchanged kernel body, and `hc_pre_batch_dev` dispatches the split for
every `t`. The verify workspace scratch grows to `tmax*24*32`. Decode (M1) launches the same
geometry as before.

## Fixed binary

`fix-gate-served-r1`: gate binary `c7952f84...`, served defaults, full run.

- verdict (b): greedy spec == plain, 160/160 tokens literal identity, sequential and batched arms.
- verdict (c): 14 cells, 77 logit rows, 3206 cache-class comparisons, ALL BIT-IDENTICAL.
- verdict (d): 3 ring classes, 0 mismatching.
- `GPU DSPARK GATE [PASS]`.

Fixture: `dspark-fx-tape416.json` (416-token prompt from the gate source tape).

## Served cells on the fixed binary

> **Rates invalid (2026-09-23):** measured on the power-braked pair (`../power-brake/POWER-BRAKE.md`).
> The identity verdicts stand; the healthy-pair served cells are in `../REBASELINE.md`.

memra-server from the fix tree (`spec-fix-r1/binary.sha256`), same bench, prompts and box as the
pre-fix baseline (`research/dsv4f-bringup-20260923/BASELINE.md`). Single runs, 256 max tokens,
c1, streaming, usage-authoritative counts, 250 ms telemetry.

| arm | cell | ok | decode tok/s (1/TPOT p50) | TTFT p50 ms | pre-fix decode tok/s | pre-fix TTFT p50 ms |
|---|---|---|---|---|---|---|
| plain | greedy c1 | 8/8 | 15.86 | 537 | 15.86 | 549 |
| plain | sampled c1 | 8/8 | 15.28 | 539 | 15.28 | 552 |
| plain | greedy c1 ignore-eos | 4/4 | 15.85 | 538 | | |
| DSpark | greedy c1 | 8/8 | 22.07 | 695 | 22.00 | 707 |
| DSpark | sampled c1 | 8/8 | 19.26 | 710 | 18.23 | 720 |
| DSpark | greedy c1 ignore-eos | 4/4 | 22.02 | 695 | | |

Greedy text sha256 (first 16 hex), fixed binary:

| idx | plain | DSpark | pre-fix plain |
|---|---|---|---|
| 0 | aea6e69ed4151b24 | aea6e69ed4151b24 | 6bd935354b38b161 |
| 1 | 26ac8df7db88c947 | 26ac8df7db88c947 | 6bd092ba5334120c |
| 2 | 850f75ed4e3adb23 | 850f75ed4e3adb23 | 65399f0a59f8d1e5 |
| 3 | a7784b9a5589ca05 | a7784b9a5589ca05 | 9d5751569fc92bb8 |
| 4 | 7568f9b35944585c | 7568f9b35944585c | 437790e6b2d71a7e |
| 5 | 9dd1aedd90dcbced | 9dd1aedd90dcbced | 052f1203e9050edb |
| 6 | 4fd72390cf1cd748 | 4fd72390cf1cd748 | 846790995da33c6a |
| 7 | 937f04d8eb280816 | 937f04d8eb280816 | a0efb32a48373fcf |

The 4 ignore-eos requests repeat prompts 0..3 with the same hashes in both arms. Served DSpark
greedy now equals served plain greedy on every request.

Plain greedy text moved against the pre-fix plain: prefill rows now take the HC24 split class,
so the prompt KV and hidden lineage carry different (split-class) bits. Decode speed is unchanged;
the plain TTFT drop of 12 ms is inside single-run noise and is not claimed.

DSpark sampled c1 moved from 18.23 to 19.26 tok/s. Single runs on each side, so this is not
claimed as a speedup without an A/B.

## HC split gate on the fixed tree

`dsv4_hc_dot_split_gate --defaults` (TP/EP one-token decode, graph replay, the gate that owns the
HC split door) on the fix tree plus #657 (rebuilt source tape) plus the census repair below.
Binary `a15d7e05...`, receipts `hc-gate/`.

The first run on this tree (`hc-gate/stale-490-*`, binary `e528f736...`) failed at gate line 169
in both arms: `left: 0 right: 43`. The census still expected 43 fused
`dsv4_norm_rope_f32_fixed_order_kernel` nodes per forward segment, and memra #490 had deleted that
kernel with its door. Its norm and rotation now run as the separate rmsnorm and rope nodes, so the
census moved with it:

| node, per rank | segment 0 | segment 2 | segment 3 | segment 1 (head) |
|---|---|---|---|---|
| fused norm/RoPE, old census | 43 | 43 | 43 | 0 |
| fused norm/RoPE, now | 0 (deleted) | 0 | 0 | 0 |
| rmsnorm, old census | 86 | 128 | 148 | 1 on rank 1 |
| rmsnorm, now | 129 | 171 | 191 | 1 on rank 1 |
| rope, old census | 107 | 107 | 107 | 0 |
| rope, now | 150 | 150 | 150 | 0 |
| HC split partial / reduce (arm ON) | 86 / 86 | 86 / 86 | 86 / 86 | 0 / 0 |

Each of the 43 fused nodes became one rmsnorm node and one rope node (+43 each). The new counts
are read from the captured DOTs (`hc-gate/*/qual-graphs.tar.xz`, full manifest in
`dots.sha256`).

| arm | policy | verdict | identity steps | refusals | decode tok/s, 5 rows |
|---|---|---|---|---|---|
| unset | `hc_dot_split=true slices=16` | PASS | 256 | 8 | 21.64 to 21.75 |
| `MEMRA_DSV4_HC_DOT_SPLIT=0` | `hc_dot_split=false slices=0` | PASS | 256 | 8 | 21.28 to 21.40 |

This gate pins its own program (TP/EP, device sampler, small-kernel diet) and makes no
performance claim; the rates are listed only as run records, taken on the braked pair. It does not exercise the multi-row
split that #660 adds. The multi-row path is covered by the DSpark gate verdicts above and the
served cells.

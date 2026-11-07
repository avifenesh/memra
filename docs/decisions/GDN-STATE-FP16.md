# fp16 GDN recurrent state (2026-11-08)

**Status:** rejected on the quality gate; the default stays fp32 on every device. Implementation
and receipts live on the research branch `research/gdn-state-fp16`
([`research/gdn-state-fp16-20261107/`](https://github.com/avifenesh/memra/tree/research/gdn-state-fp16/research/gdn-state-fp16-20261107)),
not on main. Follows [GDN-STATE-BF16.md](GDN-STATE-BF16.md).

## Question

Both bf16 forms of the Qwen3.8-27B GDN state lost long HumanEval (155/164 against a same-binary
fp32 158) to the per-token rounding of decode and verify. fp16 keeps the bytes and rounds about 8x
finer (10 mantissa bits against 7), but its range is 65504 at the top with subnormals below
6.1e-5. Does the state fit, and does the finer rounding keep the panel?

## Step 1: range (the census)

`MEMRA_GDN_STATE_CENSUS` copied every live session's 48 x 48 x 128 x 128 state to the host between
server ticks on GB10: 310 samples over the 131,550-token panel prime, its decode, 90 panel task
sessions, and a cold 130,801- and 257,766-token prime.

- Largest |x| 101.7 (layer 0; every other layer below 7.8): 644x headroom, no overflow. It does
  not grow with depth past 110K tokens.
- 10.6% of elements sit in fp16's subnormal range and 1.2e-4 would flush to zero (each at most
  3e-8, so less than 1e-7 of any head's energy). The worst head carries 6.8% of its energy below
  2^-14 (head RMS 8.6e-5).
- One fp16 rounding: relative L2 error per layer median 2.07e-4, worst head 4.13e-4. One bf16
  rounding: median 1.65e-3, worst head 3.37e-3. bf16/fp16 per head: median 8.0, min 5.8.

fp16 holds the state; no scale is needed.

## Step 2: the fp16 decode class

The bf16 decode class (fp32 prime and captures, one rounding at decode start, 16-bit decode and
verify with per-token rounding) with fp16 storage. Gates on GB10: fp32 class byte-identical to
main (prefill logits and run-gen numeric lines), kernel-check 369 OK including the nine 16-bit
cells against the fp16 oracle bit for bit, run-spec K=1..8 identical to plain in both classes,
restore == unrestored and the packed verify == chained rows in the GPU tests, both classes.

| run (same binary, fresh boot each) | long HumanEval | long GSM8K |
|---|---:|---:|
| fp16 decode class, run 1 | 156 / 164 | 97 / 100 |
| fp32 control | 158 / 164 | 97 / 100 |
| fp16 decode class, run 2 | 156 / 164 | 97 / 100 |
| bf16 decode class (earlier lane, two runs) | 155 / 164 | 97 / 100 |

Both fp16 runs gave the same 264 answers byte for byte, and the fp32 control matched the bf16
lane's fp32 control answer for answer on another binary: the spread is 0. fp16 fails HumanEval/91
and /129 against the control and gains nothing (p = 0.50, 2 vs 0). bf16 failed those two and /120.

## Decision

Rejected under the gate set for the arm: both runs in band (156 and 97 are at the band floor)
and not below the same-binary control by more than the measured spread. They are 2 tasks below
against a spread of 0. Nothing lands on main; the decode tick and TTFT cells were not run.

What the result says beyond the verdict: fp16 changes as many answers as bf16 (45 against 43 of
264) and loses one task fewer. On this deterministic panel, rounding 8x finer does not keep the
greedy trajectories on fp32's path any longer. It moves which answers diverge, and the gate cannot
tell a real cost from a fixed unlucky sample at 2 tasks.

## Rejected alternatives and open doors

- **bf16 in either form**: GDN-STATE-BF16.md.
- **A per-layer or per-head scale for fp16**: not needed. Nothing approaches the range limit,
  and the small heads already round at most about twice the normal-range error.
- **Not built:** deterministic stochastic rounding keyed on (layer, head, element, position). It
  keeps decode == verify, and it is unbiased where round-to-nearest stalls a slowly decaying
  element. Also not built: an fp32/fp16 mixed class with fp32 in the early layers that hold the
  smallest states.
- **A larger or sampled long-context quality measure** could resolve a 2-task difference. That
  changes the gate, which is the owner's decision.

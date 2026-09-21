# Lockstep: why a stream's bytes depended on its peer count (#577)

Verdict: two row-count-dependent programs sat in `moe_ffn_lockstep`, neither of them the
router (#565) and neither the gathered `m_e`-row expert call. The shared expert ran as one
`mrows`-wide matmul, and the CPU companion's multi-row ABI re-split a row's expert sum across
per-expert tickets. With the shared expert per row and the CPU experts on the one-job-per-row
program, `run_lockstep` M=4 reproduces the M=1 logits bit for bit over 33 steps on Hy3, same
prompt or four distinct prompts. Instruments and the probe matrix are below.

## Rig, artifact, binaries

One rented RTX PRO 6000 Blackwell WS (97887 MiB, driver 595.84), Ryzen 9 9950X, 188 GB RAM,
CUDA 13.1.2, 2026-09-21. `Tiyuvta/Hy3-NVFP4@0af425172b7a`, SHA256SUMS verified (README.md is
the only mismatch and is not a weight). Single-GPU frozen-residency regime, native CPU expert
companion, `NVIDIA_TF32_OVERRIDE=0`, greedy, 32 new tokens, one process at a time behind the
GPU lock. Stream 0 is always P0 ("Explain speculative decoding briefly.", 20 tokens); the mixed
file adds three prompts of 25, 23 and 26 tokens. Drivers `raw/ab577*.sh`, binaries and commits
in `raw/*/binaries.sha256`, `raw/*/commits.txt`.

| Binary | Tree | Dir |
| --- | --- | --- |
| probe | main `ea08bc7f8` + instruments (logits dump, MoE input trace on the lockstep path, two doors) | `raw/ab577/` |
| probe2 | probe + shared expert per row | `raw/ab577b/` |
| probe3 | probe2 + CPU experts one job per row by default, split door removed (this lane's head) | `raw/ab577c/` |

## Instruments

- `MEMRA_LOCKSTEP_LOGITS_DUMP`: stream 0's full logits row after the prime and after every
  step, raw f32. `compare_logits.py` diffs two dumps bitwise per step (first differing step,
  max |d|, argmax, top-2 margin). Token identity had hidden this: the earlier A/B called M=4
  same-prompt "identical to M=1" on tokens, and its logits differ from step 1.
- `MEMRA_MOE_INPUT_TRACE_DIR` now fires on the lockstep path too; `compare_trace.py` finds the
  first (step, layer) whose stream-0 MoE input differs.
- `MEMRA_MOE_TRACE` (existing) for the routes.

## Probe matrix (probe binary, `raw/ab577/`), stream 0 against its M=1 run

| Cell | Logits | Note |
| --- | --- | --- |
| M=2 mixed | IDENTICAL, 33 steps | |
| M=3 mixed | differ from step 1, max d 0.90 | |
| M=4 mixed | differ from step 1, max d 1.11, every element | argmax flips at step 17 (M=1 top-2 margin 0.077) |
| M=4 same prompt | differ from step 1, max d 0.89 | tokens happened to agree for 32 steps |
| M=4 mixed, `SPLIT_GROUPS=1` | identical to M=4 mixed | gathered `m_e`-row expert call is per-row exact |
| M=4 mixed, `CPU_ROWS=0` | differ from step 1, max d 0.59 | multi-row ABI is one of two sites |
| M=4 mixed, both doors | identical to `CPU_ROWS=0` | |

First divergent stream-0 MoE input: lockstep step 0, layer 2 (max |d| 4.5e-8, 2999 elements),
so layer 1's MoE block already differs and 78 of 79 layers differ at step 0; the head sees O(1)
logit differences after 79 layers. Routes: same expert sets, first order swap at (step 0,
layer 19), a near-tie downstream of the drift. This is a reduction-order class compounding
through the depth, not a routing defect.

## Fix, measured (`raw/ab577b/`, `raw/ab577c/`)

| Binary, cell | Logits vs M=1 |
| --- | --- |
| probe2 M=1 | identical to probe M=1 (the fix does not touch m=1) |
| probe2 M=4 mixed | differ, max d 0.94 (the CPU multi-row site remains) |
| probe2 M=4 mixed, `CPU_ROWS=0` | IDENTICAL, 33 steps |
| probe2 M=2 mixed | IDENTICAL |
| probe2 M=4 same | differ, max d 0.84 |
| probe3 M=1 | identical to probe M=1 |
| probe3 M=2 mixed | IDENTICAL, 33 steps |
| probe3 M=3 mixed | IDENTICAL, 33 steps |
| probe3 M=4 mixed | IDENTICAL, 33 steps |
| probe3 M=4 same prompt | IDENTICAL, 33 steps |
| probe3 M=4 mixed, `MEMRA_LOCKSTEP_CPU_ROWS=1` (opt-in arm) | differ from step 1, max d 0.94 |

Throughput on probe3 (single runs, for the record): M=1 4.09, M=2 4.89, M=3 5.38, M=4 mixed
5.60 tok/s aggregate; the opt-in multi-row arm reads 5.74 on the same cell (+2.5%), the probe
binary's batched shared expert plus multi-row arm read 5.63. Exactness costs about half a
percent here.

## Reading

1. The shared expert: `e.matmul(gate_shexp, zbatch, mrows)` (and up, down) chose its
   reduction program by row count; M=2 stayed on the M=1 program, M=3 and M=4 did not. Per row
   with `m = 1` is the single-sequence decode chain's own call. Lockstep rows are 1..=16.
2. The CPU companion: the M=1 run sums a row's CPU experts inside one job; the multi-row ABI
   issues one ticket per shared expert and the Rust side sums tickets in expert order, a
   different FP order for the same row. It is the M3 amortization arm (+4.8% aggregate at
   m=4, research/moe/draft-head-and-concurrency-lanes.md). It stays reachable as
   `MEMRA_LOCKSTEP_CPU_ROWS=1` until it carries an exactness proof against the M=1 order.
3. The `SPLIT_GROUPS` door measured no effect and is not shipped.
4. `decode_step_lockstep` has one caller, the harness; memra-server does not run it. The law
   this closes is the one in memra's CLAUDE.md: one numeric program per request, batched vs
   solo being a named pair.

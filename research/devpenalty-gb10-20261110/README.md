# The device penalty arm on GB10 (2026-11-10)

Verdict: on GB10 the device penalty arm (`MEMRA_SERVE_DEVPENALTY`) is flat for one penalised
sampled request at a time on both models measured, and faster for four at once: 11.8% on
Qwen3.8-27B (aggregate 37.49 to 41.90 tok/s) and 47.7% on Qwen3.6-35B-A3B (88.97 to 131.41
tok/s), nine of nine boot pairs each. With it on, every seeded prompt writes one text in every
boot and the seeded-invariance gate passes with `--penalised device` on both models. On the 27B
the arm lifts plain batched decode at four sessions past K=3 (41.9 to 49.3 against 43.1 tok/s),
so the spec cost model serves such a group plain, and its requests then get their first token
together: mean first token four at once 1.77 to 2.12 s, with requests 3.6 s shorter. The default
flips on for the dense and the MoE Qwen3.5 class on GB10, each on its own cell.

Decision record: [`docs/decisions/GB10-DEVICE-PENALTY-DEFAULT.md`](../../docs/decisions/GB10-DEVICE-PENALTY-DEFAULT.md).

## The cell

`tools/pk_pen.py`, one server boot per call and per GPU lease; `tools/pk_pen_summary.py` reads
its rows.

- **Device:** one DGX Spark (GB10, sm_121a). The timed cells ran on one unit; the first four
  boots of the 27B cell had started on the other unit and are kept as a note (`note-other-unit/`).
- **Binary:** the build of memra#1172's code (the position-keyed sampled verify), the same in
  both arms. Arms: `MEMRA_SERVE_DEVPENALTY=0` (off) and `=1` (on).
- **Request shape:** the Qwen non-thinking vendor shape, temperature 0.7, top_p 0.8, top_k 20,
  presence_penalty 1.5, a fixed seed per prompt, thinking off, 300-token answers, context 8,192,
  four sessions.
- **Per boot:** three prompts one at a time (34, 1.2k and 1.2k tokens), then two groups of four
  sent at once (two 34-token and two 1.2k-token prompts each).
- **Boots:** six per model, off on on off off on. Ratios are on over off on medians of per-boot
  medians (N = 3 boots per arm); U counts the boot pairs (of nine) the on arm wins.
- **Thermal regime:** while busy the GPU ran at 46 to 77 C on the 27B and 56 to 66 C on the MoE
  model, the SM clock at a median of 2,177 to 2,184 MHz in every boot. On the 27B 2 to 4 of
  about 650 busy samples per boot carry clock event reason 0x4; on the MoE model none does
  (`telemetry-*.csv`, 250 ms; `tools/pk_pen_summary.py` prints the per-boot lines).

## Qwen3.8-27B (dense) (`q27/`)

boots: 1 off, 2 on, 3 on, 4 off, 5 off, 6 on; requests 66, errors 0

| class | measure | off (median of per-boot medians, range) | on | on over off | U |
|---|---|---|---|---:|---:|
| c1 | decode tok/s per request | 20.89, 20.89-20.89 | 20.92, 20.88-20.98 | 1.001 | 6/9 |
| c1 | TTFT, 34-token prompt (s) | 0.216, 0.216-0.217 | 0.215, 0.214-0.216 | 0.994 | 0.5/9 |
| c1 | TTFT, 1.2k-token prompts (s) | 1.313, 1.312-1.314 | 1.311, 1.310-1.314 | 0.999 | 3/9 |
| c1 | tokens per second over the request | 19.18, 19.16-19.19 | 19.19, 19.15-19.28 | 1.001 | 5.5/9 |
| c4 | aggregate tok/s, four at once | 37.49, 30.40-37.57 | 41.90, 41.23-42.11 | 1.118 | 9/9 |
| c4 | decode tok/s per request | 10.16, 10.05-10.59 | 11.65, 11.24-11.72 | 1.147 | 9/9 |
| c4 | TTFT, 34-token prompts (s) | 0.311, 0.307-0.913 | 0.321, 0.310-1.436 | 1.034 | 6/9 |
| c4 | TTFT, 1.2k-token prompts (s) | 3.295, 2.200-3.297 | 2.862, 2.817-3.349 | 0.869 | 5/9 |
| c4 | request time (s) | 30.95, 30.18-31.04 | 27.36, 27.35-27.76 | 0.884 | 0/9 |

Time per output token after the first, over requests (p50, p95, p99, ms): c1 48.0, 51.6, 51.6
off and 48.0, 51.7, 51.7 on; c4 98.8, 103.3, 164.0 off and 87.1, 94.6, 98.1 on.

The draft length the spec cost model serves shifts with the arm, and the shift follows its
realized rates. From the `[spec-cost] realized` lines of the server logs, one value per boot
(tok/s):

| route | off (boots 1, 4, 5) | on (boots 2, 3, 6) |
|---|---|---|
| speculating at K=3, four sessions | 42.8, 43.3, 43.3 | 42.9, 43.1, 43.3 |
| plain batched decode, four sessions | 41.9, 42.2, 41.6 | 49.3, 49.3, 49.4 |
| plain batched decode, three sessions | 33.7, 33.4, 33.5 | 38.3, 38.5, 38.3 |

The arm leaves the speculative route where it was (its verify draws are on the device in both
arms) and lifts plain batched decode by 17.7%. With the arm off K=3 is 2 to 4% ahead of plain;
with it on plain is 14% ahead of K=3. The model's choices (`[spec-cost] n=.. rung=.. -> K=..`
and `[spec-gate] demoted session` lines):

- First group of four, every boot of both arms: K=3 on the boot prior (64.1 to 65.1 predicted
  against 46.1 to 46.8 for plain), K=3 measured at 42.8 to 43.3, then the four requests moved
  to plain decode at 97 to 137 generated tokens of 300. The plain prior is close to what the on
  arm serves (49.3) and above what the off arm serves (41.6 to 42.2).
- Second group, on, every boot: plain from arrival (37.7 to 38.3 predicted at the rung against
  49.3 to 49.4 realized for plain).
- Second group, off: K=3 for the whole group in boots 1 and 4 (43.3 and 43.7 against 41.9 and
  41.8 for plain). In boot 5 the first request, alone at arrival, was moved to plain at its
  first token and ran beside three peers at K=3: 6.12 tok/s for that request, 24.37 for the
  group.

The same groups by wall clock, prefill included. The route is from the lines above, not from
the `k` field of `runs.jsonl`: that field is the first `[spec-k]` line within one token of the
request's prompt length, the rung at arrival, and two requests of a group whose prompts differ
by one token can report the same line.

| arm | group | route | aggregate tok/s (three boots) |
|---|---|---|---|
| on | first of the boot | K=3, then plain | 40.11, 38.42, 39.72 |
| on | second of the boot | plain from arrival | 44.10, 44.03, 44.08 |
| off | first of the boot | K=3, then plain | 36.58, 36.45, 36.42 |
| off | second of the boot | K=3 (two boots) | 38.56, 38.53 |
| off | second of the boot | one request plain beside three at K=3 (one boot) | 24.37 |

The model's choice is right in each arm by the rates it measured: plain with the arm on, K=3
with it off. The mixed group of boot 5 is a cost-model case of its own (a request priced alone
while its peers arrive), not an effect of the arm; with the arm on the same arrival took the
whole group to plain.

First tokens. One at a time they do not move. Four at once they move with the route: a group
served plain from arrival gives the three requests behind the first their first token together.
Per group, sorted (s):

| arm | group | 34-token prompts | 1.2k-token prompts |
|---|---|---|---|
| off | first | 0.20 0.21 0.21 0.41 0.41 1.75 | 1.55 2.98 3.75 3.75 5.05 5.05 |
| off | second | 0.20 0.20 0.20 0.40 1.62 1.63 | 1.42 1.43 1.62 2.84 2.84 2.85 |
| on | first | 0.20 0.21 0.23 0.41 0.44 5.77 | 1.54 1.67 2.77 2.97 3.83 5.05 |
| on | second | 0.20 0.20 0.20 2.64 2.65 2.65 | 2.86 2.86 2.87 2.87 2.87 2.87 |

Mean over the second groups 1.44 s off and 2.14 s on; over all 24 requests 1.77 and 2.12 s;
over the twelve 34-token requests 0.62 and 1.32 s. The per-boot medians of the table above do
not show it. The same requests finish 3.6 s earlier with the arm on (30.95 to 27.36 s).

Texts: with the arm off, two of the eight prompts sent four at once wrote two texts across
boots; with it on, every prompt wrote one text in every boot.

Gate, arm on, `--penalised device` (`q27/gate.log`): PASS, 54 rows, 10 prompts. The penalised
prompt writes one text in plain decode, at K=0, 1, 2, 3 and in the demoted run, the text
speculation alone wrote with the arm off (`--penalised host` in memra#1172's receipts, where
plain decode and the demoted run each wrote another).

## Qwen3.6-35B-A3B (MoE) (`q36moe/`)

`Arch::Qwen35Moe`, pack `qwen35_moe`, the IQ4_XS GGUF with its MTP head. The same cell, the same
binary.

boots: 1 off, 2 on, 3 on, 4 off, 5 off, 6 on; requests 66, errors 0

| class | measure | off (median of per-boot medians, range) | on | on over off | U |
|---|---|---|---|---:|---:|
| c1 | decode tok/s per request | 70.34, 70.11-70.35 | 70.37, 70.30-70.55 | 1.000 | 7/9 |
| c1 | TTFT, 34-token prompt (s) | 0.106, 0.106-0.107 | 0.106, 0.105-0.106 | 0.992 | 1/9 |
| c1 | TTFT, 1.2k-token prompts (s) | 0.580, 0.576-0.587 | 0.592, 0.587-0.595 | 1.021 | 9/9 |
| c1 | tokens per second over the request | 66.10, 66.03-66.23 | 66.17, 65.99-66.31 | 1.001 | 5/9 |
| c4 | aggregate tok/s, four at once | 88.97, 88.81-89.35 | 131.41, 131.40-131.61 | 1.477 | 9/9 |
| c4 | decode tok/s per request | 24.48, 24.12-24.82 | 37.74, 37.66-38.01 | 1.542 | 9/9 |
| c4 | TTFT, 34-token prompts (s) | 0.353, 0.150-0.539 | 0.509, 0.501-0.536 | 1.440 | 6/9 |
| c4 | TTFT, 1.2k-token prompts (s) | 1.470, 1.460-1.487 | 1.431, 1.425-1.447 | 0.973 | 0/9 |
| c4 | request time (s) | 13.46, 13.38-13.48 | 9.10, 9.09-9.11 | 0.676 | 0/9 |

Time per output token after the first (p50, p95, p99, ms): c1 14.3, 14.8, 14.8 off and 14.3,
14.9, 14.9 on; c4 40.7, 44.5, 44.6 off and 27.1, 30.1, 30.1 on.

No draft-length shift on this model: the cost model served all 48 requests sent four at once
plain in both arms (none reports an acceptance; a request that arrived alone was moved to plain
at its first token when its peers came). The whole difference is the sampler. Realized plain
rate at four sessions: 99.0, 99.2 and 101.9 tok/s off, 153.0, 154.9 and 155.1 on, against a boot
prior of 157 to 158. Per token of a request the arm removes 14.5 ms here (4/99.2 against
4/154.9 s) and 14.3 ms on the 27B (4/41.9 against 4/49.3 s). One at a time every request
speculated in both arms, with the same acceptance per prompt (0.43 to 0.50).

Two first-token measures move by a small amount with the arm on:

- One at a time, the 1.2k-token prompts: 0.580 to 0.592 s (12 ms, nine of nine pairs). The
  34-token prompt is unmoved (0.106 s).
- Four at once, the 34-token prompts: the per-boot medians read 0.353 against 0.509 s, but the
  two short requests of a group land on admission slots, not on a rate. Per boot, sorted, off:
  0.10 0.11 0.97 1.29, 0.10 0.11 0.19 0.98, 0.10 0.20 0.50 0.51; on: 0.10 0.50 0.50 0.98,
  0.10 0.51 0.51 0.97, 0.10 0.10 0.97 0.97. The same four values (about 0.1, 0.2, 0.5 and 1.0
  s) in both arms, mean 0.43 off and 0.53 on over twelve requests each, 0.95 and 0.97 over all
  24 requests. The 1.2k-token prompts of the same groups are 39 ms earlier with the arm on
  (1.470 to 1.431 s, nine of nine).

Texts: with the arm off, three of the eight prompts sent four at once wrote two texts across
boots; with it on, every prompt wrote one text in every boot.

Gate, arm on, `--penalised device` (`q36moe/gate.log`): PASS, 54 rows, 10 prompts, one text per
prompt in plain decode, at K=0, 1, 2, 3 and in the demoted run (acceptance per drafted token
0.689, 0.528, 0.401 at K=1, 2, 3).

## The default itself (`q27/gate-default.log`, `q36moe/gate-default.log`)

The same gate on the build of this change, with no environment override and `--penalised
device`: PASS, 54 rows, on the 27B and on the MoE model. It passes only if the default resolves
on for the device: the penalised prompt writes the text it wrote with the arm forced on (the
same hash in `gate/runs.jsonl` and `gate-default/runs.jsonl`), in plain decode, at K=0, 1, 2, 3
and in the demoted run. The build is of the default's final logic; the tests and comments of
the commit came after it.

## Limits

One Spark unit for the timed cells, 300-token answers, at most four requests at once, one
checkpoint per class (the 9B and Ornith take their class's default without a cell). The cell
kept the first chunk and the total time per request, so there are no inter-token latency
percentiles. Not measured: eight or more at once, and the other families, which stay off on
GB10. The four boots on the other unit (`note-other-unit/`): one at a time 19.24 and 19.26 tok/s
over the request off, 19.20 and 18.91 on; four at once 36.71 and 36.71 (first groups), 23.97 and
38.60 (second groups) off, 40.13 and 39.63, 44.07 and 44.00 on.

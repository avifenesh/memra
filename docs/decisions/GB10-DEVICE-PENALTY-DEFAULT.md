# The device penalty arm is on for the dense and MoE Qwen3.5 classes on GB10 (2026-11-10)

**Status:** adopted on GB10, keyed by the device name, for the dense `Arch::Qwen35` class
(measured on Qwen3.8-27B) and the `Arch::Qwen35Moe` class (measured on Qwen3.6-35B-A3B), each on
its own cell. Every other family stays off on GB10. `MEMRA_SERVE_DEVPENALTY=0` stays the
rollback seam. Receipts:
[`research/devpenalty-gb10-20261110/`](../../research/devpenalty-gb10-20261110/).

**Cost to know about, for interactive latency on the 27B:** with the arm on, plain batched
decode beats K=3 at four sessions, so the spec cost model serves a group of four plain, and a
group served plain from arrival gets its first tokens together. The three requests behind the
first then see their first token at 2.64 to 2.87 s, where a speculating group staggers them
(1.42, 1.62 and 2.84 s). Mean first token four at once: 1.77 s off, 2.12 s on (1.44 against
2.14 s in the groups served plain from arrival). The same requests finish 3.6 s earlier (30.95
to 27.36 s) and throughput is 11.8% higher. One request at a time is unaffected, and the MoE
model shows none of it (plain in both arms). `MEMRA_SERVE_DEVPENALTY=0` restores the earlier
first tokens at the earlier throughput.

## Question

`MEMRA_SERVE_DEVPENALTY` moves a penalised sampled request's plain draws from the host sampler
to the device. Its default was on only where a pair had a receipt: the dense Qwen3.5 class on the
RTX 5090 and RTX PRO 6000 classes. On GB10 it was off, unmeasured. Two things made that worth
measuring now:

- The Qwen non-thinking vendor shape carries `presence_penalty 1.5`, so the default-shaped
  request on GB10 was the host-drawn class.
- Since memra#1154 a seeded sampled request writes one text on the MTP route at every draft
  length and across a demotion, for every request plain decode draws on the device. With the
  arm off a penalised request on GB10 is not one of them: plain decode, speculation and a
  demoted run wrote three texts in the invariance gate.

## The cell

One DGX Spark, context 8,192, four sessions, thinking off, the vendor non-thinking shape
(temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5) with a fixed seed per prompt,
300-token answers. One binary, the arms `MEMRA_SERVE_DEVPENALTY=0` and `=1`, six boots per model
off on on off off on, one boot per lease. Per boot: three prompts one at a time (34, 1.2k and
1.2k tokens), then two groups of four sent at once (two 34-token and two 1.2k-token prompts
each). 66 requests and 0 errors per model. On over off on medians of per-boot medians, N = 3
boots per arm; U counts the boot pairs (of nine) the on arm wins. Thermal regime: the GPU at 46
to 77 C while busy on the 27B and 56 to 66 C on the MoE model, the SM clock at 2,177 to 2,184
MHz in every boot.

## Measurement: the dense class

Qwen3.8-27B NVFP4 + Q5_K GGUF with its MTP head.

| shape | measure | off | on | on over off | U |
|---|---|---:|---:|---:|---:|
| one at a time | decode tok/s per request | 20.89 | 20.92 | 1.001 | 6/9 |
| one at a time | TTFT, 34-token prompt (s) | 0.216 | 0.215 | 0.994 | 0.5/9 |
| one at a time | TTFT, 1.2k-token prompts (s) | 1.313 | 1.311 | 0.999 | 3/9 |
| four at once | aggregate tok/s | 37.49 | 41.90 | 1.118 | 9/9 |
| four at once | decode tok/s per request | 10.16 | 11.65 | 1.147 | 9/9 |
| four at once | request time (s) | 30.95 | 27.36 | 0.884 | 0/9 |
| four at once | TTFT, 34-token prompts (s) | 0.311 | 0.321 | 1.034 | 6/9 |
| four at once | TTFT, 1.2k-token prompts (s) | 3.295 | 2.862 | 0.869 | 5/9 |

Time per output token after the first, over requests (p50, p95, p99, ms): one at a time 48.0,
51.6, 51.6 off and 48.0, 51.7, 51.7 on; four at once 98.8, 103.3, 164.0 off and 87.1, 94.6,
98.1 on.

Flat one at a time, 11.8% more aggregate throughput four at once.

**The draft length shifts with the arm, and the shift follows the realized rates.** On this
card the spec cost model owns K. Its `[spec-cost] realized` lines give the rate of each route,
one value per boot (tok/s):

| route | off (three boots) | on (three boots) |
|---|---|---|
| speculating at K=3, four sessions | 42.8, 43.3, 43.3 | 42.9, 43.1, 43.3 |
| plain batched decode, four sessions | 41.9, 42.2, 41.6 | 49.3, 49.3, 49.4 |
| plain batched decode, three sessions | 33.7, 33.4, 33.5 | 38.3, 38.5, 38.3 |

The arm leaves the speculative route where it was (its verify draws are on the device in both
arms) and lifts plain batched decode by 17.7% (41.9 to 49.3 tok/s). That reverses the order of
the two routes: with the arm off K=3 is 2 to 4% ahead of plain, with it on plain is 14% ahead
of K=3. What the model did:

- First group of four, every boot of both arms: it starts at K=3 on its boot prior (64.1 to
  65.1 predicted against 46.1 to 46.8 for plain), measures K=3 at 42.8 to 43.3 and moves the
  four requests to plain decode at 97 to 137 generated tokens of 300. The plain prior it moves
  on is close to what the on arm serves (49.3) and above what the off arm serves (41.6 to
  42.2).
- Second group, arm on, every boot: plain from arrival (37.7 to 38.3 predicted at the rung
  against 49.3 to 49.4 realized for plain).
- Second group, arm off: K=3 for the whole group in two boots (43.3 and 43.7 against 41.9 and
  41.8 for plain). In the third boot the first request, alone at arrival, was moved to plain at
  its first token and ran beside three peers at K=3: 6.12 tok/s for that request and 24.37 for
  the group.

The same groups by wall clock, prefill included:

| arm | group | route | aggregate tok/s (three boots) |
|---|---|---|---|
| on | first of the boot | K=3, then plain | 40.11, 38.42, 39.72 |
| on | second of the boot | plain from arrival | 44.10, 44.03, 44.08 |
| off | first of the boot | K=3, then plain | 36.58, 36.45, 36.42 |
| off | second of the boot | K=3 (two boots) | 38.56, 38.53 |
| off | second of the boot | one request plain beside three at K=3 (one boot) | 24.37 |

So the model's choice is right in each arm by the rates it measured: plain with the arm on, K=3
with it off. The one mixed group of the off arm is a cost-model case of its own (a request
priced alone while its peers arrive), not an effect of the arm; with the arm on the same
arrival took the whole group to plain.

**First tokens four at once move with the route.** One at a time they do not move. A group
served plain from arrival gives the three requests behind the first their first token together,
at 2.64 to 2.87 s in all three on boots; the speculating second group of the off arm staggers
them (1.42 or 1.43, 1.62 or 1.63 and 2.84 or 2.85 s in two boots; 0.40, 1.62 and 2.84 s in the
third). Mean first token over the twelve requests of those groups: 1.44 s off, 2.14 s on. Over
all 24 requests sent four at once: 1.77 s off, 2.12 s on (over the twelve 34-token requests
0.62 and 1.32 s, one 5.77 s request of a first group included). The medians in the table do not
show it. The same requests finish 3.6 s earlier with the arm on.

**The text.** With the arm off, two of the eight prompts sent four at once wrote two texts
across boots. With it on every prompt wrote one text in every boot. The invariance gate with the
arm on and `--penalised device`: PASS, 54 rows; the penalised prompt writes one text in plain
decode, at K=0, 1, 2, 3 and in the demoted run.

## Measurement: the MoE class

Qwen3.6-35B-A3B IQ4_XS GGUF with its MTP head (`Arch::Qwen35Moe`, pack `qwen35_moe`), the same
cell and binary.

| shape | measure | off | on | on over off | U |
|---|---|---:|---:|---:|---:|
| one at a time | decode tok/s per request | 70.34 | 70.37 | 1.000 | 7/9 |
| one at a time | TTFT, 34-token prompt (s) | 0.106 | 0.106 | 0.992 | 1/9 |
| one at a time | TTFT, 1.2k-token prompts (s) | 0.580 | 0.592 | 1.021 | 9/9 |
| four at once | aggregate tok/s | 88.97 | 131.41 | 1.477 | 9/9 |
| four at once | decode tok/s per request | 24.48 | 37.74 | 1.542 | 9/9 |
| four at once | request time (s) | 13.46 | 9.10 | 0.676 | 0/9 |
| four at once | TTFT, 34-token prompts (s) | 0.353 | 0.509 | 1.440 | 6/9 |
| four at once | TTFT, 1.2k-token prompts (s) | 1.470 | 1.431 | 0.973 | 0/9 |

Time per output token after the first (p50, p95, p99, ms): one at a time 14.3, 14.8, 14.8 off
and 14.3, 14.9, 14.9 on; four at once 40.7, 44.5, 44.6 off and 27.1, 30.1, 30.1 on.

Flat one at a time, 47.7% more aggregate throughput four at once. There is no draft-length
shift to read here: the cost model served all 48 requests sent four at once plain in both arms
(a request that arrived alone was moved to plain at its first token when its peers came), so
the difference is the sampler alone. Its realized plain rate at four sessions is 99.0, 99.2 and
101.9 tok/s off and 153.0, 154.9 and 155.1 on, against a boot prior of 157 to 158: the prior
prices the device draw. Per token of a request at four sessions the arm removes 14.5 ms here
(4/99.2 against 4/154.9 s) and 14.3 ms on the 27B (4/41.9 against 4/49.3 s): the same cost, a
larger share of a model that decodes 3.4 times faster.

Two first-token measures move. One at a time, the 1.2k-token prompts start 12 ms later (0.580
to 0.592 s, nine of nine pairs); the 34-token prompt does not move. Four at once, the 34-token
prompts land on admission slots (about 0.1, 0.2, 0.5 and 1.0 s) in both arms, and the median
over a boot's four requests picks a slot, not a rate: the mean over twelve requests is 0.43 s
off and 0.53 s on, the mean over all 24 requests 0.95 and 0.97 s, and the 1.2k-token prompts of
the same groups start 39 ms earlier with the arm on.

**The text.** With the arm off, three of the eight prompts sent four at once wrote two texts
across boots; with it on every prompt wrote one text in every boot. The invariance gate with
the arm on and `--penalised device`: PASS, 54 rows.

## The default itself

The invariance gate on the build of this change, with no environment override and `--penalised
device`: PASS, 54 rows, on the 27B and on the MoE model. It passes only if the default resolves
on for the device: the penalised prompt writes the text it wrote with the arm forced on.

## What was chosen and what was not

- On for the dense Qwen3.5 class on GB10: flat one at a time, 11.8% more throughput and 3.6 s
  shorter requests four at once, and it puts the default-shaped request on GB10 under the
  position key. Its cost is the first token of a group the cost model serves plain from
  arrival: 0.7 s on the mean over such a group, 0.35 s over all requests sent four at once.
- On for the Qwen3.5 MoE class on GB10, by its own cell, not by the 27B's: flat one at a time,
  47.7% more four at once, one text per prompt.
- The key is the class, as on the other cards: one checkpoint per class was measured, and the
  other checkpoints of a class (the 9B in the dense class, Ornith in the MoE class) take its
  default on GB10 without a cell of their own.
- Keyed by the device name ("GB10"), as the other pairs of this default are keyed by the card
  class. The `120a` build family is not the key: it also builds for cards with no receipt.
- Not flipped: every other family on GB10 (Qwen3, Qwen3 MoE, Gemma and the rest), an
  `Arch::Qwen35` plan that carries experts, and the MoE class on the RTX 5090 and RTX PRO 6000
  classes. No cell ran for them.
- Not measured: eight or more at once, inter-token latency percentiles (the cell kept the first
  chunk and the total per request), a second Spark unit for the timed cell. Four boots of the
  27B cell on the other unit are kept as a note: one at a time 19.24 and 19.26 tok/s over the
  request off, 19.20 and 18.91 on; four at once 36.71 and 36.71 (first groups), 23.97 and 38.60
  (second groups) off, 40.13 and 39.63, 44.07 and 44.00 on.

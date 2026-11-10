# The MTP draft's leftover inputs for the removed accept test (memra#1169, 2026-11-10)

Verdict: exact, and close to flat. Every seeded text and every acceptance count is the same
before and after, on both devices, on the captured and on the eager draft chain, alone and in a
pooled pair. On GB10 the pure-temperature sampled class decodes 0.3% faster (0.32 to 0.44 ms off
a 107 to 109 ms round, the after arm faster in 105 of 108 boot pairs); the filtered class gains
0.1% (63 of 108, the size of the greedy control's own spread); the penalised class and the
greedy control do not move. On the RTX 5090 the timed cell does not resolve a change: the card
ran throttled, the untouched greedy control itself read 1.022 and the two graph-drafted sampled
classes 0.997 and 1.002. A sampled round on GB10 stays 7 to 10 ms longer than a greedy one (107
to 111 ms against 99.5 to 101.8 ms): these leftovers were under half a millisecond of that gap,
so the gap is somewhere else. The change lands as a dead-code removal.

Model: Qwen3.8-27B NVFP4 + Q5_K GGUF with its MTP head, context 8,192, two sessions, thinking
off, every switch at the device default. Devices: an RTX 5090 Laptop GPU (sm_120a) and GB10, one
DGX Spark (sm_121a). Arms are two binaries per device: `before`, 4aec19e824 (the head of
memra#1172, the position-keyed sampled verify, before it landed), and `after`, the same tree plus
this change. The change is one patch (stable patch id c916fefe46ae); main gained an unrelated
DSpark drafter fix (memra#1159) between the measurement and this branch's rebase onto it, which
neither binary carries and no cell here reaches.

## What the change removes

Since memra#1154 `HybridModel::spec_round_finish` accepts a sampled draft token when it equals
the target's own position-keyed draw. It no longer reads the draft's distribution. The draft
side still produced that distribution's inputs every sampled round:

| where | what went | per round at K=3 |
|---|---|---|
| sampled draft graph, pure temperature (single head and chain) | the device copy of the head logits into a retained slot after each replay; after the chain, one `filter_stats` launch and three scalar readbacks per slot | 3 device copies, 3 launches, 9 readbacks |
| sampled draft graph, filtered (top_k, top_p, min_p) | the device copy per replay and three scalar readbacks per slot (the in-graph statistics) | 3 device copies, 9 readbacks |
| the capture body | the in-graph copy of the head logits into a persistent buffer that only the retention read | 3 in-graph copies |
| eager sampled chain (a penalised request, or any request when a graph cannot be captured) | the retained copy of the head logits for an unpenalised request, the readback of the kept mass z, the two pushes | up to 3 device copies, 3 readbacks |
| pooled draft | the same per member, and the post-chain statistics pass | as above, per member |
| state | `SpecRoundState.draft_stats`, `SpecRoundState.draft_logits`, `DraftGraphCtx.q_slots` (K vocabulary rows per sampled session, allocated at capture), `DraftGraphCtx.g_q` | K + 1 vocabulary rows of device memory per sampled session |

What stays: the in-graph `filter_stats` and filtered perturb nodes, which shape the draft's own
draw, and in the eager chain the statistics launch and the readback of the row maximum and the
threshold, which the perturb takes as host scalars. Greedy draft arms, the other speculative
routes and the kernels are untouched.

Two consequences outside the draft arms:

- The capture bootstrap estimate (`draft_capture_bootstrap_estimate`, used only until a model
  has measured a capture) loses its sampled-tail term of K + 2 vocabulary rows. It also feeds
  the greedy capture gate, whose threshold is lower by that many bytes before the first
  measurement.
- The measured per-session draft-state high-water no longer counts the K retained rows. In the
  RTX 5090 gate logs it reads 41 MB where the before binary reads 44 MB, and admission charges a
  speculative session 43 MB instead of 46 MB.

The pooled-draft probe (`decode-batch-bench --pooled-draft`) compared two fields that no longer
exist, "filter stats" and "q rows". It now compares draft tokens, trimmed ids (a p-min-discarded
draw included), the Philox counter, the draft KV length on host and device, the draft KV bytes
the round wrote and a follow-up head step over them. It no longer sees a step's head logits or
its filter statistics directly, only through the draw: a difference between the captured
graph's head row and the pooled row that flips no draw is not seen. Its first-step check of the
rows forward against per-session eager steps is unchanged. Its `DRAFT-TIME` lines are not
comparable with older receipts, because both drafts lost work.

## Exactness

The draft tokens must not change, so every text and every acceptance count must be the same on
both binaries. `tools/gatecmp.py` compares two gate receipts row by row (text sha, completion
tokens, acceptance).

| device | cell | before | after | rows compared | same text sha, tokens and acceptance |
|---|---|---|---|---:|---:|
| RTX 5090 | `tools/seeded-invariance-gate.py`, 7 boots (`rtx5090/gate-*.log`) | PASS | PASS | 54 | 54 |
| RTX 5090 | the same gate with `MEMRA_SPEC_NOGRAPH=1`, the eager draft chain (`rtx5090/gate-eager-*.log`) | PASS | PASS | 54 | 54 |
| RTX 5090 | pooled-draft probe, 7 cells (`rtx5090/probe/cells.txt`) | | PASS | | |
| GB10 | the gate with `--penalised host`, 7 boots (`gb10/gate-*.log`) | PASS | PASS | 54 | 54 |
| RTX 5090 | the perf cell, 12 prompts in 6 boots per arm | | | 144 | one text and one acceptance per prompt across the 12 boots |
| GB10 | the perf cell, 12 prompts in 6 boots per arm | | | 144 | one text and one acceptance per prompt across the 12 boots |

- The gate's rows: four lone prompts (temperature 1 long and short, temperature 0.7 with top_p
  0.95, a frequency penalty), two prompts sent at once, a two-turn conversation, a parked-prompt
  pair, at K=0, 1, 2, 3 and across a demotion to plain decode. Every text equals the plain
  reference on both binaries.
- The pooled cohort: in each binary's gate the three K-pinned boots log
  `[spec-pool] pooled MTP cohort engaged: 2 sessions` and
  `pooled rows draft engaged: 2 members (arms [GraphSampled, GraphSampled])` for the pair; with
  the graph off the arms are `[Eager, Eager]`. The eager gate's 54 rows also equal the default
  gate's 54 rows on the after binary (graph and eager chains draft the same tokens).
- The probe's cells on the after build: greedy ragged K, mixed greedy and sampled, sampled with a
  p-min stop, filtered, penalised, and the eager chain sampled and filtered. Seven of seven print
  `ROWS-STEP ... BITWISE EQUAL` and `ROUND-DRAFT ... tokens, ids, counters, draft-KV, follow-up:
  BITWISE EQUAL`; no `MISMATCH` line.
- The penalised prompt drafts on the eager chain at the default settings, so the default gate
  covers that chain on both devices; on GB10 plain decode draws it on the host, which is what
  `--penalised host` tells the gate.

## Decode rate and round time

Twelve boots per device, before after after before three times, one boot per GPU lease, 250 ms
`nvidia-smi` telemetry per boot. Per boot twelve cold prompts, one at a time, 400-token answers
(`tools/ab_cell.py`: the memra#1154 cell's nine prompts plus a penalised class):

- `t1`: temperature 1 (the pure-temperature sampled draft graph)
- `chat`: temperature 0.7, top_p 0.95 (the filtered sampled draft graph)
- `greedy`: temperature 0, the control
- `pen`: temperature 0.7, top_p 0.8, top_k 20, presence_penalty 1.5 (the eager sampled chain)

each at 34, 1.2k and 3.6k prompt tokens. Both arms write the same text with the same
acceptance in every class, so tokens per round is one number per class and the decode ratio is
the round-time change alone. Ratios are after over before on the per-arm medians over boots;
"wins" counts the (after boot, before boot) pairs the after arm decodes faster in; tokens per
round is 1 + K x acceptance per drafted token; round ms is 1000 x tokens per round / median
decode tok/s (`tools/round_summary.py`).

### GB10 (`gb10/perf/`, N = 6 boots per arm, 144 requests, 0 errors)

Thermal regime: steady. Every boot ran at a 2,184 MHz SM clock, 65 to 67 C median (75 to 78 C
maximum) and 32 to 33 W; the unit served other leases between the boots.

| class | K | decode tok/s before | after | ratio | wins | tokens per round | round ms before | after | change ms |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| t1-short | 3 | 20.32 | 20.39 | 1.003 | 36 of 36 | 2.17 | 106.97 | 106.63 | -0.34 |
| t1-1k | 3 | 22.09 | 22.16 | 1.003 | 33 of 36 | 2.37 | 107.46 | 107.14 | -0.32 |
| t1-4k | 3 | 21.23 | 21.32 | 1.004 | 36 of 36 | 2.32 | 109.41 | 108.98 | -0.44 |
| chat-short | 3 | 20.70 | 20.69 | 1.000 | 17 of 36 | 2.24 | 108.37 | 108.40 | +0.03 |
| chat-1k | 3 | 22.18 | 22.22 | 1.002 | 25 of 36 | 2.43 | 109.61 | 109.42 | -0.20 |
| chat-4k | 3 | 22.37 | 22.39 | 1.001 | 21 of 36 | 2.48 | 110.92 | 110.77 | -0.15 |
| greedy-short | 3 | 21.61 | 21.59 | 0.999 | 11 of 36 | 2.15 | 99.52 | 99.58 | +0.07 |
| greedy-1k | 3 | 23.62 | 23.63 | 1.000 | 15 of 36 | 2.35 | 99.62 | 99.57 | -0.04 |
| greedy-4k | 3 | 23.12 | 23.09 | 0.999 | 9 of 36 | 2.35 | 101.70 | 101.79 | +0.09 |
| pen-short | 3 | 19.96 | 19.98 | 1.001 | 21 of 36 | 2.16 | 108.32 | 108.24 | -0.08 |
| pen-1k | 3 | 20.93 | 20.91 | 0.999 | 15 of 36 | 2.27 | 108.69 | 108.79 | +0.10 |
| pen-4k | 3 | 21.28 | 21.28 | 1.000 | 13 of 36 | 2.37 | 111.57 | 111.57 | +0.00 |

| pool | classes | decode ratio, geometric mean | round ms change, mean |
|---|---:|---:|---:|
| temperature 1 (pure-temperature graph draft) | 3 | 1.003 | -0.36 |
| temperature 0.7, top_p 0.95 (filtered graph draft) | 3 | 1.001 | -0.11 |
| both graph-drafted sampled classes | 6 | 1.002 | -0.24 |
| penalised (eager sampled draft) | 3 | 1.000 | +0.01 |
| greedy control | 3 | 1.000 | +0.04 |

The pure-temperature class is the one that lost a kernel launch per slot as well as the
readbacks, and it is the only one whose gain is clear of the control. Before this change a
sampled round ran 107.0 to 110.9 ms and a greedy round 99.5 to 101.7 ms; after it, 106.6 to
110.8 ms. The 7 to 10 ms a sampled round costs over a greedy one on GB10 is not in the work
this change removes.

### RTX 5090 (`rtx5090/perf/`)

N = 6 boots per arm, 144 requests, 0 errors. K is the device's own: 3 on the short prompt, 2 on the
two long ones.

Thermal regime: hot and throttled in every boot. The card is shared and did not cool before the
first boot (73 C after a ten-minute wait for 62 C). Busy medians 86 to 87 C (89 to 92 C
maximum), SM clock 1,335 to 1,522 MHz (mean 1,404 MHz over the before boots, 1,418 MHz over the
after boots), 124 to 138 W, thermal throttle flags set in all twelve boots. A class's decode rate
ranges over 3 to 20% of its median from boot to boot.

| class | K | decode tok/s before | after | ratio | wins | tokens per round | round ms before | after | change ms |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| t1-short | 3 | 67.38 | 67.27 | 0.998 | 19 of 36 | 2.37 | 35.20 | 35.26 | +0.06 |
| t1-1k | 2 | 64.78 | 64.84 | 1.001 | 18 of 36 | 2.18 | 33.65 | 33.62 | -0.03 |
| t1-4k | 2 | 58.74 | 58.28 | 0.992 | 15 of 36 | 2.04 | 34.77 | 35.04 | +0.28 |
| chat-short | 3 | 55.61 | 54.96 | 0.988 | 17 of 36 | 2.20 | 39.54 | 40.01 | +0.46 |
| chat-1k | 2 | 61.89 | 62.38 | 1.008 | 22 of 36 | 2.25 | 36.35 | 36.07 | -0.28 |
| chat-4k | 2 | 57.98 | 58.64 | 1.011 | 23 of 36 | 2.12 | 36.56 | 36.16 | -0.41 |
| greedy-short | 3 | 55.45 | 57.77 | 1.042 | 27 of 36 | 2.15 | 38.69 | 37.14 | -1.55 |
| greedy-1k | 2 | 63.69 | 64.80 | 1.018 | 26 of 36 | 2.17 | 34.05 | 33.46 | -0.59 |
| greedy-4k | 2 | 62.91 | 63.32 | 1.007 | 22 of 36 | 2.13 | 33.82 | 33.60 | -0.22 |
| pen-short | 3 | 55.15 | 56.83 | 1.030 | 27 of 36 | 2.32 | 42.13 | 40.89 | -1.25 |
| pen-1k | 2 | 58.91 | 59.67 | 1.013 | 23 of 36 | 2.17 | 36.91 | 36.44 | -0.47 |
| pen-4k | 2 | 55.92 | 56.58 | 1.012 | 22 of 36 | 2.10 | 37.61 | 37.18 | -0.43 |

| pool | classes | decode ratio, geometric mean | round ms change, mean |
|---|---:|---:|---:|
| temperature 1 (pure-temperature graph draft) | 3 | 0.997 | +0.10 |
| temperature 0.7, top_p 0.95 (filtered graph draft) | 3 | 1.002 | -0.08 |
| both graph-drafted sampled classes | 6 | 1.000 | +0.01 |
| penalised (eager sampled draft) | 3 | 1.018 | -0.72 |
| greedy control | 3 | 1.022 | -0.79 |

Not resolved on this card. The greedy control, whose code is untouched, reads 1.022 (0.79 ms a
round), more than any sampled class moves. The penalised class moves with it (1.018), and the two
graph-drafted sampled classes read 0.997 and 1.002. No per-class difference is significant at
N = 6: the after arm wins 15 to 27 of 36 boot pairs, and 31 are needed at p = 0.05 (two-sided
Mann-Whitney). The removed work is on the order of 0.1 ms of a 34 to 42 ms round here; a cell
with this spread cannot see it. What the RTX 5090 contributes is the exactness above.

## An earlier form, not kept

Before the base branch gained its trimmed-head fix, this change also moved the eager sampled
chain's draw to the device-statistics kernel (`gumbel_perturb_filtered_col`), which removed the
two remaining readbacks per slot there. That form passed the same exactness cells against its own
base (the gate and the eager gate on the RTX 5090, 54 of 54 rows each, the gate on GB10, 54 of
54, and the probe). Its GB10 cell was stopped after three boots (one before, two after) when the
base moved: the penalised class read 1.000, 1.000 and 1.001 (`earlier-form/gb10-perf-runs.jsonl`,
N = 1 and 2, not a result). The base's eager chain now draws through a host-scalar perturb that
takes the trimmed head's id map; there is no device-statistics twin of it, and adding a kernel
was outside this change. So the two readbacks stay, with no measured loss.

## Limits

One model, two devices, 400-token answers, one request at a time in the timed cell. Not
measured: the RTX PRO 6000, a multi-head chain model, a trimmed draft head, concurrent requests
(the pooled draft is covered for exactness only), and the VRAM a parked sampled session gives
back beyond the two high-water lines quoted above. The before and after binaries were built from
4aec19e824 and from that tree plus this patch; the branch's head adds comment and doc edits and
sits on a later main.

## Files

- `rtx5090/gate-{after,before}/`, `rtx5090/gate-eager-{after,before}/`, `gb10/gate-{after,before}/`:
  `runs.jsonl` and the server log of every boot; the gate's printed table is the `.log` beside
  each directory.
- `rtx5090/probe/`: the probe's seven cell logs and `cells.txt`.
- `rtx5090/perf/`, `gb10/perf/`: `runs.jsonl`, `boots.log`, the server log and the telemetry of
  every boot.
- `tools/`: `ab_cell.py` (the cell), `round_summary.py` (the tables above), `gatecmp.py`,
  `probe_cells.sh`, `exact_5090.sh` and `driver_5090.sh` (the RTX 5090 leases; `no_landing.sh`
  is its wait for a landing's cells), `gb10-build.sh`, `gb10-gates.sh`, `gb10-boot.sh` (run on
  the GB10 host under the caller's lease). The gate is `tools/seeded-invariance-gate.py` at the
  repo root; the thermal table comes from
  `research/position-keyed-sampling-20261110/tools/pk_thermal.py`.

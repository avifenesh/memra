# GB10 single-user decode, round 5 (2026-11-09)

Device: DGX Spark (GB10, sm_121a), two units. Model: Qwen3.8-27B GGUF (NVFP4 trunk, Q5_K output head) with the masked MTP draft (32,768-row head, Q4_K/Q6_K block), 262,144 context, served the way the launcher serves it: the MTP speculative path at the automatic K (K=3 for these cold requests), 32-token bursts. Binary: main 65fb297a59 for the scout, main a8a980a017 as lever 1's base.

The question: where a single user's decode time goes against the part's bandwidth roofline, and which exact levers close the gap.

## The scout

Three cells, one per lease:

- `decode` (unit 2): one cold request each at about 1k, 131k and 260k prompt tokens, `max_tokens` 512, every server line timestamped, `MEMRA_TICK_TRACE=1` for the per-burst rounds. Every answer is about 100 tokens and ends on EOS.
- `nsys1k` (unit 1) and `nsys131k` (unit 2): the same server under Nsight Systems, one request each, and the decode window after the first token.

Receipts: `scout/` (`decode-unit2/tab.txt`, `nsys*/kernels.txt`, `nsys*/copies-gaps.txt`, the 1k round's op timeline `nsys1k-unit1/round6.txt`, the tensor byte tables `gguf-bytes-*.txt`). Tools: `tools/`.

### c1 decode

| prompt | TTFT | decode after the first token | inside full bursts | tokens per round | acceptance per drafted token | ms per round |
|---:|---:|---:|---:|---:|---:|---:|
| 1,036 | 1.13 s | 22.1 tok/s | 29.8 tok/s | 2.65 | 0.54 | 89 |
| 131,051 | 117.7 s | 10.3 tok/s | 13.5 tok/s | 2.59 | 0.52 | 192 |
| 259,951 | 313.1 s | 6.9 tok/s | 9.4 tok/s | 2.77 | 0.58 | 294 |

Single runs. "Inside full bursts" uses the burst walls of every burst but the first (which holds the prime) and the last (cut by the stop).

### One round against the roofline

The floor prices each part's bytes at 226 GB/s, the part's measured streaming ceiling for scattered slabs (`bw-roofline`, research/batch-spec-20261101/GB10-PLAIN-DECODE.md; 234 GB/s contiguous, 273 GB/s nominal). ms per round from the nsys windows: 44 rounds at 1k, 42 at 131k.

| part | bytes per round | floor | 1k | 131k |
|---|---|---:|---:|---:|
| trunk matvecs, verify at T=4 (NVFP4) | 13.70 GB | 60.6 | 65.1 | 65.5 |
| output head, verify (Q5_K) | 0.874 GB | 3.87 | 5.08 | 5.09 |
| draft: 3 steps of the MTP block and its 32k head | 1.07 GB | 4.75 | 5.5 | 5.6 |
| verify attention, 16 layers | unique KV 0.03 / 3.89 GB, read 4 times | 0.14 / 17.2 | 0.82 | 99.65 |
| draft attention, the MTP layer, 3 steps | 0.006 / 0.73 GB | 0.03 / 3.2 | 0.1 | 4.9 |
| GDN scan and conv (fp32 state read and write) | 0.30 GB | 1.34 | 2.3 | 2.2 |
| rollback copies (per-round snapshot, per-column state clones, partial-accept restore) | 0.78 GB | 0 | 5.4 | 5.7 |
| norms, quantize, glue, argmax | | | 1.9 | 1.9 |
| launch gaps and host | | | 3.1 | 3.3 |
| round | | 70.7 / 91.0 | 89.5 | 195.8 |

KV bytes are 1,856 per token per full-attention layer (q8_0 K, q5_1 V). At the measured acceptance the floor is 37.5 / 28.5 / 25 tok/s at 1k / 131k / 260k, against 29.8 / 13.5 / 9.4 inside bursts and 22.1 / 10.3 / 6.9 end to end.

### The three biggest gaps

1. **Verify attention reads the KV once per verify row.** `fa_decode_vec_q_seqs_v4_deep` runs one grid z per verify row, and each row walks the whole cache: 15.6 GB per round at 131k for 3.89 GB of unique KV. That is 82 ms per round over the floor at 131k (42% of the round) and about 165 ms at 260k (56%).
2. **The last burst ran past EOS.** The burst's round flush answered only the admission yield, so a request's last burst ran out to its full 32-token target after EOS. The measured requests spent 13, 11 and 13 rounds on tokens nobody sees: 1.16, 2.1 and 3.8 s, about a fifth of each decode. Averaged over answer lengths it is about 6 rounds per request.
3. **Rollback copies and launch gaps at short context.** 0.78 GB of device copies per round keep the recurrent state rewindable, and with the launch gaps they cost about 8.5 ms of the 89.5 ms round at 1k.

### Levers, ranked by expected gain and exactness

| # | lever | exactness | expected gain | size |
|---|---|---|---|---|
| 1 | the burst ends at the round its public stream ends (EOS or a stop id; a gone client too) | visible stream identical: it is already complete | about 6 rounds per request on average, 11-13 measured | small, server |
| 2 | multi-row verify attention: one block reads each KV tile once for all verify rows | bit-identical per row (same walk, same split partition) | 131k: -55 to -80 ms per round; 260k: -110 to -160 | kernel |
| 3a | per-column state snapshots stay in the scan's slab (no clones) | bit-identical (same bytes) | -471 MB, about -2.8 ms per round | small |
| 3b | a per-layer state slot ring selected through the state pointer table (no snapshot, no restore) | bit-identical | about -2.6 ms more | medium |
| 4 | K re-priced on GB10 (the cold rungs were priced on other devices) | greedy byte-identical; sampled stays distribution-exact, the stream moves | to measure | measurement |
| 5 | matvec schedules toward the ceiling (head 76%, draft 85%, trunk 93%) | bit-identical schedules only | at most about 6 ms per round | kernel |
| 6 | one graph for the eager single-session round | bit-identical | about -2 ms per round | medium |

## Lever 1: the burst ends at the round its public stream ends

`SpecBurstEmit::flush`, the MTP burst's round hook, answered only the admission yield. After the round that committed EOS or a stop id it kept answering "continue", so every such request's last burst ran to its full 32-token target on tokens nobody sees, and the response's end waited for it. `spec_round_continues` now ends the burst at that round, through the exit a reached burst count and the admission yield already take. The public stream is complete at that point, so no visible token can move. A gone client ends the burst the same way. The single-session and pooled bursts share the hook, so both stop early. A request that ends on its token budget saves nothing: the burst target never exceeds the request's room, so the engine's own count already ends that burst.

### Was running past EOS a guard?

No. It was an omission that one later mechanism leaned on, and that mechanism no longer does.

- **History.** The hook's verdict was added for the admission yield alone (`MEMRA_ADMIT_YIELD`, 2026-08-06; its FLAGS.md row and `research/admission-20260806/VERDICT.md` list the empty flushes it polls and say nothing about EOS). The lines predate the repository's history import (49d1d6f659, 2026-09-01). The three later commits that touched them moved or trimmed them without a reason to continue: 392863ff9d (the round-cadence twin on this route), 4fbf093c2a (the extraction into `SpecBurstEmit` for the pooled cohort) and adbc1a1539 (the per-burst seam's removal; "poll-only" had named that seam). The sibling routes end at EOS on purpose (`glm_spec.rs`, `dflash.rs`).
- **What runs after the burst is the same.** The post-burst settle reads the returned burst through `spec_visible_prefix`, and the burst always holds the EOS round. Usage, finish reason, the fed and sampler mirrors and the tool-call tail are unchanged. Stop strings are checked after the burst and still let it run to its target. Logprobs requests never reach this route, and n > 1 waits outside it.
- **No new end state.** The admission yield could already end a burst at any round, the EOS round included, and EOS could already land in a burst's last round (GB10 g1k below).
- **The one thing that leaned on it: the spec pool's exact probe.** The parked session keeps every committed token. With about 25 tokens of overshoot after EOS the probe missed, and the turn took the affinity rewind. A removed door that trimmed the overshoot for budget-ended turns (`MEMRA_SPEC_BUDGET_CLAMP`, FLAGS.md "Removed doors, 2026-11-04") showed what a hit cost then: resumes at the parked end of a recurrent trunk, and warm turns that left the cold text. main 1991314b86 (memra#1093, landed with #1139) closed that: an exact or text hit on a recurrent trunk rewinds to the on-grid turn checkpoint or primes cold, and every pool resume starts the request's sampler counters as a cold session does. So the length of the parked tail no longer decides what a resumed turn computes. The warm == cold pair cells below check it at this lever's head.

### What else moves

- **Acceptance telemetry.** The rounds after EOS drafted on a finished answer and accepted little. They no longer run, so a request's `usage.spec` and the /metrics acceptance windows read higher for the same text (RTX 5090 g1k: 72 of 177 drafted tokens accepted over 59 rounds before, 65 of 126 over 42 rounds after). Acceptance figures from before and after this change do not compare. Where the cost model owns K (GB10), its acceptance input no longer includes those rounds.
- **A response whose first token is EOS** runs no round, so it carries no `usage.spec` block.

Served B/T, B = main a8a980a017, T = the lever, one fresh server per cell, B T T B B T, one cell per lease. GB10 is unit 2 (`eos-unit2/`). The RTX 5090 runs the same model and draft at a 32k context (`eos-5090/`). Requests per cell:

- `g1k`: greedy, about 1k prompt tokens;
- `g1k-t2`: its second turn, resumed from the parked session;
- `s1k`: sampled at temperature 0.8, seed 1234;
- `g131k`: greedy at 131k (GB10 only).

Medians of 3 per arm. The spread is max minus min.

| device | request | B total | T total | change | decode change | last burst, B / T | spread B / T |
|---|---|---:|---:|---:|---:|---|---:|
| GB10 | g1k | 4.784 s | 4.770 s | -0.3% | -0.2% | 14 / 14 rounds (EOS in the burst's last round) | 0.006 / 0.006 |
| GB10 | g1k-t2 | 1.667 s | 0.868 s | -47.9% | -64.2% | 14 / 5 rounds for 18 tokens | 0.003 / 0.000 |
| GB10 | s1k | 4.692 s | 4.339 s | -7.5% | -10.0% | 13 / 9 rounds for 23 tokens | 0.012 / 0.004 |
| GB10 | g131k | 129.889 s | 127.519 s | -1.8% | -19.1% | 17 / 5 rounds for 12 tokens | 0.075 / 0.096 |
| RTX 5090 | g1k | 2.504 s | 2.041 s | -18.5% | -27.0% | 20 / 3 rounds for 6 tokens | 0.139 / 0.023 |
| RTX 5090 | g1k-t2 | 0.757 s | 0.470 s | -37.9% | -67.6% | 15 / 5 rounds for 18 tokens | 0.031 / 0.013 |
| RTX 5090 | s1k | 2.517 s | 2.085 s | -17.2% | -26.8% | 18 / 3 rounds for 8 tokens | 0.090 / 0.061 |

- **Visible output is unchanged.** Every request served the same text in all six cells of both arms on both devices: greedy, sampled, and the resumed second turn.
- **The gain depends on where EOS lands.** It is the rounds after the EOS round. The GB10 g1k answer ended in its burst's last round, so it has nothing to save.
- **A round costs the same.** Inside full bursts it is 146.5 / 146.4 ms on GB10 and 30.8 / 31.2 ms on the 5090 (`*/rounds.txt`), each change inside its spread.

The session-gate oracle is an engine gate, and the lever does not touch the engine. On GB10 it passes at its defaults (K=2, 24 tokens per turn) on both builds. At K=3 with 48 tokens per turn it reports the same turn-2 mismatch on both builds, main included (`eos-unit2/session-gate-*.log`). So that mismatch is main's, not the lever's. Its reference re-primes the whole history, while the session continues from decoded state; since main 1991314b86 the served path never resumes a recurrent trunk at its parked end (the grid law, memra#1093). The mismatch is filed as memra#1146.

## The cold-long rung on GB10: K=2 against K=3 (lever 4's first cell, for memra#1093)

memra#1144 found that a cold sampled prompt of 1,024 tokens or more decodes faster at K=2 than at the cold-long rung's K=3 on the RTX 5090 (from 3.6k) and on the RTX PRO 6000 Max-Q (every length). This cell asks the same on GB10, with that lane's harness plus a GB10 prompt set and the served route's draft (`tools/cold_long_ab.py`, `PROMPT_SET=gb10`, `CELL_EXTRA`; the per-boot telemetry table is `research/spec-k-cold-long-20261109/analyze.py`).

- **Binary and route:** main a8a980a017, the served GB10 route (NVFP4 trunk, Q5_K head, the masked MTP draft, `MEMRA_PREFIX_STABLE_BOUNDARY=1`). Arm k3 is the default; arm k2 sets `MEMRA_SPEC_K_COLD_LONG=2`, the existing measurement door.
- **Boots:** 12, in the order k3 k2 k2 k3 three times, one boot per lease, unit 1. The first driver stopped at boot 7 with one request served; that boot was rerun whole, and `kcl-unit1/runs.jsonl` holds complete boots only.
- **Requests per boot:** nine cold prompts, one at a time. T=1 with a fixed seed per prompt: two of about 1.24k tokens, two of 3.6k, two of 5.8k. One greedy prompt of 3.6k. Two T=1 prompts of 34 tokens. Thinking off, `max_tokens` 400, and every answer reached the cap. TTFT is the first content chunk; the decode rate is the tokens after it over the time after it.
- **Clock:** the SM clock read 2,184 MHz in every boot and the GPU 65-69 °C (`kcl-unit1/analysis.md`). GB10 did not throttle, so the raw rates compare directly.

Per class, each boot's median, then the median over the arm's six boots (`tools/kcl_u.py`). U counts, over the 36 pairs of a k2 boot and a k3 boot, how often the k2 boot is faster (for TTFT, lower):

| class | k3 decode (median of boot medians, range) | k2 decode | k2/k3 raw | U raw | TTFT k2/k3 | U TTFT (k2 lower) |
|---|---|---|---:|---:|---:|---:|
| T=1 1.3k | 23.37 (23.05-23.44) | 23.06 (23.04-23.07) | 0.987 | 4/36 | 1.002 | 9/36 |
| T=1 3.6k | 23.62 (23.28-23.66) | 22.88 (22.88-22.90) | 0.969 | 0/36 | 1.000 | 15/36 |
| T=1 5.8k | 23.46 (23.44-23.50) | 22.71 (22.70-22.73) | 0.968 | 0/36 | 1.001 | 8.5/36 |
| T=1 all long | 23.46 (23.40-23.50) | 22.98 (22.97-23.00) | 0.980 | 0/36 | 1.000 | 15/36 |
| greedy 3.6k | 26.34 (26.31-26.36) | 26.29 (26.27-26.29) | 0.998 | 0/36 | 0.999 | 22/36 |
| T=1 short (reference) | 21.77 (21.75-21.79) | 22.72 (22.70-22.73) | 1.044 | 36/36 | 1.000 | 19/36 |

Per prompt (`kcl-unit1/per-prompt.md`):

| prompt | class | prompt tokens | k3 decode, median (range) tok/s | k2 decode | k2/k3 | k3 / k2 acceptance | texts per arm (6 boots) |
|---|---|---:|---|---|---:|---|---|
| rivers-1k | t1-long | 1,239 | 22.88 (22.57-22.96) | 22.94 (22.92-22.96) | 1.002 | 0.404 / 0.540 | 1 / 1 |
| bridges-1k | t1-long | 1,240 | 23.85 (23.52-23.91) | 23.18 (23.15-23.19) | 0.972 | 0.434 / 0.557 | 1 / 1 |
| glaciers-4k | t1-long | 3,566 | 22.23 (21.91-22.29) | 22.38 (22.37-22.39) | 1.007 | 0.395 / 0.535 | 1 / 1 |
| harbors-4k | t1-long | 3,697 | 25.01 (24.65-25.04) | 23.38 (23.37-23.41) | 0.935 | 0.499 / 0.593 | 1 / 1 |
| volcanoes-6k | t1-long | 5,727 | 23.30 (23.28-23.34) | 22.39 (22.38-22.41) | 0.961 | 0.453 / 0.557 | 1 / 1 |
| canyons-6k | t1-long | 5,928 | 23.62 (23.59-23.66) | 23.02 (23.02-23.05) | 0.975 | 0.455 / 0.587 | 1 / 1 |
| islands-4k | greedy-long | 3,565 | 26.34 (26.31-26.36) | 26.29 (26.27-26.29) | 0.998 | 0.460 / 0.617 | 1 / 1 |
| comets-short | t1-short | 34 | 22.30 (22.28-22.33) | 22.73 (22.69-22.74) | 1.019 | 0.383 / 0.535 | 1 / 1 |
| tides-short | t1-short | 34 | 21.24 (21.22-21.27) | 22.72 (22.70-22.74) | 1.069 | 0.346 / 0.535 | 1 / 1 |

- **K=2 does not win on GB10.** Four of the six long sampled prompts are 2.5% to 6.5% slower at K=2, two are flat (+0.2%, +0.7%), and the greedy prompt is flat (-0.2%). By class the 3.6k and 5.8k prompts lose 3.1% and 3.2% with K=3 faster in 36 of 36 pairs, and the 1.3k prompts lose 1.3% (32 of 36).
- **A reading of why it differs from sm_120a.** A K=2 round drafts one token less, and acceptance per drafted token rises from 0.40-0.50 to 0.54-0.59. On GB10 the draft is about 6% of a round and the trunk verify dominates (the roofline table above), so the extra verified row of K=3 costs little and its extra accepted tokens pay.
- **What the 36 pairs measure.** Seeds are fixed, so every boot of an arm serves the same text for a prompt (one text hash per prompt and arm). The pairs show that the machine repeats to 0.1-0.3%; the sample of texts is six sampled prompts and one greedy.
- **The greedy text is K-invariant:** one hash across both arms.
- **TTFT does not move** (ratios 0.999 to 1.002).
- **The door reaches further on GB10.** Every `[spec-k]` line on GB10 reads `source=cost-model`, and with the door at 2 the 34-token prompts ran K=2 as well. Those two prompts were 1.9% and 6.9% faster at K=2. So the short prompts are not a same-K reference here, and the harness's normalized columns are left out. It is also a lead for the GB10 cost model's short-prompt K, not a result: two prompts.

For memra#1093's flip: the cold-long K=2 default does not carry to GB10. GB10 keeps K=3 on that rung.

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
| 1 | the burst ends at the round its public stream ends (EOS, stop id, budget) | visible stream identical; burst size is content-neutral | about 6 rounds per request on average, 11-13 measured | small, server |
| 2 | multi-row verify attention: one block reads each KV tile once for all verify rows | bit-identical per row (same walk, same split partition) | 131k: -55 to -80 ms per round; 260k: -110 to -160 | kernel |
| 3a | per-column state snapshots stay in the scan's slab (no clones) | bit-identical (same bytes) | -471 MB, about -2.8 ms per round | small |
| 3b | a per-layer state slot ring selected through the state pointer table (no snapshot, no restore) | bit-identical | about -2.6 ms more | medium |
| 4 | K re-priced on GB10 (the cold rungs were priced on other devices) | greedy byte-identical; sampled stays distribution-exact, the stream moves | to measure | measurement |
| 5 | matvec schedules toward the ceiling (head 76%, draft 85%, trunk 93%) | bit-identical schedules only | at most about 6 ms per round | kernel |
| 6 | one graph for the eager single-session round | bit-identical | about -2 ms per round | medium |

## Lever 1: the burst ends at the round its public stream ends

`SpecBurstEmit::flush`, the MTP burst's round hook, answered only the admission yield. After the round that committed EOS, a stop id or the last budgeted token, it kept answering "continue", so every request's last burst ran to its full 32-token target on tokens nobody sees, and the response's end waited for it. `spec_round_continues` now ends the burst at that round, the exit the admission yield already takes. Burst size is content-neutral, so the visible stream cannot move. The single-session and pooled bursts share the hook, so both stop early.

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

The session-gate oracle is an engine gate, and the lever does not touch the engine. On GB10 it passes at its defaults (K=2, 24 tokens per turn) on both builds. At K=3 with 48 tokens per turn it reports the same turn-2 mismatch on both builds, main included (`eos-unit2/session-gate-*.log`). So that mismatch is main's, not the lever's. Its reference re-primes the whole history, while the session continues from decoded state; the served path never resumes a recurrent trunk at its parked end (the grid law, memra#1093). The mismatch is filed for triage as its own item.

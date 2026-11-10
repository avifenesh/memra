# GB10 single-user decode after round 5: re-scout and the next levers (2026-11-10)

Device: DGX Spark (GB10, sm_121a), two units. Model: Qwen3.8-27B GGUF (NVFP4 trunk, Q5_K output head) with the masked MTP draft (32,768-row head), 262,144 context, served the way the launcher serves it: the MTP speculative path at the automatic K (K=3 for these cold requests), 32-token bursts. Binary: main fcd66f8963 (round 5's levers 1 and 3a are in it) plus lever 2, the verify rows walk (memra#1162, the code of 4b4bb62abd).

The question is the first scout's (`research/gb10-decode-20261109/README.md`): where a single user's decode time goes against the part's bandwidth roofline, now that the burst ends at EOS, the verify column states stay in the scan's slabs and a deep verify reads the KV once. This note is the measurement and the designs. Nothing here is built.

## The cells

Same cells and prompts as the first scout, one per lease:

- `decode` (unit 2): one cold request each at about 1k, 131k and 260k prompt tokens, `max_tokens` 512, every server line timestamped, `MEMRA_TICK_TRACE=1`. The three answers are the first scout's texts, byte for byte (same sha256).
- `nsys1k`, `nsys131k`, `nsys260k` (unit 1): the same server under Nsight Systems, one request each, and the decode window after the first token. The 260k profile is new.
- `k3`, `k4`, `k5` (unit 1): both cold K rungs pinned to N through their own env (`MEMRA_SPEC_K_COLD_SHORT`, `MEMRA_SPEC_K_COLD_LONG`), one 1k and one 131k cold request each.
- `sp256`, `sp512`, `sp1024` (unit 2): `MEMRA_FA_SPLIT=N`, one 131k and one 260k cold request each (design 1's probe).

Receipts: `scout/`. Tools: `tools/` here and the first scout's `research/gb10-decode-20261109/tools/` (`r5_tab.py`, `r5_decode.py`, `r5_gaps.py`, `r5_round.py`).

## c1 decode

| prompt | TTFT | decode after the first token | first scout | inside full bursts | tokens per round | ms per round | first scout |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,036 | 1.13 s | 31.05 tok/s | 22.1 | 31.1 tok/s | 2.65 | 85.3 | 89 |
| 131,051 | 117.4 s | 17.00 tok/s | 10.3 | 17.2 tok/s | 2.59 | 150.8 | 192 |
| 259,951 | 313.9 s | 12.65 tok/s | 6.9 | 13.0 tok/s | 2.77 | 213.8 | 294 |

Single runs. Rounds, drafted and accepted counts per burst are the first scout's, token for token: the levers changed no value. The last burst now runs 3, 2 and 4 rounds where it ran 17, 14 and 18. End to end the decode is 1.41, 1.66 and 1.82 times the first scout's.

## One round against the roofline

Floors price each part's bytes at 226 GB/s, as in the first scout. ms per round from the profiles: 34 rounds at 1k, 32 at 131k, 30 at 260k. First scout values in parentheses.

| part | floor | 1k | 131k | 260k |
|---|---:|---:|---:|---:|
| trunk matvecs, verify at T=4 (NVFP4) | 60.6 | 66.0 (65.1) | 65.4 (65.5) | 65.2 |
| output head, verify (Q5_K) | 3.87 | 5.07 (5.08) | 5.11 (5.09) | 5.15 |
| draft: 3 steps of the MTP block and its 32k head | 4.73 | 5.7 (5.5) | 5.7 (5.6) | 5.8 |
| verify attention, 16 layers: the walk | 0.14 / 17.2 / 34.2 | 0.75 | 50.4 (90.7) | 98.5 |
| verify attention: the combine | | 0.07 | 9.0 (8.9) | 19.3 |
| draft attention, the MTP layer, 3 steps | 0.03 / 3.2 / 6.4 | 0.08 | 4.9 (4.9) | 9.8 |
| GDN scan and conv | 1.33 | 2.4 (2.3) | 2.4 (2.2) | 2.5 |
| rollback copies (per-round snapshot, partial-accept restore) | 0 | 2.5 (5.4) | 2.7 (5.7) | 2.5 |
| norms, quantize, glue, argmax | | 1.9 (1.9) | 2.1 (1.9) | 2.2 |
| launch gaps and host | | 2.4 (3.1) | 2.5 (3.3) | 2.9 |
| round | 70.7 / 91.0 / 111.1 | 86.9 (89.5) | 150.1 (195.8) | 213.7 |

The floor is 83% of the served round at 1k (70.7 of 85.3 ms), 60% at 131k and 52% at 260k. At the measured acceptance the floor is 37.5 / 28.5 / 25 tok/s against 31.1 / 17.2 / 13.0 inside bursts.

Two notes on the rows. "Launch gaps and host" holds the memsets too (0.10 / 0.33 / 0.74 ms per round). The long profiles hold one event that is not a round: the prefix cache publishes the prime boundary at the first burst boundary, one gap of 83 ms at 131k and 172 ms at 260k. The table leaves it out (`parts.txt` lists it apart); the first scout's 131k copies and gaps rows had it inside.

## What the round is made of now

1. **At 1k the round is the weights.** Trunk, head and draft matvecs are 76.8 of 86.9 ms and sit at 91.8%, 76% and 83% of the ceiling. Everything else is 10 ms: rollback copies 2.5, launch gaps and host 2.4, GDN 2.4, glue 1.9, attention 0.9. No part outside the matvecs is above 3% of the round. At 1k the verify still takes the seqs launch (the rows walk starts at 2,048 keys for this shape).
2. **Inside the trunk the weak launch is `ssm_out`.** Window means against each class's bytes (`kernels.txt`): the gate and up pair is 1.4 ms above its floor (29.8 against 28.4), the grouped q/k/v and gate launches 0.6 (13.6 against 13.0), the two single-matrix launches 3.4 (22.6 against 19.2). Per launch in round 6 of the 1k and 131k profiles (`round6.txt`): `ffn_down` (50.1 MB) takes 240 to 245 us against 222 at the ceiling; the 17.7 MB output projection takes 118 to 120 us after the GDN scan (`ssm_out`, 48 layers) and 85 to 92 us after attention (`attn_output`, 16 layers), against 78. Same kernel, same bytes, same shape. So of the 3.4 ms about 1.9 is `ssm_out`, 1.3 `ffn_down` and 0.2 `attn_output`.
3. **The deep verify walk reads the KV once and is no longer bandwidth bound.** Lever 2's kernel-check timing rows on unit 1 at 131k (GQA 6, T=4; memra#1162, `research/gb10-verify-rows-20261109/kernel-check/`) time one attention call, walk and combine: four groups of one row 6.34 ms, two groups of two rows 4.40 ms, one group of four rows 3.71 ms. One more KV read costs 0.7 to 1.0 ms (243 MB is 1.08 ms at 226 GB/s) and one row's own work 0.6 to 0.75 ms, 0.14 of it its share of the combine. In the served round the walk is 3.17 ms per call: about 1 ms is the read and the rest is the four rows' work, roughly 35 of the walk's 50.4 ms per round. That work is each (row, head) warp's score (64 dp4a per key) and its V accumulate (256 ordered multiply-adds per key), plus its partial write.
4. **Partials are 45% of an attention call's bytes at depth.** At 128 keys per split a 131k cache has 1,024 splits. The partials of one verify call are 1,024 splits x 96 (row, head) x 1,032 B = 101 MB: the walk writes them, the combine reads them. The KV itself is 243 MB. The combine is at its own bandwidth: 101 MB in 0.567 ms is 178 GB/s. Splits grow with the context, so the share is the same at 260k (201 MB of partials written and read against 482 MB of KV; the combine takes 1.22 ms per call).
5. **Rollback copies are down to 2.5 ms.** Lever 3a took the column copies out (641 MB per round at 1k to 172 MB). What is left is the per-round snapshot of every linear layer's conv and ssm state (`snapshot_into`: 96 copies, 157 MB, 1.5 ms, every round) and the partial-accept restore (`copy_batch_uniform_f32`, two launches, 1.39 ms in the rounds that have one, 0.84 to 1.0 ms averaged).
6. **The host is not in the way.** The gap between a round's last kernel and the next round's first is 1.7 ms at every context, 1.5 ms of it the snapshot copies, 0.2 ms host (`roundgap.txt`). Launch gaps under 5 us sum to 1.75 to 1.8 ms per round over about 1,150 launches.
7. **The prefix-cache publication stalls a long cold request once.** `prefix_insert_from_spec_boundary` copies the 16 trunk KV layers and the draft plane up to the prime boundary: 34 zeroed allocations and 34 device copies, 4.13 GB at 131k (copies 37.8 ms, zeroing 22.1 ms, the whole gap 83 ms) and 8.2 GB at 260k (76.1, 44.7, 172 ms). The plain path's publisher (`prefix_snapshot`) allocates without zeroing and copies in one batched launch.
8. **The draft attention zeroes its partials on every call.** `fa_decode_dc_q8` memsets the partial pool's live range before each of the three draft steps: 25 MB per call at 131k, 50 MB at 260k, 0.23 and 0.62 ms per round.

## K probes (unit 1, single requests)

| K | 1k: rounds | tokens per round | ms per round | decode | 131k: rounds | tokens per round | ms per round | decode |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 40 | 2.67 | 85.7 | 30.9 tok/s | 39 | 2.62 | 150.4 | 17.1 tok/s |
| 4 | 38 | 2.82 | 88.8 | 31.4 tok/s | 39 | 2.62 | 166.3 | 15.3 tok/s |
| 5 | 37 | 2.89 | 93.2 | 30.8 tok/s | 37 | 2.76 | 201.5 | 13.5 tok/s |

One cold greedy request per cell, the same two prompts; the texts are identical at every K. Rounds count the whole answer; ms per round is over every burst but the first, which holds the prime (`tools/r6_ktab.py`).

- At 1k a longer draft buys 2 and 3 rounds of 40 and costs 3.1 and 7.5 ms per round. The decode rate is flat inside a single run's noise (30.9, 31.4, 30.8).
- At 131k K=4 buys no round and costs 15.9 ms per round: one more verify row is 11 ms of attention work plus a draft step. K=5 costs 51 ms: six verify rows do not fit one row group at GQA 6 (five rows per block), so the round reads the KV twice.
- K=6 and K=2 were not run: the unit went to a landing record. Round 5 measured K=2 against K=3 on six prompts of 1.3k to 5.8k tokens and K=2 lost; K=2 at 131k is unmeasured. Lever 2 made a verify row cheaper, which favours more rows, not fewer.

K=3 stays on both cold rungs on GB10. A short-context gain from K=4, if any, is under 2% and would need the six-prompt cell to see.

## Next levers, ranked

Gains are ms per served round at 1k / 131k / 260k (85.3 / 150.8 / 213.8 ms).

| # | lever | exactness | expected gain | size | first step |
|---|---|---|---|---|---|
| 1 | a deeper split rung for deep caches (512 keys per split past a depth floor) | numeric: every path keeps the one partition, so speculative stays bit-identical to plain; the value differs from today's in summation order at those depths | 0 / -12.2 (8.1%) / -24.6 (11.5%), measured by the probe at 512 keys | small: one ladder rung per device, numeric qualification | done: the served probe through the existing `MEMRA_FA_SPLIT` door (design 1) |
| 2 | state slot ring: the verify scan writes its columns into a ring of T+1 state slots and the round's start moves by index (round 5's 3b) | bit-identical (the same bytes are read) | -2.3 at every context (2.7% / 1.5% / 1.1%) | medium | design 2 |
| 3 | the deep verify walk's per-row work | bit-identical per warp | unknown; the bound is the 35 / 68 ms of row work at 131k / 260k | kernel | an ncu profile of `fa_verify_rows_v4_deep` at 131k |
| 4 | trunk matvec schedules, `ssm_out` first | bit-identical schedules only | `ssm_out` up to -1.9 (-1.3 if it only reaches `attn_output`'s time), `ffn_down` up to -1.3, the pair up to -1.4 | kernel | find why the same launch costs 30 us more after the GDN scan |
| 5 | publish the spec prime boundary the way the plain path does | bit-identical | the 83 / 172 ms gap of a long cold request down to about its copy time (38 / 76 ms) | small | design 3 |
| 6 | one graph for the eager single-session round (round 5's 6) | bit-identical | at most -1.8 (the launch gaps) | medium to large | after 2: the ring changes what a round launches |
| 7 | output head, draft and GDN schedules | bit-identical schedules only | at most -1.2, -0.9 and -1.0 | kernel | |

Lever 1 is the only one that moves the long round by more than 3%: the probe measured +8.6% decode at 131k and +14.5% at 260k. It is not bit-identical to today's output, so it needs a decision before it is built. Levers 2 and 5 are exact and designed below. Levers 3 and 4 need an instruction-level profile before a design: both kernels were tuned in earlier rounds and the remaining gap has no single known cause.

## Design 1: a deeper split rung

**The gap.** `fa_split_keys` (lib.rs) gives 128 keys per split past 16,384 keys on every device. Its own history says why the ladder tapers with depth: the split count grows with the context and "the combine + partial-buffer cost dominates" (the 40k sweep that set 128, 2026-07-05; the 2026-08-02 re-sweep moved the short rungs for the same reason). The top rung was priced on plain decode at 40k. One row's partials are then a sixth of a call's bytes. At 131k and 260k with a four-row verify that reads the KV once they are 45%.

**The change.** One more rung in `fa_split_keys`: S keys per split from depth D on. S = 512 from the probe; D at or above 65,536 (at 512 keys a 65,536-key cache still has 128 splits per KV head), set per device from timing rows.

**Who reads the ladder.** The plain walk, the draft's `_dc` twin, the seqs launch, the rows walk and their combines take the split from `fa_split_keys`, so they move together and no kernel changes. Captured graphs size their partials from the same function at capture, and the straddle checks compare rungs, so both follow. Prime attention does not read the ladder. Two things to carry:

- `fa_decode_dcw_rows` (flash_attn.cu) spells the big-rig ladder as a literal, and its launchers refuse under `MEMRA_FA_SPLIT` or below 128 SMs. It does not run on GB10. A rung on the RTX PRO 6000 has to be mirrored in that literal, or its twins diverge.
- The draft's `_dc` launch keys its split on the scratch capacity bucket, not the live length, so it crosses D before the trunk does. The one-partition law holds per launch; kernel-check's twin cells have to cover that crossing.

**Exactness.** Speculative stays bit-identical to plain at every depth: kernel-check's FA cells pin the twins against each other at whatever rung the ladder gives, and `run-spec` re-gates self-consistency. The combined output at depths past D differs from today's in the order the partial sums are added (fewer, longer partials). The code names the class: a split-ladder change is a numeric-class door, "new tape + battery, exactly like every other split-ladder change". A greedy stream can flip on a near-tie at those depths.

There is no bit-identical way to cut the partial bytes on today's partition. The combine adds the splits' partials in split order from the global maximum. A block that walked several splits could not fold them into that sum: it would need the running value of the splits before it.

**Expected before the probe.** At 512 keys the combine reads a quarter of the bytes: -6.8 ms at 131k, -14.5 at 260k. The walk writes 76 MB less per call at 131k: up to -5.4 ms per round at the bandwidth price. The draft's partials shrink the same way (-0.5 to -1 ms). Sum: -7 to -13 ms at 131k, -15 to -27 at 260k. Plain decode at depth gains the combine's share too.

**The probe (no code).** `MEMRA_FA_SPLIT=N` forces N keys per split at every depth. One boot per N, one cold request at 131k and one at 260k. ms per round is over every burst but the first; the ladder's row is the `decode` cell through the same tool:

| keys per split | 131k: ms per round | decode | 260k: ms per round | decode | texts |
|---:|---:|---:|---:|---:|---|
| 128 (the ladder) | 150.7 | 17.00 tok/s | 213.7 | 12.65 tok/s | |
| 256 | 142.5 (-8.3, -5.5%) | 17.57 tok/s (+3.4%) | 195.6 (-18.1, -8.5%) | 13.89 tok/s (+9.8%) | both differ by a few words |
| 512 | 138.5 (-12.2, -8.1%) | 18.47 tok/s (+8.6%) | 189.1 (-24.6, -11.5%) | 14.49 tok/s (+14.5%) | both identical to the ladder's |
| 1,024 | 136.7 (-14.0, -9.3%) | 19.23 tok/s (+13.1%) | 186.3 (-27.4, -12.8%) | 14.59 tok/s (+15.3%) | 131k identical; 260k differs in its last clause |

Unit 2, single runs, the same two prompts. TTFT is unchanged (117.3 to 117.4 s and 312.5 to 313.9 s across the four).

- At 512 keys both greedy texts are byte-identical to the ladder's, and so are the rounds, drafted and accepted counts (39 rounds each): on these prompts the summation order moved no token.
- At 256 and 1,024 keys the numeric class shows. At 1,024 the 131k answer is identical but took 38 rounds for 39 (one draft token moved), and the 260k answer ends "to continue the cycle." where the ladder's ends "to create a looping pattern." (109 tokens for 110, both stop on EOS). At 256 the 260k answer is the 1,024 one and the 131k answer says "identifier" where the ladder's says "counter" (40 rounds). Every answer still returns both needles.
- 256 gets about two thirds of 512's gain. 1,024 adds 1.3% and 1.5% per round over 512.

The probe lands at the top of the estimate. It does not say where the rung should start: `MEMRA_FA_SPLIT` moves every depth.

**Then.** The rung in `fa_split_keys` (S and D from kernel-check timing rows per device: GB10, the RTX 5090, the RTX PRO 6000 Max-Q; a device where it does not pay keeps 128), kernel-check all green, the numeric gate, `run-spec` K=1..8 on both roster models, served B/T at 131k and 260k with ms per round as the primary signal and TTFT as no-regression.

## Design 2: the state slot ring

**Today.** The packed verify scan (`gdn_scan_s128_tsnap` and its conv twin, rounds with T >= 2) reads each linear layer's state through the state pointer table, writes the state after each of the first T-1 columns into a column slab from the round's scratch pool, and writes the state after the last column over the live buffer (in place at even T, into `ssm_state_alt` at odd T). Because the live state is overwritten, each round first copies every linear layer's conv and ssm state into the session's snapshot (`Cache::snapshot_into`: 96 device copies, 157 MB, 1.5 ms). A partial accept of j columns copies column j-1's state from the slab back into the live buffers (`commit_verified_prefix`: two `copy_batch_uniform_f32` launches, 1.39 ms). A full accept copies nothing. A round that keeps no column restores the snapshot and replays through the per-row arm.

So 157 MB of state moves once every round and a second time on a partial accept, only to sit where the next scan expects it.

**The change.** For the eager single-session burst, a ring of T+1 state slots per linear layer (conv and ssm), taken from the scratch pool for the burst and sized by the burst's largest T. The first round of a burst starts from the live buffers. The scan reads its start state from the start pointer and writes column c's state to ring slot (s + 1 + c) mod (T + 1), the last column included, where s is the start's slot (no slot when the start is the live buffer): it never writes the state it started from. After the verify:

- j columns kept, 1 <= j <= T: the next start is the slot of column j-1. No copy.
- no column kept: the start is unchanged, so nothing was lost and no snapshot was needed.

The kernel change is where a column's state lands: ring base + ((s + 1 + c) mod (T + 1)) x stride for every column, instead of slab + c x stride for the first T-1 and the live or alt buffer for the last. The start pointer goes through the pointer table the scan already resolves at launch. Values and their order are untouched, so each state is the same bytes as today's.

**When the live buffers must hold the state.** Everything outside the packed scan reads `rl.conv_state` / `rl.ssm_state` (and the alt buffer's parity). The start slot is materialized into the live buffers, one batched copy, at each of these points:

- the burst's exit, normal or by error (plain decode, the prefix-cache capture, park and restore and the pooled handoff read the live state after it);
- before the replay of a round that kept no column (the legacy arm runs the per-row program on live and alt); today that round pays a restore copy anyway;
- before any round that does not take the packed scan (T=1, the per-row arm) and before a verify-graph round in the same burst (`refresh_tables` reads the live handles).

Swapping buffer handles with the live state at exit, as the scan's canonical/alt swap already does, would remove that copy too; it is a second step.

A burst is about 12 rounds, so the copies fall from 2.5 ms per round to about 0.2.

**Memory.** The ring is pool scratch, not session state: T+1 slots (785 MB at K=3) where the pool holds T-1 column states today (471 MB), 314 MB more for the engine, nothing per session. The session's snapshot and alt buffers stay: the other paths use them.

**Bit identity and gates.** kernel-check: the scan with ring destinations against today's slab and live buffer, for every start slot and every T up to the largest K (every column's state bit-equal). `run-spec` K=1..8 on both roster models (partial accepts, full accepts and the no-column round). The pooled-cohort stream cell and a served cell that mixes eager and verify-graph rounds in one burst. Served B/T at 1k and 131k, ms per round inside full bursts, texts identical.

**Scope.** The eager single-session round, where the copies were measured. The pooled rows path and the captured verify graph keep today's slab.

## Design 3: the spec prime boundary's publication

`prefix_insert_from_spec_boundary` (worker.rs) allocates a zeroed plane per KV layer and copies into it with one `memcpy_dtod` each: 34 allocations, 34 memsets and 34 copies for 4.13 GB at 131k. `prefix_snapshot`, the plain path's publisher, takes its planes from `prefix_plane_alloc` (no zeroing) and copies all of them in one `copy_batch_items_u8` launch.

The fix: publish the speculative boundary through that plane allocator and batched copy, the draft plane as one more item. Every byte of a published plane up to the boundary is written by the copy, so nothing reads an unwritten byte.

Expected: the zeroing goes (22 / 45 ms) and 34 allocations become one pass. The copy itself is already at the ceiling (4.13 GB read and written in 37.8 ms), so the gap falls from 83 / 172 ms to about 38 / 76. One thing to keep: today a failed draft-plane copy drops only the draft plane; the batch has to keep that isolation or refuse the entry whole.

Gate: the prefix-cache restore cells (a second request on the published prefix serves the same bytes as on today's entry) and the decode cell's longest inter-round gap.

## Not levers

- **K on GB10.** K=3 stays on both cold rungs (the probes above).
- **The combine's schedule.** It is at 178 GB/s on its own bytes. Its cost is the bytes (lever 1), not the loop.
- **The host round hook.** 0.2 ms per round.
- **The draft attention's partial zeroing** (finding 8, 0.2 / 0.6 ms per round). The deep `_dc` kernel writes every (head, split) partial its combine reads, so the zeroing is redundant for values. But `fa_decode_dc_q8` is a shared launcher (trunk graphs, six kernel arms), and its comments say token-graph retargeting finds attention children by their three-memset signature. Not worth a change on its own.

## Receipts

- Every cell: `pf.jsonl`, `guard.log` (the binary's sha256 and the server env), `trace.txt` (the server log's spec, prefix-cache and burst lines).
- `scout/decode-unit2/`: `tab.txt` (`r5_tab.py`).
- `scout/nsys1k-unit1/`, `scout/nsys131k-unit1/`, `scout/nsys260k-unit1/`: `kernels.txt` (`r5_decode.py`), `copies-gaps.txt` (`r5_gaps.py`), `parts.txt` (`tools/r6_parts.py`, the table above), `roundgap.txt` (`tools/r6_roundgap.py`), `round6.txt` (`r5_round.py`, one round's launches in order; 1k and 131k), `big.txt` (`tools/r6_big.py`, the publication's copies; 131k and 260k).
- `scout/k-unit1/kN/`, `scout/split-unit2/spN/`: `tab.txt`; the K and split tables are `tools/r6_ktab.py` over these directories.
- `tools/r6_cell.sh`, `tools/r6_driver.sh`: the cells and their one-lease-per-cell driver. `tools/r6_export.py` wrote `scout/`.
- Profiles (`prof.sqlite`), GPU telemetry and memory samples stay on the machine.

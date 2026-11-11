# The spec prime boundary publishes its planes without zeroing them (GB10 decode round 6, lever 5)

Verdict: shipped as a small measured gain; the design's expectation did not hold. The publisher's planes skip the memset and keep one device copy each. On GB10 with Qwen3.8-27B the one pause after a long cold request's first burst falls by 13.0 ms at a 131k prompt on the first burst boundary (77.4 to 64.4 ms, -17%) and by 14.1 ms on the publisher's time on the scheduler thread (65.3 to 51.3 ms, -22%), every cell of the head under every cell of main. At 260k: 157.4 to 118.8 ms (-25%) and 142.2 to 112.3 ms (-21%), three cells of each. Under Nsight Systems the pause between burst 1's last kernel and burst 2's first falls from 72.8 to 58.6 ms at 131k. Round time and every served text are unchanged; TTFT is unchanged at 131k and 0.1 s of 313 s higher at 260k on every head cell, a measured cost of unknown cause (the publication runs after the first token). The design (no memset and one batched copy launch for all planes) took 7.0 ms off at 131k: its launch waits for the last allocation and then runs under the next burst. The published entry is byte-identical to main's (18 entries, round-trip digests equal), the head's warm == cold pair cases serve main's answers and bursts, with main's published-entry digests on the sampled cases, and a plane whose copy failed is never published (a forced failure, with its red arm). What is left at 131k is 43 ms of device copies (4.45 GB at about 210 GB/s, read plus write) and 16 ms in which the device waits for seven slow allocation calls.

Device: DGX Spark (GB10, sm_121a), two units. Model: Qwen3.8-27B GGUF (NVFP4 trunk, Q5_K output head) with the masked MTP draft (32,768-row head), 262,144 context, served as the launcher serves it: the MTP speculative path, K=3 on a cold request, 32-token bursts. Dates: 2026-11-10 and 11. Arms, all sm_121a builds from one tree and one target directory on unit 1 (`build/`); the cells after the rebase are in "After the rebase":

| arm | commit | the publisher |
|---|---|---|
| B | main 2628ece940 | a zeroed plane per K and V, one device copy per plane |
| W | 3afa740d13, this PR's code | planes without a memset, one device copy per plane |
| U | a25c08f05e | W's program as a two-line change to main (the probe W was built from) |
| T | ee37f8d3b4 | planes without a memset, every plane in one batched copy launch (the design) |

U and T are not on this branch. They are archived on `bank/spec-boundary-publish-20261110/a25c08f05e` and `bank/spec-boundary-publish-20261110/ee37f8d3b4`.

The design is `research/gb10-decode-20261110/README.md`, "Design 3: the spec prime boundary's publication", and its finding 7. It proposed arm T and expected the pause to fall from 83 to about 38 ms at 131k. That did not happen. What the cells show is below.

## The change

A cold speculative request publishes its prompt's KV to the prefix cache once, at the first burst boundary (`prefix_insert_from_spec_boundary`, `worker.rs`). Per attention layer and for the MTP draft plane the publisher allocated a zeroed K plane and a zeroed V plane (`alloc_zeros`) and then overwrote every byte of each with its own device copy: 34 allocations, 34 memsets and 34 copies on Qwen3.8-27B (16 trunk layers and the draft plane), 4.13 GB at a 131k prompt.

At the head the planes come from `prefix_plane_alloc`, the plain snapshot's allocator (design B1 of `research/spill-a-20260919/DAY59.md`): no memset for a plane with bytes, the zeroed 1-byte buffer for a plane without. Each plane is exactly `pos * tok_bytes` long, as before, and its copy writes `[0, pos * tok_bytes)`, all of it. So the entry holds the bytes it held before. The copies stay one per plane, through one helper (`spec_plane_copy`). There is no switch.

Nothing else about the publication moves: the boundary position, the stable-boundary capture, the recurrent planes (the capture's own buffers), the boundary logits and hidden row, the latent planes of an MLA or DSA cache, the DSpark or GLM-5 drafter tail (exported by the caller) and the insert's budget check, which still runs after the copies are enqueued. The same publisher serves the DSpark and GLM-5 boundary captures (their callers pass no draft plane), so their trunk planes lose the memset too.

### Failure isolation

Unchanged, and now it carries more weight, because a plane that is not written holds whatever its allocation held.

| event | before | at the head |
|---|---|---|
| a trunk plane cannot be allocated | the insert is abandoned | the same |
| a trunk plane's copy fails | the insert is abandoned | the same |
| the draft source is shorter than the boundary | the entry publishes trunk-only, with its log line | the same |
| a draft plane cannot be allocated, or its copy fails | the entry publishes trunk-only | the same |
| a plane has no bytes | a zeroed 1-byte buffer, no copy | the same (the allocator's zero-byte arm) |

The guarantee is in two places. `spec_plane_copy` returns true only when the copy of the whole plane was enqueued, or the plane has no bytes. Each of its four call sites drops the plane on false: the trunk arm returns from the publisher before the entry exists, the draft arm returns no plane. A census pins both arms, and a GPU cell forces a copy to fail (below).

## Exactness

The claim has two halves: a published entry holds the bytes it held on main, and a plane that was not written is never published.

**CPU census, fail before and pass after** (`cpu-checks.txt` at c232c9132a on main 66ee31fcb6, `cpu-checks-d117bf8607.txt` and `cpu-checks-3afa740d13.txt` on the earlier mains; a CPU build host, sm_120a build). `worker::tests::spec_boundary_publish_allocates_no_zeroed_plane` reads the publisher's source: no `alloc_u8` and no `alloc_zeros` in it, four `prefix_plane_alloc` calls and four `spec_plane_copy` calls (trunk K and V, draft K and V), the trunk arm's `return` and the draft arm's `return None` on a failed copy, the helper's body, and the allocator's two arms. On main's production code with the head's tests (`tools/red_worker.py`) it fails with `the spec publisher calls engine.alloc_u8(`; at the head it passes.

**GPU unit cells** (`unit-cells/3afa740d13/verdicts.txt`, GB10 unit 1; test binaries of main's production code with the head's tests, of the head, and of the red arm):

- `spec_boundary_publish_is_the_copy_program`: a live cache filled with random bytes on a tiny hybrid config and on the 27B's own geometry (17 attention layers, the last one standing for the unexecuted MTP layer, 48 recurrent layers), boundaries at 1 and 97 tokens, the live cache at the boundary and 34 rows past it, each published with a draft plane, with a draft source one byte short and with none. Every trunk and draft plane of the entry equals `[0, pos * tok_bytes)` of its source, the byte count is the planes' sum, and a short or absent draft source publishes trunk-only. It passes on main's publisher and on the head's, and the 18 entries' round-trip digests (trunk KV, recurrent planes, draft plane, hidden row, logits; `digests-base.txt`, `digests-head.txt`) are the same 18 lines. That is the first half, plane by plane.
- `spec_boundary_publish_drops_a_plane_whose_copy_failed`: a test-only fault makes one `spec_plane_copy` call report failure and copy nothing. A failed trunk copy (the first K, the first V, the last V; on the tiny config's two trunk planes the last V is the first V) publishes no entry and leaves the cache empty. A failed draft copy (K, then V) publishes the entry trunk-only, its trunk planes equal to their sources. Green at the head on both configs (2 and 32 trunk planes). **Red arm**: the head with the publisher ignoring the failed copy (`tools/red_unwritten.py`) fails the cell with `an entry was published although the copy of the first trunk K failed` (exit 101). That is the second half.
- `an_unwritten_plane_is_charged_to_the_insert_budget_reading`: the insert budget on unified memory reads free memory right after the copies are enqueued. A 2 GiB plane moves that reading from 118,550 to 116,468 MiB when it is allocated and not at all when it is written, so the budget does not depend on whether a copy has run.
- `b1_snapshot_and_restore_are_the_copy_program`, the plain snapshot's cell, stays green at the head (it shares the allocator).

The copy-program, budget and B1 cells also ran at arm T's head (`unit-cells/ee37f8d3b4/`): green, 18 digests equal to main's.

**Warm == cold pair cells** (`tools/specpub_pairs.py`, the pair cells of `research/spec-pool-grid-20261109/` with a deep chat shape; GB10 unit 2, binaries copied from unit 1, sha256 equal on both). One warm boot serves a conversation in order; turn 1 is cold and its speculative session publishes at the first burst boundary; each later warm turn must hit an entry the spec publisher wrote and restore a speculative session from it, trunk and draft plane (`[prefix-cache] hit`, `spec restore: N of M prompt tokens + draft plane from cache`, N equal to an `insert (spec-boundary)` line's token count and to no other insert's); one cold boot per later turn primes that turn's prompt from scratch; every warm answer must be byte-equal to its cold twin. Every turn is bounded at 64 tokens, so no session parks and the prefix cache is the only warm path. Turn 3 restores from an entry a restored session published. Short is 1,000 prompt tokens, deep 135,545. Greedy cases run in the served shape; sampled cases (temperature 1.0, seed 1234) run with `MEMRA_PREFIX_SPLIT_TRACE=1`, which hashes every published entry's trunk KV, recurrent state and boundary logits.

On main 2628ece940 (`pairs/unit2/table.txt`, one file per case next to it; W is 3afa740d13's binary):

| case | arm | later turns equal to cold | restored from a spec-published entry, with its draft plane | answers (sha256, per turn) |
|---|---|---|---|---|
| short, greedy, 3 turns | B | = / = | 960 of 1,083; 1,056 of 1,171 | f1fb5d52 / ce72af3e / 8c7ed81f |
| short, greedy, 3 turns | T | = / = | 960 of 1,083; 1,056 of 1,171 | f1fb5d52 / ce72af3e / 8c7ed81f |
| short, greedy, 3 turns | W | = / = | 960 of 1,083; 1,056 of 1,171 | f1fb5d52 / ce72af3e / 8c7ed81f |
| short, sampled, 3 turns, digests | W | = / = | 960 of 1,082; 1,056 of 1,169 | cde53ba3 / fcbcfe74 / 5b17104a |
| deep, greedy, 3 turns | B | = / = | 135,520 of 135,628; 135,584 of 135,716 | a99396c5 / a13b08a4 / 2d03a4e6 |
| deep, greedy, 3 turns | W | = / = | 135,520 of 135,628; 135,584 of 135,716 | a99396c5 / a13b08a4 / 2d03a4e6 |
| deep, sampled, 2 turns, digests | B | = | 135,520 of 135,612 | 5ac95257 / 7e34878c |
| deep, sampled, 2 turns, digests | W | = | 135,520 of 135,612 | 5ac95257 / 7e34878c |

Every case is warm == cold on every arm, and every warm turn restored from an entry the spec publisher wrote. Where two arms ran a case they served the same answers with the same bursts (rounds, drafted and accepted counts per burst), and on the deep sampled case main's and the head's published entries have the same state digests (trunk KV, recurrent state, boundary logits; the entry a restored session published at 135,584 tokens is dac1cf80... on both arms and on both arms' cold boots). Main's short sampled case did not run on 2628ece940; it ran after the rebase, on main 11f3c0920b, next to the head's (below).

**Served texts.** Every timed cell of every arm served the same text at its context (the section above): 18 cells at 131k, 6 at 260k.

## Served B/T

One cold request per cell on a fresh server: a 1k warm-up request, then the context's request, greedy, 512 max tokens, every server line timestamped, `MEMRA_TICK_TRACE=1` (`tools/specpub_cell.sh`, `tools/specpub_ab.py`; the prompts are the split rung lane's served cells', token for token). GB10 unit 1, one cell per lease. Medians with spreads (max - min); N per arm in each table. Thermal regime: the unit at its stock power and clocks (maximum SM clock 3,003 MHz in every cell's `guard.log`), not pinned; no per-cell temperature or clock trace was recorded, and the arms alternate within each run so a drift reaches both.

The publication runs between the request's first burst and its second. Two readings of it come from the server's own lines (`tools/specpub_tab.py`):

- **the publisher on the scheduler thread**: from burst 1's `[tick-spec]` line to the `[prefix-cache] insert (spec-boundary)` line;
- **the first burst boundary over its rounds**: burst 1's line to burst 2's line, minus burst 2's rounds at the rate of the later bursts. It also counts publication work that is still running on the device when burst 2 starts. It is the noisier one: the later bursts are 11 rounds at 131k, and one cell swings by about 10 ms.

### 131k (131,046 prompt tokens, entry 131,008 tokens, 4,290.5 MB)

18 cells in three runs of one campaign: B T T B B T, then U B T U T B U, then W B W B W.

| | B (7 cells) | W (3) | U (3) | T (5) |
|---|---:|---:|---:|---:|
| first burst boundary over its rounds, ms | 77.4 (12.6) | 64.4 (11.5) | 55.7 (7.1) | 70.4 (10.3) |
| publisher on the scheduler thread, ms | 65.3 (3.6) | 51.3 (3.6) | 52.6 (2.8) | 22.1 (2.6) |
| burst 1 line to burst 2 line, ms | 2005.3 (14.6) | 1994.4 (12.2) | 1990.2 (6.0) | 1997.0 (13.0) |
| burst 2, ms per round (14 rounds) | 138.62 (1.19) | 138.63 (0.78) | 138.56 (0.47) | 141.06 (0.96) |
| later bursts, ms per round (11 rounds) | 137.72 (1.08) | 138.33 (1.23) | 138.01 (0.43) | 137.64 (1.59) |
| TTFT, s | 117.862 (0.208) | 117.888 (0.161) | 117.927 (1.109) | 117.817 (0.185) |

- **W against B: -13.0 ms (-17%) on the boundary, -14.1 ms (-22%) on the publisher's time.** Every W cell is under every B cell on both.
- U is W's program and overlaps W on every row. Its three cells read -21.7 ms (-28%) on the boundary estimator: the favourable end of that estimator's spread, not a different result. On the publisher's time U is -12.8 ms.
- **T, the design, takes 7.0 ms off the boundary (-9%).** Its publisher returns in 22.1 ms: the same allocations, then one table upload and one launch. The copy launches themselves are not what it saves (W's 34 launches take 1.5 ms of host time in the profile); W's allocation calls take 52.7 ms on the host with copies running beside them, T's whole publisher 22.1 ms with none (no profile of T, so the cause is not measured). But the launch cannot start before the last plane is allocated, and the copy then runs under burst 2: burst 2 costs 141.06 ms per round on T against 138.62 on B, +2.4 ms on each of 14 rounds. Every U cell is under every T cell on the boundary (53.3 to 60.4 against 60.8 to 71.1).
- No regression on W: burst 2 and the later bursts cost what B's do, TTFT is equal, and all 18 cells served one text (sha256 e3b0c1e81f9a, 95 tokens) with one burst structure (rounds, drafted and accepted counts). One U cell has a TTFT of 118.98 s against 117.8 to 117.9 elsewhere; its decode readings sit with the other U cells.

### 260k (259,946 prompt tokens, entry 259,904 tokens, 8,357.4 MB)

B W W B B W, one campaign. A release record took the unit after the fifth cell; the sixth (06-W) ran the next night on the same binaries, unit and prompt.

| | B (3 cells) | W (3) | W - B |
|---|---:|---:|---:|
| first burst boundary over its rounds, ms | 157.4 (18.2) | 118.8 (14.4) | -38.7 (-25%) |
| publisher on the scheduler thread, ms | 142.2 (1.5) | 112.3 (4.7) | -29.9 (-21%) |
| burst 1 line to burst 2 line, ms | 3166.2 (33.9) | 3119.4 (22.3) | -46.7 |
| burst 2, ms per round (16 rounds) | 189.10 (2.11) | 187.69 (1.42) | -1.42 |
| later bursts, ms per round (30 rounds) | 188.00 (1.02) | 187.54 (0.50) | -0.46 |
| TTFT, s | 313.551 (0.252) | 313.647 (0.291) | +0.096 |

Every W cell is under every B cell on both readings of the pause. With five cells (before 06-W) the figures were -33.6 ms (-21%) and -28.5 ms (-20%); the sixth cell (114.5 and 112.3 ms) is W's lowest on the boundary. The round costs the same, and all six cells served one text (sha256 5275df37e59f, 134 tokens) with one burst structure. Every W cell has a higher TTFT than every B cell, by 0.07 to 0.6 s of 313 s. The change does not reach prefill; with three cells per arm this is reported as measured.

Per cell: `served/bt131k/table.txt`, `served/bt260k/table.txt`.

### TTFT: the publication runs after the first token

The 260k TTFT ordering (every W cell above every B cell, +0.096 s on the medians) is kept as a measured cost of unknown cause, not called drift. What the record can say about it:

- **The publication runs after the first token is sent, in every cell of every arm** (`tools/specpub_ttft_order.py` on the per-cell `ab.jsonl` and `trace.txt`; client and server share one clock). The first token goes out 12 to 13 ms (260k) and 23 to 25 ms (131k) after burst 1 starts; the `insert (spec-boundary)` line comes 2,349 to 2,383 ms (260k) and 1,534 to 1,581 ms (131k) after the first token. The publisher's allocations and copies are inside that window, so they cannot delay the first token, and moving them later would change nothing for TTFT.
- **The head changes no code that runs before the first token.** Its production diff is the publisher (`prefix_insert_from_spec_boundary`) and its copy helper (`spec_plane_copy`); `prefix_plane_alloc` is main's existing allocator for the plain snapshot. Prefill, the prime and the stable-boundary capture are main's code on both arms.
- **131k shows no ordering**: W 117.888 (0.161) against B 117.862 (0.208); W's cells (117.784 to 117.945 s) sit across B's (117.803 to 118.011 s), one below all of them.
- With three cells per arm, every W above every B happens by chance in 1 of 20 orderings when the arms do not differ. Prefill takes 313 s here and its spread within one arm is 0.25 to 0.29 s.

## Under Nsight Systems

One profile per arm, B and W, at 131k (131,072-token target, 512 max tokens, one cold request on a fresh server under `nsys profile -t cuda,nvtx`; `tools/specpub_nsys.sh`), GB10 unit 1, 2026-11-11, the same binaries as the served cells. The pause is the longest gap between two decode kernels of the request's decode window: burst 1's last kernel to burst 2's first. `tools/specpub_gap.py` lists what runs inside it (`nsys/*/gap.txt`), `tools/specpub_split.py` splits it into host and device time (`nsys/*/split.txt`). One profile per arm: single runs, read next to the served cells' medians.

| inside the pause | B | W |
|---|---:|---:|
| the pause, ms | 72.83 | 58.63 |
| plane copies (64 MB or more): count, ms, MB | 34, 39.64, 4,133.6 | 34, 39.55, 4,133.6 |
| small copies: count, ms, MB | 205, 2.88, 314.0 | 205, 2.90, 314.0 |
| memsets: count, ms, MB | 135, 23.79, 4,290.5 | 101, 0.62, 157.0 |
| device busy (copies, memsets, kernels), ms | 66.32 | 43.07 |
| device idle, ms | 6.52 | 15.56 |
| host in `cuMemAllocAsync` while the device idles, ms | 5.55 | 14.54 |
| allocation calls on the host: count, ms | 134, 66.22 | 134, 52.72 |
| of them 1 ms or more: count, ms | 7, 66.01 | 7, 52.48 |
| the longest five, ms | 16.30, 12.84, 12.68, 11.54, 6.84 | 16.01, 13.60, 7.32, 7.10, 3.37 |
| copy launches on the host, ms | 1.40 | 1.52 |

The 34 large memsets (23.2 ms of device time, 4.13 GB) are gone at the head, and 14.2 ms of the pause with them: the rest of their time was already under allocation calls. The copies are the same on both arms, about 210 GB/s counting a read and a write. The publisher's host side is almost all allocation: 127 of its 134 allocation calls take 0.25 ms together, and seven take 52.5 ms. The last runtime call inside the pause ends with it on both arms, so the host is on the path the whole time.

## What is left of the pause

At 131k on the head's profile, 58.6 ms:

- **43.1 ms of device work**, 42.4 ms of it the copies (4.45 GB). At about 210 GB/s, read plus write, this is near what GB10's memory moves; a faster copy will not remove it. Only a publication that does not copy the planes (the entry taking the session's own pages up to the boundary, with the session writing past it) does. Copying under the next burst instead costs the burst: arm T's copy under burst 2 added 2.4 ms to each of its 14 rounds.
- **15.6 ms in which the device waits for the host**, 14.5 ms of it inside `cuMemAllocAsync`. Seven allocation calls take 52.5 ms, mostly hidden under the copies. If allocation returned at once, the pause would fall to the copies' 43 ms. The engine already sets the default pool's release threshold to the maximum at boot (`memra-engine` `lib.rs`), so freed blocks stay cached; seven slow calls among 127 fast ones read as the pool reserving new memory for planes no cached block fits. The pool's reserved bytes around the publisher were not recorded, so that reading is not proven here. Arm T's publisher, the same allocations with no copy running beside them, returned in 22.1 ms in all, so the copies running beside W's allocations may be what makes them slow; T was not profiled.

At 260k there is no profile. The publisher's time at the head is 112.3 ms and the boundary 118.8 ms; the copies of 8.36 GB alone would take about 80 ms at the 131k rate (an estimate from the 131k profile, not a reading).

Both are beyond this change. memra#1221 sizes them from the profile above.

## After the rebase

The branch was rebased twice: from main 2628ece940 onto 11f3c0920b (v0.141.1), then onto 66ee31fcb6. Main moved `worker.rs` in between: the speculative stop strings, the burst-entry yield bound and the DSpark park shape (by 11f3c0920b), then the speculative cost model's tie-break (`SpecCostEntry`, `spec_cost_verdict`, by 66ee31fcb6). The code commit applied without a conflict both times and is the same patch (d117bf8607 on 11f3c0920b, c232c9132a on 66ee31fcb6), and none of those changes reaches the publisher (`prefix_insert_from_spec_boundary`), the boundary captures' drain helpers or the allocator. Which main each cell used: every cell above ran on binaries from 2628ece940 (B) and from 3afa740d13 (W, the same code on that main); the cells below ran on 11f3c0920b (B') and d117bf8607 (W'). No cell ran on 66ee31fcb6; the CPU checks did (`cpu-checks.txt`, at c232c9132a).

Main's short sampled pair case, parked for the new main, ran on it next to the head's, plus one 131k cell per arm, on unit 1 (B' = main 11f3c0920b, W' = d117bf8607, this PR's code on that main; `tools/spark_build_11f3.sh`, `tools/l5_11f3.sh`, `build/build-*-11f3c0920b.log`, `build/build-head-d117bf8607.log`):

- **The short sampled pair case on main 11f3c0920b, both arms** (unit 1, `pairs/unit1-11f3/`; the files name the arms B and W, here B' and W'). Main's case, parked for the new main, and the head's: warm == cold on both later turns on both arms, each restored from a spec-published entry with its draft plane (960 of 1,082 and 1,056 of 1,169 tokens), answers cde53ba3 / fcbcfe74 / 5b17104a on both. B', W' and the head's case on main 2628ece940 (W, unit 2) served the same answers with the same bursts and published the same entry digests (`pairs/short-sampled-main11f3-vs-head.txt`), across both mains and both units.
- **One 131k cell per arm on main 11f3c0920b** (unit 1, `served/bt131k-11f3/`, the same cell as the campaign above: B' then W', one cell per lease). Single cells, so read them against the campaign's spreads:

| | B' (1 cell) | W' (1 cell) | W' - B' | the 2628ece940 campaign, B and W medians |
|---|---:|---:|---:|---:|
| first burst boundary over its rounds, ms | 78.5 | 60.9 | -17.6 | 77.4 / 64.4 |
| publisher on the scheduler thread, ms | 64.9 | 52.1 | -12.8 | 65.3 / 51.3 |
| burst 2, ms per round | 139.40 | 138.09 | -1.31 | 138.62 / 138.63 |
| TTFT, s | 117.894 | 117.822 | -0.072 | 117.862 / 117.888 |

  On both pause readings and on TTFT each cell sits inside its arm's range on the earlier main (B 74.9 to 87.5 and 64.1 to 67.7 ms, W 54.2 to 65.7 and 50.1 to 53.7 ms); burst 2 is 0.10 ms per round above B's range on B' and 0.41 below W's on W'. Both served the same text (e3b0c1e81f9a) with the same burst structure, and published at the first burst boundary after the first token (1,579 and 1,561 ms after it). The change measures the same on the new main.

## Not covered

- **Not run:** `tools/spec-on-cache-hit-gate.sh qwen` on main and on the head (its own 9B artifact). The pair cases above cover the same path, a speculative restore from a spec-published entry, on the 27B.
- **Inter-token latency percentiles.** The pause is one event per request, read from the server's timestamped lines; no ITL or TPOT p50/p95/p99 was recorded.
- **Other devices.** Every cell here is GB10. The change is not keyed on a device: an RTX 5090 or RTX PRO 6000 loses the same memsets. No timed cell ran there.
- **Other publishers of the same function.** The DSpark and GLM-5 boundary captures call the publisher without a draft plane. The unit cell's no-draft case is their call shape on a two-plane cache, and it is byte-equal to main's. No DSpark or GLM-5 model was served. Latent (MLA or DSA) planes keep their own snapshot calls, which this change does not touch.
- **Multi-card placements** (pipeline stages on several cards). The copies are the per-plane driver copies main already issues there; only the allocation call changed. Not run.
- **The remaining pause**, 58.6 ms at 131k under the profiler (43.1 ms of copies and memsets, 15.6 ms of waiting for allocation calls) and about 112 to 119 ms at 260k, is not addressed (above).

## Which binary each cell ran on

- **B, W, U, T servers and the test binaries** were built on unit 1 from one source tree (main 2628ece940, copied with `git archive`) and one target directory: an arm is that tree with the arm's `crates/memra-server/src/worker.rs`, so the engine compiled once and each arm recompiled the server crate only (`tools/spark_build.sh`, `tools/spark_build_arm.sh`, `build/*.log`, `build/bins-sha256.txt`). The head's other change, one row's wording in `docs/FLAGS.md`, does not reach a binary: `build.rs` reads only the flag names from that file.
- **B is main 2628ece940 exactly**, the main the branch sat on when these cells ran.
- **W = 3afa740d13**, this PR's code commit on that main. The timed cells at 131k (14-W, 16-W, 18-W) and at 260k, the profiles, the head's unit cells and the head's pair cells ran its binary (sha256 9662e2229317...). Unit 2 ran a copy of it with the same sha256.
- **B' = main 11f3c0920b and W' = d117bf8607** (the same code commit rebased onto it) were built on unit 1 from a second tree at the same path and the same target directory (`tools/spark_build_11f3.sh`); they ran the cells under "After the rebase".
- **U = a25c08f05e** is main with two lines changed (the planes from `prefix_plane_alloc`). W is the same program behind `spec_plane_copy`, plus tests. U's three cells at 131k are in the table as their own arm.
- **T = ee37f8d3b4** is the design as approved: `prefix_snapshot_batch` extended with the draft plane and called once by the publisher. Its five cells at 131k, its unit cells and one pair case are kept as the record of the arm that lost.
- **Test binaries.** `basetests-3afa740d13` is main's production code with the head's tests (`tools/red_worker.py`); `unwritten-3afa740d13` is the head with the publisher ignoring a failed copy (`tools/red_unwritten.py`).
- **CPU checks** ran on a CPU build host with the CUDA 13.1 compiler and an sm_120a build, at 3afa740d13, at d117bf8607 on 11f3c0920b and at c232c9132a on 66ee31fcb6 (`cpu-checks.txt` is the last; `cpu-checks-d117bf8607.txt` and `cpu-checks-3afa740d13.txt` the earlier).

## Receipts

- `cpu-checks.txt` (c232c9132a), `cpu-checks-d117bf8607.txt`, `cpu-checks-3afa740d13.txt`: fmt, workspace clippy, the memra-server suite (and from d117bf8607 on the memra-engine suite), the census at the head and on the fail-before tree.
- `build/`: the build logs of every arm and `bins-sha256.txt`.
- `unit-cells/3afa740d13/`, `unit-cells/ee37f8d3b4/`: `verdicts.txt`, the digest lines of both sides and each cell's log.
- `served/bt131k/`, `served/bt260k/`: per cell `ab.jsonl`, `guard.log` (the binary's sha256 and the server env) and `trace.txt` (the server log's `[tick-spec]` and `[prefix-cache]` lines with their timestamps); `campaign.log`; `table.txt` (`tools/specpub_tab.py`).
- `pairs/unit2/`: one file per case (`tools/export_cells.py`: the command, the exit code, the verdict line, each turn's token counts and spec counters, every boot's prefix-cache lines) and `table.txt` (`tools/pairs_tab.py`).
- `nsys/nsys131k-B/`, `nsys/nsys131k-W/`: `gap.txt` (`tools/specpub_gap.py`), `split.txt` (`tools/specpub_split.py`), `big.txt` (every copy and memset of 64 MB or more), `pf.jsonl`, `guard.log`, `trace.txt`. The profiles themselves stay on the machine.
- `served/bt131k-11f3/`, `pairs/unit1-11f3/`: the cells after the rebase.
- `tools/`: the cells (`specpub_cell.sh`, `specpub_ab.py`, `specpub_pairs.py`, `unit_gpu_cells.sh`, `specpub_nsys.sh`), their readers (`specpub_tab.py`, `pairs_tab.py`, `specpub_gap.py`, `specpub_split.py`, `specpub_ttft_order.py`), the builds (`spark_build.sh`, `spark_build_arm.sh`, `spark_build_11f3.sh`, `red_worker.py`, `red_unwritten.py`, `cpu_checks.sh`, `engine_lib_rerun.sh`), and the drivers that take one unit lease per cell and yield to landing and release tickets (`specpub_driver.sh`, `l5_11f3.sh`, `unit_cell.sh`, `landing_quiet.sh`, `fetch_cells.sh`, `fetch_small.sh`, `export_cells.py`).
- Full server logs, GPU telemetry and memory samples stay on the machines.

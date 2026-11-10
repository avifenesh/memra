# The spec prime boundary publishes its planes without zeroing them (GB10 decode round 6, lever 5)

Verdict: shipped as a small measured gain; the design's expectation did not hold. The publisher's planes skip the memset and keep one device copy each. On GB10 with Qwen3.8-27B the one pause after a long cold request's first burst falls by 13.0 ms at a 131k prompt on the first burst boundary (77.4 to 64.4 ms, -17%) and by 14.1 ms on the publisher's time on the scheduler thread (65.3 to 51.3 ms, -22%), every cell of the head under every cell of main. At 260k: 157.4 to 123.8 ms (-21%) and 142.2 to 113.7 ms (-20%), three cells of main and two of the head. Round time, TTFT and every served text are unchanged. The design (no memset and one batched copy launch for all planes) took 7.0 ms off at 131k: its launch waits for the last allocation and then runs under the next burst. The published entry is byte-identical to main's (18 entries, round-trip digests equal), and a plane whose copy failed is never published (a forced failure, with its red arm). About 51 ms of the pause is left at 131k: 34 allocations and 34 copies.

Device: DGX Spark (GB10, sm_121a), two units. Model: Qwen3.8-27B GGUF (NVFP4 trunk, Q5_K output head) with the masked MTP draft (32,768-row head), 262,144 context, served as the launcher serves it: the MTP speculative path, K=3 on a cold request, 32-token bursts. Date: 2026-11-10. Arms, all sm_121a builds from one tree and one target directory on unit 1 (`build/`):

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

**CPU census, fail before and pass after** (`cpu-checks.txt`, dev desk, sm_120a build). `worker::tests::spec_boundary_publish_allocates_no_zeroed_plane` reads the publisher's source: no `alloc_u8` and no `alloc_zeros` in it, four `prefix_plane_alloc` calls and four `spec_plane_copy` calls (trunk K and V, draft K and V), the trunk arm's `return` and the draft arm's `return None` on a failed copy, the helper's body, and the allocator's two arms. On main's production code with the head's tests (`tools/red_worker.py`) it fails with `the spec publisher calls engine.alloc_u8(`; at the head it passes.

**GPU unit cells** (`unit-cells/3afa740d13/verdicts.txt`, GB10 unit 1; test binaries of main's production code with the head's tests, of the head, and of the red arm):

- `spec_boundary_publish_is_the_copy_program`: a live cache filled with random bytes on a tiny hybrid config and on the 27B's own geometry (17 attention layers, the last one standing for the unexecuted MTP layer, 48 recurrent layers), boundaries at 1 and 97 tokens, the live cache at the boundary and 34 rows past it, each published with a draft plane, with a draft source one byte short and with none. Every trunk and draft plane of the entry equals `[0, pos * tok_bytes)` of its source, the byte count is the planes' sum, and a short or absent draft source publishes trunk-only. It passes on main's publisher and on the head's, and the 18 entries' round-trip digests (trunk KV, recurrent planes, draft plane, hidden row, logits; `digests-base.txt`, `digests-head.txt`) are the same 18 lines. That is the first half, plane by plane.
- `spec_boundary_publish_drops_a_plane_whose_copy_failed`: a test-only fault makes one `spec_plane_copy` call report failure and copy nothing. A failed trunk copy (the first K, the first V, the last V) publishes no entry and leaves the cache empty. A failed draft copy (K, then V) publishes the entry trunk-only, its trunk planes equal to their sources. Green at the head on both configs (2 and 32 trunk planes). **Red arm**: the head with the publisher ignoring the failed copy (`tools/red_unwritten.py`) fails the cell with `an entry was published although the copy of the first trunk K failed` (exit 101). That is the second half.
- `an_unwritten_plane_is_charged_to_the_insert_budget_reading`: the insert budget on unified memory reads free memory right after the copies are enqueued. A 2 GiB plane moves that reading from 118,550 to 116,468 MiB when it is allocated and not at all when it is written, so the budget does not depend on whether a copy has run.
- `b1_snapshot_and_restore_are_the_copy_program`, the plain snapshot's cell, stays green at the head (it shares the allocator).

The copy-program, budget and B1 cells also ran at arm T's head (`unit-cells/ee37f8d3b4/`): green, 18 digests equal to main's.

**Warm == cold pair cells** (`tools/specpub_pairs.py`, the pair cells of `research/spec-pool-grid-20261109/` with a deep chat shape; GB10 unit 2, binaries copied from unit 1, sha256 equal on both). One warm boot serves a conversation in order; turn 1 is cold and its speculative session publishes at the first burst boundary; each later warm turn must hit an entry the spec publisher wrote and restore a speculative session from it, trunk and draft plane (`[prefix-cache] hit`, `spec restore: N of M prompt tokens + draft plane from cache`, N equal to an `insert (spec-boundary)` line's token count and to no other insert's); one cold boot per later turn primes that turn's prompt from scratch; every warm answer must be byte-equal to its cold twin. Every turn is bounded at 64 tokens, so no session parks and the prefix cache is the only warm path. Turn 3 restores from an entry a restored session published. Short is 1,000 prompt tokens, deep 135,545. Greedy cases run in the served shape; sampled cases (temperature 1.0, seed 1234) run with `MEMRA_PREFIX_SPLIT_TRACE=1`, which hashes every published entry's trunk KV, recurrent state and boundary logits.

Run so far (`pairs/unit2/table.txt`, one file per case next to it):

| case | arm | later turns equal to cold | restored from a spec-published entry, with its draft plane | answers (sha256, per turn) |
|---|---|---|---|---|
| short, greedy, 3 turns | B | = / = | 960 of 1,083; 1,056 of 1,171 | f1fb5d52 / ce72af3e / 8c7ed81f |
| short, greedy, 3 turns | T | = / = | 960 of 1,083; 1,056 of 1,171 | f1fb5d52 / ce72af3e / 8c7ed81f |
| deep, greedy, 3 turns | B | = / = | 135,520 of 135,628; 135,584 of 135,716 | a99396c5 / a13b08a4 / 2d03a4e6 |
| deep, sampled, 2 turns, digests | B | = | 135,520 of 135,612 | 5ac95257 / 7e34878c |

Main is warm == cold on every case, and every warm turn restored from an entry the spec publisher wrote. On the short greedy case arm T served main's three answers with main's bursts (rounds, drafted and accepted counts per burst). On main's deep sampled case the entry a restored session published at 135,584 tokens has the same state digest as the entry the cold boot of that turn published (dac1cf80...).

**Not run yet: the head's four cases** (short and deep, greedy and sampled, with the digest comparison against main's sampled cases) and main's short sampled case. Unit 2 was taken by another lane's cells, which have first call on it. They are the first cells to run when it frees. Until then the head's warm path rests on the unit cells above: the entry is the same bytes, plane by plane, so a restore reads what it read on main.

**Served texts.** Every timed cell of every arm served the same text at its context (the section above): 18 cells at 131k, 5 at 260k.

## Served B/T

One cold request per cell on a fresh server: a 1k warm-up request, then the context's request, greedy, 512 max tokens, every server line timestamped, `MEMRA_TICK_TRACE=1` (`tools/specpub_cell.sh`, `tools/specpub_ab.py`; the prompts are the split rung lane's served cells', token for token). GB10 unit 1, one cell per lease. Medians with spreads (max - min).

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
- **T, the design, takes 7.0 ms off the boundary (-9%).** Its publisher returns in 22.1 ms, because one launch replaces 34 copy calls. But the launch cannot start before the last plane is allocated, and the copy then runs under burst 2: burst 2 costs 141.06 ms per round on T against 138.62 on B, +2.4 ms on each of 14 rounds. Every U cell is under every T cell on the boundary (53.3 to 60.4 against 60.8 to 71.1).
- No regression on W: burst 2 and the later bursts cost what B's do, TTFT is equal, and all 18 cells served one text (sha256 e3b0c1e81f9a, 95 tokens) with one burst structure (rounds, drafted and accepted counts). One U cell has a TTFT of 118.98 s against 117.8 to 117.9 elsewhere; its decode readings sit with the other U cells.

### 260k (259,946 prompt tokens, entry 259,904 tokens, 8,357.4 MB)

B W W B B W was planned. A release record took the unit after the fifth cell, so the set is B W W B B: three B cells and two W cells.

| | B (3 cells) | W (2) | W - B |
|---|---:|---:|---:|
| first burst boundary over its rounds, ms | 157.4 (18.2) | 123.8 (10.2) | -33.6 (-21%) |
| publisher on the scheduler thread, ms | 142.2 (1.5) | 113.7 (4.7) | -28.5 (-20%) |
| burst 1 line to burst 2 line, ms | 3166.2 (33.9) | 3128.3 (17.7) | -37.8 |
| burst 2, ms per round (16 rounds) | 189.10 (2.11) | 188.39 (1.41) | -0.71 |
| later bursts, ms per round (30 rounds) | 188.00 (1.02) | 187.78 (0.47) | -0.22 |
| TTFT, s | 313.551 (0.252) | 313.790 (0.291) | +0.239 |

Every W cell is under every B cell on both readings of the pause. The round costs the same, and all five cells served one text (sha256 5275df37e59f, 134 tokens) with one burst structure. Both W cells have a higher TTFT than the three B cells, by 0.07 to 0.6 s of 313 s. The change does not reach prefill; with two W cells this is reported as measured.

Per cell: `served/bt131k/table.txt`, `served/bt260k/table.txt`.

## Under Nsight Systems

Not run. One profile per arm (B and W) at 131k is queued behind the release record on unit 1 (`tools/specpub_nsys.sh`, `tools/specpub_gap.py`): the pause between the last kernel of burst 1 and the first of burst 2, the memsets and device copies inside it, and the runtime calls that overlap it. Until it runs, the split of the pause is the served cells' (next section).

## What is left of the pause

At the head the pause is the publisher's 34 allocations and 34 device copies:

| | 131k | 260k |
|---|---:|---:|
| the publisher on the scheduler thread, B | 65.3 ms | 142.2 ms |
| the same at the head (W) | 51.3 ms | 113.7 ms |
| arm T's publisher: 34 allocations without a memset, one table upload, one launch | 22.1 ms | not run |

Arm T's publisher time is the nearest reading of the allocations alone that the served cells give: about 22 ms for 4.13 GB at 131k. The rest of W's 51 ms is the 34 copy calls, which return on the scheduler thread before burst 2 starts (W's burst 2 costs what its later bursts do, within a round's noise). The first scout's profile of main priced the copies at 37.8 ms and the zeroing at 22.1 ms of an 83.1 ms pause at 131k (`research/gb10-decode-20261110/scout/nsys131k-unit1/parts.txt`); the served cells here see 14 ms of that zeroing come off the scheduler thread.

So the remaining pause at 131k is about 51 ms, roughly 22 ms of allocation and 30 ms of copy calls, and about 114 ms at 260k. A profile of B and W would split it exactly (allocation calls, copies, what overlaps); it has not run (below). Removing it needs a different mechanism than this change: planes that are not allocated at the boundary (a pool, or the entry taking the session's own pages), or a copy that does not sit between two bursts.

## Not covered

- **Not run yet at this push:** the head's warm == cold pair cases, the Nsight Systems profiles, the sixth 260k cell (06-W), and `tools/spec-on-cache-hit-gate.sh qwen` on main and on the head.
- **Other devices.** Every cell here is GB10. The change is not keyed on a device: an RTX 5090 or RTX PRO 6000 loses the same memsets. No timed cell ran there.
- **Other publishers of the same function.** The DSpark and GLM-5 boundary captures call the publisher without a draft plane. The unit cell's no-draft case is their call shape on a two-plane cache, and it is byte-equal to main's. No DSpark or GLM-5 model was served. Latent (MLA or DSA) planes keep their own snapshot calls, which this change does not touch.
- **Multi-card placements** (pipeline stages on several cards). The copies are the per-plane driver copies main already issues there; only the allocation call changed. Not run.
- **The remaining pause**, about 51 ms at 131k and 114 ms at 260k, is not addressed (above).

## Which binary each cell ran on

- **B, W, U, T servers and the test binaries** were built on unit 1 from one source tree (main 2628ece940, copied with `git archive`) and one target directory: an arm is that tree with the arm's `crates/memra-server/src/worker.rs`, so the engine compiled once and each arm recompiled the server crate only (`tools/spark_build.sh`, `tools/spark_build_arm.sh`, `build/*.log`, `build/bins-sha256.txt`). The head's other change, one row's wording in `docs/FLAGS.md`, does not reach a binary: `build.rs` reads only the flag names from that file.
- **B is main exactly.** The branch sits on main 2628ece940, the commit B was built from.
- **W = 3afa740d13**, this PR's code commit. The timed cells at 131k (14-W, 16-W, 18-W) and at 260k, the head's unit cells and the head's pair cells ran its binary (sha256 9662e229...). Unit 2 ran a copy of it with the same sha256.
- **U = a25c08f05e** is main with two lines changed (the planes from `prefix_plane_alloc`). W is the same program behind `spec_plane_copy`, plus tests. U's three cells at 131k are in the table as their own arm.
- **T = ee37f8d3b4** is the design as approved: `prefix_snapshot_batch` extended with the draft plane and called once by the publisher. Its five cells at 131k, its unit cells and one pair case are kept as the record of the arm that lost.
- **Test binaries.** `basetests-3afa740d13` is main's production code with the head's tests (`tools/red_worker.py`); `unwritten-3afa740d13` is the head with the publisher ignoring a failed copy (`tools/red_unwritten.py`).
- **CPU checks** ran on a CPU build host at 3afa740d13 with the CUDA 13.1 compiler and an sm_120a build.

## Receipts

- `cpu-checks.txt`: fmt, workspace clippy, the memra-server suite, the census at the head and on the fail-before tree.
- `build/`: the build logs of every arm and `bins-sha256.txt`.
- `unit-cells/3afa740d13/`, `unit-cells/ee37f8d3b4/`: `verdicts.txt`, the digest lines of both sides and each cell's log.
- `served/bt131k/`, `served/bt260k/`: per cell `ab.jsonl`, `guard.log` (the binary's sha256 and the server env) and `trace.txt` (the server log's `[tick-spec]` and `[prefix-cache]` lines with their timestamps); `campaign.log`; `table.txt` (`tools/specpub_tab.py`).
- `pairs/unit2/`: one file per case (`tools/export_cells.py`: the command, the exit code, the verdict line, each turn's token counts and spec counters, every boot's prefix-cache lines) and `table.txt` (`tools/pairs_tab.py`).
- `tools/`: the cells (`specpub_cell.sh`, `specpub_ab.py`, `specpub_pairs.py`, `unit_gpu_cells.sh`, `specpub_nsys.sh`), their readers (`specpub_tab.py`, `pairs_tab.py`, `specpub_gap.py`), the builds (`spark_build.sh`, `spark_build_arm.sh`, `red_worker.py`, `red_unwritten.py`, `cpu_checks.sh`), and the driver that takes one unit lease per cell and yields to landing and release tickets (`specpub_driver.sh`, `unit_cell.sh`, `landing_quiet.sh`, `fetch_cells.sh`, `fetch_small.sh`, `export_cells.py`).
- Full server logs, GPU telemetry and memory samples stay on the machines.

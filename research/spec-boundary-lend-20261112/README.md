# The spec prime boundary lends the session's rows to its prefix entry (memra#1221 part 2)

Verdict: adopted. The MTP publisher no longer allocates or copies anything at the speculative prime
boundary: the prefix entry reads the session's rows `[0, pos)` in place, and if the session ends
first its plane moves to the entry. On GB10 with Qwen3.8-27B and its MTP draft the boundary stall
falls from 59 to 8 ms at a 131k prompt and from 118 to 6 ms at 260k (every cell of the head under
every cell of main), the publisher from 52 and 115 ms to under 0.2 ms, the pool no longer grows
inside it (544 and 1,408 MB on main), and from the boundary on the device holds the prompt's rows
once instead of twice (4.2 and 8.2 GB less pool used). Rounds, TTFT and every served text are
unchanged; the agentic replay serves the same texts with no leak on GB10 and the RTX 5090. Record:
[`docs/decisions/SPEC-BOUNDARY-LEND.md`](../../docs/decisions/SPEC-BOUNDARY-LEND.md).

Device: DGX Spark (GB10, sm_121a), two units, `u1` and `u2` in the directory names. Model:
Qwen3.8-27B GGUF (NVFP4 trunk, Q5_K output head) with the masked MTP draft (32,768-row head),
served as the GB10 launcher serves it (`tools/serve.sh`): the MTP speculative path, 262,144 context,
`MEMRA_PREFIX_STABLE_BOUNDARY=1`, no host tier. RTX 5090 Laptop GPU (sm_120a) for the
no-regression and agentic cells. Date: 2026-11-12 and 13. Page cache unfilled in every cell
(part 1's filled-cache condition is not this change's subject).

## Arms

| arm | binary | what it is |
|---|---|---|
| B | `B-dae162ddef` | main (11b368f7d7's crates; dae162ddef is the same tree with part 1 cherry-picked before it landed, `git diff` empty) with part 1's trace probe (`tools/probe_i782.py`: pool reserved/used per plane) |
| Z | `Z-dae162ddef` | B with the publisher returning at its entry: no plane, no copy, no entry. The floor of any publication that does not allocate or copy |
| O | `O-dae162ddef` | B with the copies on the engine's copy stream after an event (the allocations unchanged), joined at the next tick top |
| S | `S-dd9437e445` | the head: the lending publisher. Its production code is the PR's (the commits after it change tests and docs only) |
| BT, ST | `BT-dae162ddef`, `ST-S-6fd1564e74` | B and S with the leak-check probe (`tools/trim_probe.py`: a file drives the stock trim path and prints the pool's used bytes and the cache's charges) |

## Measure first: where the boundary's time and memory go

Arms B, Z and O, one cell per lease, B Z O O Z B on each unit; one cold request per cell on a fresh
server (a 1k warm-up, then the context's request, greedy, 512 max tokens, every server line
timestamped; `tools/specpub_cell.sh`, `tools/specpub_ab.py`, the prompts of lever 5's cells token
for token). Readings (`tools/d_tab.py`): the **stall** is the first burst boundary over its rounds
(burst 1's line to burst 2's, less burst 2's rounds at the later bursts' rate), the **publisher** is
burst 1's line to the insert line (Z: to its return line). Medians (spread), N = 2 per arm.

| 131k, u2 (`gb10/d131k-u2/`) | B | O | Z |
|---|---:|---:|---:|
| stall, ms | 63.2 (9.5) | 52.3 (2.6) | -2.9 (0.5) |
| publisher, ms | 50.6 (0.8) | 51.0 (0.8) | 0.0 (0.0) |
| burst 2, ms per round | 138.47 (0.27) | 138.25 (0.42) | 137.50 (0.26) |
| later bursts, ms per round | 137.57 (0.34) | 138.16 (0.65) | 137.72 (0.22) |
| TTFT, s | 117.807 (0.170) | 117.811 (0.142) | 118.365 (1.007) |

| 260k, u1 (`gb10/d260k-u1/`) | B | O | Z |
|---|---:|---:|---:|
| stall, ms | 122.5 (0.1) | 124.4 (0.5) | 6.5 (3.2) |
| publisher, ms | 111.0 (1.4) | 114.0 (2.1) | 0.0 (0.0) |
| burst 2, ms per round | 187.67 (0.03) | 187.57 (0.13) | 187.29 (0.15) |
| later bursts, ms per round | 186.97 (0.09) | 186.92 (0.00) | 186.90 (0.35) |
| TTFT, s | 313.529 (1.251) | 312.943 (0.198) | 312.888 (0.301) |

Every cell served one text per context (e3b0c1e81f9a at 131k, 5275df37e59f at 260k) with one burst
structure.

- **The publisher is allocation-bound.** Part 1's trace probe reads the default pool around every
  plane. At 131k, 27 of the 34 planes come from cached blocks in microseconds and 7 grow the pool,
  2 to 16 ms each, 544 MB in all; the allocation calls are 49 to 50 ms of the publisher's 50.6. At
  260k, 9 allocations take 109 to 115 ms and grow the pool 1,408 MB. The 34 device copies (4.13 and
  8.20 GB, about 40 and 80 ms of device time) already run under those calls.
- **So copying on a side stream does not help.** O moves every copy to the engine's copy stream
  (`tools/arms_1221b.py`): the allocations stay, the publisher does not shorten, the stall falls by
  11 ms at 131k and rises by 2 ms at 260k, and burst 2's rounds do not move (the copies had already
  finished under the allocations). Part 1's arm H showed that the pool's growth does not overlap
  burst 1 on GB10 either.
- **Only a publication that allocates and copies nothing reaches the floor.** Z, which publishes
  nothing, takes the stall to -3 and 6.5 ms.
- **Memory.** On main the entry's planes add 4.13 GB (131k) and 8.20 GB (260k) of pool used from
  the boundary to the end of the session (B: used 25.42 -> 29.56 GB and 30.31 -> 38.51 GB across
  the publisher). MemAvailable moves less (1.1 and 1.9 GB): most of the entry is carved from blocks
  the prime's transients left cached, which the pool keeps either way; the 4.13 and 8.20 GB are
  what other sessions on the device can no longer take without the pool growing again.

## The change against main, same unit, interleaved

B against S, B S S B B S, one cell per lease, the same cell as above.

| | 131k, u2: B | S | S - B | 260k, u1: B | S | S - B |
|---|---:|---:|---:|---:|---:|---:|
| stall, ms | 59.1 (7.9) | 7.9 (5.6) | -51.2, every S under every B | 118.1 (14.0) | 5.5 (15.1) | -112.6, every S under every B |
| publisher, ms | 51.6 (0.4) | 0.0 (0.0) | -51.5, every S under every B | 115.3 (1.5) | 0.1 (0.1) | -115.2, every S under every B |
| the lend itself (`[specpub]`), ms | | 0.07 | | | 0.12 | |
| pool growth inside the publisher, MB | 544 | 0 | | 1,408 | 0 | |
| pool used at the boundary, GB | 29.56 | 25.39 | -4.17 | 38.51 | 30.28 | -8.23 |
| burst 2, ms per round | 138.24 (0.16) | 138.27 (0.60) | +0.03, overlap | 187.53 (0.35) | 187.62 (0.69) | +0.08, overlap |
| later bursts, ms per round | 137.84 (0.63) | 137.71 (0.19) | -0.13, overlap | 187.34 (0.45) | 187.42 (0.38) | +0.08, overlap |
| burst 1 wall, ms | 1534.5 (2.4) | 1534.9 (11.1) | +0.5, overlap | 2241.4 (1.4) | 2248.6 (11.4) | +7.2, every S over every B |
| TTFT, s | 117.543 (0.109) | 117.605 (0.338) | +0.062, overlap | 313.058 (0.435) | 313.328 (0.448) | +0.270, overlap |
| total, s | 122.557 (0.100) | 122.551 (0.342) | -0.006, overlap | 324.032 (0.436) | 324.199 (0.452) | +0.167, overlap |
| MemAvailable low after burst 1, GB | 86.51 | 87.52 | +1.01 | 77.57 | 79.30 | +1.73 |

Medians (spread), N = 3 per arm. One text per context on every cell of both arms (e3b0c1e81f9a,
5275df37e59f) and one burst structure. B's pool readings are its probe's at the publisher's exit, S's
its `[specpub]` line at the publication; the entry is charged 156.9 MB on S (the capture's recurrent
planes) and 4,290.5 / 8,357.4 MB on B.

- The stall falls to the Z arm's floor at both contexts, every S cell under every B cell; the
  publisher's 51 and 115 ms are gone and nothing is allocated at the boundary.
- Rounds, TTFT and the total are unchanged within the spreads. One reading is reported as measured:
  at 260k burst 1's wall is 2241.4 ms on B and 2248.6 ms on S, every S cell over every B cell by 3 to
  14 ms (0.3%). Burst 1 runs before the publication and nothing is lent during it (the 1k warm-up's
  entry is lent, its 34 books walked once per tick, microseconds); at 131k the same reading is
  1534.5 against 1534.9 ms. Read as order and noise, not a cost.
- **RTX 5090** (`rtx5090/rtx28k/`, 150 W, B S S B B S S B, a 28,006-token prompt at ctx 32,768): one
  text (d22600fc844a); rounds 33.67 (1.98) against 33.38 (0.12) ms; TTFT 24.223 (1.784) against
  24.247 (6.360) s; the entry charged 156.9 MB instead of 1,039.4 MB. One S cell's TTFT was 29.9 s:
  one prime chunk (rows 16k to 20k) took 8.5 s in its own kernels against 3.5 to 4.0 s for its
  neighbours; the prime has no publication and no lend in it, and the four cells added after it
  (with 250 ms GPU telemetry) read 23.5 to 25.4 s on both arms.

## Exactness

### The entry is main's entry, plane for plane (GPU unit cells, u2)

`unit/unit-S-6fd1564e74/` (`tools/unit_cells.sh`; the server, engine and KV lib test binaries of the
head, a tiny hybrid config and the 27B's own geometry, `MEMRA_B1_MODEL`):

- `spec_boundary_publish_is_the_copy_program`, now run on both plane sources. The copying
  publication's 18 round-trip digests are lever 5's lines byte for byte (lever 5 showed them equal
  to main's), and every lending twin (`assert_lent_twin`) reads its source's rows in place (its
  device pointer is the session plane's, 34 planes on the 27B geometry), holds the copy's bytes, is
  charged only its own planes (156.9 MB of 159.9 MB on the 27B at 97 rows) and has the copy's
  round-trip digest. So a lent entry restores exactly what main's entry restores.
- `spec_boundary_publish_drops_a_plane_whose_copy_failed`, both sources: a trunk plane that does not
  lend (a test fault) publishes no entry, a draft plane that does not lend publishes the entry
  trunk-only, as for a failed copy.
- `a_lent_plane_outlives_its_lender_and_its_last_reader_frees_it`: a rollback and a recurrent
  rewind to under the lent rows refuse and leave every length as it was; a rollback to the lent rows
  passes; `into_pooled` of a lent plane refuses and the plane moves to its loan. The lender dropped,
  the charge-only reconcile charges the adopted planes' capacity exactly once (8,077,568 bytes of
  K/V capacity at the 27B's geometry, to the one holder), the entry restores the session's rows into
  a fresh cache, and evicting it frees at least what was charged. An entry evicted while its lender
  lives frees only its own planes, and the lender then frees its rows. The pool's used bytes end at
  their baseline (33,554,432 before and after).
- `adopted_planes_compact_only_when_that_frees_more_than_it_copies`: 97 lent rows under a 256-row
  plane compact (34 planes on the 27B; charge = the copy's bytes; digest = the copy publication's);
  200 under 256 stay adopted, charged their capacity. Pool used at its baseline after each.
- `the_mtp_scratch_refuses_a_length_below_its_lend` (engine): under a 40-row lend, lengths 0 and 39
  refuse through `set_len` and `set_plane_len` with the length unchanged; 40 and 45 pass.
- `writes_below_a_live_lend_refuse_and_the_lender_hands_its_plane_on` and
  `a_release_keeps_the_lent_bytes_of_an_on_demand_plane` (KV crate, on the crate's own device seam):
  the pooled guards and hand-off as above; on an on-demand VMM plane backed in four extents, a
  release past 0 keeps the two lent extents and frees the others, and the adopted plane holds only
  the lent extents.
- `b1_snapshot_and_restore_are_the_copy_program` and
  `an_unwritten_plane_is_charged_to_the_insert_budget_reading` (lever 5's) still pass.

**Red arms** (`unit/unit-red-*/`, `tools/red_patch.py`: the head with one guard disarmed, the test
binary of its cell's crate): each cell FAILS. `rollback` (the rollback below the lend lands),
`rewind` (a recurrent rewind with planes of the cache's own shapes lands under the lend; the first
build of this cell passed no planes, so another check refused it, and it was rebuilt at fb3d350144:
`unit/unit-red-rewind-S-fb3d350144/`), `pooled` (the plane leaves its loan; the loan then cannot
hold it), `release` (a release takes the lent extents: 0 of 8 MiB mapped), `adopt` (the refused
plane is freed with its owner: the loan holds 0 bytes) and `scratch` (`set_len` below the lend
lands).

### Fails before, passes after (CPU)

The four source censuses (`cpu/red-census.txt`): the MTP publisher lends with no allocation or copy
on its lending arms while DSpark and GLM-5 copy; every caller of a write that could start below a
lent row is pinned (a runtime walk over every `.rs` file of the crates); the KV crate's guards sit
before their writes; the MTP scratch's two setters check before writing. Built on main 11b368f7d7's
production code with these tests appended (`tools/dd_red_census.sh`), all four FAIL; at the head
all four pass, with the compaction rule's table test.

### Warm equals cold on lent entries (pair cells)

`tools/pairs_1221b.py` (`tools/pairs_run.sh` on GB10, `tools/rtx_pairs.sh` on the RTX 5090): one
server per case; turn 1 publishes at its speculative boundary, every later turn is served warm
(restored from that entry with the draft plane) and then cold on a fresh server, and the two texts
must be equal byte for byte; the `-digest` cases add the entries' round-trip digests
(`MEMRA_TICK_TRACE`). Two read paths:

- **borrowed** (the default config): a bounded turn parks, so the next turn's hit reads the parked
  lender's rows (at the hit: 34 books, 34 lent planes live, nothing adopted);
- **adopted** (`MEMRA_REUSE_POOL=0`): no parking, the lender drops its planes at retirement and the
  entry holds them (at the hit: 0 lent live, 33.8 MB adopted).

The first runs' `adopted` case read a borrowed entry (bounded turns park under the default
config); the table labels each case by what its trace read, and the `nopark` cases were added for
the adopted path. Turn 2 of the `borrowed` cases is sampled (temperature 1.0, seed 7);
`S-adopted-nopark-digest` samples every turn (1.0, seed 1234); greedy otherwise. `deep-adopted`
(head only, ctx 262,144) publishes a 135k-token entry; its read path was not traced.

GB10, u1 (`gb10/pairs-u1/`, `table.txt`; `S-*` and `S-adopted-*` the head, `B-*` main):

| run | case | later turns == cold | restored (of prompt) with draft plane | answers per turn | entry digests | at the hit: books / lent live / adopted MB | read |
|---|---|---|---|---|---|---|---|
| S-short | adopted | = / = | 960 of 1,083, 1,056 of 1,171 | f1fb5d52 / ce72af3e / 8c7ed81f | - | 34 / 34 / 0.0 | borrowed |
| S-short | borrowed | = | 960 of 1,275 | 347edd7b / 54327aef | - | 34 / 34 / 0.0 | borrowed |
| S-short | borrowed-digest | = | 960 of 1,275 | 347edd7b / 54327aef | d543aabbaed3, 37dda33d31b0 | 34 / 34 / 0.0 | borrowed |
| B-short | adopted | = / = | 960 of 1,083, 1,056 of 1,171 | f1fb5d52 / ce72af3e / 8c7ed81f | - | - | - |
| B-short | borrowed | = | 960 of 1,275 | 347edd7b / 54327aef | - | - | - |
| B-short | borrowed-digest | = | 960 of 1,275 | 347edd7b / 54327aef | d543aabbaed3, 37dda33d31b0 | - | - |
| S-deep | deep-adopted | = | 135,520 of 135,628 | a99396c5 / a13b08a4 | - | - | - |
| nopark | S-adopted-nopark | = / = | 960 of 1,083, 1,056 of 1,171 | f1fb5d52 / ce72af3e / 8c7ed81f | - | - | - |
| nopark | B-adopted-nopark | = / = | 960 of 1,083, 1,056 of 1,171 | f1fb5d52 / ce72af3e / 8c7ed81f | - | - | - |
| nopark | S-adopted-nopark-digest | = | 960 of 1,082 | cde53ba3 / fcbcfe74 | d543aabbaed3, e15a5f50306d | 34 / 0 / 33.8 | adopted |

RTX 5090 Laptop, 150 W (`rtx5090/pairs-S/`, `pairs-B/`, `pairs-nopark/`, `pairs-table.txt`):

| run | case | later turns == cold | restored (of prompt) with draft plane | answers per turn | entry digests | at the hit: books / lent live / adopted MB | read |
|---|---|---|---|---|---|---|---|
| S | adopted | = / = | 960 of 1,083, 1,056 of 1,171 | 1f1a5d34 / 87b7e911 / 12d0fea8 | - | 34 / 34 / 0.0 | borrowed |
| S | borrowed | = | 960 of 1,275 | b8fd8d30 / 79dc2d57 | - | 34 / 34 / 0.0 | borrowed |
| S | borrowed-digest | = | 960 of 1,275 | b8fd8d30 / 79dc2d57 | ce71138cb228, e9632f046071 | 34 / 34 / 0.0 | borrowed |
| B | adopted | = / = | 960 of 1,083, 1,056 of 1,171 | 1f1a5d34 / 87b7e911 / 12d0fea8 | - | - | - |
| B | borrowed | = | 960 of 1,275 | b8fd8d30 / 79dc2d57 | - | - | - |
| B | borrowed-digest | = | 960 of 1,275 | b8fd8d30 / 79dc2d57 | ce71138cb228, e9632f046071 | - | - |
| nopark | S-adopted-nopark | = / = | 960 of 1,083, 1,056 of 1,171 | 1f1a5d34 / 87b7e911 / 12d0fea8 | - | 34 / 0 / 33.8 | adopted |
| nopark | B-adopted-nopark | = / = | 960 of 1,083, 1,056 of 1,171 | 1f1a5d34 / 87b7e911 / 12d0fea8 | - | - | - |
| nopark | S-adopted-nopark-digest | = | 960 of 1,083 | 572a8a03 / bfa1ae7c | ce71138cb228, 25004bd4b715 | 34 / 0 / 33.8 | adopted |

- Every warm turn equals its cold twin, on both devices, on both read paths, greedy and sampled,
  and at 135k on GB10.
- The head and main serve the same answer at every turn and publish the same entry digests
  (`borrowed-digest`: d543aabbaed3, 37dda33d31b0 on GB10 and ce71138cb228, e9632f046071 on the RTX
  5090, on both arms).
- An adopted entry is the borrowed entry and main's entry: the no-park case's 960-token entry has
  the same digest (d543aabbaed3 on GB10, ce71138cb228 on the RTX 5090).

### Exactness gates (GB10, u2, the head)

The repo's four gates on the head's server, one per lease (`gb10/gates-u2/`, `tools/gate_run.sh`),
with the Qwen3.8-27B MTP model:

| gate | verdict | what it ran |
|---|---|---|
| `tools/seeded-invariance-gate.py --penalised device` | PASS (54 rows, 10 prompts, cases ref, 0 to 3, demote) | spec path: 26 lent `spec-boundary` publications, 6 spec restores |
| `tools/spec-burst-boundary-gate.py` | PASS (229 rows, grid default, 1:2, 3:2, 3:3) | spec path: 80 lent `spec-boundary` publications, 16 spec restores |
| `tools/prefix-exact-resend-gate.py` | PASS (resend 2/2, growth 2/2, identical 9/9, tails 4, covered 4/4, V1 to V5) | plain path: seed entries, tails, the new charge books |
| `tools/prefix-restore-identity-gate.py` | PASS (identical 5/5, grid 5/5, V1 V2) | plain path: restore points |

No rewind was refused below a lend in any gate. The two plain-path gates do not publish at a
speculative boundary; they cover the entry plane type (`PlaneBuf`) and the real-byte accounting
on entries that own their planes. The lending path's restores are covered by the pair cells above
and the agentic replay below.

## Agentic replay and the leak check

The 29-turn agentic replay of the 2026-11-05 decision cell (its workload and driver,
`research/c8-workspace-20261108/harness/agent_turns.py`: a prime pass at 1 token, then the timed
pass), on the MTP route rather than the decision cell's DFlash route, so the lending publisher is
in play: ctx 32,768, 8 sessions, prefix cache 4,000 MB, no host tier, c = 2, thinking on,
temperature 0, seed 1234, 768 tokens, `MEMRA_TICK_TRACE=1`. Boots BT ST ST BT, one per lease. After
the replay the leak-check probe drops every evictable pool through the stock trim path (parked
sessions and prefix entries, then the pool trimmed back to the driver) and prints the pool's used
bytes and the cache's charges (`tools/trim_probe.py`, the same patch on both arms;
`tools/agentic_cell.sh`, `tools/rtx_agentic.sh`, `tools/agentic_tab.py`). The workload file is
not published (agent session content, as in the decision cell); its sha256 is `5fe1afe2afdc3db0`
(full in `rtx5090/agentic/*/guard.log`), and the receipts keep per-request rows without prompt or
generated text (each text's sha256 only).

| GB10, u1 (`gb10/agentic-u1/`) | 01-BT | 02-ST | 03-ST | 04-BT |
|---|---:|---:|---:|---:|
| answered | 29/29 | 29/29 | 29/29 | 29/29 |
| generated tokens | 9,665 | 9,665 | 9,665 | 9,665 |
| TTFT median, s | 2.413 | 2.477 | 2.409 | 2.404 |
| aggregate tok/s | 33.47 | 33.54 | 33.45 | 33.44 |
| `[prefix-loan] reconcile` lines, charged == real | - | 87 of 87 | 87 of 87 | - |
| plane adoptions / compactions | - | 1,904 / 0 | 1,904 / 0 | - |
| rewinds refused below a lend | 0 | 0 | 0 | 0 |
| pool used at boot -> after the trim, MiB | 18,301 -> 18,402 | 18,301 -> 18,402 | 18,301 -> 18,402 | 18,301 -> 18,402 |
| charged bytes after the trim | 0 | 0 | 0 | 0 |

Every request's text is the same on all four boots (29 of 29).

| RTX 5090 Laptop, 150 W (`rtx5090/agentic/`) | 01-BT | 02-ST | 03-ST | 04-BT |
|---|---:|---:|---:|---:|
| answered | 27/29 | 29/29 | 29/29 | 27/29 |
| generated tokens | 8,897 | 9,396 | 9,396 | 8,897 |
| TTFT median, s | 3.141 | 3.055 | 3.337 | 3.519 |
| `[prefix-loan] reconcile` lines, charged == real | - | 80 of 80 | 80 of 80 | - |
| plane adoptions / compactions | - | 1,428 / 0 | 1,428 / 0 | - |
| rewinds refused below a lend | 0 | 0 | 0 | 0 |
| pool used at boot -> after the trim, MiB | 16,529 -> 16,661 | 16,529 -> 16,661 | 16,529 -> 16,661 | 16,529 -> 16,661 |
| CUDA out of memory | 2 | 0 | 0 | 2 |

The 27 requests main answered have the same text on all four boots; the head's two boots agree on
all 29.

- **No leak.** After every evictable pool is dropped, the pool's used bytes are the same to the byte
  on main and on the head, on both devices (the 101 to 132 MiB above the boot baseline are the
  serving allocations both arms keep), and the cache's charges are 0.
- **Charged equals real at every reconcile**: the cache's `total_bytes` against the bytes counted
  independently from the entries' own planes and the distinct adopted planes (`trace_loans`), 87
  and 80 lines per boot, every one equal.
- **No double count**: memory admission reads the device's free bytes from the device and is
  offered the cache's charges (`evictable_bytes`, what `evict_all` frees), and those charges equal
  the bytes the entries and adopted planes really hold at every reconcile above, so a lent plane is
  counted once: as the session's memory while the session holds it, as the cache's charge after.
- **The lend floor cost nothing here**: no rewind was refused, so no request went to the cold path
  because of it.
- **Main's two OOMs** on the 5090 are the same two requests in both main boots (the warm admission of
  the 3,905-token prompt and the decoding session beside it): the cohort's pooled verify runs eager
  with graph replay suspended and fails, and both sessions had streamed, so neither is parked. Filed
  as memra#1266 (not memra#1255 or memra#1258). The head's boots do not reach it: at that admission
  the head holds 2.3 GB less pool (19,778 against 22,113 MB used; 1,413 against 560 MB driver free),
  because its entries read the parked sessions' rows instead of copying them. That moves this
  replay away from the edge; it does not fix the uncovered transient.

### Fleet replay (u1)

`tools/fleet-replay.py`'s agent-shaped traffic made greedy (`tools/fleet_replay_t0.py`; each ok
line carries its text's sha256): 12 conversations in 6 pairs that share a 3-4k-token system prefix
under 4 tenant salts, 2-4-turn bursts, requests one at a time, seed 1234, 80 requests, on the same
server config as the agentic replay (`tools/fleet_cell.sh`, `gb10/fleet-u1/`). Entries are lent by
live sessions, read by the other conversation of the pair while their lenders are parked, adopted
when the pool evicts a lender, and evicted by LRU under the cache budget.

| | BT | ST |
|---|---:|---:|
| answered | 80/80 | 80/80 |
| prefix hits / spec restores | 61 / 61 | 62 / 62 |
| LRU evictions | 57 | 55 |
| `[prefix-loan] reconcile` lines, charged == real | - | 75 of 75 |
| plane adoptions / compactions, books at most | - | 2,074 / 0, 748 |
| pool used at boot -> after the trim, MiB | 18,301 -> 18,301 | 18,301 -> 18,301 |

All 80 texts equal across the arms. The head evicts two entries fewer and serves one more hit: a
borrowed entry is charged only its own planes while its lender is parked.

## CPU checks

On a CPU build host (sm_120a build, CUDA 13.1 compiler), the PR's crates (`cpu/full.txt`, the
logs `cpu/full-*.log.gz`): `cargo fmt --check`, workspace clippy `-D warnings` on all targets, the
coalescer contract, memra-server 1,228 passed (16 GPU cells ignored), memra-kv 89 (2), memra-engine
lib 744 (66), memra-gguf lib 417 (2), and the portable suites (450 passed, 0 skipped of budget 0).
An earlier run failed the portable skip census on a VMM cell that printed a skip line; that cell
now requires VMM instead. The four censuses on main: `cpu/red-census.txt`.

## Not covered and limits

- **Devices.** The timing cells are GB10 only (two units, each comparison on one unit). The RTX 5090
  ran the agentic replay and a no-regression cell (above); no RTX PRO 6000 cell ran. On the PRO 6000
  Workstation Edition the session planes are on-demand VMM planes: the lend and the hand-off cover
  them (the KV crate's on-demand cell ran on GB10's VMM), and an adopted plane keeps only its lent
  extents, but no served cell ran there.
- **Routes.** The MTP publisher lends; DSpark and GLM-5 keep copying, and their parks' rewinds are
  not checked against a lend. Pipeline-parallel stage-owned caches lend on each plane's own stream;
  no multi-card cell ran.
- **The write floor is on the functions that lower a length** (`Cache::rollback`,
  `Cache::rewind_recurrent_from`, the MTP scratch's `set_len` and `set_plane_len`), on
  `into_pooled` and on `release_beyond`; the census pins every caller. The spec round loop's own
  length writes raise a length or restore it to the committed rows, which are at or past the
  boundary by construction (the invariant the copy already relied on for one burst); they are not
  checked one by one.
- **Over-hold.** An adopted plane is charged its whole capacity until it is compacted (only at the
  tick top and after retirement, with the engine) or evicted. A bounded request holds a few hundred
  rows past its entry; the compaction rule keeps an open-output session from holding more than
  twice its entry.
- **Admission's arrival term.** The partial reclaim on discrete cards still counts the entry an
  arriving request will publish at its full logical bytes (`reclaim_arrival_seed_bytes`); with the
  lend a speculative arrival's entry needs only its recurrent planes at publication, so the term
  frees a little more prefix cache than needed. Unchanged from main, conservative, and left for its
  own item.
- **Page cache.** Every cell ran with the page cache as the unit had it (unfilled). Part 1's filled
  condition grows the pool more on main; the lend allocates nothing in either.

## Receipts

- `gb10/d131k-u2/`, `gb10/d260k-u1/`: the B/Z/O measurement; `gb10/ab131k-u2/`, `gb10/ab260k-u1/`:
  the A/B. Per cell `ab.jsonl` (the requests, their TTFT, totals and text sha256), `guard.log` (the
  binary's sha256 and the server env), `server.log.gz` (every line timestamped), `mem.jsonl.gz`
  (MemAvailable at 1 Hz); `campaign.log`; `table.txt` (`tools/d_tab.py`).
- `gb10/unit/`: the GPU unit cells of the head (`unit-S-6fd1564e74`, `unit-S-fb3d350144`) and the red
  arms (`unit-red-*`): `verdicts.txt`, `cell-lines.txt`, one log per cell.
- `gb10/gates-u2/`: the four gates' logs and receipts; `gb10/pairs-u1/`: the pair cells' verdicts,
  per-turn rows and server logs; `gb10/agentic-u1/`, `gb10/fleet-u1/`: per boot the rows (no prompt
  or generated text), `summary.txt`, the server log, `metrics.json`, `table.txt`.
- `rtx5090/rtx28k/`, `rtx5090/agentic/`, `rtx5090/pairs-*`: the same on the RTX 5090.
- `cpu/`: `full.txt` (the CPU chain), `red-census.txt` (the four censuses on main), the test logs.
- `tools/`: the arms (`arms_1221b.py`, `probe_i782.py`, `trim_probe.py`, `red_patch.py`), the builds
  (`spark_build*.sh`, `spark_red.sh`, `dd_*.sh`), the cells (`specpub_cell.sh`, `serve.sh`,
  `agentic_cell.sh`, `fleet_cell.sh`, `gate_run.sh`, `pairs_run.sh`, `pairs_1221b.py`,
  `unit_cells.sh`, `red_cells.sh`, `rtx_*.sh`), the drivers that take one unit lease per task and
  yield to landing and release tickets (`unit_cell.sh`, `landing_quiet.sh`, `chain.sh`), and the
  readers and stagers (`d_tab.py`, `agentic_tab.py`, `pairs_1221b_tab.py`, `receipts.py`,
  `stage_gates.py`).
- GPU telemetry and the build trees stay on the machines.

# A windowless drafter's window wrapped: the DSpark KV ring and five served-route sums (memra#1159)

## What broke

`dspark_q38_gate` panicked in `DflashDraft::ingest_ctx` -> `Engine::copy_into` on Qwen3.8-27B with the q38 DSpark agentic drafter, right after `[dflash-kv-ring] grow keep_from=0 kept=0 phys_rows=15 (was 5)`. The DFlash2 drafter on the same gate passed. The served route with that drafter does not boot at all (below).

## Cause

The drafter has no sliding layer: its config has `sliding_window: null` and five `full_attention` layers, and the loader reads that as `sliding_window = usize::MAX`. Its window, `sliding_window + block_size`, is meant to be `usize::MAX` (it reads every row). Wherever that sum, or arithmetic on it, was plain, it wrapped in release builds (no overflow checks) and would panic in a debug build.

The KV ring (`crates/memra-engine/src/dflash.rs`):

- `dflash_kv_phys_rows`: `2 * window_rows + block` = `2^64 - 2 + 7`, which wraps to 5 rows for block 7. `DflashKv::new` allocated 5 rows per layer instead of `cap + block`, and admission charged 5 rows (204,800 B) per session whatever the cap.
- `ring_plan`, grow arm: `kept + add + window_rows` wraps to `kept + add - 1`, one row short of the write.

The gate's first prime ingest is 16 rows at `len = 0`. The ring grows to `0 + 16 + usize::MAX = 15` rows, and `ingest_ctx` copies 16 rows (16,384 f32) into the 15-row K plane: one row, 1,024 f32 (4 KiB), past the end. cudarc's `slice_mut` refuses the range and the gate panics. The verify window plays no part: the panic is in the prime, before the first round, and it reproduces with the default `ConfidenceSlot` window too. Both sums came in with the ring (3e5ba4617e, #301).

The served route (`crates/memra-server/src/worker.rs`) had five more plain `sliding_window + block_size` sums, all behind the prefix-restore doors (`dspark_prefix_restore_on()`, default ON on GB10 and the RTX PRO 6000 Max-Q, OFF on the RTX 5090 unless `MEMRA_DSPARK_PREFIX_RESTORE=1`). For this drafter each gave a window of 6:

1. the admission restore route (`dspark_floored_tail_too_short` on an entry's tail),
2. the admission reclaim spare (the tail window `pin_reclaim_spare` keeps entries by),
3. the restorable-hit probe (`dspark_entry_restorable`; computed on every request with a DSpark drafter, so a debug build would panic there on every request, though on main a debug build dies earlier, at `2 * window_rows` in `dflash_kv_phys_rows` during boot calibration),
4. the restore block (`dspark_floored_tail_too_short` and its refusal line),
5. the cold-drafter conversion of a vetoed plain hit (`dspark_cold_drafter_refusal`).

## What a served request could hit

- On main (fcd66f8963): nothing gets past the ring. `memra-server` with `MEMRA_DSPARK_SPEC=1` and this drafter dies at boot: the boot calibration primes a dspark session (`run_boot_calibration` -> `dspark_spec_session_new`), the ring grows from 5 to 255 rows for a 256-row tap batch, `copy_into` panics on the GPU worker, and the log then shows `respawn attempt 1/1 in 2s (reloading weights)` and `FATAL: worker init failed: worker died during init`. Run below (at fcd66f8963). Had it booted, every cold dspark admission would have panicked the same way, and `new_cold_at` and `from_tail` build the same 5-row ring.
- With the ring fixed but the five sums left plain (this PR's first push, 9a4233ce4b): the server boots and cold requests serve exactly. With the restore doors on, the window of 6 does the damage. A vetoed plain hit whose suffix is 6 rows or more takes the cold-drafter conversion: `new_cold_at` builds a drafter with a context floor at the boundary, and the round attention outside DFlash2 ignores the floor, so the drafter attends zero rows where a cold prime has the real history. That session's tail carries the floor, and sites 1, 3 and 4 then call it restorable, so a later request restores a drafter that sees a shorter context than a cold prime. Greedy output stays exact (verify decides every token); a sampled request's drafts, and so its stream, differ from the cold reference. Site 2 keeps such entries as reclaim spares. This is from the code; no run on that intermediate build was made.
- With this fix: every site takes `DflashCfg::window_rows()`, which saturates. A windowless drafter never takes the conversion (`ShortSuffix`) and restores only a full tail (floor 0). `forward_round` refuses a KV with a floor on the floor-blind arm, so the shape cannot run silently if a future path builds one.

## Fix

- `DflashCfg::window_rows()` saturates, and every window comes from it: the three engine copies and the five worker sites.
- `dflash_kv_phys_rows` saturates: a windowless drafter gets `cap + block` rows, the door-off size, and never compacts or grows. Admission charges what the session allocates (40,960 B per row at this geometry).
- The grow in `ring_plan` saturates. Plane sizes are a checked product (`dflash_kv_plane_words`) in `DflashKv::new`, `grow_cap` and the grow, so a windowless grow (`usize::MAX` rows, unreachable) is refused by name.
- `forward_round` refuses a KV with a context floor outside the DFlash2 arm.
- The boot line says a windowless drafter's charge grows with the cap. The prefix-entry tail comments say its tail is its whole history (40,960 B per token): charged to the entry like any plane, so eviction and the whole-budget refusal hold and fewer entries fit; a failed device copy publishes the entry trunk-only.

Finite windows never overflowed, so DFlash2 plans, allocations and decisions are unchanged.

## Costs and limits

- The tail is copied before it is charged. `export_tail` allocates and copies `end x 40,960 B` on the device, the entry is charged after that, and an entry larger than the whole prefix budget is refused after both copies (`refuse_oversize`). Trunk planes behave the same way. For a windowless drafter the tail is now larger than the trunk planes (1,229 MB against about 1,057 MB at 30k tokens). Skipping the export when trunk plus tail cannot fit needs the per-token trunk plane size before the trunk copy, which only `prefix_insert_from_spec_boundary` computes; that is more than a few lines and is not done here. It costs capacity, never correctness.
- Admission, read from the code: `estimate_spec` comes from the K gate alone, and the draft plane (1,229 MB at a 30k cap, about 10.7 GB at 262k for this drafter) is added to the request's cost. If that cost cannot fit an idle card, the request gets HTTP 400 `context_length_exceeded` naming the largest admissible context; it is not retried as plain. With peers active it defers (the bounded memory defer ends in a 429 with Retry-After when armed). One gap predates this change: the eager-arm path (`MEMRA_SPEC_CAPTURE_GATE`, default on) admits a request whose cost fits without the draft-state term, assuming that term is a graph capture the pre-capture gate can refuse later. For a dspark model the term is the drafter KV, which `dspark_prime_start` allocates unconditionally, so on that path the plane is not charged. That holds for DFlash2's 170 MB too and is cap-linear for a windowless drafter. Changing it changes production admission for DFlash2, so it is left for its own change.
- The worker census: `every_window_read_on_the_served_route_is_a_listed_safe_one` lists every production read of `sliding_window` (four pass-throughs to `dspark_spec_prompt_fits`, which only takes `ctx_cap.min` of it, and one `>` comparison) and fails on any other read, so any arithmetic form or alias fails it. `the_window_census_fails_on_every_form_of_window_arithmetic` adds each form to the real production source and shows the census red. The per-site pins name `window_rows()`, which main lacks, so on main they fail by construction: they show the text changed, and the predicate tests carry the behaviour.

## Runs

CPU, on a dev box with no GPU (`cpu-checks.txt`). The fail-before tree is main 8a3e4904c6 plus only the two appended test modules (`dflash_windowless_ring_tests` in dflash.rs, `windowless_drafter_window_tests` in worker.rs), which use no API main lacked:

- `failbefore-engine-tests.log`: `FAILED. 0 passed; 4 failed`. The ring is `5` rows where `134` belong (cap 127); a window between `usize::MAX / 2` and `usize::MAX` (its double alone overflows) gets `21`; the windowless admission charge is `204800` B, not `335831040`; the grow plans `Grow { keep_from: 0, phys_rows: 15 }`.
- `failbefore-server-tests.log`: `FAILED. 1 passed; 6 failed`. The read census lists main's production reads of `sliding_window` and fails on the six arithmetic ones (`cfg.sliding_window + cfg.block_size`, `|d| d.cfg.sliding_window + d.cfg.block_size` twice, `draft.cfg.sliding_window + draft.cfg.block_size` twice, `let window = draft.cfg.sliding_window + draft.cfg.block_size`). The five per-site pins fail by construction (see Costs and limits). The census self-test passes on main by design: it adds each arithmetic form to the production source and shows the census red, which tests the census, not the code.
- On the branch: engine `dflash` tests `66 passed; 0 failed`, the memra-server suite 1172 passed, 0 failed, 11 ignored, clippy with `-D warnings` clean. The consequence at each site is tested on its predicate at the site's window, against main's wrapped window: site 5 (conversion `ShortSuffix`, `None` at main's 6), sites 1 and 4 (a floored tail is too short, but not at 6), sites 2 and 3 (a floored-tail entry is not restorable, but is at 6); a finite window and a full tail behave as before. Those use the new accessor, so they have no main run.

GPU, RTX 5090 Laptop, one lease per run, card under 75 C at start (`cell.sh`, `serve-cell.sh`, `driver3.sh`; the caller supplies the binaries, artifacts, output directory and lease command). Base runs: the gate from main fcd66f8963 with only a test module added (no non-test change) and the server from main fcd66f8963, the main this branch started from; main was not rebuilt at 8a3e4904c6. Fixed runs: binaries built for sm_120a from d68780eb2e, the parent of the commit that adds these receipts.

| run | build | drafter, env | result |
|---|---|---|---|
| `base-ladder-eager.log` | gate, main fcd66f8963 | DSpark agentic, `MEMRA_DSPARK_VT=ladder` | `[dflash-kv-ring] grow keep_from=0 kept=0 phys_rows=15 (was 5)`, panic in `copy_into` <- `ingest_ctx`, rc 101 |
| `base-default-eager.log` | gate, main fcd66f8963 | DSpark agentic, default window | the same grow line and panic, rc 101 |
| `serve-base-ladder.log` + `-server-excerpt.log` | server, main fcd66f8963 | DSpark agentic, `MEMRA_DSPARK_VT=ladder` | `server died` at boot: the boot calibration's prime grows the ring 5 -> 255 and panics in `copy_into`; the log then shows `respawn attempt 1/1` and `FATAL: worker init failed: worker died during init`; boot line `0MB at a 30k cap` |
| `serve-fix-ladder.log` + excerpt | server, fix | DSpark agentic, `MEMRA_DSPARK_VT=ladder` (served verify graphs on by default) | `dspark-serve-smoke: ALL GREEN`, r1..r3 byte-identical to the spec-off boot; boot line `1229MB at a 30k cap (ring on, but the drafter has no sliding layer: grows with the cap)` |
| `serve-fix-default.log` + excerpt | server, fix | DSpark agentic, default window | `ALL GREEN` |
| `serve-fix-restore-ladder.log` + excerpt | server, fix | DSpark agentic, ladder, `MEMRA_DSPARK_PREFIX_RESTORE=1` | `ALL GREEN`; tail-carrying entries published and one whole-entry restore with the full drafter tail (`DSPARK restore: 69 of 69 prompt tokens + draft tail from cache`) |
| `fix-ladder-eager.log` | gate, fix | DSpark agentic, `MEMRA_DSPARK_VT=ladder` | `== dspark_q38_gate: ALL EXACT`, no ring line |
| `fix-ladder-vg.log` | gate, fix | DSpark agentic, ladder, `MEMRA_DSPARK_VERIFY_GRAPH=1` | `ALL EXACT`, `[fullg-debug]` shows the graph walk armed |
| `fix-default-eager.log` | gate, fix | DSpark agentic, default window | `ALL EXACT` |
| `fix-dflash2-vg.log` | gate, fix | DFlash2, `MEMRA_DSPARK_VERIFY_GRAPH=1` | `ALL EXACT`, `spec_sha256 == plain_sha256` on every prompt, the same digests as the DSpark runs |

Every fixed gate run prints `spec_sha256` equal to `plain_sha256` on all three prompts (`956d702c...` for ids-a and ids-c, `ed23df04...` for ids-b). The served smoke's greedy digests (`3accd719...`, `f5feea34...`, `03081186...`) equal the spec-off oracle in every fixed run. The tail cost shows in the excerpts: a 71-token entry is 159.0 MB with restore off (trunk only) and 161.9 MB with it on, a 2.9 MB tail (71 x 40,960 B). The tok/s in the gate logs are single runs and support no performance claim.

## Coverage added

`tools/fast-gate/models.tsv` gains the `cmd` probe `dsparkq38` (this gate, this drafter, `MEMRA_DSPARK_VT=ladder MEMRA_DSPARK_VERIFY_GRAPH=1`), and `map.tsv` routes `dflash.rs` and the gate source to it. The runner checks a `cmd` row's absolute-path arguments before running it: when one is missing the probe line says `SKIP (no artifact at ...)`, the command never runs, and the run still exits non-zero, like every missing model. The check comes after the build (the pre-build refusal loop skips `cmd` rows); the existing `gwstress` row now skips in the runner instead of in its own script (same verdict, different message); relative paths, env-column paths and `--x=/path` arguments are not checked. fast-gate's GPU selection is advisory, and no push-gated battery loads a DSpark drafter.

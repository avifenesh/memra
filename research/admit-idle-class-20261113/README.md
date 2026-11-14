# An idle server's refusal class and its booking on unified memory (memra#1304)

**Verdict: fixed.** On GB10 an idle server short of host memory answered a 1k request with HTTP 400
`context_length_exceeded`. Two causes, both fixed. Admission booked the prefix cache's whole
allowed growth (about 50 GB here) on an idle box with an empty cache, so a box with 20 GB free
refused a 1k request. And the idle refusal was a 400 whatever the shortfall's cause. Now an idle
box books none of the cache's growth (a busy box books all of it, as before), and an idle refusal
is a 400 only for a request the box's post-load capacity cannot hold; short of the moment's
memory it is a 503 `overloaded` with `Retry-After: 5`. Record:
[docs/decisions/IDLE-REFUSAL-CLASS.md](../../docs/decisions/IDLE-REFUSAL-CLASS.md).

The [review fold](#review-fold-2026-11-14) adds: on unified memory a prefix publication that
copies reserves its room before it allocates (a DFlash cell with 6.4 GB free refuses its 4.1 GB
copy cleanly and serves the same text); both idle refusals name the largest context that fits
now; and the discrete cell, the native-choices `kv` phase on an RTX 5090, where the idle server
under another process's memory now answers 503 `overloaded` with `Retry-After` and the size that
fits.

## Conditions

- One DGX Spark (GB10, sm_121a, unified memory), Qwen3.8-27B GGUF (NVFP4 trunk, Q5_K output head)
  with the masked MTP draft, 262,144 context, the served defaults (the unified prefix-cache budget
  derived: 50,530 MB in these boots), served by memra#1221's `serve_short.sh` (memory guard at
  1 GiB). One fresh server per cell, one cell at a time, under a `sparklock` lease.
- Arms: `main` = origin/main `01960ee1b1` (binary sha256 `444ee195...`); `head` = the branch's
  first build (`d8525064...`); `final` = the branch's final commit (`d625f12a...`), which moves one
  `let` above `let drained` so the drafter wiring pin reads its source contiguously; the same
  program. Each cell's `guard.log` has the binary's sha256.
- The recipe (memra#1221, memra#1257's short cells): the server reports ready, then
  `mem_holder.py` takes `MemAvailable` down to the cell's target in 256 MB steps, then the
  requests (`tools/idle_req.py`: memra#1221's g1k request, 1,031 prompt tokens, greedy, 512 max
  new, and on a refusal the status, the retry headers and the error body). Pair cells send a g28k
  request and, once its prime has started, the g1k one.
- The g1k request is charged `required` = 2,170 MB (524 MB cost, about 35 MB draft state, the 1,536 MiB
  transient reserve), as the head's `idle memory short` lines print. The post-load capacity read
  at boot was 104,140 to 104,956 MB.

## Cells

Cells `01`-`11` ran main (`M`) and the first head build (`H`); cells `12`-`17` repeat every head
cell on the final build (`F`). `MemAvailable` is `/proc/meminfo`'s, in MiB, just before the first
request (the pair cells' reading at the second request is in their `run.log`). Text: the sha256
prefix of the answer. The admission column quotes the server's lines (MB = 10^6 bytes).

| cell | arm | MemAvailable before (MiB) | request | HTTP | code | Retry-After | tokens, finish | text | admission |
|---|---|---|---|---|---|---|---|---|---|
| 01-M | main | 1,802 | g1k | 400 | context_length_exceeded | - | - | - | capacity reject, available 0 MB |
| 02-H | head | 1,805 | g1k | 503 | overloaded | 5 | - | - | idle memory short, available 2025 MB < required 2170 MB, capacity 104140 MB |
| 03-H | head | 1,563 | g1k | 503 | overloaded | 5 | - | - | idle memory short, available 1777 MB < required 2170 MB, capacity 104171 MB |
| 04-M | main | 1,542 | g1k | 400 | context_length_exceeded | - | - | - | capacity reject, available 0 MB |
| 05-M | main | 2,593 | g1k | 400 | context_length_exceeded | - | - | - | capacity reject, available 0 MB |
| 06-H | head | 2,580 | g1k | 200 | - | - | 127, stop | 26ca98d4 | admitted |
| 07-H | head | 99,232 | g1k | 200 | - | - | 127, stop | 26ca98d4 | admitted |
| 08-M | main | 99,251 | g1k | 200 | - | - | 127, stop | 26ca98d4 | admitted |
| 09-M | main | 20,183 | g28k | 400 | context_length_exceeded | - | - | - | capacity reject, available 0 MB |
|  |  |  | g1k | 400 | context_length_exceeded | - | - | - |  |
| 10-H | head | 20,183 | g28k | 200 | - | - | 120, stop | 0b725b4c | 15 VRAM defer lines (1 active) |
|  |  |  | g1k | 200 | - | - | 127, stop | 26ca98d4 |  |
| 11-H | head | 99,987 | g1k | 400 | context_length_exceeded | - | - | - | capacity reject, available 104941 MB, capacity 104951 MB |
| 12-F | final | 1,817 | g1k | 503 | overloaded | 5 | - | - | idle memory short, available 2043 MB < required 2170 MB, capacity 104187 MB |
| 13-F | final | 1,549 | g1k | 503 | overloaded | 5 | - | - | idle memory short, available 1753 MB < required 2170 MB, capacity 104197 MB |
| 14-F | final | 2,585 | g1k | 200 | - | - | 127, stop | 26ca98d4 | admitted |
| 15-F | final | 99,238 | g1k | 200 | - | - | 127, stop | 26ca98d4 | admitted |
| 16-F | final | 20,184 | g28k | 200 | - | - | 120, stop | 0b725b4c | 15 VRAM defer lines (1 active) |
|  |  |  | g1k | 200 | - | - | 127, stop | 26ca98d4 |  |
| 17-F | final | 99,991 | g1k | 400 | context_length_exceeded | - | - | - | capacity reject, available 104948 MB, capacity 104956 MB |

- **The recipe (MemAvailable about 1.8 GB).** Main refuses with 400 and `x-should-retry: false`
  (`available 0MB`), reproducing memra#1257's `01-T` and `02-B`. Head refuses with 503
  `overloaded`, `Retry-After: 5`: with nothing booked the reading is 2,025 MB, still under the
  2,170 MB charge, and the line names the capacity it fits.
- **Short even with nothing booked (about 1.5 GB).** Main 400, head 503.
- **Short only with the growth booked (about 2.6 GB).** Main 400 (`available 0MB`). Head admits;
  the request completes with 127 tokens, `stop`, and the text every admitted g1k in this record
  and in memra#1257's wrote (`26ca98d4...`). The draft-graph capture is skipped by its own guard
  (effective free 2,816 MB under its 3,140 MB floor) and the request decodes eagerly, as in
  memra#1257's cache-off cells.
- **Ample memory.** Main and head both admit at once and write the same text: admission is
  unchanged where the booking fit.
- **A busy box (about 20 GB, two requests).** Main refuses both with 400. Head admits the g28k on
  the idle box; the g1k arriving 3 s later finds it active, books the cache's whole growth again
  (`VRAM defer: 1 active, effective free 0MB`), waits FIFO for 17 s and completes after it with
  the same text. This is the overload guard the booking exists for, unchanged.
- **A charge above the capacity (teeth door).** With `MEMRA_ADMIT_RESERVE_MB=200000` the g1k is
  charged about 210 GB against a 104,951 MB capacity: head answers the 400
  `context_length_exceeded` (`capacity reject: ... available 104941MB, post-load capacity
  104951MB`). The 400 path is live.
- No cell logged an out-of-memory line, in the server or in the kernel (`kernel-oom-count.txt`).

- **The final build repeats the head.** Cells `12-F` to `17-F` give the same status, code,
  `Retry-After`, tokens and text as `02-H`, `03-H`, `06-H`, `07-H`, `10-H` and `11-H`, with
  readings within 30 MB (the g1k in `16-F` again waits 17 s behind the g28k).
- The first CPU suite on the head build failed one test: the drafter wiring pin
  (`the_drafter_refusal_is_wired_from_the_gate_to_every_dflash_route`) reads
  `let drained = ...; if drained {` contiguously, and the new `let` sat between them. The final
  build declares it above (`cpu/` holds the final run; the first run's summary is in this README
  only).

## CPU

On the dev desk (x86, the branch's final source on main `01960ee1b1`, `tools/dd-checks.sh`,
logs in `cpu/`):

- New and changed tests in `memra-server` (`worker::tests`):
  `an_idle_unified_box_books_no_prefix_growth` (cell `02-B`'s reading: nothing booked on an idle
  box and the arrival fits; the whole growth booked with a session active and it does not),
  `an_idle_refusal_is_retryable_unless_the_request_exceeds_the_post_load_capacity` (cell `01-T`'s
  reading gives `Overloaded`; the 2026-09-10 185,055-token request on a 24 GB card stays
  `ContextLength` with the largest context named; the capacity arithmetic; a per-device reading),
  `the_idle_refusal_is_classified_against_the_post_load_capacity` (the wiring: the capacity read
  after the boot calibration, the drained refusal classified at it, the reject answering through
  `idle_refusal`), and `unified_memory_admission_books_the_prefix_caches_allowed_growth` (updated
  for the idle argument, and the call site pinned to `active.is_empty()`).
- Red arms, each with the tests kept and the behavior restored in the source: R1 the booking
  ignoring idle fails the booking test (`left: 52411000000, right: 0`); R2 `idle_refusal` always a
  400 fails the class test (`left: ContextLength, right: Overloaded`); R3 the reject branch
  answering `EngineError::context_length` again fails the wiring test.
- The memra-server library suite: 1,265 passed, 0 failed, 18 ignored. Clippy (`memra-server`, all targets, `-D warnings`) and
  `cargo fmt --check` clean.
- On the rig: `tools/check-doc-links.py`, `tools/check-public-boundary.py`, `tools/check-flags.sh`
  and `git diff --check` clean.

## Review fold (2026-11-14)

An independent review of the branch at `ce99d6ebad` asked for changes; this section is their
evidence. The branch was rebased onto main `bbc6f83aea`; the fold is two code commits,
`6c2a95fdb9` and `43a727db3c` (the head these cells ran: build id `408ad3a2d420`, printed by
every server below in its `[server] build:` line).

Those ids, and `source` in the kv manifests, are the commits as measured. The branch was then
rebased once more, onto main `63c351a2f4` (the DSv4.1 CUTLASS landing), which gives them new ids.
That rebase changes 2,449 files, every one of them the landing's own: under `crates/` the
engine's DSv4 and DSv4.1 sources and kernels, its build script and manifest, the env registry
and the vendored CUTLASS headers. `crates/memra-server` is byte-identical (`worker.rs` sha256
`f7eb7100...`), and so are `dflash.rs` and `glm_spec.rs`; none of the changed engine files is on
the path these cells serve (Qwen GGUF, MTP and DFlash2). The cells and the red arms therefore
bind the rebased head by source, not by build id (the rebased tree's is `cfbd7f76c9b6`).
On the rebased tree the dev desk checks ran again (`cpu-fold/rebased/`): the memra-server
library suite 1,271 passed, 0 failed, 18 ignored; the engine's dflash tests 66 passed; clippy
(`memra-engine` and `memra-server`, all targets, `-D warnings`) and rustfmt on the touched files
clean.

What changed:

- **A publication that copies reserves its room first.** The claim above that every insert is
  clamped to live free memory held for the entry's place in the cache, not for its bytes: the
  DFlash and GLM-5 boundary publishers, an MTP publication whose plane cannot lend and a host
  promotion allocated their copy and were clamped by the insert afterwards. On unified memory
  `prefix_copy_preflight` now makes the room before the first byte (cells `28-G`, `29-G`,
  `31-G`). A discrete card keeps its path: a copy that does not fit fails as a CUDA
  out-of-memory error the publisher handles.
- **Both idle refusals name the largest context that fits now**, so a 503 that memory held for
  good never clears still tells the caller what to send, and the 400's hint is not a size that
  then gets the 503. The arithmetic counts every state a choice group builds (take 1 below).
- **The native-choices gate follows the new contract** (`tools/replay_native_choices.py`), and
  its `kv` phase is recaptured on the RTX 5090: the discrete cell this record lacked.
- A primary reading's idle capacity is pinned to the primary device; a prefix insert the live
  clamp refuses says so instead of the "larger than the entire byte budget" warning (`20-G`
  against `14-F`).

### RTX 5090: the idle refusal under another process's memory

One RTX 5090 Laptop (24 GB, driver 615.71.09, balanced profile, on AC, 78 C at the start), the
native-choices fixture (Qwen3.5-9B Q8_0, 4,096 context, eight sessions, prefix cache off), the
accounting observer built from the fold head (`43a727db3c`, sha256 `a414c93f...`, build id
`408ad3a2d420`). `tools/native_choices.py --phase kv` under a `gpulock` lease: a warm request, then
a process of the tool's own allocates all but 1 GiB of the card's free memory and an `n = 8`,
`max_tokens = 3072` request arrives on the idle server, then the allocation is freed and an
`n = 8` request runs. Receipt: `rtx5090/kv/` (the tool's directory), `rtx5090/kv.log`,
`gpu-before.txt`, `gpu-after.txt`, `verify.log`.

| step | result |
|---|---|
| boot | `[admission] idle capacity (post-load, cache empty): dev0 free 16424MB (pool used 8469MB)` |
| warm request | 200 |
| the tool's allocation | 14,957 MB taken of 16,031 MB free; 1,072 MB left |
| `n = 8` request, idle server | **503 `overloaded`, `Retry-After: 5`**: `idle memory short: ... ctx=3142 fits the post-load capacity (16424MB) but not an IDLE box's free memory now (available 1193MB, required 1517MB, largest context that fits now 712)`; the body says "the largest context that fits now is 712 tokens"; no prime, no fork, no worker row |
| allocation freed, `n = 8` again | 200, eight choices |

- The stored receipt of this cell, `research/n-choices-528-20261003/receipts/native-final-v5/kv`,
  is the same card and fixture on the `c6db` server: the same shortfall (available 1,192 MB) got
  400 `context_length_exceeded` and `x-should-retry: false`. The request fits the card at rest
  ten times over; the memory was another process's and came back seconds later.
- The verifier (`tools/replay_native_choices.py`, `check_idle_refusal`) judges the sizes the
  server logs, not the class it logged: charge within the post-load capacity means 503 with
  `Retry-After`, charge over it means 400 with `x-should-retry: false`, and the capacity must sit
  within the boot reading. The old receipt fails it (`cpu-fold/RV-frozen-kv-receipt-under-the-new-verifier.log`)
  and stays in place as that server's record; `native-final-v5/superseded.json` points the
  composite replay at this one, bound to its own source, binary, model, helpers and hardware by
  `rtx5090/kv/expected-context.json`. `tools/test_choice_verifier.py`: 25 tests pass.
- **Take 1** (`rtx5090/take1/`, the first fold commit `6c2a95fdb9`, observer `33f9ed72...`, build
  id `bcea3993f98b`): the same 503 and `Retry-After`, and no size in the message. The size hint
  divided the 324 MB overshoot by one row's 14,848 bytes per token while the group is charged nine
  states, so it shed more tokens than the request has. `43a727db3c` counts the states; take 2
  above names 712 tokens: by the server's own cost line (`14848 B/token x ctx + 2MB
  prefill-workspace + 103MB fixed`) nine states of 712 tokens are 1,040 MB, and with the 152 MB
  the 1,517 MB charge carries beyond nine states of 3,142 tokens that is about 1,192 MB, just
  under the 1,193 MB free.

### GB10: the head's cells on the fold build

The same unit, model, recipe and harness as above (the masked MTP draft), arm `fold` = the head's
sm_121a build made on the unit (sha256 `366866aa...`). `tools/campaign3.sh`; receipts in
`gb10/<cell>/`, the campaign's log in `gb10/campaign3.log`.

| cell | arm | MemAvailable before (MiB) | request | HTTP | code | Retry-After | tokens, finish | text | admission |
|---|---|---|---|---|---|---|---|---|---|
| 18-G | fold | 1,994 | g1k | 200 | - | - | 127, stop | 26ca98d4 | admitted |
| 19-G | fold | 1,534 | g1k | 503 | overloaded | 5 | - | - | idle memory short, available 1745 MB < required 2170 MB, capacity 104242 MB |
| 20-G | fold | 2,514 | g1k | 200 | - | - | 127, stop | 26ca98d4 | admitted |
| 21-G | fold | 99,290 | g1k | 200 | - | - | 127, stop | 26ca98d4 | admitted |
| 22-G | fold | 20,186 | g28k | 200 | - | - | 120, stop | 0b725b4c | 15 VRAM defer lines (1 active) |
|  |  |  | g1k | 200 | - | - | 127, stop | 26ca98d4 |  |
| 23-G | fold | 100,039 | g1k | 400 | context_length_exceeded | - | - | - | capacity reject, available 105002 MB, capacity 105007 MB |
| 27-G | fold | 1,799 | g1k | 503 | overloaded | 5 | - | - | idle memory short, available 2023 MB < required 2170 MB, capacity 104264 MB |

- Every head cell repeats: `27-G` and `19-G` are the 503s of `12-F` and `13-F` (`Retry-After: 5`,
  available 2,023 MB and 1,745 MB under the 2,170 MB charge), `20-G` and `21-G` admit and write
  `26ca98d4...`, `22-G` admits the g28k idle and defers the g1k behind it (15 defer lines, the
  texts of `16-F`), `23-G` answers the 400 (`capacity reject: ... available 105002MB, post-load
  capacity 105007MB, required 210275MB`). Neither 503 names a size, correctly: the charge's fixed
  part (the prefill workspace, the draft state and the reserve, about 2,118 MB) is over the free
  memory, so no context fits.
- `18-G` asked for the 1.8 GB recipe and was admitted: `cell.sh` released the request when
  `MemAvailable` was within 300 MiB of the target, 26 s into the holder's descent, at 1,994 MiB,
  which with the pool's cached bytes is over the charge. `cell.sh` now takes `SETTLE_MB`, and
  `27-G` is the recipe again with the request held until the holder is within 48 MiB (1,799 MiB,
  the 503). `18-G` stays as an admit at 2.0 GB.
- In `18-G` and `20-G` the admitted request's 157 MB entry is refused by the live clamp, as in
  `14-F`, and the line now names the cause: `insert refused: entry 156913664 exceeds budget 0
  (live free memory holds the insert budget under its configured 50544556032; spec-boundary,
  model q38)`. The "first prefix insert is larger than the entire byte budget" warning `14-F`
  printed is gone. The lending publication's `[specpub]` line shows what it would have held:
  `preflight_bytes=156913664 entry_bytes=156913664`.
- No out-of-memory line in any cell, in the server or in the kernel (`kernel.log` per cell from
  this campaign on; `gb10/kernel.log` is the kernel journal of the first two campaigns' window,
  read back from the unit: eight audit lines, none about memory).

### GB10: a publisher that copies

The same trunk with the DFlash2 drafter (`serve_short.sh` with `ROUTE=dflash`: the DSpark route,
whose boundary publisher copies the trunk planes and the drafter tail), one g131k request
(131,046 prompt tokens, greedy, 512 max new). The request is charged 10,176 MB (8,388 MB cost,
of which 4,479 MB is the prefill workspace, 169 MB draft state, the 1,544 MiB reserve); its
boundary entry is 4,131.5 MB.

| cell | arm | MemAvailable before (MiB) | request | HTTP | code | Retry-After | tokens, finish | text | admission |
|---|---|---|---|---|---|---|---|---|---|
| 28-G | fold | 96,013 (no holder) | g131k | 200 | - | - | 95, stop | e3b0c1e8 | admitted; published, 4,131.5 MB |
| 29-G | fold | 10,311 (regulating holder) | g131k | 200 | - | - | 95, stop | e3b0c1e8 | admitted; published, 12,137 MB free at the publication |
| 31-G | fold | 11,279 (static holder) | g131k | 200 | - | - | 95, stop | e3b0c1e8 | admitted; publication refused before allocating, 6,366 MB free |

- **Ample (`28-G`).** The publication copies and inserts: `insert (dspark-boundary): 131008
  tokens, 4131.5MB`, and the bytes the preflight counted are the entry's:
  `[specpub] Copy publication ... preflight_bytes=4131520512 entry_bytes=4131520512`.
- **Free memory just above the charge, a holder that gives memory back (`29-G`).** The harness's
  holder returns a chunk whenever `MemAvailable` falls under its target, so the unit stayed near
  10.2 GiB while the server took its KV. The arrival is admitted with nothing booked; at the
  publication 12,137 MB is free, the preflight's budget is 8,535 MB, the 4,131.5 MB copy fits
  and is published. Same text.
- **The same with a holder that does not (`31-G`).** `mem_holder.py static` stops at its target
  and holds a fixed amount, as another process's memory is; the request's own growth comes out
  of what is left (`MemAvailable` 11,279 MiB before the request, 4,642 MiB at its lowest). At the
  publication 6,366 MB is free, 2,608 MB above the headroom: `insert budget 2765MB (configured
  49153MB; live free 6366MB, headroom 3758MB, resident 0MB, new entry 157MB)`, then `insert
  refused: entry 4131520512 exceeds budget 2765296228 (live free memory holds the insert budget
  under its configured 49153134592; snapshot preflight, model q38)`. Nothing is allocated (no
  `[specpub]` line, no insert), the request completes with 95 tokens, `stop`, and the text of
  `28-G` and `29-G` (`e3b0c1e8...`). By the code before the fold this publication copied its
  4.1 GB into the 6.4 GB, which leaves the unit about 1.5 GB under its headroom, and the insert
  refused it after (read from the source, not run: see below).
- `24-G` did not boot: the first `serve_short.sh` set `MEMRA_DSPARK_PARTIAL_RESTORE`, retired on
  2026-11-07, and the boot audit refused it (no request was sent, `25-G` was skipped). The arm
  now sets no restore door (one switch, on by default on GB10 with DFlash armed); `28-G` is the
  cell again.
- Not run: the GLM-5 publisher and a host promotion on unified memory (the host tier is off by
  default there), which take the same preflight; and the branch before the fold in `31-G`'s
  conditions, which would have been a deliberate overcommit of a shared unit. The red arms
  below cover those paths in the source.

### CPU (fold)

On the dev desk (x86, CUDA 13.1, `MEMRA_CUDA_ARCH=120a`), the head's source (`worker.rs` sha256
`f7eb7100...`), logs in `cpu-fold/`:

- The memra-server library suite: 1,271 passed, 0 failed, 18 ignored (`checks/suite.log`). The
  memra-engine library suite, since `dflash.rs` and `glm_spec.rs` changed: 779 passed, 0 failed,
  66 ignored (`engine/engine-lib.log`). Clippy (`memra-engine` and `memra-server`, all targets,
  `-D warnings`) clean; `rustfmt --edition 2024 --check` on the three touched files clean.
- New tests in `memra-server` (`worker::tests`):
  `a_copy_publication_reserves_its_room_before_it_allocates` (the reviewer's case and the
  boundary in numbers through `prepare_snapshot`, and the wiring: the publication decides lend
  or copy, reserves, and only then exports the tail, slices a latent plane or allocates; the
  publishers hand the tail over as its size and its export; a host promotion reserves before it
  materializes), `a_discrete_card_keeps_its_copy_publication_order`,
  `the_idle_refusal_names_the_largest_context_that_fits_now` (a single request, the 400's two
  figures, a choice group), `a_primary_reading_takes_the_primary_devices_idle_capacity`,
  `a_live_memory_refusal_names_its_cause_and_keeps_the_shape_warning`. Updated: the three idle
  refusal tests of the first version, the insert-guard count (five sites), and
  `tail_publication_is_gated_on_the_restore_flag` (the export is a closure now; the pin follows
  it and still requires it under `dspark_prefix_restore_on()`).
- Red arms, one build each with the tests kept (`tools/mutate.py`, `tools/dd-red.sh`; logs in
  `red/`). Each fails the test it names, and the failure output contains the string in the last
  column:

| arm | from | behavior restored or broken | test that fails | its failure says |
|---|---|---|---|---|
| R1 | author | the booking ignores idle | `an_idle_unified_box_books_no_prefix_growth` | `left: 52411000000` |
| R2 | author | the idle refusal is always a 400 | `an_idle_refusal_is_retryable_unless_the_request_exceeds_the_post_load_capacity` | `left: ContextLength` |
| R3 | author | the reject branch answers `EngineError::context_length` again | `the_idle_refusal_is_classified_against_the_post_load_capacity` | `largest_now), );` |
| R4 | review 1 | the admission call site passes `false` for idle | `unified_memory_admission_books_the_prefix_caches_allowed_growth` | `active.is_empty(), prefix_cache_budget_bytes()` |
| R5 | review 1 | the capacity no longer subtracts memra's pool growth since boot | `an_idle_refusal_is_retryable_unless_the_request_exceeds_the_post_load_capacity` | `left: 6000000000` |
| RA | review 2 | a Primary reading is judged at the first boot device | `a_primary_reading_takes_the_primary_devices_idle_capacity` | `left: 2000000000` |
| R6 | fold | the size that fits now is computed on the capacity | `the_idle_refusal_is_classified_against_the_post_load_capacity` | `let largest_now = largest_context_within( admission_cap, required, available,` |
| R7 | fold | the 503 does not name the size that fits now | `the_idle_refusal_names_the_largest_context_that_fits_now` | `free memory right now; retry after the Retry-After delay` |
| R8 | fold | the 400 names only the capacity's size | `the_idle_refusal_names_the_largest_context_that_fits_now` | `the largest admissible context is 174315 tokens. Retrying will not help` |
| R9 | fold | the spec publisher allocates without reserving | `a_copy_publication_reserves_its_room_before_it_allocates` | `missing 'if to_allocate > 0 && !prefix_copy_preflight(` |
| R10 | fold | the preflight counts the bytes it is about to allocate as allocated | `a_copy_publication_reserves_its_room_before_it_allocates` | `left: (4166000000, 4166000000)` |
| R11 | fold | the DSpark publisher's tail bytes are not counted | `a_copy_publication_reserves_its_room_before_it_allocates` | `missing 'bytes: dspark.draft_kv().tail_bytes(end),'` |
| R12 | fold | a host promotion materializes its entry without reserving | `a_copy_publication_reserves_its_room_before_it_allocates` | `missing 'let hi = host_promote_reserve(` |
| R13 | fold | a live-clamp refusal spends the one-time shape warning | `a_live_memory_refusal_names_its_cause_and_keeps_the_shape_warning` | `left: (2, true)` |
| R14 | fold | the copy preflight runs on a discrete card too | `a_discrete_card_keeps_its_copy_publication_order` | `missing 'if !device_is_unified(engine) { return true; }'` |
| R15 | fold | a choice group's hint uses one row's slope | `the_idle_refusal_is_classified_against_the_post_load_capacity` | `.saturating_mul(choice_state_count as u64);` |

  `red/summary.txt` marks R10 `NOT-RED`: its test failed, at the split check before the
  reservation check my script was told to look for. With the expected string corrected it is
  rerun alone (`red/summary-R10-rerun.txt`): red as stated.
- The gate's arm: the stored `kv` receipt under the new verifier fails `kv_exhaustion`
  (`RV-frozen-kv-receipt-under-the-new-verifier.log`); with `superseded.json` the composite
  replay passes. `tools/test_choice_verifier.py`: 25 tests (two new: the refusal's contract on
  synthetic records with the old contract as a red case, and the supersession), floor raised from
  23 to 25 in `tools/validation_plan.py`, which also lists the recapture as the suite's input.
- On the rig: `tools/check-doc-links.py`, `tools/check-public-boundary.py` (0 new),
  `tools/check-flags.sh`, the support-records contract (`tools/unittest-floor.sh tools
  test_check_support_states.py 21`, `tools/check-support-states.py`), `tools/test_validation_plan.py`
  and `git diff --check` clean.
- Merged with memra PR #1310's head `716caf9006` (its fold of the parked-evict reap, which also
  edits `worker.rs`; trial merge on main `bbc6f83aea`): `worker.rs` merges without conflict, the
  merged tree's memra-server suite is 1,280 passed, 0 failed, and the merged file is
  rustfmt-clean (`cpu-fold/merged-1310/`).
- The first fold build's suite failed one pin, `tail_publication_is_gated_on_the_restore_flag`:
  moving the tail's export into a closure had taken it out from under the flag's `if`. The
  publisher builds the export under the flag again; the head's suite is the one above.

## Tools

`tools/`: `build.sh` (the sm_121a builds on the unit), `cell.sh` and `campaign.sh` /
`campaign2.sh` (the cells), `idle_req.py` (the request), `assemble.sh` (receipts here, home paths
written as `~`), `tab.py` (the table), `dd-checks.sh` (the CPU checks and red arms). They use
memra#1221's `serve_short.sh`, `mem_holder.py`, `pf_long_memra.py`, `eos_ab.py` and `tsrun.py`
from `research/spec-capture-headroom-20261112/tools/`.

The fold's: `campaign3.sh` and `campaign4.sh` (cells `18-G` to `31-G`; the lines added to
`campaign3.sh` while it ran are marked in it), `cell.sh` with `ROUTE`, `REQ`, `HOLD_S`,
`SETTLE_MB` and `HOLD_MODE`, this record's own `serve_short.sh` (the `ROUTE=dflash` arm) and
`mem_holder.py` (the `static` mode), `charge.py` (a cell's target from the charge the server
prints), `kv5090.sh` (the RTX 5090 cell under `gpulock`), `mutate.py` and `dd-red.sh` (the red
arms, one build each), `dd-checks-fold.sh`, `dd-chain.sh` and `dd-post.sh` (the dev desk checks).
`pf_long_memra.py`, `eos_ab.py` and `tsrun.py` are still memra#1221's.

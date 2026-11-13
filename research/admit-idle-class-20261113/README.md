# An idle server's refusal class and its booking on unified memory (memra#1304)

**Verdict: fixed.** On GB10 an idle server short of host memory answered a 1k request with HTTP 400
`context_length_exceeded`. Two causes, both fixed. Admission booked the prefix cache's whole
allowed growth (about 50 GB here) on an idle box with an empty cache, so a box with 20 GB free
refused a 1k request. And the idle refusal was a 400 whatever the shortfall's cause. Now an idle
box books none of the cache's growth (a busy box books all of it, as before), and an idle refusal
is a 400 only for a request the box's post-load capacity cannot hold; short of the moment's
memory it is a 503 `overloaded` with `Retry-After: 5`. Record:
[docs/decisions/IDLE-REFUSAL-CLASS.md](../../docs/decisions/IDLE-REFUSAL-CLASS.md).

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

| cell | arm | MemAvailable before (MB) | request | HTTP | code | Retry-After | tokens, finish | text | admission |
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

## Tools

`tools/`: `build.sh` (the sm_121a builds on the unit), `cell.sh` and `campaign.sh` /
`campaign2.sh` (the cells), `idle_req.py` (the request), `assemble.sh` (receipts here, home paths
written as `~`), `tab.py` (the table), `dd-checks.sh` (the CPU checks and red arms). They use
memra#1221's `serve_short.sh`, `mem_holder.py`, `pf_long_memra.py`, `eos_ab.py` and `tsrun.py`
from `research/spec-capture-headroom-20261112/tools/`.

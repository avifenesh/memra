# The pre-capture guard reads the allocatable free memory on unified memory (memra#1221)

Date: 2026-11-12. Decision record:
[`docs/decisions/SPEC-CAPTURE-HEADROOM-UNIFIED.md`](../../docs/decisions/SPEC-CAPTURE-HEADROOM-UNIFIED.md).
Follows `research/spec-boundary-publish-20261110/` (memra#1208), whose remaining pause memra#1221
sized.

**Verdict: adopted, the pre-capture guard reads the allocatable free memory (memra#1221).** On
GB10 the guard before a draft-graph capture read `cuMemGetInfo`, which leaves out page cache.
With the page cache full it trimmed the pool before a 260k prompt's capture, and the prompt's
prefix publication grew the pool back by 2.4 GB on the scheduler thread one burst later; it also
refused 1k requests' captures with about 100 GB available. On unified memory the guard now reads
the larger of the driver's reading and `MemAvailable` (the reading admission already sizes with);
a discrete card reads as before. Against main on one unit with a filled page cache, three cells
each at 260k: the publisher falls from 152.7 to 124.2 ms (-28.5 ms, -18.7%), the first burst
boundary from 165.7 to 142.7 ms (-23.0 ms), every head cell under every main cell; the pool grows
1,408 MB inside the publisher instead of 2,400; the trim fires in 0 cells instead of 3 and the 1k
capture is refused in 0 instead of 3. Rounds, TTFT and texts are unchanged. At 131k the trim fired
in one main cell of two (publisher 74.5 against 54-58 ms). The narrow trim (only the shortfall)
stays as the backstop; on its own it gave no measurable gain. The helper-thread and prime-end
reservations of the publisher's planes (the issue's lever) were measured and rejected: they move
the growth into burst 1 (+123 ms at 131k) and hold more memory. The plane copies (about 40 ms at
131k, 77 ms at 260k) are untouched.

## Conditions

- **GB10**: two DGX Spark units, `u1` and `u2` in the directory names (sm_121a), Qwen3.8-27B GGUF (NVFP4 trunk, Q5_K output
  head) with the masked MTP draft, 262,144 context, served as the launcher serves it
  (`tools/serve.sh`): the MTP speculative path, K=3 on a cold request, 32-token bursts,
  `MEMRA_PREFIX_STABLE_BOUNDARY=1`, `MEMRA_TICK_TRACE=1` (every server line timestamped by
  `tools/tsrun.py`). One cell per Spark lease, one boot per cell.
- **Cells** (`tools/specpub_cell.sh`, `tools/specpub_ab.py`): on a fresh server, g1k (a
  1,031-token prompt) then the set's cold request (g131k: 131,046 tokens, g260k: 259,946), each
  greedy to EOS with its own nonce, so neither hits the other's prefix entry.
- **Filled page cache** (`tools/fill_cache.sh`): once the server is ready and before the
  requests, the cell reads model files (largest first) until MemFree is under 3,000 MB or 120 s
  have passed. Nothing is written and no system setting is touched; the kernel reclaims these
  pages on demand. On these units MemFree falls to 2.4-5.1 GB while MemAvailable stays near
  101 GB. This is the state a GB10 host is in after it has read a few checkpoints, and the state
  in which the pre-capture guard's raw reading is low.
- **Probe arms**: every arm of the record campaigns carries arm I's trace-only lines in
  `worker.rs` (`tools/probe_i.py`, ported to 782dc471b7 as `tools/probe_i782.py`): the default
  pool's reserved and used bytes at the publisher's entry and exit, and each plane allocation's
  time and pool growth. No allocation, copy or order of calls changes (131k publisher 49.5-51.1 ms
  with the lines, 51.1 ms without).
- **Columns**: `pub` is the publisher's time on the scheduler thread (insert line minus burst 1's
  line); `stall` the first burst boundary over the rounds it holds; `growth` the pool's
  reserved bytes at the publisher's exit minus its entry; `cached` the pool's free blocks at the
  publisher's entry; `trim` the pre-capture guard's line in the cold request (`released X of Y MB`,
  the raw or allocatable reading and the requirement). `tools/specpub_tab.py` computes every
  column from the cell's trace.

## Arms

| arm | build | what it is |
|---|---|---|
| I | 68b46bfc18 + probe | main's publisher with trace lines (the morning's reference) |
| P | I + prewarm | the publisher's planes allocated and freed at the prime's end, on the scheduler thread |
| H | I + helper thread | the planes allocated and freed by a helper thread at burst 1's second commit |
| O | 3afa740d13 | memra#1208's head (lever 5), the 260k cell where the trim did not fire |
| B (`bi`) | 782dc471b7 + probe | main's guard: raw reading, trim to zero |
| W (`wi`) | 782dc471b7 + probe + the PR's first commit's spec.rs | raw reading, the trim releases only the shortfall |
| N (`ni`) | 782dc471b7 + probe, trim skipped | raw reading, the trim never runs (the trigger question) |
| V (`vi`) | 782dc471b7 + probe + the PR's spec.rs | this PR: allocatable reading, the narrow trim as backstop |
| `v` | 782dc471b7 + the PR's spec.rs | V without the probe (the gates' binary) |

**Directory labels.** The cell directories name arms by one letter, which `specpub_cell.sh` maps
to a binary per campaign (each cell's `guard.log` has its sha256):

| campaign | B | W | N | I / H / P / O |
|---|---|---|---|---|
| `probe131k`, `probe260k`, `ih131k-u1` | | | | I, H, P, O as above |
| `zt131k`, `fa260k-u1` | 782dc471b7 main, no probe | narrow trim, no probe | | |
| `pa260k-u1` | `bi` | `wi` (narrow trim) | `ni` | |
| `qa260k-u1`, `qa131k-u1`, `short-u1`, `band-u1` | `bi` | `vi` (the PR, called V in the text) | | |
| `rtx28k`, `rtx60k` | main 46920874c0 | narrow trim alone | | |
| `rtx28k-v`, `rtx28k-v2` | main 46920874c0 | the PR before review | | |

The review's change (17f4e087ef) touches only the discrete branch and the code's shape; on
unified memory it decides, trims and logs as `vi` did, so the GB10 cells and gates stand for the
head.

`spec.rs` at 782dc471b7 equals main 46920874c0's and cdfdfd6006's. V's `spec.rs` is the PR's at
6009545f71 (sha256 35f42088...); the head's differs from it only by the review change above. The
B/W/N/V builds share one tree and one target directory on u1 (`gb10/build-probe.log`,
`gb10/build-arms.log`).

## 1. Where the pause goes (main, Nsight Systems, one profile per size)

`gb10/nsys/` (`tools/specpub_nsys.sh`, `specpub_gap.py`, `specpub_split.py`; profiles of main
68b46bfc18 on unit 2, the `.sqlite` files are kept off the tree):

| | 131k | 260k |
|---|---:|---:|
| pause between burst 1's last kernel and burst 2's first, ms | 57.53 | 271.95 |
| the 34 plane copies, ms (MB) | 38.59 (4,133.6) | 76.81 (8,200.5) |
| device busy, ms | 42.07 | 80.55 |
| device idle, ms | 15.46 | 191.40 |
| host inside `cuMemAllocAsync`, 134 calls, ms | 51.58 | 261.90 |

Arm I's trace lines on the same request (`gb10/probe131k/01-I`, `gb10/probe260k/01-I`):

| | 131k | 260k |
|---|---:|---:|
| pool cached at the publisher's entry, MB | 3,626 | 5,846 |
| the published entry, MB | 4,290.5 | 8,357.4 |
| pool growth inside the publisher, MB | 544 | 2,400 |
| publisher on the scheduler thread, ms | 49.5 | 260.8 |
| pre-capture trim in the request | none | released 1,696 MB (driver 2,559 < 3,140) |

The idle time is the pool growing: the first planes are served from cached blocks in
microseconds, the last ones each grow the pool by 96-288 MB and take 8-77 ms on the host
(`gb10/probe260k/01-I/trace.txt`, the `[specpub-probe] alloc` lines). At 260k memra#1208's head
had published in 129.6 ms in a cell where the trim did not fire (`gb10/probe260k/03-O`), and
main's 261-272 ms cells were those where it did.

## 2. Moving the growth off the boundary: rejected

Both arms reserve the publisher's planes earlier, then free them, so the publisher's
allocations are served from cached blocks.

| cell | publisher, ms | stall, ms | burst 1, ms | pool reserved early, MB |
|---|---:|---:|---:|---:|
| 131k I (`probe131k/01-I`) | 49.5 | 45.1 | 1,512.3 | |
| 131k P, at the prime's end (`02-P`) | 0.7 | 44.7 | 1,624.5 | 3,150 |
| 131k H, helper thread (`04-H`) | 3.3 | 38.3 | 1,638.6 | 3,320 |
| 260k I (`probe260k/01-I`) | 260.8 | 297.7 | 2,227.8 | |
| 260k P (`02-P`) | 13.3 | 92.1 | 2,890.5 | 6,440 |
| 260k H (`04-H`) | 0.9 | 78.5 | 2,679.2 | 8,200 requested, the pool grew 6.6 GB |

Same-unit interleaved at 131k (`gb10/ih131k-u1`, three cells each, medians with spread): the
helper thread takes the publisher from 50.2 (1.5) to 3.1 (0.6) ms and the stall from 55.8 (8.3) to
42.5 (6.7) ms, and puts 123 ms on burst 1 (1,534.5 (2.9) to 1,657.5 (6.2) ms) and 2.2 ms on each of
burst 2's rounds (138.53 to 140.77). The reservation grows the pool by 3.3 GB at 131k because the
prime's transients are still alive at burst 1's second commit, so it holds more memory than the
entry needs. Both arms move the growth into the first burst and hold more memory; neither shortens
the request. P's prewarm also lands before the first token (117 ms at 131k, 641 ms at 260k).
The H and P code is not on the branch; `tools/probe_h.py` and `tools/probe_p.py` rebuild them.

## 3. The trim, and what the trigger reads

The pre-capture guard (`capture_headroom_refusal`, `spec.rs`) runs before a draft-graph capture.
On main it compared `cuMemGetInfo`'s free bytes with the capture's requirement (twice its
measured appetite plus two 1,536 MiB floors: 3,139-3,141 MiB here) and, when the driver alone was
short but cached pool blocks would cover it, trimmed the pool to zero; when even both were short
it refused the capture (the request drafts eagerly).

On GB10 `cuMemGetInfo` leaves out page cache, which the kernel hands to the driver on demand.
With the page cache filled (MemFree 2.4-5.1 GB, MemAvailable about 101 GB) the raw reading at the
cold request's capture was 2,008-2,785 MB, so the trim fired in every main cell at 260k. The
request had just grown the pool through its prime; the trim released those blocks (1,696 MB) and
the publisher grew the pool again one burst later.

## 4. The narrow trim alone (W), and no trim (N)

`gb10/pa260k-u1`: probe arms on unit 1, page cache filled, 260k, cells in the order
B W W B B W, then N twice (`08-N` ran between `04-B` and `05-B`, `07-N` last).

| cell | trim at the capture (MB) | cached at entry, MB | growth, MB | pub, ms | stall, ms | g1k capture |
|---|---|---:|---:|---:|---:|---|
| 01-B | rel. 1696 MB (driver 2599 < 3140) | 5846 | 2400 | 181.1 | 186.6 | captured |
| 02-W | rel. 416 of 1813 MB (driver 2710 < 3139) | 6423 | 1824 | 167.5 | 176.4 | refused |
| 03-W | rel. 1024 of 1813 MB (driver 2106 < 3140) | 5847 | 2400 | 153.4 | 163.3 | refused |
| 04-B | rel. 1696 MB (driver 2008 < 3140) | 5847 | 2400 | 155.6 | 167.3 | refused |
| 05-B | rel. 1696 MB (driver 2065 < 3140) | 5847 | 2400 | 150.8 | 165.8 | refused |
| 06-W | rel. 960 of 1813 MB (driver 2170 < 3139) | 5879 | 2368 | 166.8 | 175.1 | refused |
| 07-N | kept 1813 MB (driver 2785 < 3141) | 6839 | 1408 | 115.7 | 127.5 | refused |
| 08-N | kept 1813 MB (driver 2259 < 3139) | 6839 | 1408 | 123.2 | 144.0 | refused |

- **W (the shortfall only)** keeps 789-1,397 MB more in the pool than B at the trim, but about
  800 MB of that is used again before the boundary: B grows the pool back during the prime's
  tail, the capture and burst 1, where it costs nothing visible. Only what is left reaches the
  publisher: 576 MB less growth in one cell of three, 32 MB in another, none in the third.
  The publisher's time does not follow (B 150.8-181.1 ms, W 153.4-167.5 ms): the same 2,400 MB
  costs 151-181 ms on unit 1 today and cost 261 ms on unit 2 this morning, because growing the
  pool on unified memory waits for the kernel to reclaim page cache. Growth is the stable measure,
  and W does not move it enough to show in the pause.
- **N (trim skipped at the same raw reading)**: the capture went ahead at a raw reading of 2,259
  and 2,785 MB against 3,139-3,141 MB required, with 1,813 MB left cached. It succeeded: no capture
  warning on the cold request, no out-of-memory line in either server log, server exit 0, the
  lowest MemAvailable 81 GB, the same text and bursts. Growth 1,408 MB, publisher 115.7 and
  123.2 ms. The capture's driver allocations (graph instantiation, cuBLAS workspaces) took the
  memory the raw reading did not count, as the publisher's own pool growth does.
- **The same reading refuses short requests' captures.** In 7 of the 8 cells the g1k request,
  sent right after the fill, had its draft-graph capture refused before any attempt ("effective
  free 2,331-2,879 MB (driver 1,924-2,471 MB + pool-cached 407 MB) < required 3,139 MB") and
  drafted eagerly for the whole request, with about 100 GB available: B 2 of 3, W 3 of 3, N 2 of 2.

So the raw reading is the wrong trigger on unified memory: it fires the trim and the refusal
where an allocation succeeds. The narrow trim stays in the change as the backstop for a reading
that is really short.

## 5. The record: main's guard against the allocatable reading (V)

Unit 1, probe arms, page cache filled after ready, interleaved, one boot per lease.
`refused`/`captured` is the g1k request's draft-graph capture (sent right after the fill).

**260k** (`gb10/qa260k-u1`, B V V B B V):

| cell | trim at the capture (MB) | cached at entry, MB | growth, MB | pub, ms | stall, ms | burst 1, ms | ms per round | TTFT, s | g1k capture |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 01-B | rel. 1696 MB (driver 2390 < 3140) | 5847 | 2400 | 157.5 | 165.7 | 2242.4 | 187.13 | 312.947 | refused |
| 02-V | none | 6838 | 1408 | 124.2 | 142.7 | 2244.3 | 187.26 | 313.133 | captured |
| 03-V | none | 6838 | 1408 | 115.9 | 123.2 | 2247.5 | 187.03 | 313.270 | captured |
| 04-B | rel. 1696 MB (driver 2265 < 3140) | 5847 | 2400 | 152.7 | 166.8 | 2246.8 | 187.22 | 314.235 | refused |
| 05-B | rel. 1696 MB (driver 2245 < 3140) | 5847 | 2400 | 148.8 | 161.9 | 2250.2 | 187.58 | 313.236 | refused |
| 06-V | none | 6838 | 1408 | 140.4 | 148.0 | 2247.2 | 187.53 | 313.450 | captured |

| 260k, median (spread) | B | V | V - B | every V cell under every B cell |
|---|---:|---:|---:|---|
| publisher on the scheduler thread, ms | 152.7 (8.7) | 124.2 (24.5) | -28.5 (-18.7%) | yes |
| first burst boundary over its rounds, ms | 165.7 (4.9) | 142.7 (24.8) | -23.0 (-13.9%) | yes |
| pool growth inside the publisher, MB | 2,400 (0) | 1,408 (0) | -992 | yes |
| burst 1, ms | 2,246.8 (7.8) | 2,247.2 (3.2) | +0.4 | ranges overlap |
| ms per round, every burst but the first and last | 187.22 (0.44) | 187.26 (0.49) | +0.04 | ranges overlap |
| TTFT, s | 313.236 (1.288) | 313.270 (0.317) | +0.034 | ranges overlap |

One text (5275df37e59f) and one burst structure in all six cells; the same published entry
(259,904 tokens, 8,357.4 MB). No out-of-memory line in any server log.

**131k** (`gb10/qa131k-u1`, B V V B):

| cell | trim at the capture (MB) | cached at entry, MB | growth, MB | pub, ms | stall, ms | burst 1, ms | ms per round | TTFT, s | g1k capture |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 01-B | none; the 131k capture itself refused (driver 1977 + pool 1069 < 3141) | 3629 | 544 | 54.1 | 69.5 | 1525.4 | 137.88 | 117.495 | refused |
| 02-V | none | 3626 | 544 | 57.8 | 87.5 | 1521.1 | 138.95 | 117.874 | captured |
| 03-V | none | 3626 | 544 | 56.7 | 51.3 | 1517.9 | 137.25 | 117.615 | captured |
| 04-B | rel. 896 MB (driver 2154 < 3076); 15,808 MB at boot calibration | 3404 | 768 | 74.5 | 79.9 | 1522.7 | 137.36 | 117.785 | refused |

Main's guard fired in both 131k cells, three ways. In 01-B it refused the 131k request's own
capture 2.3 s before the publisher (effective 3,046 MB: driver 1,977 + pool 1,069 < 3,141), so that
request drafted eagerly: 01-B compares eager drafting with V's graph drafting, and its 544 MB of
growth and 54.1 ms publisher are not the same path as V's. In 04-B it trimmed 896 MB before the
131k capture (growth 768 MB, publisher 74.5 ms) and, during boot calibration with the page cache
already full, 15,808 MB ("driver free 2310MB < required 3200MB"). The g1k capture was refused in
both main cells. V's guard fired in none of the four checks. One text (e3b0c1e81f9a) in all four
cells and one burst structure.

## 6. Truly short: MemAvailable itself under the requirement

`gb10/short-u1` (`tools/short_cell.sh`, `tools/mem_holder.py`): unit 1 under one lease per cell,
started only with no other GPU process, container or KV-store job on the unit. A fresh server
(its own memory guard lowered from 3 GiB to 1 GiB for these cells only), then `mem_holder.py`
allocates and touches anonymous memory until `MemAvailable` is about 2 GB and holds it. The
holder runs at `oom_score_adj` 1000, gives a chunk back when the server pushes `MemAvailable`
lower, exits by itself after 300 s, and is released before the server stops. No system setting
is touched. `kernel.log` is the kernel log of each cell's window.

| cell | when the memory went short | MemAvailable at the request's end | capture guard | request | kernel OOM lines |
|---|---|---:|---|---|---:|
| 01-B | before a g1k request | 2,162 MB | not reached: admission refuses the request (HTTP 400, "available 0MB") | refused | 0 |
| 02-W | before a g1k request | 2,143 MB | not reached: admission refuses the request (HTTP 400) | refused | 0 |
| 03-B | during a g28k prime | 2,016 MB | not tested: the holder needed about 30 s and the prime took 18 s | served | 0 |
| 05-B | during a g131k prime | 1,863 MB | refused: effective 2,170 MB (raw 1,100 + pool 1,070) < 3,140 | served eagerly, text e3b0c1e81f9a | 0 |
| 06-V | during a g131k prime | 1,943 MB | refused: effective 3,040 MB (allocatable 1,970 + pool 1,070) < 3,140 | served eagerly, text e3b0c1e81f9a | 0 |

When memory is really short the head's guard still refuses the capture. It reads `MemAvailable`
(1,970 MB) where main read the raw value (1,100 MB), and both are under the requirement. No arm
logged an out-of-memory line, every server exited 0, the kernel logged no OOM in any window, and
the 131k text is the one every other 131k cell served. The 03-B row is kept for completeness.

**The band the PR opens: raw < required <= `MemAvailable`** (`gb10/band-u1`, B V V B, review
S3; `tools/band_u1.sh`, rows by `tools/band_rows.py`). The same cell with the holder's target
at 3,500 MB: once the g131k prime starts, the holder keeps `MemAvailable` between about 3,250 and
3,750 MB (it takes or gives back 256 MB chunks), so at the cold request's capture the raw reading
is under the 3,140 MB requirement and `MemAvailable` is above it. Directory label W is `vi`.

| cell | MemAvailable at the capture | capture guard | lowest MemAvailable to the request's end | request | OOM lines (server / kernel) |
|---|---:|---|---:|---|---|
| 01-B | 3,429 MB | refused: effective 2,459 MB (driver 1,389 + pool 1,070) < 3,140 | 3,429 MB | served eagerly, e3b0c1e81f9a | 0 / 0 |
| 02-W (V) | 3,463 MB | went ahead, no trim; captured (no capture warning) | 3,428 MB | served, e3b0c1e81f9a | 0 / 0 |
| 03-W (V) | 3,470 MB | went ahead, no trim; captured (no capture warning) | 3,225 MB | served, e3b0c1e81f9a | 0 / 0 |
| 04-B | 3,476 MB | refused: effective 2,478 MB (driver 1,408 + pool 1,070) < 3,141 | 3,475 MB | served eagerly, e3b0c1e81f9a | 0 / 0 |

In the band main refuses the capture and the head takes it, and the capture succeeds: the guard
logs nothing on its `Go` path, and any failed capture, refused before the attempt or failed by
the driver, prints the `[spec] WARN: draft-graph capture failed` line, which neither V cell has.
Every server exited 0 and the kernel logged no OOM in any window. The rounds are the same on both
arms (137.7-138.6 ms per round, 36 rounds, 60 of 108 drafts accepted, TTFT 117.6-117.9 s):
at 131k the eager draft costs no measurable time per round. In every cell the holder gave back
one 256 MB chunk after the capture, on both arms alike, so a memory over-ask that only the kernel
OOM killer could settle is not what these cells test; the capture's appetite here is 2 x 34 MB
under a 3,072 MB floor. With `MemAvailable` this low the prefix cache refuses the 131k entry
for its byte budget (insert budget 3,888-4,156 MB, entry 4,290 MB) in all four cells, so there is no
publication to time.

## 7. RTX 5090 Laptop GPU: the discrete path

Binaries built for sm_120a on the CPU build host from the PR tree (`cpu/build5090*.log`):
main 46920874c0 against the narrow trim alone (`rtx28k`) and against the PR's code (`rtx28k-v`, `rtx28k-v2`; the PR's code before its rebase onto cdfdfd6006, the same `spec.rs`),
Qwen3.8-27B with its MTP draft at 32,768 context and a 28,006-token cold prompt
(`tools/rtx_cell.sh`; `rtx60k`: 65,536 context, 60,016 tokens). These binaries carry the narrow
trim on every device. The review moved it to unified memory only (`capture_headroom_plan`
returns `TrimAll`, main's trim to zero, on a discrete card), so at the PR's head a discrete card
reads, decides, trims and logs exactly as main; the table test pins both branches. The trim did
not fire in any cell below, so these cells ran main's path on both arms either way.

| campaign | cells (arm: ms per round, TTFT s) | trims | g1k capture |
|---|---|---|---|
| `rtx28k` (B W W B) | B 34.49, 34.99 / W 34.58, 34.51; TTFT 21.1-23.5 | none | captured in all |
| `rtx28k-v` (B W W B) | B 34.82, 35.07 / W 35.51, 35.24; TTFT 22.3-24.1 | none | captured in all |
| `rtx28k-v2` (W B B W W B) | W 34.53, 50.16, 50.32 / B 33.44, 32.78, 50.13; TTFT 21.5-79.8 | none | captured in all |
| `rtx60k` (B W W B) | B 43.66, 46.14 / W 43.27, 44.23 | none | refused in all four, identically |

Texts are equal within each campaign (d22600fc844a at 28k, 0608579006e1 at 60k). The rounds drift
with the card's state, not the arm: in `rtx28k-v2` cells 04 to 06 (W, W, B) all ran at 50 ms per
round on a laptop GPU capped at 150-160 W while other lanes worked on the rig, and 04-W's prime
chunks took 6.6 and 22 s against 0.7 and 2.7 s in the cell before it. At 60k the card is really
short (driver 1,458 MB + pool 1,156 MB < 3,143 MB) and both arms refuse the g1k capture with the
same line: the guard still does its job on a discrete card. A 100k cell at 131,072 context does
not fit this card (main step-OOMs and then refuses the request as over capacity; `rtx100k`), so
the trim path is not reachable here for this model.

## 8. Exactness

On the PR's `spec.rs` without the probe (`v`, built on unit 1 from the same tree), Qwen3.8-27B
with its MTP draft, GB10 (`gb10/gates/`):

- **Seeded-invariance** (`tools/seeded-invariance-gate.py --penalised device`): PASS, 54 rows,
  10 prompts, cases ref, K = 0..3 and demote (`inv-v.log`).
- **Burst-boundary** (`tools/spec-burst-boundary-gate.py`): PASS, 229 rows, grid default, 1:2, 3:2,
  3:3; 1,252 sampled bursts entered on a carried token, none drew at a burst's tail (`bnd-v.log`).
- **Served texts**: every cell of every campaign above serves one text per request size across
  its arms (5275df37e59f at 260k, e3b0c1e81f9a at 131k, 26ca98d4733a for g1k).

## 9. CPU checks

On the CPU build host (`cpu/`), sm_120a build, `--release --locked`:

All on the head's code (17f4e087ef; every file under `crates/` and `tools/` matched the PR's by
sha256):

- **The red/green tests.** `pre_capture_guard_plan_by_reading_and_device` (the table over the GB10
  readings and the discrete rows, plus the pin that the guard reads `allocatable_mem_info` and
  decides through `capture_headroom_plan`) passes at the head and fails on the same tree with the
  guard on the raw reading (`cpu/census-r2-raw.txt`).
  `pre_capture_trim_releases_the_shortfall_not_the_cache` passes at the head and fails with the
  unified trim back to zero ("the pre-capture trim does not keep the pool's blocks above the
  shortfall", `cpu/census-r2-zero.txt`). Before the review the same pair ran on the first commit
  (`cpu/census-narrow-trim.txt`) and on 6009545f71 (`cpu/census-final.txt`, whose guard test the
  table test replaced).
- **The local CI CPU chain** (`cpu/full-r2.txt`): `cargo fmt --check`,
  `cargo clippy --all-targets -D warnings`, the coalescer contracts, `memra-server` (1,223
  passed), `memra-engine --lib` (743 passed), `memra-gguf --lib` (417 passed; model files absent
  on that host, so this is not the skip census), `tools/portable-suites.sh`: all exit 0.

## Reproduce

Every cell: `tools/specpub_cell.sh CAMPAIGN NN-ARM ...` under one Spark lease
(`tools/unit_cell.sh`, `tools/ab_run.sh`), with `MID="bash box/fill_cache.sh 3000"` for a filled
page cache; tables: `tools/specpub_tab.py gb10/CAMPAIGN`. Probe arms: `tools/probe_i782.py` on
`crates/memra-server/src/worker.rs` at 782dc471b7, built by `tools/spark_build_probe.sh` and
`tools/spark_build_arms.sh`. Truly short: `tools/short_cell.sh`; the band: `tools/band_u1.sh`
(`SIZE=131072 short_cell.sh band-u1 NN-ARM BIN 3500` per cell), rows by `tools/band_rows.py`.
5090: `tools/rtx_ab.sh`.
Receipts were copied by `tools/receipts.py` (home paths written as `~`, units as `u1`/`u2`, files
over 64 KB gzipped).

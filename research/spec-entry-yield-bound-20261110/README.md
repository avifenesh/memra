# A speculating request stalls under sustained arrivals: the entry-yield bound (memra#1201)

Date: 2026-11-10. Qwen3.8-27B NVFP4 + Q5_K with its MTP head, context 8,192, eight sessions,
every switch at its default unless a row says otherwise. Pass/fail style rate cells: single
runs, no timing claim beyond the counts. Decision record:
[`docs/decisions/SPEC-ENTRY-YIELD-BOUND.md`](../../docs/decisions/SPEC-ENTRY-YIELD-BOUND.md).

**Verdict: fixed by the entry-yield bound, N=1** (`SPEC_ENTRY_YIELD_MAX`). On GB10 at the
default gate a long greedy stream emitted 1 or 2 tokens in 30 s from two closed-loop clients of
one-token requests with a think time of 200 ms or less. With the bound it advances in every cell
of the same sweep (50 to 139 tokens, longest gap 1.77 s), and the short requests' p50 rises by at
most 0.10 s. Texts are prefix-equal to main's on every row. It is a liveness guarantee, not a
share policy: under this pressure the stream runs at 2 to 5 tok/s, against about 21 tok/s when
the band serializes the newcomers (memra#1211). Rejected arms: a cost-model demotion after two
silent entries, and a flush of the carried token (both pay a target step every tick and do worse
on both counts). Deferred: a zero-rate sample in the cost model's realized slot (best for both
classes here, but a zero slot never recovers; memra#1211, memra#1212).

## The defect

A speculative burst ends early when a request waits for admission (the admission yield,
`MEMRA_ADMIT_YIELD`). The verdict is read at every commit hook, including the first, which runs
before round 0. A greedy burst ends on a pending token: emitted, not yet in the caches, carried
into the next burst's first verify. A burst that enters on that carried token has emitted nothing
at its first hook. If a request waits there, the burst runs no round and emits nothing. When the
next request is already waiting at the next burst's entry, and the one after that, the session
makes no progress for as long as the arrivals keep coming.

History, read from the tags' trees (`git log -S` stops at the 2026-09-01 catch-up sync):

- the greedy pending carry is in every release from v0.60.0;
- the admission yield's verdict at the burst's entry (`keep_going = flush_commit(...)` before
  `while keep_going && ...`) from v0.71.0;
- the calibrated spec cost model, which supplies the default-configuration trigger, from
  v0.139.0.

## Cell

`tools/stall_cell.py`, one server boot per call and several cells in it. In each cell a long
greedy streamed request (a 34-token prompt, max_tokens 3000) runs alone for 8 s, so it has run
rounds and holds a carried token. Then N closed-loop clients send one-token chat requests for
30 s, each waiting the think time after every answer. Then the long request runs alone for 12 s
and the cell hangs up on it. Per cell: the long request's tokens before, in and after the
window, the longest gap between its tokens inside the window, the short requests served and
their latency, and from the server log (`MEMRA_SPEC_SETUP_TRACE=1`) the long request's
continuation bursts in the window, how many ran no round and how many of those emitted nothing
("silent"). The long text is written out. The stream is cut at a time, not a token count, so
greedy texts are compared by prefix (`tools/summarize.py --ref`). Every greedy row in this
directory is prefix-equal to main's text of the same cell.

## Main, the think-time sweep (GB10)

`gb10/sweep-main/`: one DGX Spark (GB10), main 2628ece940 (`memra-0.140.0-d1b6c9994da6`), one
boot, twelve cells, greedy, default gate.

| clients | think ms | long tokens in window | longest gap s | silent / continuation bursts | short requests | short p50 s | short p95 s |
|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | 1000 | 630 | 0.31 | 0 / 24 | 24 | 0.297 | 0.305 |
| 1 | 200 | 431 | 0.32 | 0 / 61 | 61 | 0.306 | 0.309 |
| 1 | 50 | 230 | 0.32 | 0 / 98 | 98 | 0.258 | 0.261 |
| 1 | 0 | 230 | 0.32 | 0 / 98 | 98 | 0.308 | 0.312 |
| 2 | 1000 | 501 | 0.53 | 6 / 48 | 49 | 0.220 | 0.312 |
| 2 | 200 | **1** | **30.03** | 141 / 142 | 143 | 0.219 | 0.224 |
| 2 | 50 | **1** | **30.23** | 142 / 143 | 144 | 0.369 | 0.376 |
| 2 | 0 | **1** | **30.28** | 72 / 73 | 145 | 0.417 | 0.426 |
| 4 | 1000 | 228 | 0.95 | 70 / 95 | 98 | 0.237 | 0.243 |
| 4 | 200 | **1** | **30.48** | 48 / 49 | 146 | 0.633 | 0.648 |
| 4 | 50 | **1** | **30.61** | 48 / 49 | 147 | 0.779 | 0.795 |
| 4 | 0 | **2** | **30.62** | 36 / 37 | 147 | 0.833 | 0.844 |

- One client never stalls the long request: its next short request arrives only after the
  previous one is answered, so none is waiting when the long request's burst begins.
- From two clients up, with a think time of 200 ms or less, the long request emits 1 or 2 tokens
  in 30 s. It recovers when the arrivals stop (287 to 306 tokens in the 12 s after).
- No demotion in the boot. At every concurrency the cost model prices speculation above plain:
  n=2 42.8 against 25.6 tok/s, n=3 57.4 against 36.7, n=4 77.9 against 46.8, n=5 84.4 against
  55.9.
- Every short request is admitted to the speculative path (1,312 `[spec-k] ... source=cost-model`
  lines). The admission yield steps sessions that have emitted nothing first, so their first
  bursts run ahead of the long request. They finish, their clients send again at once, and a
  request is waiting when the long request's burst begins.

Two more readings of the logs. The realized-rate sampler takes a sample only between two ticks
in the same steady regime that emitted at least one token, so under these arrivals no sample
lands and the boot-table prediction stands. The acceptance window the verdict reads is
refreshed only by bursts that ran a round, so during the stall the verdict keeps a stale
history.

## RTX 5090, main (default gate)

`rtx5090/main/`: one RTX 5090 Laptop GPU, main 2628ece940 (sm_120a build, same content id),
one boot.

| clients | think ms | long tokens in window | longest gap s | silent bursts | short requests | short p50 s |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0 | 1197 | 0.29 | 0 | 68 | 0.888 |
| 2 | 200 | 1262 | 0.29 | 0 | 56 | 0.868 |
| 4 | 0 | 175 (demoted at 5 active >= HIGH=4) | 0.30 | 0 | 344 | 0.350 |

The 27B's boot calibration is refused on this card (memra#1191), so the LOW/HIGH band decides,
and it admits every short request to plain decode (468 `K=0 source=concurrency`). Those
requests prefill after the long request's burst, so the long request runs at full speed and the
short requests wait out its burst. At four clients the band demotes. No stall in the default
configuration on this card.

## Workarounds on releases before the fix (GB10, main)

`gb10/workarounds/`, one boot each.

| setting | clients | think ms | long tokens in window | longest gap s | short requests | short p50 s | short p95 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `MEMRA_ADMIT_YIELD=0` | 2 | 200 | 690 | 0.51 | 21 | 2.792 | 3.226 |
| `MEMRA_ADMIT_YIELD=0` | 2 | 0 | 690 | 0.51 | 21 | 2.991 | 3.306 |
| `MEMRA_ADMIT_YIELD=0` | 4 | 0 | 559 | 0.94 | 47 | 2.321 | 3.620 |
| `MEMRA_SPEC_GATE_MODEL=0` | 2 | 0 | 678 | 0.32 | 38 | 1.611 | 1.916 |
| `MEMRA_SPEC_GATE_MODEL=0` | 2 | 200 | 696 | 0.33 | 34 | 1.616 | 2.006 |
| `MEMRA_SPEC_GATE_MODEL=0` | 4 | 0 | 171 (demoted) | 0.28 | 340 | 0.356 | 0.362 |

Both remove the stall. The band (`MEMRA_SPEC_GATE_MODEL=0`) is the cheaper one: the short
requests wait for the long request's burst (1.6 s at two clients) instead of 2.3 to 3.0 s, and at
four clients both classes do better than on main. Its standing cost is the cost model's own
choices: at two concurrent agentic sessions the band's aggregate is about 18% lower
(`research/batch-spec-20261101/GB10-AGENTIC.md`, 23.5 against 28.8 tok/s).

## The arms (GB10, one probe binary)

`tools/probe-arms.patch` is the probe on main + memra#1196's fix: one binary whose arms switch
per cell through a knob file (`PROBE1201_KNOBS`, re-read every 200 ms), so one boot runs every
arm. `N=<n>` is the bound; `STARVED=2` demotes a session, when the cost model owns the gate,
after two silent entries in a row (the session's realized rate over them is zero); `FLUSH=1`
has no bound and flushes the carried token (one target step) at a yielding entry so the entry
emits the next token; `SLOT=1` puts a zero-rate sample into the cost model's realized slot at
the tick's concurrency on a session's second and later silent entry in a row; `BAND=1` hands
the gate to the band. Rows: `gb10/arms/stall.jsonl`; boots `probe4` (four clients, ABBA, two
runs per arm), `probelo` (two clients) and `probeslot` (N=0, the band and the slot arm; the
slot arm writes the cost model's slots, so it ran last).

Long-stream tokens in the 30 s window / longest gap s / short requests' mean latency s:

| arm | 4 clients, 0 ms | 2 clients, 0 ms | 2 clients, 200 ms |
|---|---|---|---|
| off (main + memra#1196) | 1, 1 / 30.7 / 0.83, 0.84 | 1 / 30.3 / 0.42 | 1 / 30.1 / 0.22 |
| N=0 | 83 / 0.97 / 0.84 | 143 / 0.52 / 0.51 | 228 / 0.51 / 0.42 |
| **N=1** | **58, 53 / 1.77 / 0.74, 0.75** | **74 / 0.94 / 0.47** | **142 / 0.73 / 0.32** |
| N=2 | 31, 31 / 2.61 / 0.81 | 50 / 1.36 / 0.45 | 93 / 0.93 / 0.29 |
| N=4 | 41, 38 / 4.27 / 0.70 | 26 / 2.20 / 0.44 | 57 / 1.35 / 0.26 |
| STARVED=2 | 33, 33 / 2.56 / 0.89 | 62 / 1.33 / 0.49 | 106 / 0.91 / 0.36 |
| FLUSH=1 | 35, 34 / 0.92 / 0.90 | 64 / 0.50 / 0.49 | 106 / 0.49 / 0.37 |
| N=2 + STARVED=2 | 34, 33 / 2.55 / 0.89 | | |
| BAND=1 | 169 / 0.28 / 0.36 | 645 / 0.32 / 1.67 | 654 / 0.33 / 1.68 |
| SLOT=1 | 150 / 3.59 / 0.39 | 189 / 1.88 / 0.31 | 267 / 0.22 / 0.25 |

Short p50 at four clients: off 0.83 to 0.85, N=0 0.93, N=1 0.83, N=2 0.83, N=4 0.83, STARVED
0.90, FLUSH 0.91, BAND 0.36, SLOT 0.36. Short p95 at four clients: off 0.84 to 0.87, every N
0.93.

- At four clients every arm that runs rounds ends in a demotion: once rounds run, the 30 s
  acceptance window ages below the 64 rounds a verdict needs, and at n=5 the verdict falls back
  to the band (`n * (rung + 1) > 16` rules out exploration), which demotes at 5 >= HIGH. The
  moment it happens varies between arms, so the four-client rows mix speculative and plain
  ticks. At two clients nothing but STARVED and SLOT demotes.
- The trade (`tools/tradeoff.py`, two-client cells, long tokens gained over off per second of
  added short-request latency): N=0 11.9 and 12.4, N=1 11.7 and 12.5, N=2 11.1 and 11.8, N=4
  10.0 and 11.6; STARVED 7.2 and 7.1; FLUSH 6.7 and 6.7; BAND 14.3 and 14.0. The bound sits on
  one line whatever N is; N sets the split. STARVED and FLUSH pay a target step every tick, which
  is a worse rate on both counts.
- SLOT does best for both classes here: the zero samples drive the verdict at that concurrency
  to K=0 (`predicted 0.0 vs plain 57.6 tok/s`), so the long session demotes and later newcomers
  are admitted to plain decode. It does not ship as built. A speculative slot is refreshed only
  by speculative ticks at its own concurrency, so a zero slot is never measured again, and that
  concurrency is served plain until restart. The two-client 200 ms cell shows it: its slot was
  already zero from the cell before, and the session demoted at its first tick. Neighbors also
  inherit the zero through `priced()` (memra#1212 on how it picks between equal neighbors).
  memra#1211 takes the slot price with a decay or time-to-live and its own normal-concurrency
  proof.

## The choice: the bound at N=1

The bound is the fix. A session yields at a burst's entry with nothing emitted at most once in a
row; at its next such entry it runs one round, and that round's hook yields. It acts only after
a silent entry yield, whoever owns the gate and on either sampling class. N=1 rather than N=0
or N=2: the trade is the same line, and N=1 keeps the longest gap under 1.8 s in every measured
cell (2.6 s at N=2, 4.3 s at N=4) while a waiting request pays at most one round every second
burst. N=0 doubles that cost (short mean +23% and +87% at two clients, against +12% and +43%) and
takes the newcomer-first ordering away at every carried entry.

What it guarantees and what it does not: no outright stall, and a longest gap under about 1.8 s
in these cells. It is a liveness guarantee, not a throughput policy. Under this pressure the
long stream still runs at about 2 to 5 tok/s (74 and 142 tokens in 30 s at two clients), against
about 21 tok/s when the band serializes the newcomers behind its bursts. The share between the
two classes is memra#1211.

## The fix, the same sweep (GB10)

`gb10/sweep-fix/` and `gb10/sweep-main141/`: main v0.141.0 (3de59bbf80, `memra-0.141.0-d792eacd8d45`)
against the fix (7c9a0244b8, `memra-0.141.0-dd78f75bcc5a`), one boot each, built on the same
unit, run on the two GB10 units in the same window. Main v0.141.0 reproduces 2628ece940's sweep
cell for cell.

| clients | think ms | long tokens in 30 s, main -> fix | longest gap s | short requests | short p50 s | short p95 s | fix demoted |
|---:|---:|---|---|---|---|---|---|
| 1 | 1000 | 631 -> **630** | 0.31 -> 0.31 | 24 -> 24 | 0.301 -> 0.302 | 0.304 -> 0.308 | no |
| 1 | 200 | 427 -> **424** | 0.32 -> 0.31 | 60 -> 62 | 0.307 -> 0.306 | 0.312 -> 0.31 | no |
| 1 | 50 | 229 -> **229** | 0.32 -> 0.32 | 97 -> 97 | 0.259 -> 0.26 | 0.262 -> 0.264 | no |
| 1 | 0 | 229 -> **230** | 0.32 -> 0.32 | 97 -> 98 | 0.31 -> 0.31 | 0.312 -> 0.312 | no |
| 2 | 1000 | 491 -> **488** | 0.53 -> 0.52 | 50 -> 50 | 0.22 -> 0.219 | 0.31 -> 0.311 | no |
| 2 | 200 | 1 -> **139** | 30.04 -> 0.73 | 143 -> 116 | 0.219 -> 0.321 | 0.225 -> 0.33 | no |
| 2 | 50 | 1 -> **94** | 30.21 -> 0.73 | 144 -> 125 | 0.369 -> 0.466 | 0.376 -> 0.474 | no |
| 2 | 0 | 1 -> **74** | 30.21 -> 0.94 | 145 -> 130 | 0.416 -> 0.439 | 0.421 -> 0.519 | no |
| 4 | 1000 | 228 -> **228** | 0.95 -> 0.95 | 98 -> 98 | 0.238 -> 0.24 | 0.254 -> 0.344 | no |
| 4 | 200 | 1 -> **50** | 30.49 -> 1.56 | 146 -> 136 | 0.633 -> 0.729 | 0.645 -> 0.743 | no |
| 4 | 50 | 1 -> **59** | 30.66 -> 1.57 | 147 -> 159 | 0.782 -> 0.781 | 0.796 -> 0.893 | yes |
| 4 | 0 | 1 -> **56** | 30.66 -> 1.77 | 147 -> 160 | 0.833 -> 0.835 | 0.844 -> 0.936 | yes |

- Every cell advances. The longest gap is 1.77 s (four clients, 0 ms), against 30.0 to 30.7 s on
  main. The twelve long texts are prefix-equal to main's.
- One client is unchanged: no burst entry meets a waiting request, so the bound never acts. At
  1000 ms the long request's tokens are level (488 against 491 at two clients, 228 against 228 at
  four). The bound acts there too (silent bursts 23 -> 2 and 70 -> 47), but the stream was not
  starved to begin with.
- What the short requests pay where the bound fires: at two clients p50 +0.10 s at 200 ms, +0.10
  s at 50 ms, +0.02 s at 0 ms; at four clients p50 +0.10 s at 200 ms and level at 50 and 0 ms,
  p95 +0.09 to +0.10 s. At four clients 50 and 0 ms the default gate demotes the long request
  part way through the window (the acceptance window ages out, see the arms), after which the
  short requests run faster than on main.

## Exactness

The bound moves where a burst ends, never what a round computes, and where a burst ends is
content-neutral. Evidence:

- every greedy long text above (main sweep, arms, fix sweep, workarounds) is prefix-equal to
  main's text of the same cell;
- the seeded-invariance gate and the burst-boundary gate on the fix binary: in progress, rows to
  `gb10/gates/` with the verdicts here;
- the session gate is an engine binary and the engine is unchanged by this fix (the diff touches
  `crates/memra-server/src/worker.rs` only).

## Normal concurrency

In progress: the c = 1, 2, 4, 8 closed-loop cell (`tools/normal_cell.py`) on main v0.141.0 and on
the fix, and on the probe at N=1 to count how often the bound acts there.

## Binaries

Built on one GB10 unit from clean trees: main 3de59bbf80 `memra-0.141.0-d792eacd8d45` (sha256
8bbab751...), the fix `memra-0.141.0-dd78f75bcc5a` (sha256 0701bbdc...; its `crates/` tree is the
fix commit's), the probe `memra-0.140.0-499bcd947707` (sha256 0a9cd2b7...; main 2628ece940 + memra#1196's fix + `tools/probe-arms.patch`; the first build of the probe, 5ec3fffa..., was refused at boot by the env audit and ran no cell).
The RTX 5090 main build is sm_120a with content id `memra-0.140.0-d1b6c9994da6`, the same as the
GB10 build of 2628ece940.

## Files

- `tools/stall_cell.py`: the cell. `tools/normal_cell.py`: the normal-concurrency cell.
  `tools/summarize.py`, `tools/tradeoff.py`: the tables. `tools/probe-arms.patch`: the probe.
- `gb10/sweep-main/`: main's sweep. `rtx5090/main/`: the RTX 5090 on main. `gb10/workarounds/`:
  `MEMRA_ADMIT_YIELD=0` and `MEMRA_SPEC_GATE_MODEL=0`. `gb10/arms/`: the probe arms.
- Each directory holds `stall.jsonl` (rows), `guard.log` (binary hash, device, temperature and
  load at boot), the server logs, and `texts/` (the long request's text per cell).

# A DFlash request is never admitted with its draft plane uncharged (memra#1173)

Verdict: fixed. On main the VRAM gate admitted a DFlash request whose full charge did not fit
on the charge without its drafter KV plane, and the plane was still allocated: 1,197 MB
uncharged for a drafter with no sliding layer at a 29,110-token request, 169 MB for the
DFlash2 drafter. A step-OOM replay of a DFlash session was priced plain and primed a drafter
anyway. The gate now refuses such a request its drafter and serves it plain for its life, and
`admit` builds a drafter session only for a request whose plane the gate charged. Decision
record: [`docs/decisions/DRAFTER-ADMISSION-REFUSAL.md`](../../docs/decisions/DRAFTER-ADMISSION-REFUSAL.md).

Card: one RTX 5090 Laptop GPU (24 GB). The enforced power limit read 150 or 160 W (each run's
`gate.log` or `guard.log` records it) and was not changed. Model: Qwen3.8-27B NVFP4 + Q5_K
with its MTP head. Drafters: the q38 DSpark agentic export (no sliding layer, called
"windowless" below) and the q38 DFlash2 export (sliding window). Every gate run is a single
run; cell tables show each boot.

## Binaries

| label | tree | binary sha256 (first 16) | version string |
|---|---|---|---|
| main | `46920874c0` | `de1e84621861a15f` | `memra-0.141.1-24e7e0144192` |
| change | this PR's code on `46920874c0` (`cbb61aef13`; `2ae03f732a` changes only the gate tool) | `71a7b8173fcbd94e` | `memra-0.141.1-089bd6b58f1a` |
| main, before the rebase | `340a5e5bfa` | `0f98341aa6660d9d` | `memra-0.141.1-5b4e956cdb89` |
| change, before the rebase | this PR's code on `340a5e5bfa` (`9dd7d190ab`) | `377bb2745e0f0845` | `memra-0.141.1-3f56310c5e4b` |

The PR head is the same lane patches rebased onto `cdfdfd6006` (the landing of memra#1227),
`git range-diff` identical for all five code and tool commits. That landing changes the tick's
phase (a) order and `spec_burst_pre`/`spec_burst_post` in `worker.rs`; none of its hunks is in
or next to the admission gate, `admit`, `park_requeue` or the verdict functions, so the device
runs were not repeated on it. The CPU checks ran again on that tree (`cpu/head-cdfdfd6006/`).

The rebase onto `46920874c0` brought main's byte-based reclaim settle and pending-tail booking,
which feed the headroom the gate reads; the verdict code did not change. The gates and the
agentic c = 2 replay ran on the rebased pair. The normal-concurrency cells, the plain c = 1
boots and the agentic c = 4 replay ran on the pair before the rebase and are labeled so.

Earlier development runs (base `1d3be5b0df`, this change's first commits, and the gate tool
before its review rework) are superseded and not kept here; nothing in this directory or the
record cites them.

## What is here

| path | what |
|---|---|
| `GATES.md` | per-step tables and verdicts of every `tools/drafter-admission-gate.py` run |
| `gates/<run>/` | `summary.json`, `steps.jsonl`, `VERDICTS.txt`, the three server logs (measured, plain reference, replay; gzipped), the ballast log and the wrapper's `gate.log` (binary sha, card, power limit) |
| `CELLS.md` | the normal-concurrency cells and the plain c = 1 boots, one row per boot and c |
| `cells/<boot>/`, `cells-plain/<boot>/` | `normal.jsonl` (per request: tokens, TTFT, inter-token gaps, text sha), the server log, 1 Hz telemetry, `guard.log` |
| `AGENTIC.md` | the agentic replay at c = 4 and c = 2, one row per boot |
| `agentic-c4/<boot>/`, `agentic-c2/<boot>/` | `cells.log`, per-request rows (no prompt or generated text), `/metrics`, the server log, 1 Hz telemetry, `guard.log` |
| `cpu/` | the CPU red and green arms (`summary.txt`, each arm's patch and log), the builds and the full CPU checks |
| `harness/` | the wrappers that ran the gates, cells, replays and CPU arms, and the table scripts |

## Gate runs

`GATES.md` has every step. `tools/drafter-admission-gate.py` at this PR's head; the wrapper's
exit code is the `gate <run> rc=` line at the end of each `gate.log`.

| run | binary | route, drafter | exit | result |
|---|---|---|---|---|
| `agentic-main` | main | DFlash, windowless (plane 1,197 MB) | 1 | V2 V3 V4 V7 V8 FAIL: 3 `eager-arm admit ... 1197MB uncharged`; both busy band steps HTTP 400; the idle band step not placeable (the driver could not give its ballast); the replay speculated on a plain charge; idle window steps left 2,120 and 2,122 MB of effective free against a 2,156 MB reserve. No OOM |
| `agentic-fix` | change | same | 0 | PASS V1 to V8: refusals idle-fits 2, idle-short 1, busy-fits 1, busy-unfundable 1, all served plain; the replay plain |
| `dflash2-main` | main | DFlash2 (plane 169 MB) | 1 | V2 V3 V4 V7 V8 FAIL: 3 `eager-arm admit ... 169MB uncharged`; the idle band step and both busy band steps HTTP 400; the replay speculated on a plain charge. No OOM |
| `dflash2-fix` | change | same | 0 | PASS V1 to V8: refusals idle-fits 2, idle-short 2, busy-fits 1, busy-unfundable 1, all served plain; the replay plain |
| `mtp-fix` | change | MTP (draft state 39 MB) | 0 | PASS V1, M2, M3, V5, V6: the eager arm twice, each capture refused pre-attempt, same text; peak pool use is information only on this route |

No MTP run on main: this change does not touch the MTP route's verdict (the table test pins it).

## Cells

`CELLS.md`, before the rebase (main `340a5e5bfa` against this change on it). DFlash2 drafter,
closed-loop c = 1, 2, 4, 8, 6 requests per client, greedy chat, 160 new tokens, streamed; boots
main, change, change, main. No request took either arm or failed, and the four boots wrote the
same text per prompt at every c. tok/s main | change: c1 70.23, 70.63 | 70.65, 70.60; c2 74.62,
77.75 | 77.82, 77.79; c4 116.09, 120.77 | 120.98, 119.13; c8 140.65, 146.19 | 148.90, 147.64.

`cells-plain/`: the c = 1 cell with `MEMRA_SERVE_SPEC=0` on the change's binary, two boots:
38.23 and 38.16 tok/s, the decode rate of a request refused its drafter on this shape (54% of
the drafter's 70.6).

## Agentic replay

`AGENTIC.md`. The 2026-11-05 decision cell's config: 29 agentic turns of 1.4k to 17k prompt
tokens, DFlash2, ctx 32,768, 8 sessions, prefix cache 4,000 MB, T = 1 seed 1234, 768 tokens;
boots main, change, change, main.

- c = 4 (`agentic-c4/`, before the rebase): every request priced plain (the wave is above the
  spec gate's low mark), so neither arm is reachable. 29/29 per boot, 10,293 tokens each; main
  44.56, 46.37, change 46.34, 46.90 tok/s.
- c = 2 (`agentic-c2/`, the rebased pair): every request priced `path=spec` with its plane
  charged; 0 refusals, 0 eager-arm lines, 0 VRAM defer lines on both binaries. tok/s main 35.27,
  change 34.41, change 37.28, main 38.02. One step OOM in `main-133251` and one in
  `fix-134632`, at the same point, with both sessions charged in full (not this change's
  subject; memra#1255). The replay after it: main priced it plain and decoded it with the drafter; the change
  served it plain (`[dspark] declined: admission priced this request plain`). That boot's
  sampled replay wrote its plain stream, so it totals 9,765 tokens against 10,818 in the other
  three.

## CPU

A CPU host, the tree on main `46920874c0` (the gates' change binary was built from it). `cpu/summary.txt`: the two
memra#1173 tests (`the_vram_gate_never_admits_a_dflash_request_without_its_draft_plane`,
`the_drafter_refusal_is_wired_from_the_gate_to_every_dflash_route`) pass on the tree, and each
of nine red arms fails one of them:

| arm | what it removes |
|---|---|
| `red-arm` | the DFlash route's own second arm (the request takes the MTP eager arm) |
| `red-busy` | the `busy-fits` refusal (the request defers instead) |
| `red-unfundable` | the `busy-unfundable` refusal (the request defers instead) |
| `red-stale` | the drained re-read (an idle card refuses on its first read) |
| `red-idle` | the `idle-short` refusal (the idle card rejects the request) |
| `red-route` | the funded term in `admit`'s DFlash route |
| `red-funded` | the gate's write of `drafter_funded` (every request counts as funded) |
| `red-price` | the plain pricing of a refused request |
| `red-requeue` | the refusal's mark on the request |

`cargo fmt --check`, `cargo clippy --all-targets -D warnings`, the memra-server suite (1,212
passed, 0 failed, 14 ignored, plus 7 and 3 integration tests), the flags census and the
numeric-scope check pass (`cpu/chainB.log`, `cpu/check-*.log`).

On the PR head's tree (main `cdfdfd6006`): fmt, clippy (`check-clippy2.log` reran it with the
server sources touched), the memra-server suite (1,215 passed, 0 failed, 14 ignored, the two
memra#1173 tests among them), the flags census and the numeric-scope check pass
(`cpu/head-cdfdfd6006/`). The red arms were not rerun there: the lane's patches are identical.

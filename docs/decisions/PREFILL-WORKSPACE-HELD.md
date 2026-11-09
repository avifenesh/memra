# Admission charges the prefill workspace a prime holds, on the RTX 5090 Laptop GPU (2026-11-08)

Date: 2026-11-08. Branch `c8r2/prefill-held`, receipts `research/c8-workspace-20261108/`. The
follow-up the c8 record named ([`C8-SERVING-24GB.md`](C8-SERVING-24GB.md)). Card: one RTX 5090
Laptop GPU (24 GB), Qwen3.8-27B NVFP4 + Q5_K with its MTP head, `MEMRA_CTX=32768`.

## Problem

Admission charged every request a prefill workspace of the widest prime call (4096 rows) on an
empty prime slab, plus hiddens for the whole prompt (`PrimeWorkspaceShape::admission_bytes`). At
agentic c8 that was 66-80% of each request's charge: 2,229 MB of 3,101 MB at a 17,166-token prompt
on main's seven-plane formula, 2,147 MB on this tree's six-plane one. The VRAM gate deferred such a
request while 3.3-5.0 GB was free, against a need of 5,048 MB (the charge plus the 1,974 MB
transient reserve).

A probe build of main `2288c4bf82` (`research/c8-workspace-20261108/harness/ws_probe_patch.py`,
never landed) logged the pool's peak growth (`CU_MEMPOOL_ATTR_USED_MEM_HIGH`, reset at the call's
start) around every outermost prime call, in three boots: agentic c8, the agentic prime pass at c1,
and the mixed shape.

| prime call | calls | peak growth during the call |
|---|---|---|
| plain 1024-row tick beside a 4096-row slab (agentic c8) | 306 | 292-297 MB (290 KB per row) |
| 4096-row MTP walker chunk beside the slab (agentic prime pass, c1) | 24 | median 1,170 MB, max 1,187 MB |
| concat batch prime of 2-6 fresh prompts, at most 512 rows (mixed) | 42 | median 67 MB, max 171 MB (444 KB per row) |
| the boot calibration's 4096-row MTP chunk, slab 2048 -> 4096 rows | 1 per boot | 1,942 MB, 763 MB of it the new slab |

- The retained per-device prime slab grew to 4096 rows (1,527 MB on main, 1,443 MB here) during
  the boot calibration and never grew while serving: 2 of the 434 c8 calls, 2 of the 90 c1 calls
  and 2 of the 63 mixed calls grew it, all at boot.
- A plain call returned only the hiddens of its own rows (21 MB per 1024 rows), freed by its
  caller. No plain call held a prompt-long stack; only the MTP walker allocates one.
- The concat batch prime does not use the slab. Its planes are its own, about 444 KB per row of
  the batch, the sum of its members' takes.

## Chosen

On the RTX 5090 Laptop GPU (`prefill_held_for_device`, boot line `[admission] prefill
workspace: held|cold`), a single-device request whose prime runs `prime_layers` on the retained
slab, at the default 4096-row chunks the boot calibration probes, is charged what its prime
allocates beside the resident slab (`PrimeWorkspaceShape::held_bytes`, `admission_prime_hold`):

- **Plain route.** One take of the scheduler's per-tick budget (1024 rows beside a peer, the 8192-
  row solo widening when no session shares the worker, an explicit `MEMRA_PREFILL_TICK` always)
  with that take's hiddens. The take's slab part is still charged, because a take of the
  configured budget can join a concat batch prime, whose planes are its own.
- **MTP and DFlash walkers** (`estimate_spec`): the 4096-row chunk's per-call extras and the
  whole-prompt stack. The resident slab covers the chunk's slab part. The calibrated transient
  reserve measures that route's 4096-token prime at boot, the DFlash walker's per-chunk tap
  buffers included.
- **A slab smaller than the prompt's widest call** is charged as the whole new slab, because
  `prime_slabs_get` allocates it before it drops the old one.
- **A concat batch**, interactive or dark-lane, stops adding takes where its planes (the model's
  `call_row_bytes` plus `prompt_row_bytes`, 458 KB per row on this model, times the batch's rows)
  would pass effective free memory less the transient reserve floor (`concat_takes_within`), since
  no admission charge books a batch's members together. Effective free counts pool-cached bytes,
  and a batch's widest plane (about 434 MB at 6,234 rows) is one block; the reserve kept above the
  batch is the margin for that.
- **The prefix cache gives memory back before every prime call** on the keyed card. Seeds
  publish after prime calls, and with the held charge the free memory they take is the room the
  next call needs. In one run of the version without this (`runs/c8-v2/06-B`), three seeds of
  1,538 MB in one prefill phase took driver free to 52 MB and the next prefill call failed out of
  memory, ending three streams; a second time in the measured pass ended one. The c8 record's
  yield (`prefix_yield_transient_reserve`) ran only before the decode step. Now:
  - before phase (a), the whole reserve after any insert, as before the decode step: phase (a)
    runs the MTP and DFlash walkers, the spec bursts and the fanout leader primes after what the
    tick top and the previous tick published;
  - right before each prime call, that call's need (`yield_prefix_for_prime`): inside
    `prefill_tick` at the take it computed, and before every fanout leader prime at its prefix.
    The need is the cold charge of the call's rows (`prime_call_need`): 470 MB at a 1024-row
    take, where the probe measured at most 297 MB, and 1.96 GB at the 8192-row solo widening,
    whose 4096-row calls peak near 1.2 GB. The read takes no fence unless it is short.
- **The deadline order's estimate, on every card** (`deadline_prefill_rows`). With the held charge
  a burst of cold prompts is admitted at once instead of queueing. The deadline order read that as
  no backlog and estimated the round robin as if finishing peers left, which the c8 record had
  measured 2x short. The prime pass's slowest first token sat at 85-89 s in nine of nine runs
  against the 90 s deadline: a 17,101-token prompt was admitted 0.05 s after it arrived, then
  shared prefill eight ways.
  - Without waiting requests the estimate now keeps the tick's current takes for every take the
    row still needs. Its first token comes in the tick of its last take.
  - In a closed loop a finishing peer is replaced, and one that leaves only shortens the tick.
  - On the c8 cell's sharing (one run) it gives 69-87 s against the 69 s measured.
  - With requests waiting nothing changes.
  - A rescue whose fill would take every row holds nobody back and keeps the round robin.
- **Unchanged:** the reserve is sized from the cold charge as before. Every other card, unified
  memory, multi-device placements, the gemma and Step35 programs, monolithic primes and chunks
  wider than 4096 or set explicitly (`MEMRA_PRIME_CHUNK`) keep the cold charge, byte for byte. A
  held charge is never above the cold one.

The held charge at the 17,166-token prompt is 476 MB beside a peer (2,147 MB cold).

## Measured

These are interleaved fresh-boot A/B runs: A = main `0c25263389`, B = this change on it (`research/c8-workspace-20261108/RESULTS.md`, N=3 per arm unless stated):

| shape | A (main) | B (this change) |
|---|---|---|
| agentic c8, measured pass: first token p50 / p99 | 22.0 / 64.7 s | 14.5 / 39.1 s |
| agentic c8, measured pass: tok/s (median run) / E2E p50 | 47.76 / 53.7 s | 49.52 / 52.7 s |
| agentic c8, prime pass: first token p50 / slowest per run | 31.0 s / 66.0-75.4 s | 39.3 s / 60.0-62.6 s |
| agentic c8: failures, OOM lines, lowest driver free | 0, 0, 947 MiB | 0, 0, 307 MiB |
| greedy c8 (N=1): byte-identical outputs | | 29 + 29 of 29 + 29 |
| mixed shape: tok/s (median run), TTFT p50, identical rows | 120.64, 953 ms | 124.58, 932 ms, 103 of 103 |
| agentic c4, measured pass: first token p50 / p99, tok/s per run | 3.1 / 19.9 s, 47.6-50.2 | 3.1 / 19.8 s, 48.1-50.1 |
| agentic c1 (N=1), measured pass: first token p50, tok/s | 2.4 s, 33.84 | 2.3 s, 34.53 |
| DFlash route, grow-rewind replay (ABBA): c1 second turns resumed; c4 tok/s; c8 HTTP 408s | 3 of 11; 35.52 / 33.85; 3 and 2 | 11 of 11; 45.71 / 45.44; 0 and 0 |

- **Measured pass.** The first token comes 34% sooner at the median and 40% sooner at p99. Aggregate decode throughput rises 3.7%. Per-token decode is slower (TPOT p50 135 against 96 ms), because more sessions decode together. Whole requests end at the same median.
- **Prime pass.** This pass is a cold burst of 29 prompts. Its slowest first token falls in every run. Its median rises 27%: in a burst, the deadline order is close to oldest first.
- **c4 and c1.** Both shapes are flat. At c4 the cold charge never deferred, so the held charge has nothing to admit sooner.
- **The DFlash route** (the grow-rewind replay, c1 to c8 in each boot). No OOM line and no step-OOM park in any arm. On B the lowest driver-free readings the server logged were 100 and 132 MiB, against 452 and 484 MiB on A. Both were the before side of an admission trim or a reserve settle. The pool held cached blocks, and the trim or settle returned them before the prime ran: after a trim, the lowest reading was 1,508 MiB on B against 2,020 MiB on A. The 250 ms telemetry floor was 487 and 519 MiB on B, against 839 and 871 MiB on A.

## Rejected

- **The held charge with the reserve yield only before the decode step** (`runs/c8-v2/`, three
  pairs on `b868464d9b`): measured-pass first-token p50 14.6 s against 23.4 s, but four prefill
  out-of-memory failures in one of its three runs, as above. Fixed by the yield before every prime
  call.
- **The whole reserve before every prime call and every batch round** (`C1c`, `47e9d340a8`,
  `runs/*-c1c/`). It had no failures and no OOM line. It was superseded, not refuted: each prime
  call now yields only its own need, and the batch rounds bound themselves on a fresh reading. Its
  mixed-shape median (114.13 against 118.27 tok/s) overlapped A's runs. The final tree is flat
  there.
- **The held charge with the old deadline estimate** (`C1e`, `runs/c8-c1e/`). The measured pass
  gained as above. The cold-burst prime pass put its slowest first token at 86-89 s in every run,
  against the 90 s deadline.
- **A fixed 2x margin on that estimate, keyed to this card.** It was never measured. It was a
  calibration constant from one card and one workload, and the derived estimate replaced it.
- **The yield before every prime call sized by the batch formula, with no reserve yield before
  phase (a) unless an MTP session was priming** (`C1d`, `bb7f308cef`; review). It missed the
  DFlash walkers, which prime in phase (a) with no `spec` state, and the fanout leader prime, a
  plain slab prime of up to one take. Its need, the batch's 458 KB per row times the take, was
  3.76 GB at the 8192-row solo widening, about three times what that call holds.
- **The first version of this lane** (`runs/c8-v1/`, one diagnostic pair, then reviewed out). It
  charged one take on every single-device card, gave walker routes the slab credit by route alone,
  and did not bound concat batches. Review found three problems:
  - a batch of six 1039-row takes needs about 2.86 GB of planes, against the 2.45 GB the last
    admission left (its own charge plus the reserve);
  - one card's evidence changed admission on every card;
  - a slab regrow allocates the new slab beside the old one.
  All three are fixed above.
- **A slab credit alone** (the widest call less the resident slab, the whole-prompt stack kept).
  A concat batch does not use the slab, so a credit on its members' charge would leave a batch of
  fresh prompts charged below its planes.
- **Dropping the per-call term and leaving every transient to the reserve.** Prime calls run one
  at a time, but a batch's planes are the sum of its members'. The formula's 458 KB per row bounds
  both the 290 KB per row a slab call holds beyond the slab and the 444 KB per row of a batch.
- **Charging a measured per-row transient instead of the formula.** It would need a boot
  measurement per route and model, and the formula already bounds both paths from above.
- **Every card at once, or the whole RTX 5090 class.** One card's evidence sets one card's default
  (AGENTS.md, per-hardware arm selection). The desktop RTX 5090 primes through the qwen prime graph
  on its 170 SMs and keeps a chunk graph per session between ticks, which the probe never saw.
  Unified memory (GB10) treats memory differently, and the PRO 6000 editions and Hopper have no
  cell.

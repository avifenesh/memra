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
- **Unchanged:** the reserve is sized from the cold charge as before. Every other card, unified
  memory, multi-device placements, the gemma and Step35 programs, monolithic primes and chunks
  wider than 4096 or set explicitly (`MEMRA_PRIME_CHUNK`) keep the cold charge, byte for byte. A
  held charge is never above the cold one.

The held charge at the 17,166-token prompt is 476 MB beside a peer (2,147 MB cold).

## Measured

Pending: the interleaved A/B in `research/c8-workspace-20261108/RESULTS.md`.

## Rejected

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

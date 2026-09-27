# WP-B day 11: the 32k residual classified by its series on both card classes

Repository: **avifenesh/memra**, branch `lane/spill-b-20260919`.
Source for every cell on both cards: **`1566d4f81a5f58d39c96a78ba987c34cfc66ccf3`**
(`feat(kv-tier-gate): --reclaim-cycles N ...` on top of the merge of `origin/main` `34d6bce35`,
PR #573, which carries the day-10 code). **Not merged, released, deployed, serving-qualified, or a
default promotion.** Every cell is N=1, `executed-not-qualified` or `refused` in the collector's
vocabulary, and no timing is compared between the two cards.

## What changed in the gate

`kv-tier-gate --case active --tiers host --same-program --kv-allocator vmm --reclaim-diagnostic
--reclaim-cycles N` (N >= 2) runs N demote/restore roundtrips in ONE process on ONE cache under the
same tokenwise `decode_step_h` program, and records per cycle the driver-free bytes before demote,
after demote and after restore, the reclaim delta and the residual. `cycle-<k>/` holds the full
day-10 receipt set for each cycle; `reclaim-cycles.tsv` and `reclaim-cycles.txt` hold the series.
The series class (`kv_tier_gate/reclaim_contract.rs`, `classify_cycles`) follows the lead ruling:
exactly one granule after cycle 1 and identical through cycle N, with (a) to (c) holding every
cycle and no drift of the process free baseline, is `one-time-driver-mapping-metadata`; a residual
(or the bytes still unreturned against the first baseline) that grows is `growing-residual`; zero
everywhere is `none`; anything else stays `unclassified`. The per-cycle G1 line is unchanged
(`reclaimed = vmm_granularity != 0 && reclaim_observed && observation.residual == 0`) and the
series `g1_reclaim_qualified` is the AND of the cycles. Refusals (exit 2, `REFUSED:` last line):
no `--reclaim-diagnostic`, a pooled allocator, a duplicate, N < 2, a missing value or junk.
No new `MEMRA_*` read. Option text: `docs/TESTING.md`, kv-tier-gate section.

## Source and binary identity

| Card | Native source | Binary SHA-256 | build / clippy / kv+tier tests |
| --- | --- | --- | --- |
| one RTX PRO 6000 Blackwell, 96 GB, 600/600 W | `1566d4f81a5f58d39c96a78ba987c34cfc66ccf3` | `e85a6ab7305fa1937e68eac81fac6539944ad531b401a37afa61f905996cbc03` | 0 / 0 / 0 (276 passed) |
| RTX 5090 Laptop GPU, local rig, `power.limit [N/A]`, `power.max_limit 175.00 W` | `1566d4f81a5f58d39c96a78ba987c34cfc66ccf3` | `b47d025891dfce8abf5226de73c16b4c52d1a79b84826106175f4992ae076094` | 0 / 0 / 101 (storage flake, see below) |

Artifact on both cards: `1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a`
(Qwen3.8-27B native NVFP4/Q5K GGUF). Frozen target-card bundle `BOX3-BASELINES.json`
`72db65b6a52784b14a54e3c76f8bfdd1aa58f02b8e7a159e2e27696cbc831990`, unchanged. Direct VMM
construction engaged on every executed cell (`construction=direct`, `empty_plane_swap=false`,
34 VMM / 0 pooled planes at position 0).

## Cells

All cells: `--case active --context <ctx> --tiers host --same-program --kv-allocator vmm
--reclaim-diagnostic --reclaim-cycles 5`, through `tools/tier-battery.py` with the rig's lock
(`pro-single`: `/tmp/memra-gpu.lock`; `rtx5090`: `/tmp/memra-5090.lock`), 250 ms telemetry, one
compute process in every before/after snapshot. Raw receipts: `pro-single-day11/` (mirror of the
target card's `b-day11` receipts, with the collector's `--validate` output in each `validate.json`)
and `rtx5090-day11/`. Final logits are gzip archived losslessly; the replay hashes decoded bytes.

| Card | Cell | Collector status | Elapsed | GPU temperature | Power draw |
| --- | --- | --- | ---: | --- | --- |
| PRO 6000 | `cycles-32768` | `executed-not-qualified` | 695.3 s | 33 to 51 C | 32 to 361 W (cap 600 W) |
| PRO 6000 | `cycles-8192` | `refused` | 128.8 s | 35 to 51 C | 50 to 361 W |
| PRO 6000 | `cycles-8192-retry` | `refused` | 128.8 s | 33 to 51 C | 33 to 362 W |
| 5090 Laptop | `cycles-32768` | `executed-not-qualified` | 959.6 s | 58 to 87 C | 26 to 172 W (max 175 W) |
| 5090 Laptop | `cycles-8192` | `refused` | 184.3 s | 59 to 86 C | 15 to 173 W |

Elapsed and thermal figures are conditions of each cell, recorded per card; they are not a
comparison and no median is claimed (N=1 everywhere).

## Per-cycle table, 32k, one RTX PRO 6000 Blackwell (bytes verbatim)

| Cycle | Free before demote | Free after demote | Free after restore | Released chunks | Reclaim delta | Reacquired | Residual | Restore residual | (a)-(c) | per-cycle G1 line |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 85,313,847,296 | 86,217,719,808 | 85,313,847,296 | 905,969,664 | 903,872,512 | 903,872,512 | **2,097,152** | 0 | true | `false` |
| 2 | 85,313,847,296 | 86,217,719,808 | 85,313,847,296 | 905,969,664 | 903,872,512 | 903,872,512 | **2,097,152** | 0 | true | `false` |
| 3 | 85,313,847,296 | 86,217,719,808 | 85,313,847,296 | 905,969,664 | 903,872,512 | 903,872,512 | **2,097,152** | 0 | true | `false` |
| 4 | 85,313,847,296 | 86,217,719,808 | 85,313,847,296 | 905,969,664 | 903,872,512 | 903,872,512 | **2,097,152** | 0 | true | `false` |
| 5 | 85,313,847,296 | 86,217,719,808 | 85,313,847,296 | 905,969,664 | 903,872,512 | 903,872,512 | **2,097,152** | 0 | true | `false` |

`free_before_drift_bytes` is 0 in every cycle; the mapped-VA probe reads
`mapped_va_release_delta_bytes=0`, `mapped_unmap_delta_bytes=0`, `mapped_va_roundtrip_equal=true`
in every cycle; every cycle's `restored-prefix-state.tsv` equals the suspended `prefix-state.tsv`;
all seven continuation surfaces match the frozen target-card 32k bundle.

## Per-cycle table, 32k, RTX 5090 Laptop GPU (bytes verbatim)

| Cycle | Free before demote | Free after demote | Free after restore | Released chunks | Reclaim delta | Reacquired | Residual | Restore residual | (a)-(c) | per-cycle G1 line |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 8,784,248,832 | 9,688,121,344 | 8,784,248,832 | 905,969,664 | 903,872,512 | 903,872,512 | **2,097,152** | 0 | true | `false` |
| 2 | 8,784,248,832 | 9,688,121,344 | 8,784,248,832 | 905,969,664 | 903,872,512 | 903,872,512 | **2,097,152** | 0 | true | `false` |
| 3 | 8,784,248,832 | 9,688,121,344 | 8,784,248,832 | 905,969,664 | 903,872,512 | 903,872,512 | **2,097,152** | 0 | true | `false` |
| 4 | 8,784,248,832 | 9,688,121,344 | 8,784,248,832 | 905,969,664 | 903,872,512 | 903,872,512 | **2,097,152** | 0 | true | `false` |
| 5 | 8,784,248,832 | 9,688,121,344 | 8,784,248,832 | 905,969,664 | 903,872,512 | 903,872,512 | **2,097,152** | 0 | true | `false` |

Same probe readings and same in-process identity as on the PRO card. This laptop card has no
frozen bundle of its own: against the rented RTX 5090 day-6 bundle the artifact, plan, prompt and
program identities match and **the 128 generated tokens are identical**, while `prefix-state.tsv`,
`final-state.tsv`, `logits.tsv` (0 of 129 rows equal) and `final-logits.f32le` differ. The replay
reports that and holds the cell to in-process identity only; it can never carry a G1 label.

## Verdicts (verbatim, `verify-day11.py`)

```text
pro-single/cycles-32768: ACTIVE-32K physical reclaim/restore bit-identical across 5 cycles, residual 2097152 B each cycle, class one-time-driver-mapping-metadata, not G1 PASS
pro-single/cycles-8192: ACTIVE-8K reclaim-cycles control not executed: REFUSED: diagnostic could not re-reserve original VMM address
pro-single/cycles-8192-retry: ACTIVE-8K reclaim-cycles control not executed: REFUSED: diagnostic could not re-reserve original VMM address
rtx5090/cycles-32768: ACTIVE-32K physical reclaim/restore bit-identical across 5 cycles, residual 2097152 B each cycle, class one-time-driver-mapping-metadata, not G1 PASS
rtx5090/cycles-8192: ACTIVE-8K reclaim-cycles control not executed: REFUSED: diagnostic could not re-reserve original VMM address
```

Gate console, both 32k cells, verbatim:

```text
RECLAIM-CYCLES: class=one-time-driver-mapping-metadata cycles=5 granule=2097152 residual_first=2097152 residual_last=2097152 g1_reclaim_qualified=false
ACTIVE_COPY_RESTORE_CAPTURED; reclaim qualification incomplete; see metrics; continuation comparison pending; not G1 PASS committed=32768 generated=128
```

## Class each card shows

Both card classes show **`one-time-driver-mapping-metadata`** at 32k: the residual is exactly one
VMM granule (2,097,152 B) after cycle 1 and byte-identical through cycle 5, criteria (a) to (c)
hold in every cycle, free VRAM returns to the identical baseline after every restore, and the
process free baseline does not drift. It does not grow, so it is not a per-cycle leak; it does not
return on unmap or on VA free (day-10 probe, repeated here in every cycle); it returns on remap.
This is the first time both cards carry the same nonzero class. It is recorded, not promoted.

## The 8k control did not execute on either card

The 8k series control refused in cycle 1, after the full 8,064-token prefill, inside the mapped-VA
probe (`KvPlane::probe_demoted_va_release`, day-10 code): after unmapping the retained chunks and
freeing the reservation, `cuMemAddressReserve` with the original base as the fixed-address request
returned a different address, the probe freed that unexpected reservation and refused. Verbatim:

```text
REFUSED: diagnostic could not re-reserve original VMM address
```

Deterministic on the PRO card (two attempts, 128.8 s each, identical console) and reproduced once on
the 5090 laptop. The 32k probe re-reserved its base in all 32 planes in all 5 cycles on both cards.
At 8k the reservations are small (K 5 granules, V 3 granules) against 17 and 12 at 32k; the driver
treats the fixed address as a hint, and this lane does not know its policy. This is the fail-closed
path working as designed: the plane stays suspended, the reservation ownership is tracked for
cleanup, no token executed against the suspended cache, no `ACTIVE.txt`, no cycle row. The raw
receipts, `validate.json` (`refused`) and `REFUSED.txt` are preserved. Consequence: **no 8k
residual series exists on either card**. The day-10 `ACTIVE-8K G1 PASS` cells ran without the
mapped-VA probe and are unchanged. The lead decides whether an 8k series control is wanted without
the mapped-VA probe (a CLI change to this gate); this session did not alter the probe or the
option after seeing the refusal.

## What stays `not G1 PASS`

- 32k on both cards: `g1_reclaim_qualified=false` in every cycle and for the series. Tightening (e)
  is untouched; a classified nonzero residual is recorded with its class and bytes and does not
  qualify. Lifting (e) for `one-time-driver-mapping-metadata` is the lead's call, now that both
  card classes carry the class.
- The 8k series control on both cards: not executed (refused), so it neither passes nor fails.
- The 5090 laptop 32k cell additionally has no frozen bundle for its card.

## Checks actually run

| Check | Result |
| --- | --- |
| `cargo fmt --all -- --check` | PASS |
| `cargo test -p memra-tier -p memra-kv --offline` (dev, local, CPU quota) | first run: `storage` `day4::review_catalog_recovers_pending_with_and_without_published_root` `Err(Busy)` at `day4.rs:309` while a co-tenant `local-ci` ran; rerun of `--test storage --test reclaim`: 53 + 14 passed |
| `cargo clippy -p memra-engine -p memra-tier -p memra-kv --offline --all-targets -- -D warnings` (dev, local) | PASS |
| `bash tools/check-flags.sh`; `git diff --check` | PASS |
| `python3 research/spill-b-20260919/test-day11.py` | 8 passed |
| Native PRO release build, release clippy `-D warnings`, release kv+tier tests | 0 / 0 / 0, 276 passed (`pro-single-day11/build/`) |
| Local release build, release clippy `-D warnings` | 0 / 0 (`rtx5090-day11/build/`) |
| Local release kv+tier tests | `tests.log`: 1 failed (`day4.rs:471` `Err(Busy)`); `tests-rerun.log` on the quiet rig: 3 failed, all `Err(Busy)` at `day4.rs:143`, `:240`, `:471`; 273 passed |
| Final battery on the quiet rig, CPU quota (`run-day11-checks.py`: fmt, kv+tier check, kv+tier tests, scoped clippy `-D warnings`, `test-day11.py`, `verify-day11.py --require-complete`, `git diff --check`, `check-flags.sh`) | all 8 exit 0; 276 tests passed, 0 failed (`day11-checks/final/checks.json`) |
| Full GPU exactness/serving/PRO-pair battery; 8k series control | NOT RUN / not executed |

The storage `Err(Busy)` failures are in lane A's `crates/memra-tier/tests/storage/day4.rs` and
`src/object_store` (`evict` re-locks after unlocking the shared lifetime lock; `CatalogStore::open`
after a simulated crash). DAY9.md recorded the same `day4.rs:471` failure on the rented RTX 5090
with cause unresolved. It reproduces on this rig (tmpfs `/tmp`, 24 cores) in release and dev
profiles, and did not reproduce on the PRO card (276 passed). Nothing in this lane touches that
code; it is reported to the lead as an intermittent, rig-sensitive test, not relabelled.

## Boundaries and record

- The first local 32k launch was stopped by this session at prompt position 512 (a tool-side time
  cap, not the rig) and relaunched detached; the stopped attempt is kept under
  `rtx5090-day11/interrupted-attempt-cycles-32768/` with a note and carries no result.
- No lock name other than the two canonical ones; every GPU command went through the collector; no
  `--no-verify`, no skip variable, no touch of `/root/artifacts`, `/root/memra-spill`, other lanes'
  worktrees or `main`.
- Push: the pre-push perf-ci freshness gate refuses this branch (`engine files touched after the
  last perf-ci battery`, base `7d213551a`, the engine files coming from the `origin/main` merge).
  The lead pushes with the logged override; the SHAs are in `STATE.md`.
- `day11-raw-manifest.json` seals both raw directories; `verify-day11.py` replays journals, lock,
  power constancy, co-tenancy, build/source/binary identity, direct construction, continuation
  surfaces, every cycle's chunk census, probe accounting, arithmetic and flags, the series table and
  class against the pure rule, and refused cells verbatim. It does not run CUDA.
- This session used approximately **1.8 agent-hours** of the 8-hour day-11 budget, including builds,
  prefill waits and lock hygiene.

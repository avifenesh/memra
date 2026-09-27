# WP-B day 12: lead ruling 6 implemented, and the 32k series printed the classified label on both card classes

Repository: **avifenesh/memra**, branch `lane/spill-b-20260919`.
Gate source for every cell on both cards: **`c7dd20cc53f126787c550b86f24d8a3d04daed54`**
(`feat(kv-tier-gate): series-level g1_reclaim_qualified per lead ruling 6 ...` on top of `d1844b3c0`,
the pushed day-11 tip). Docs, `verify-day12.py` and the CPU battery: `152c736cf`; this record and the
raw receipts: the data commit after it. **Not merged, released, deployed, serving-qualified, or a default
promotion.** Every cell is N=1, `executed-not-qualified` in the collector's vocabulary, and no timing is
compared between the two cards.

## What changed in the gate (lead ruling 6, implemented exactly)

`kv_tier_gate/reclaim_contract.rs::series_verdict` computes the series-level verdict of a
`--reclaim-cycles N` run. `g1_reclaim_qualified=true` with a nonzero residual is allowed only when all of:
N >= 5 (`series_min_cycles=5`); `residual_series_class = one-time-driver-mapping-metadata`; criteria (a) to
(c) hold in every cycle; the restored prefix is bit-identical in every cycle; baseline drift is 0. Then, and
only then, the gate prints `ACTIVE-32K G1 PASS (classified one-time-driver-mapping-metadata, N cycles)` as
its status line (the `K` tag follows the committed context, so a 16k series cannot print `32K`), writes the
same line as the first line of `ACTIVE.txt`, and records `series_label=<label>` in `reclaim-cycles.txt`
(`series_label=not-printed` otherwise). A series whose every cycle is exact (class `none`) is `true` with no
new label. A single roundtrip with a nonzero residual, a series shorter than 5, any other class, a drifting
baseline, a differing restore and any pooled run stay `false` / `not-applicable-pooled` with their existing
status lines. The per-cycle line is untouched:
`reclaimed = vmm_granularity != 0 && reclaim_observed && observation.residual == 0` (tightening (e)), and it
is `false` in every cycle on both cards below. Criteria (a) to (d) are unchanged; (e) stays in force for every
other shape. `active.rs::write_cycles` takes the context and returns the verdict; the gate loop prints the
label only when `write_cycles` produced it. No CLI change, no probe change, no new `MEMRA_*` read.

One place the series field is stricter than day 11's plain AND of the per-cycle lines: exact cycles over a
drifting baseline are `growing-residual` or `unclassified` and stay `false`. No observed cell has that shape.

Tests: `reclaim_contract` unit test (N=4 false; N=5 with one drifting cycle false; N=5 `growing-residual`
false; two-granule and never-returned shapes false; one non-identical restore false; N=5 exact `none` true
without a label; pooled `not-applicable-pooled`; context tag). `crates/memra-tier/tests/reclaim/day12.rs`
replays the committed day-11 series bytes of both card classes offline (label; every prefix shorter than five
false; drift; growing; unclassified; restore identity; every one-byte mutation; pooled) and the day-10 8k
receipt as an exact series. `day11.rs` expectations are unchanged; one comment changed because "the class
never promotes" now applies to the per-cycle line only. `verify-day12.py` enforces the same series rule on
the receipts and additionally requires that the label was PRINTED by the gate as the final status line,
exactly once, with `ACTIVE.txt` carrying the same line; `test-day12.py` holds 11 adversarial arms.

## Source and binary identity

| Card | Native source | Binary SHA-256 | release build / clippy / kv+tier tests |
| --- | --- | --- | --- |
| one RTX PRO 6000 Blackwell, 96 GB, 600/600 W | `c7dd20cc53f126787c550b86f24d8a3d04daed54` | `efc52ef21ac615a4bd78df5a72a1a8f28b15369ffabd7f1a55a3dba4fe9b4930` | 0 / 0 / 0 (280 passed; `dirty.txt` empty) |
| RTX 5090 Laptop GPU, local rig, `power.limit [N/A]`, `power.max_limit 175.00 W` | `c7dd20cc53f126787c550b86f24d8a3d04daed54` | `3f492fbac79206b633a387acad47b06eb691f008a6b9ef57712312efca81af67` | 0 / 0 / 101 (278 passed, 2 failed in lane A's `storage` `day4.rs:143` and `:471`, see below; `dirty.txt` lists only the untracked day-12 python files and receipt directory, no engine source) |

Artifact on both cards: `1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a`
(Qwen3.8-27B native NVFP4/Q5K GGUF). Frozen target-card bundle `BOX3-BASELINES.json`
`72db65b6a52784b14a54e3c76f8bfdd1aa58f02b8e7a159e2e27696cbc831990`, unchanged. Direct VMM construction
on both cells (`construction=direct`, `empty_plane_swap=false`, 34 VMM / 0 pooled planes at position 0).
The binary hashes differ from day 11 because the gate source changed; the local binary carries the new
format strings (`series_label`, `series_min_cycles`, the label).

## Cells

Both cells: `--case active --context 32768 --tiers host --same-program --kv-allocator vmm
--reclaim-diagnostic --reclaim-cycles 5`, through `tools/tier-battery.py` with the rig's lock (`pro-single`:
`/tmp/memra-gpu.lock`, acquired on attempt 0, no contention with lane C; `rtx5090`: `/tmp/memra-5090.lock`,
attempt 0), 250 ms telemetry, no co-tenant compute process in the before/after snapshots. The 8k series
control was not rerun (lead ruling 7, below). Raw receipts: `pro-single-day12/` (mirror of the target card's
`b-day12` receipts, collector `--validate` output in `validate.json`) and `rtx5090-day12/`. Final logits are
gzip archived losslessly; the replay hashes decoded bytes.

| Card | Cell | Collector status | Elapsed | GPU temperature | Power draw |
| --- | --- | --- | ---: | --- | --- |
| PRO 6000 | `cycles-32768` | `executed-not-qualified` | 696.6 s | 33 to 51 C | 32 to 362 W (cap 600 W) |
| 5090 Laptop | `cycles-32768` | `executed-not-qualified` | 940.6 s | 53 to 87 C | 9 to 174 W (max 175 W) |

Elapsed and thermal figures are conditions of each cell, recorded per card; they are not a comparison and no
median is claimed (N=1 everywhere).

## Per-cycle table, 32k, one RTX PRO 6000 Blackwell (bytes verbatim; identical to day 11)

| Cycle | Free before demote | Free after demote | Free after restore | Released chunks | Reclaim delta | Reacquired | Residual | Restore residual | (a)-(c) | per-cycle G1 line |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 85,313,847,296 | 86,217,719,808 | 85,313,847,296 | 905,969,664 | 903,872,512 | 903,872,512 | **2,097,152** | 0 | true | `false` |
| 2 | 85,313,847,296 | 86,217,719,808 | 85,313,847,296 | 905,969,664 | 903,872,512 | 903,872,512 | **2,097,152** | 0 | true | `false` |
| 3 | 85,313,847,296 | 86,217,719,808 | 85,313,847,296 | 905,969,664 | 903,872,512 | 903,872,512 | **2,097,152** | 0 | true | `false` |
| 4 | 85,313,847,296 | 86,217,719,808 | 85,313,847,296 | 905,969,664 | 903,872,512 | 903,872,512 | **2,097,152** | 0 | true | `false` |
| 5 | 85,313,847,296 | 86,217,719,808 | 85,313,847,296 | 905,969,664 | 903,872,512 | 903,872,512 | **2,097,152** | 0 | true | `false` |

## Per-cycle table, 32k, RTX 5090 Laptop GPU (bytes verbatim; identical to day 11)

| Cycle | Free before demote | Free after demote | Free after restore | Released chunks | Reclaim delta | Reacquired | Residual | Restore residual | (a)-(c) | per-cycle G1 line |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 8,784,248,832 | 9,688,121,344 | 8,784,248,832 | 905,969,664 | 903,872,512 | 903,872,512 | **2,097,152** | 0 | true | `false` |
| 2 | 8,784,248,832 | 9,688,121,344 | 8,784,248,832 | 905,969,664 | 903,872,512 | 903,872,512 | **2,097,152** | 0 | true | `false` |
| 3 | 8,784,248,832 | 9,688,121,344 | 8,784,248,832 | 905,969,664 | 903,872,512 | 903,872,512 | **2,097,152** | 0 | true | `false` |
| 4 | 8,784,248,832 | 9,688,121,344 | 8,784,248,832 | 905,969,664 | 903,872,512 | 903,872,512 | **2,097,152** | 0 | true | `false` |
| 5 | 8,784,248,832 | 9,688,121,344 | 8,784,248,832 | 905,969,664 | 903,872,512 | 903,872,512 | **2,097,152** | 0 | true | `false` |

On both cards `free_before_drift_bytes` is 0 in every cycle; the mapped-VA probe reads
`mapped_va_release_delta_bytes=0`, `mapped_unmap_delta_bytes=0`, `mapped_va_roundtrip_equal=true` in every
cycle; every cycle's `restored-prefix-state.tsv` equals the suspended `prefix-state.tsv`. `reclaim-cycles.txt`
on both cards: `residual_series_class=one-time-driver-mapping-metadata`, `free_before_drift_last_bytes=0`,
`all_cycles_reclaim_observed=true`, `all_cycles_restored_bit_identical=true`, `g1_reclaim_qualified=true`,
`series_min_cycles=5`, `series_label=ACTIVE-32K G1 PASS (classified one-time-driver-mapping-metadata, 5 cycles)`.

## The label each card printed (verbatim)

Gate console, final two lines, identical on both cards:

```text
RECLAIM-CYCLES: class=one-time-driver-mapping-metadata cycles=5 granule=2097152 residual_first=2097152 residual_last=2097152 g1_reclaim_qualified=true
ACTIVE-32K G1 PASS (classified one-time-driver-mapping-metadata, 5 cycles) committed=32768 generated=128
```

- one RTX PRO 6000 Blackwell: `ACTIVE-32K G1 PASS (classified one-time-driver-mapping-metadata, 5 cycles)`
- RTX 5090 Laptop GPU: `ACTIVE-32K G1 PASS (classified one-time-driver-mapping-metadata, 5 cycles)`

## Verdicts (verbatim, `verify-day12.py --require-complete`)

```text
pro-single/cycles-32768: ACTIVE-32K G1 PASS (classified one-time-driver-mapping-metadata, 5 cycles)
rtx5090/cycles-32768: ACTIVE-32K G1 PASS (classified one-time-driver-mapping-metadata, 5 cycles)
```

Continuation, reported separately from the series verdict: the PRO cell `matches the frozen target-card
bundle (BOX3-BASELINES.json)` on all seven surfaces. The laptop cell, as on day 11: against the rented RTX
5090 day-6 bundle the artifact, plan, prompt and program identities match and the 128 generated tokens are
identical, while `prefix-state.tsv`, `final-state.tsv`, `logits.tsv` and `final-logits.f32le` differ; this
card has no frozen bundle of its own, so its continuation identity is in-process only. Ruling 6 conditions
the label on the series (class, N, (a) to (c), identical restore, drift 0), which both cards meet; the
frozen-bundle fact is recorded next to it, not folded into it.

## The 8k control (lead ruling 7)

Not rerun. On day 11 the mapped-VA probe refused to re-reserve the original VMM address for the small 8k
planes (K 5 granules, V 3) on both cards, deterministically, freed the stray reservation and refused as
designed (`REFUSED: diagnostic could not re-reserve original VMM address`). Fail-closed; no probe or CLI
change; recorded as the open item in `docs/decisions/KV-PHYSICAL-RECLAIM.md`. The 8k evidence stays the
single-roundtrip `ACTIVE-8K G1 PASS` with residual 0 on both cards (day 10 PRO, day 9 rented RTX 5090).

## Checks actually run

| Check | Result |
| --- | --- |
| `cargo fmt --all -- --check` | PASS |
| `cargo test -p memra-tier --test reclaim --offline` (dev, local, CPU quota) | 18 passed (3 new `day12` arms, 1 new `reclaim_contract` series test) |
| `cargo clippy -p memra-engine -p memra-tier -p memra-kv --offline --all-targets -- -D warnings` (dev, local) | PASS |
| `bash tools/check-flags.sh`; `git diff --check`; `python3 tools/check-public-boundary.py check` | PASS (boundary: 0 new matches) |
| `python3 research/spill-b-20260919/test-day12.py` | 11 passed |
| Native PRO release build, release clippy `-D warnings`, release kv+tier tests (`pro-single-day12/build/`) | 0 / 0 / 0, 280 passed |
| Local release build, release clippy `-D warnings` (`rtx5090-day12/build/`) | 0 / 0 |
| Local release kv+tier tests (`rtx5090-day12/build/tests.log`) | 278 passed, 2 failed: `storage` `day4::catalog_gc_reference_count_charges_tombstone_crash_replay` (`day4.rs:143:73`) and `day4::review_gc_cancel_removes_staging_and_corrupt_staging_is_noop` (`day4.rs:471:25`), both `called Result::unwrap() on an Err value: Busy`, lane A code, the rig-sensitive flake DAY11.md reported; not touched |
| Final battery on the quiet rig, CPU quota (`run-day12-checks.py`: fmt, kv+tier check, kv+tier tests, scoped clippy `-D warnings`, `test-day12.py`, `verify-day12.py --require-complete`, `git diff --check`, `check-flags.sh`) | all 8 exit 0; 280 tests passed, 0 failed (`day12-checks/final/checks.json`) |
| Full GPU exactness/serving/PRO-pair battery; 8k series control | NOT RUN / not rerun |

## Boundaries and record

- Every GPU command went through the collector with the rig's canonical lock; no third lock name; no bare
  GPU run; no `--no-verify`; no skip variable; no touch of `/root/artifacts`, `/root/memra-spill`, other
  lanes' worktrees or `main`. The native checkout was synced by git bundle to `c7dd20cc5` and left clean; no
  B tmux session remains on either rig; the bundle file was removed.
- Nothing relaxed beyond ruling 6: no PASS for N < 5, for a single roundtrip with a residual, for any other
  class, for a drifting baseline, for a differing restore or for a pooled run. The per-cycle line and the
  probe are unchanged. `INTEGRATION-DAY11.md` and the HANDOVER were not touched.
- `day12-raw-manifest.json` seals both raw directories (150 files); `verify-day12.py` replays journals, lock,
  power constancy, co-tenancy, build/source/binary identity, direct construction, continuation surfaces,
  every cycle's chunk census, probe accounting, arithmetic and flags, the series table, the class, the
  ruling-6 verdict and the printed label. It does not run CUDA.
- Push: refused by `tools/hooks/pre-push` (perf-ci freshness gate), verbatim `pre-push: engine files touched after
  the last perf-ci battery.` with base `d1844b3c030b413dda6dc85c10bbf7c5c4f17a46` and the engine files
  `crates/memra-engine/src/bin/kv_tier_gate.rs`, `kv_tier_gate/active.rs`, `kv_tier_gate/reclaim_contract.rs`;
  every other pre-push arm (perf board, flags census, releasability, docs registry, public boundary) passed.
  No override used; the lead pushes the tip with the logged override. The unpushed tip is the data commit at
  the top of `git log` (`STATE.md`).
- Local builds and tests ran under `systemd-run --user --scope -p CPUQuota=1200% -p MemoryMax=28G`.
- Time, measured anchors (UTC, from the receipts): native build started 22:03, both cells ran 22:05 to 22:20,
  CPU battery 22:23, data commit 22:26. Approximately **0.8 agent-hours** of the 6-hour day-12 budget from the
  first read to the amended data commit, including builds, prefill waits and lock hygiene.

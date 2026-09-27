# Session C day eleven: the installer catalog comes from the model plan

Scope: door doc pending item 3, first half (installer generality), plus the lead's
self-review finding 4 nits (`research/spill-lead-20260919/PR-INTEG6-SELF-REVIEW.md`).
Everything measured here is N=1 `executed-not-qualified` development evidence on one
RTX PRO 6000 Blackwell (600 W envelope) through the canonical collector; nothing is a
support state. Same numeric program as day ten: the bank bytes, records and checksums
for the approved artifact are unchanged, and the parity is replayed below.

## What changed

`6db8ac122` (catalog): `Engine::install_expert_bank_gate` no longer spells
`blk.N.ffn_{gate,up,down}_exps.weight`. The new `memra_gguf::expert_banks` module
(`crates/memra-gguf/src/expert_banks.rs`) derives an `ExpertBankCatalog` from the
compiled `ModelPlan` and the model pack's GGUF `TensorContract`: for every MoE block of
the plan (trunk layers by position, then MTP blocks by depth) it selects the three bank
requirements by semantic id (`LayerTensor::MoeExpert{Gate,Up,Down}Bank`), carries their
declared `QuantAux` requirements along, and binds that sub-contract against exactly the
census rows those names match through `TensorContract::bind`. The installer
(`banked_residency/native.rs`) takes the contract from `model_packs::for_config(&model.cfg)`
(`TensorContract::for_plan` when no pack matches), the census from
`source::census_from_gguf`, and walks `catalog.blocks()`; the loaded `HostExps` supply
only the bytes and the router mask, and every retained record is still compared
byte-for-byte with the loaded expert (`native expert bytes differ from pinned GGUF`
stays a failure). No checkpoint name is spelled in the installer, no architecture name is
consulted, no `MEMRA_*` read is added.

Typed refusals (`REFUSED: experts-via-tier expert catalog refused: <detail>`, exit 2 in
both gate binaries), each with a CPU unit test in `expert_banks.rs`:

| condition | refusal detail (verbatim `Display`) |
|---|---|
| plan has no MoE block | `the compiled plan has no MoE expert projections` |
| contract has no entry for a bank | `tensor contract has no entry for Layer { index: 0, tensor: MoeExpertDownBank }` |
| contract has two entries for a bank | `tensor contract has 2 entries for Layer { index: 0, tensor: MoeExpertGateBank }` |
| bank tensor absent from the artifact | the contract's `Missing` text for `Layer { index: 1, tensor: MoeExpertUpBank }`, accepted name `blk.1.ffn_up_exps.weight` |
| bank tensor duplicated in the artifact | the contract's `DuplicateCensusName` text for `blk.0.ffn_down_exps.weight` |
| bank tensor shape-incompatible | the contract's `ShapeMismatch` text (expected `[64, 32, 4]`, actual `[64, 32, 8]`) |
| artifact carries a scale plane for a bank | `artifact carries expert scale planes the consumer does not declare: blk.0.ffn_gate_exps.input_scale, blk.0.ffn_up_exps.scale` |
| loaded bank carries `macros` or `fp8_blk` | `loaded bank <name> carries scale planes (macro or block scales) the native installer does not consume` (native check, no CPU fixture can build `HostExps`) |
| contract compiled for HF | `checkpoint dialect HfSafetensors is not GGUF` |
| plan/model disagreement | `loaded model has N layers, compiled plan has M`; `plan layer N routes experts but the loaded layer is dense`; `loaded MTP head routes experts but the compiled plan has no MTP expert bank` (native checks) |

Scale admission is not landed: a scale-bearing artifact or bank is refused, never banked
payload-only (door doc item 3, second half). The installer also prints, once the catalog is
bound and before `installed`:
`[experts-via-tier] catalog blocks=<n> banked=<n> projections=<n> catalog_sha256=<hex>
records=<n> records_sha256=<hex>`. `catalog_sha256` is SHA-256 over
`ExpertBankCatalog::identity()` (one line per projection: block, plan layer, projection,
name, GGUF shape, storage, bytes); `records_sha256` chains the per-record checksums in
catalog order; `banked` is 40 of 41 blocks under `run-gen`, which loads without the MTP
head.

`76f78c569` (finding 4 nits): `expert_bank_cli` splits each argument at `=` and matches
the key exactly, so `--expert-bank-host-bytes-x=1` and `--expert-bank-gpu-bytesx=1` are
`unknown expert bank flag "..."; expected --experts-via-tier, --expert-bank-host-bytes=<bytes>
or --expert-bank-gpu-bytes=<bytes>` and `--experts-via-tier=1` is `--experts-via-tier takes no
value` (usage errors, `Error:` exit 1; six look-alikes tested in `day10.rs`). The eight-byte
slot tail pad is `banked_residency::SLOT_TAIL_PAD_BYTES`, imported by `moe_cache.rs` for
`hard_slot_bytes`, the exact and auto slot sizing, the allocation and the size-class plan
(ten literals replaced, every one was 8; `slru-synthetic.json` re-pinned to the new
`moe_cache.rs` hash with all 2013 rows identical). `banked_residency` is a `#[doc(hidden)]
pub mod`; `run-gen` and `run-spec` reach `expert_bank_cli` and `refusal_reason` through it
and the crate root re-exports nothing from the gate. `6defcd604`: `run-day11.py`.

CPU gate on the tree (this rig, `systemd-run` CPU cap): `cargo fmt --all -- --check` clean;
`cargo test -p memra-gguf --offline --lib` 290 pass (10 new in `expert_banks`);
`cargo test -p memra-tier --offline` bank 61 pass; `cargo clippy -p memra-gguf -p memra-tier
--offline --all-targets -- -D warnings` clean; `DOCS_RS=1 cargo clippy -p memra-engine
--offline --lib --bin run-gen --bin run-spec --target x86_64-unknown-linux-gnu -- -D warnings`
clean; `tools/check-flags.sh` 864 names, no uncovered; `git diff --check` clean.

## Parity: plan-derived catalog == literal day-ten catalog

Two derivations of the approved artifact's catalog, compared on the target card:

- Plan-derived, from the running binary: the `catalog_sha256` the installer printed in
  every cell, `2204b15974f6c5e7794d6f4912af1ec6961f52bf53f6f82325f94c123bdefde8`
  (41 blocks, 123 projections).
- Literal, from memra's own census: `run-day11.py --inspect` built `memra-cli` (no CUDA)
  and ran `memra model inspect <artifact> --against qwen35moe` (receipt
  `pro-single-day11/inspect/`: `family=qwen35_moe tensors=753`; `artifact.lock` records
  `binding=passed`, `tensor_count=753` and a `census_sha256` equal to the SHA-256 of the
  committed `tensor-census.tsv`; the artifact identity itself is the installer's
  `installed artifact_sha256=` line in every cell). `verify-day11.py` rebuilds the day-ten spelling from
  `tensor-census.tsv` alone (every `blk.N` with the three `ffn_*_exps.weight` tensors,
  N < 40 trunk by position, N = 40 the MTP block, confirmed by `blk.40.nextn.eh_proj.weight`),
  in `identity()` form, and hashes it: `2204b15974f6c5e7794d6f4912af1ec6961f52bf53f6f82325f94c123bdefde8`.

Equal. The first and last identity rows are
`trunk:0 0 Gate blk.0.ffn_gate_exps.weight 2048x512x256 IQ3_S[256] 115343360` and
`mtp:0 40 Down blk.40.ffn_down_exps.weight 512x2048x256 Q4_K[256] 150994944` (tab
separated): the artifact mixes IQ3_S, IQ4_XS and Q4_K across projections and layers, and
the catalog binds each one through `QuantConstraint::Weight` with its own storage. The
CPU unit test `plan_derived_catalog_matches_the_literal_day10_spelling` proves the same
equality on a synthetic qwen3_5_moe plan with an MTP block (names, order, shapes, blocks).

Bank identity: the day-eleven `installed` line is byte-identical to the day-ten exact cell,
`[experts-via-tier] installed artifact_sha256=df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf host_slots=16 max_expert_bytes=860160`,
and the gen tape equals the day-nine native `default-gen-off` control and the day-ten exact
cell (`verify-day11.py`). Record digests: gen (40 blocks, 30,720 records)
`8084706a52804d906afb7d9599d6ff44a7dd13c3b37af5fffb9caca72e0a459d`; spec (41 blocks,
31,488 records) `e14759067def912b3f162d2429e34cacbb176c16dd1fdd51a3a20a7dde8af8e4`,
identical in the refused and the exact cell. Day ten printed no record digest, so these are
the first anchors; the equality to day ten rests on the catalog identity, the record-by-record
byte check against the loaded experts, the identical `installed` line and the identical tape.

## Target-card cells (`pro-single-day11/`, remote `c-day11/`)

Build: `/root/wt-c` fast-forwarded to `6defcd604` (git bundle over the lead's ssh master);
`cargo build --release -j 16 -p memra-engine --bin run-gen --bin run-spec` into
`CARGO_TARGET_DIR=/root/wt-c/target-day11` (nvcc 13.2, rustc 1.97.1; `build.json`,
`build.log`). Day-eleven binaries: run-gen
`79125c9a8d9be7eb9c74d07c79147cfb4ee3dba4be61e12e4938b1e01cf73f8b`, run-spec
`a81c2a029873426b557317a407b26629922235dcd2e8cac55416616fdf23453e`. The frozen day-nine
(`target/`) and day-ten (`target-day10/`) binaries were never rebuilt; `binary-manifest.json`
and `binary-postcheck.json` record all three pairs before and after the cells.

Receipts are mirrored byte-for-byte from the remote `c-day11/` directory; the two 52 MB
files of the exact-eight spec cell (`command.log`, its console echo) are committed gzipped
and `verify-day11.py` hashes the decompressed bytes against the capture's `raw_log` entry.
Every cell: `run-day11.py --cells` through `tools/tier-battery.py --rig pro-single` only
(canonical `/tmp/memra-gpu.lock`, 250 ms telemetry, 600 W limit recorded, attempt 1 each,
no lock wait needed), `env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 <gate> <artifact> 55 88 13
--experts-via-tier [budget]`, no `MEMRA_MOE_SLOTS`. One slot is 860,168 bytes; the
eight-slot minimum is 6,881,344 bytes.

| cell | gate, flag | collector | verbatim |
|---|---|---|---|
| a `gen-default` | `run-gen`, default budgets | `executed-not-qualified`, exit 0, 91.0 s | `prefill argmax=198  decode argmax=198  logit maxdiff=6.482e-1  MATCH`; `[experts-via-tier] catalog blocks=41 banked=40 projections=123 catalog_sha256=2204b159…fde8 records=30720 records_sha256=8084706a…459d`; `[experts-via-tier] installed artifact_sha256=df27a780…7adf host_slots=16 max_expert_bytes=860160`; `[expert-gpu-slru] slots=90705 allocated_bytes=78021538440 evictions=0`; `[experts-via-tier] physical_reads=30720 owner_close=Ok(())` |
| c `spec-refuse-7` | `run-spec`, `--expert-bank-gpu-bytes=6021176` (7 slots) | `refused`, exit 2, 80.7 s | `REFUSED: experts-via-tier GPU bank budget cannot hold the eight-slot minimum (requested 6021176, minimum 6881344, ceiling 77968398729)` as the final stderr line; before it only the model-load mirror line and `[experts-via-tier] catalog blocks=41 banked=41 projections=123 catalog_sha256=2204b159…fde8 records=31488 records_sha256=e1475906…`; no `gpu_bank_budget`, no `installed`, no `[expert-host-slru]`, no `loaded`, no tape |
| b `spec-exact-8` | `run-spec`, `--expert-bank-gpu-bytes=6881344` (8 slots) | `executed-not-qualified`, exit 0, 298.4 s | `=== SELF-CONSISTENCY PASS ===`; `[experts-via-tier] gpu_bank_budget bytes=6881344 slots=8 hard_ceiling=77968398729`; `[experts-via-tier] installed artifact_sha256=df27a780…7adf host_slots=16 max_expert_bytes=860160`; `loaded qwen35moe (41 layers, nextn=1)`; K=1..8 each `self-consistency: PASS (identical to plain target)`; `[expert-gpu-slru] slots=8 allocated_bytes=6881344 evictions=674569`; `[experts-via-tier] physical_reads=670848 owner_close=Ok(())` |

Cell a: with default budgets the native slot sizing is untouched (90,705 slots from free
VRAM, zero GPU evictions, `physical_reads=30720`: every record read exactly once into the
16-record host bank, `rereads=0`), the same shape as the day-nine `default-gen-on` cell.

Cell c: the catalog is bound and its line printed before the budget check, as on day ten
(the catalog walk reads the mmap'd tensor bytes for the record checksums; no bank, CUDA
slot or positioned read exists yet). The ceiling in the message is the hard ceiling measured
on this card at install time (77,968,398,729 bytes; day ten saw 78,022,085,820 with a
different free-VRAM reading), so the same command refuses with a different ceiling on a
different card or with other tenants.

Cell b: `[experts-via-tier] gpu_bank_budget bytes=6881344 slots=8 hard_ceiling=77968398729` precedes
`installed`; the K ladder is 1..8 with every row `self-consistency: PASS (identical to plain
target)`, and the spec tape and the eight acceptance rows equal the day-nine native
`default-spec-off` control on the same card (`verify-day11.py`), so the eight-slot bank under
eight speculative rounds is the same numeric program as the native cache. Eviction accounting
under an eight-slot GPU bank and a sixteen-record host bank across prefill plus eight rounds: GPU
evictions **674,569**, host evictions **670,832**, physical reads
**670,848** (every host miss is a read, `physical_reads == misses`), re-reads
**654,444**, `owner_close=Ok(())`. The exact minimum bank thrashes on purpose and stays
correct; the per-round `tok/s` lines in the log are pressure diagnostics, not a performance number.

Collector `--validate` on the receipt directory: `validate.log`, exit 0, `"kind": "capture-integrity", "cells": 3, "failed_commands": 0,
"refused_commands": 1, "qualification": false`.

`verify-day11.py`: PASS (`VERDICT.json`): build receipt identity and its own target
dir, frozen binaries unchanged before and after, `git diff --quiet 6defcd604 <worktree head>
-- crates/` empty, inspect receipt hashes and artifact lock, literal catalog digest equal to
the printed `catalog_sha256` in all three cells, gen `installed` line equal to day ten, gen
tape equal to the day-nine control and the day-ten exact cell, spec tape and acceptance
equal to the day-nine `default-spec-off` control, refusal token final with the arithmetic
above and no work after it, exact cell 8 slots / 6,881,344 bytes / `SELF-CONSISTENCY PASS`
/ K ladder 1..8 / evictions and re-reads present, postcheck after every cell.

## Boundaries

- N=1 per cell, one card class, development evidence; no medians, no timing claim, no
  cross-box comparison. The exact-eight spec cell is a correctness and eviction-accounting
  cell under extreme pressure, not a performance number.
- The hash lock to the one approved artifact stays; the catalog is plan-derived but the
  door still admits exactly that artifact. Scale admission (door doc item 3, second half)
  is pending: `.scale` / `.input_scale` census rows and loaded `macros` / `fp8_blk` planes
  are typed refusals.
- The catalog is GGUF-only (`checkpoint dialect HfSafetensors is not GGUF`), matching the
  door (GGUF path only).
- Push: the perf-ci pre-push gate refuses this lane (engine files touched after the last
  perf-ci battery: `kv_tier_gate.rs`, `active.rs`, `tier_transfer_gate.rs`, `lib.rs`,
  `tier_transfer.rs` from the merged main, plus this lane's files). No `MEMRA_SKIP_PERF_CI`,
  no `--no-verify`; the lead pushes. The tree reached the target card as a git bundle over
  the lead's ssh master.

Effort: approximately 3.5 agent-hours.

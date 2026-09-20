# Integration — days 7–8 (`lane/spill-integ4-20260920`)

Lead: @agent-c07799. Base: `origin/main` `a5744c72` (#535 P0 registry merged upstream; replayed the
whole lane on it, clean). Lane tips merged: A `06fda471`, B `2694ecca`, C `6ebeaeba`, D `7f7bf547`,
E `4d89a243`, F `eca3139e`.

## Lead-owned edits in this integration

| file | change | why |
|---|---|---|
| `crates/memra-engine/src/lib.rs` | `pub mod tier_transfer;` | A's fragment (`research/spill-a-20260919/CUDA-TRANSFERS.md`) |
| `crates/memra-engine/Cargo.toml` | `[[bin]] tier-transfer-gate`, `[[bin]] h2d-probe` | A / F fragments (hyphenated names, no auto-discovery underscores) |
| `crates/memra-engine/src/bin/kv_tier_gate.rs` | drop the `#[path = "../tier_transfer.rs"]` twin, `use memra_engine::tier_transfer;` | B's stop-gap until the lib module existed; `active.rs` keeps `super::tier_transfer::…` |
| `crates/memra-engine/build.rs` | DOCS_RS branch emits `MEMRA_MMQ_ARCHIVE_HASH=docs-rs-stub` | upstream `lib.rs:882` `env!`-binds the hash; the docs-only build had no value for it (docs.rs would fail) |
| `crates/memra-gguf/src/source.rs` (`5ffe935c`) | gate `AsRawFd` import to linux | its only use is under `cfg(target_os = "linux")`; unblocked macOS `clippy -D warnings` for every lane |

## What the lanes landed (evidence in each lane's day report; all rented-5090, `executed-not-qualified`)

- **A** — native owner-stream `CudaTransfers` (`tier_transfer.rs`) + `tier-transfer-gate`: all frozen v1/v1.1/v1.2
  schedules **PASS natively**; additive `take_device` (same-pointer hand-back) and `retire_source` (source freed
  while the host image stays live) **PASS**; byte round trips 4 KiB…256 MiB N=1 PASS; governor zero after drain.
  `research/spill-a-20260919/day7/RESULTS.md`.
- **B** — HostPrefix v2 applied on lane (711 native server tests); `kv-tier-gate --case active` bound to native
  demote/restore ownership (`active.rs`) — **not yet run** (day 8b in flight). `native-patch-check.py` spelling
  fixed to `kv-tier-gate`; refusal lines conform to D's token contract.
- **C** — Qwen3.6-35B-A3B IQ4_XS artifact (`ARTIFACT_SHA256_MATCH`); **experts-via-tier run-gen `MATCH`** with
  byte-identical 32-token tapes (`physical_reads=28758`), **run-spec K=1..8 `SELF-CONSISTENCY PASS`**;
  `qwen4exp-gpu-gate --rows-via-tier --device-publish` **BIT-IDENTICAL** (16 cases / 6144 values, 16 device
  uploads, exclusive hand-back); owner-thread proxy registry keeps `Engine: Sync` without `unsafe`. Day-7 report
  is being committed on the lane (agent died before the commit); receipts are in `6ebeaeba`.
- **D** — legacy identity / forced-tiny teeth / failure gates **ALL GREEN under the collector's inherited lock**,
  matching B's bare controls verbatim; refusal token contract (`REFUSED:` at line start; generic `Error:`+exit 2
  is *failed*); storage captures bound to the object filesystem; G2 N=1 plumbing ran. `DAY8-VERIFICATION.md`.
- **E** — PR #560 review's three medium findings verified fixed (B/D); **FREEZE-V1.3** (additive
  `take_device`/`retire_source`, default `Unsupported` bodies, CPU schedules); native-conformance receipt schema.
- **F** — `h2d_probe` (N=1 plumbing, native build) + **NVMe VM spend: NO-GO** until in-guest block ancestry is
  provable (`NVME-DECISION.md`); `--copies` extension in flight.

## Battery (this tree, Mac, offline)
`cargo test -p memra-tier -p memra-kv`: 253 passed / 0 failed · clippy `-D warnings` (tier, kv, gguf all-targets;
engine lib+bins on `x86_64-unknown-linux-gnu` with `DOCS_RS=1`) clean · fmt clean · `check-flags.sh` clean ·
publish census OK (12/12) · docs registry census OK · perf board up to date.
`git diff --check` flags whitespace only inside raw receipt logs and a banked `.diff` (hash-bound receipts; left
byte-exact on purpose). `tools/test_sft_scale.py` fails 5/10 identically on pristine main — unrelated, not absorbed.

## Toolchain note (A)
BOX2 stable Rust 1.98 raises 20 `chunks_exact_to_as_chunks` lints in memra-tier/memra-gguf; the repo pins 1.97.1
(`rust-toolchain.toml`) and CI clippy is clean there. Ruling: no lint churn for an unpinned toolchain; revisit when
the pin moves.

## Gates
No G0–G7 gate advanced by this integration. Next: B's active verdict (G1 needs observed device reclaim), C's
pressure/eviction cells, D's G2 once F's `--copies` lands, NVMe M1 still blocked on a provable rig.

## CI round 1 on PR #563 (`068279c8`) — two natives-only failures, fixed in the follow-up commit

| job | failure | fix |
|---|---|---|
| clippy | `items_after_test_module` — B's HostPrefix v2 patch appended `reserve_tier_image` below `mod tests` in `crates/memra-server/src/admit_memory.rs` | moved the item above the test module (mechanical; B-owned file, lead-applied) |
| publish-dryrun | `tier-transfer-gate` path-included `memra-tier/tests/contracts/conformance.rs`; the packaged memra-engine tarball has no sibling crate's `tests/` | promoted the schedules to **`memra_tier::conformance`**, an unconditional public module: the contract crate ships the suite every backend must run (a feature + self dev-dependency variant was tried first and refused by `workspace-publish-census.sh`, which orders only `[dependencies]`; a public module needs neither). Every path include of `conformance.rs` / `revision_v1{1,2,3}.rs` (tier bank/peer/storage/contracts tests, kv lib tests + `contracts_v12`, the native gate) now uses the library module. Same code, same schedules; no test lost (253/253). |

Test-only cross-crate `#[path]` includes remain (`storage_bench.rs`, `banked_residency.rs`, `ple_rows_tier.rs`, `support.rs`) — they live in test targets, which `cargo publish` verify does not build.

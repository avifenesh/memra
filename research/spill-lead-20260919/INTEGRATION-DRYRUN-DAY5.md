# Generic spill integration dry-run — day 5

Repository **avifenesh/memra**. Date: 2026-09-19. Lane E owns this report only;
A/B/C/D remain implementation owners. **CPU union PASS; native qualification
UNRUN.** No merge to main, scratch push, release, deployment or support promotion.

## Exact input and union identities

Fetched origin immediately before constructing the scratch worktree. An earlier
fetch timed out connecting to GitHub; a bounded retry succeeded before any push
or dry-run claim. The following are the refreshed refs used, not moving labels:

| Input | Exact commit |
|---|---|
| E v1.2 schedules + docs + lane receipt | `fa22213526af78e97a33fcb417aff37e57dc64cf` |
| origin/lane/spill-a-20260919 | `7f104abf4f06ad2f4258f0b4c74c366058d6a789` |
| origin/lane/spill-b-20260919 | `fcb10942519cb485386eb15c2c3de79978f8ca62` |
| origin/lane/spill-c-20260919 | `082727f389ff1c0e601b8ef5ce061c8750cab692` |
| origin/lane/spill-d-20260919 | `645602c83ded280bae830a1647e002e8520a8773` |
| Scratch checked HEAD | `11c12dcc4186c9fba02ee00d6f9425e7d6c03348` |
| Scratch tree | `8a35903daa026220f6d4b40f11fa64f373735385` |

Used a separate `lane/spill-e-dryrun-20260919` branch/worktree. Merged A → B → C → D
with `git merge --no-ff` (ort), starting at E's input above. **All four merges were
conflict-free.** No manual resolution, owner-file replacement, additive module/test
union repair, or runtime edit was needed. `integration-day5/union.json` retains
the actual full parent graph, tree identity and clean post-check status. Scratch
commit is deliberately not pushed; reconstruct the same tree from the input SHAs
in this order (commit timestamps/SHAs of a new replay need not match).

## Executed checks

Reproducer from the union checkout:
`python3 research/spill-lead-20260919/verify-v12.py --out <new-directory>`.
Raw combined stdout/stderr is losslessly retained under `integration-day5/*.log.gz`;
`checks.json` records exact argv, UTC starts, exits and raw/archive SHA-256s.
`source-manifest.json` binds all tier/KV Rust sources; `changed-source-manifest.json`
also binds the union's changed crate/tool files, including unbuilt native slices.

| Check | Actually ran | Result |
|---|---|---|
| cargo fmt --all -- --check | Yes | PASS |
| tier + KV all-targets offline check, macOS | Yes | PASS |
| same check, x86_64-unknown-linux-gnu | Yes | PASS, cross-check only |
| cargo test -p memra-tier -p memra-kv --offline --no-fail-fast | Yes | **243 passed; 0 failed/ignored** |
| tier + KV Clippy --all-targets --no-deps -- -D warnings | Yes | PASS |
| git diff --check | Yes | PASS |
| bash tools/check-flags.sh | Yes | PASS, 864 runtime names; no new reads |
| bash tools/docs-registry-census.sh | Yes | PASS, 122 kernel-file refs, router 40/60 lines |
| independent canonical wire/payload fixture pins | Yes | PASS, all eleven |
| frozen contracts/fixtures/root manifest/lock diff vs b3487a03 | Yes | PASS, byte-unchanged |
| python3 -B -m unittest discover -s crates/memra-tier/tests/battery -p 'test_*.py' | Yes | **52 passed**, collector/receipt/bootstrap stubs; not GPU execution |
| engine/server build, nvcc, native CUDA, model/serving batteries | **No** | No GPU/nvcc in this lane; not inferred from CPU green |
| Linux execution, direct-I/O, physical NVMe, PRO pair/four-card | **No** | Remain separate qualification gates |

Rust test counts: KV unit 60, new v1.2 KV integration 2; tier unit 2, bank 46,
contracts 52, peer 18, placement 6, storage 53, compile-fail doctests 4. Compared
to E alone (239), the union adds B's one seam test and C's three native-source
bridge CPU tests. No existing schedule was weakened or skipped.

## Conflicts, failures and remaining owner seams

No merge conflict or requested CPU check failure occurred. The pre-existing
macOS dependency warning at `crates/memra-gguf/src/source.rs:20` (unused AsRawFd)
remains; requested no-deps package Clippy passes. It is not suppressed or repaired
outside lane scope.

CPU pass does **not** make this raw lane union a standalone engine-build-ready tip:

| Surface / precise source | Owner / disposition |
|---|---|
| `research/spill-c-20260919/DAY5.md:35–47` requires `mod ple_rows_tier;` and `mod banked_residency;` in engine `src/lib.rs`; neither is present in the union. | Lead integration, with C's native ownership. C's gate hook refers to the new module; apply the owned fragment before an engine build. This is a source-inspected missing seam, not a captured compiler failure. |
| `research/spill-b-20260919/GATE-BINARY.md:7–16` requires the named `kv-tier-gate` Cargo bin stanza. | Lead integration, with B. Without it Cargo auto-discovery uses `kv_tier_gate`. No executable rename silently performed in dry-run. |
| B's HostPrefix patch remains a separately qualified/applied patch, not activated by merging the lane. | B + lead: exact patch/source review and native execution remain separate. Existing legacy-prefix evidence does not prove generic active-tier materialization. |
| C's gate-only row seam and D/A collection handoffs have separate receipts. | C/D/A own them. Imported historical native logs were not independently rerun or promoted here. |

No `.cu` or FFI file changed in the E work or this union delta. No kernel inventory
amendment, performance board update, format substitution, io_uring implementation,
new-model work or runtime default promotion follows from this report.

## Cleanup and handoff

The user explicitly requested abandoning the scratch after reporting. Its checked
worktree was clean; the scratch worktree and local branch are removed after saving
these receipts. No scratch branch is pushed. The E delivery branch stays available
for lead integration; only E's docs, schedules and receipts are committed there,
not this A/B/C/D union. Final remote SHA and cleanup read-back are in the handoff.

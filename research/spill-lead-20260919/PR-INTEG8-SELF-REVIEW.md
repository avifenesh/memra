# integ8 self-review (lead, 2026-09-20)

Read in full: the crate diff vs `main` `30e9c7a38` (A day 10: `memra-tier/tests/storage/{mod,day4,telemetry}.rs`; C day 11: `memra-gguf/src/expert_banks.rs`, `banked_residency.rs`, `banked_residency/native.rs`, `moe_cache.rs`, `lib.rs`, `run_gen.rs`, `run_spec.rs`, `tests/bank/day10.rs`; B day 11 and 12: `kv_tier_gate/cli.rs`, `kv_tier_gate/reclaim_contract.rs`,
`kv_tier_gate/active.rs`, `bin/kv_tier_gate.rs`, `memra-tier/tests/reclaim/{main,day11}.rs`, `memra-tier/Cargo.toml`).
Docs and receipts spot-checked against `verify-day11.py`.

## Findings
1. **Serving path untouched.** Every change is inside the `kv-tier-gate` binary and the tier test crate. No
   memra-server or engine library symbol changes; `Cache`, `KvAllocator`, `KvPlane` untouched.
2. **CLI fails closed.** `--reclaim-cycles` needs an ASCII integer >= 2 (`+3`, `3 `, `0x10`, `2.5`, empty, a flag as
   value all refuse with one `REFUSED:` line, junk never echoed), refuses a duplicate, refuses without
   `--reclaim-diagnostic`, refuses a pooled allocator (explicit or default), a baseline case and a second storage
   route. One cycle is the existing single roundtrip, not a series.
3. **Classification matches ruling 5 and never promotes.** `classify_cycles`: granule 0 is `not-applicable-pooled`;
   fewer than two cycles is `unclassified`; the one-time class needs every cycle `bounded_no_leak`, `free_before`
   identical to the first cycle (drift 0) and residual exactly one granule; all-zero residuals with the same
   steadiness is `none`; monotone growth of either the residual series or the unreturned-against-first-baseline series
   is `growing-residual`; everything else `unclassified`. The per-cycle G1 line
   `reclaimed = vmm_granularity != 0 && reclaim_observed && observation.residual == 0` is unchanged and the series
   `g1_reclaim_qualified` is the AND of the cycles, so the day-11 receipts print `false` with the class. Unit tests
   cover the day-10 PRO bytes for 8k and 32k, N=2..6, growing, leaking, drifting, exact-then-short.
4. **Gate loop.** Each cycle writes its full receipt under `cycle-<k>/`, must restore the prefix bit-identically
   before the next cycle (error names the cycle), and the last cycle's `restored-prefix-state.tsv` is copied to the
   run root so the existing verifiers keep their paths. `write_cycles` emits `reclaim-cycles.tsv` (one row per cycle
   with reclaimed, reacquired, residual, unreturned, drift, class, per-cycle G1) and `reclaim-cycles.txt`
   (`residual_series_class`, series bytes, drift, `g1_reclaim_qualified`). `fs::create_dir` (not `_all`) refuses a
   reused cycle directory, which is right for a fresh collector out dir.
5. **Nits (not blocking).** `Roundtrip.g1_reclaim_qualified: String` duplicates the boolean `reclaimed` in text form
   (kept for the receipt line; fine). `granularity` for the series is the max over cycles; every cycle reports the same
   value today, a mixed series would be a driver anomaly worth a refusal rather than a max (note for day 12).
   `USAGE` line is now long; the gate prints it on a parse error only.
6. **C, plan-derived catalog.** `expert_bank_catalog` refuses a non-GGUF dialect, a plan without MoE projections, a
   missing or ambiguous contract entry (including a requirement whose match mode is not `OneOf`), and binds through the
   contract so shape, layout and census errors are the contract's own text; scale planes the consumer does not declare
   refuse. `ExpertBankCatalog::identity` is the plan's block, checkpoint layer, projection, bound name, shape, storage
   and bytes, hashed as `catalog_sha256`. In `native.rs` the installer walks `catalog.blocks()` (three per block, same
   block id, else `InvalidLayout`), maps trunk blocks by position and refuses a dense loaded layer where the plan
   routes experts; an MTP MoE block with no loaded head is skipped (`continue`), depth > 0 with a head refuses. The
   skip is right for a run without the head, and the `installed` line carries the banked count, so a reader sees it.
   Typed refusals go through `ExpertBankRefusal`, so they keep the `REFUSED:` exit-2 contract. No name is spelled in
   the installer; the day-10 literal spelling survives only in `verify-day11.py` as the parity oracle.
7. **C, nits.** `expert_bank_cli` now parses `key=value` once, matches `--experts-via-tier`, `--expert-bank-host-bytes`,
   `--expert-bank-gpu-bytes` exactly, and turns any other `--expert-bank-*` or `--experts-via-tier*` spelling into a
   usage error (six look-alikes tested); a bare door flag with a value is an error. `SLOT_TAIL_PAD_BYTES = 8` lives in
   `banked_residency.rs` because the tier bank tests compile that file verbatim; `moe_cache.rs` imports it.
   `#[doc(hidden)] pub mod banked_residency`; the gate binaries import through the module path.
8. **A, storage test fence.** `tests/storage/mod.rs` adds a process-wide `RwLock`: `OwnedDirectory::new()` holds it
   shared for the directory's lifetime, `OwnedDirectory::spawning()` / `Fence::exclusive()` hold it exclusively around
   the four tests that spawn a child. This serialises only spawn against lock re-acquisition; no engine file, no
   frozen schedule, no assertion changed. The strace evidence names the exact `clone3`/`close`/`flock EAGAIN` window.
9. **Boundary scanner parity.** The checkout scan prefilters with `git grep --text -P` over raw bytes; the commit and
   ref scans did not, and scanned ignore-decoded text where compressed bytes glued into a provider name the raw-byte
   walker never saw. `raw_bytes_prefilter` gives the commit and ref scans the same first question (rule sources
   compiled as bytes patterns, fail open into the text scan if one will not compile; all shipped rules compile), so
   both halves judge one candidate set. `scan_secret_bytes` itself is unchanged, so no blob already on `main` changes
   status (an `errors="replace"` attempt was rejected because it surfaced four `final-logits.f32le.gz` receipts).
   Existing `CommitBlobTests` unchanged and green. The two allowlist pins that existed only for the artefact are
   removed. Gate tooling change inside an integration PR: flagged in the body for the owner.
10. **B day 12, series verdict.** `series_verdict` is the only source of the classified label; it requires N >= 5,
   the one-time class, every cycle `bounded_no_leak`, restore hash equal to the suspended hash in every cycle, and
   `free_before` identical to the first cycle; an exact series is true without a label; pooled stays
   `not-applicable-pooled`. The per-cycle `reclaimed` line is untouched. Unit test covers N=4, one drifting cycle,
   growing, other classes, differing restore, exact, pooled; `day12.rs` replays both cards' committed bytes with
   mutations. Matches ruling 6 word for word, plus the stated tightening (exact cycles over a drifting baseline are
   `false`), which is stricter, not looser.

## Verification this review relied on
CPU battery on the integ8 tree (`integration-day11/integ8-cpu-battery/`): fmt, `cargo test -p memra-tier -p memra-kv -p memra-gguf`,
clippy `-D warnings` (engine, server, tier, kv, gguf, all-targets, Linux target, `DOCS_RS=1`), check-flags, publish
census, docs registry census, collector pytest, perf board, `git diff --check`. B's own batteries: 276 tier/kv tests on
both rigs, `test-day11.py`, `verify-day11.py --require-complete`, receipts `--validate` on both cards. This rig cannot
run the model gates (memory `local-ci-skips-on-linux-rig`); the GPU evidence is the two-card cycle series itself.

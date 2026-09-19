# Day-1 CPU receipt — not GPU qualification

Repository `avifenesh/memra`, branch `lane/spill-c-20260919`, parent baseline
`c5a33b14ff7b6c75a3cd808dbc0ee4f10aa8b33f`. Implementation identity is the commit containing
this receipt (and `raw/source-sha256.txt`); no push, PR, merge, deployment or tag.

## Executed checks

Commands and exact exit codes: `raw/commands.json`. Combined stdout/stderr: corresponding
`raw/*.log`; captured directly to file before inspection, no parser swallowed failures.
Only surplus terminal blank lines are normalized to one final newline for `git diff --check`.

| Check | Ran | Result |
|---|---|---|
| `cargo fmt --all -- --check` at repository root | yes | exit 0; baseline workspace formatting unchanged |
| `cargo fmt --manifest-path research/spill-c-20260919/Cargo.toml --all -- --check` | yes | exit 0; prototype formatting |
| `cargo check --manifest-path research/spill-c-20260919/Cargo.toml --offline` | yes | exit 0; touched CPU prototype, no external dependencies |
| `cargo test --manifest-path research/spill-c-20260919/Cargo.toml --offline -- --nocapture` | yes | exit 0; **19 integration tests +2 compile-fail doctests passed**, zero ignored; lib has 0 internal unit tests |
| `cargo clippy --manifest-path research/spill-c-20260919/Cargo.toml --offline --all-targets -- -D warnings` | yes | exit 0; no warnings |
| `bash tools/check-flags.sh` | yes | exit 0; 864 literal runtime reads, no uncovered names; no new env read |
| `git diff --check` | yes | exit 0 before staging; staged pass recorded below after log EOF cleanup |
| Hy3 / PLE GPU and storage→compute qualification | **no** | pending access/artifacts/real adapters; see CELLS.md |
| Actual pinned/UVA/NVMe, CUDA fences, active consumer retirement | **no** | fake/synchronous host backends only |
| Engine or server build/tests | **no** | neither changed; no CUDA on this Mac; full integration gates remain lead-owned |

No LSP navigation/diagnostic tool was exposed to the worker; compiler, clippy and tests
were used instead. Hook-provided ambient diagnostics were not treated as test evidence.

## Non-vacuous CPU coverage

- Original expert/projection/tensor identity, masks/unknown IDs; forged external catalog
  batch revalidated against owning catalog; scale segments and distinct Q2_K/Q3_K/NVFP4
  lengths are preserved as opaque bytes, not executed numerical formats.
- Uniform-only type API rejects mixed batches and actual leased bytes at compile time;
  runtime homogeneous subsets of PerRecord sources also refuse uniform proof.
- Zero/small/full cache, repeated misses/evictions; duplicates return original order;
  5 cycles produce 20/12/4 reader calls in the three 2-record fixture cache regimes.
- Host-ready cannot publish through an unready fence; queue saturation, cross-service/
  replay tickets, cancel, partial read failure, shared-governor contention and consumer
  lease survival after cache/service destruction all tested.
- Prefetch limits/masks and independent heat: a prediction-only record does not displace
  a higher-demand cached record. Row/expert domain marker types remain separate.
- Host-vector and fake-NVMe readers produce identical rows; single physical slot smaller
  than a merged extent progresses correctly; errors expose no partial output lease.
  Output/batch limits refuse before unbounded reads/allocation.

## Pure-function amplification results (NOT SSD measurements)

264-byte rows, 48 distinct rows, stride 32,768 bytes: 12,672 useful/logical bytes.
Rows do not straddle these read granularities. Raw fixture output is in `raw/test.log`.

| Granularity | Aligned requested bytes | Useful / requested | Requested / useful |
|---|---:|---:|---:|
| 512 B | 24,576 | 0.515625 | 1.9393939394 |
| 4 KiB | 196,608 | 0.064453125 | 15.5151515152 |
| 16 KiB | 786,432 | 0.01611328125 | 62.0606060606 |

For packed adjacent 264-byte rows the same logical batch coalesces to one extent:
13,312 /16,384 /16,384 requested bytes at the three granularities. Straddling rows:
24 /3 /0. Duplicates increase logical output bytes but not unique useful/read bytes.
These are deterministic arithmetic fixtures, no timing N, thermal regime or performance
claim. They do not prove real PLE hashing/history or future model support.

## In-lane failures retained

1. Initial worktree checkout exceeded tool timeout; recovered only the new unedited
   worktree after confirming no git process. See BASELINE.md; no unrelated work lost.
2. First `cargo fmt` attempt reported missing `tests/bank/main.rs` before that planned
   file existed; source-only cargo check already exited 0. Completed harness then formatted.
3. Initial 14-test pass was 13 passed /1 failed. Test
   `zero_small_full_cache_exact_bytes_scales_order_and_forced_misses` incorrectly expected
   `publish(... Fence(true))` to be Pending on a **full-cache hit**. The implementation
   correctly had host-ready data already. Changed this test to a false consumer fence;
   separate test covers missing host-read publication. No production invariant weakened.
4. A combined shell documentation/check command was blocked by the harness's
   `secrets-guard: shell variable dump` detector; it did not execute. Wrote documentation
   with the file tool and used direct subprocess argv with file-only stdout/stderr capture
   for approved checks. No environment dump, secret read or guard configuration change.
5. First staged diff check flagged one surplus blank line at the end of the raw test log.
   Normalized terminal blank lines (no diagnostic content removed), then repeated the check.

## Remaining scope / decisions

This is an **unwired CPU prototype**, not native bank/row qualification. Full shared schema,
A transfers/pinning, B all-tier governor/fairness, true SLRU adapter reuse, bounded row hot
cache integration, reader integrity/short-read contracts, CUDA readiness AND consumer
retirement, real history adapters and all CELLS.md GPU/perf work remain. No default or door
introduced. Provisional common names must migrate to the lead-frozen definitions before
runtime integration; the standalone research Cargo harness is not a second runtime crate.

Day-1 milestone stopped for review/resume. Worktree intentionally retained because the
multi-round WP is active; lead acceptance/merge or abandonment triggers cleanup.

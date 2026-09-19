> Day-1 record retained below. See the appended **Day-2 CPU milestone** for current scope and receipts.

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

# Day-2 CPU milestone — frozen interface migration

Repository **avifenesh/memra**, branch **lane/spill-c-20260919**.
Executed source tip: **7f28f4f1eabe4b6fb86e39c849f8fdf6b8f01b0a**.
The subsequent receipt/documentation commit does not change source or fixtures.

Commits:
- `0f52f63c`: merge accepted lead `259ff819` into this lane, **no conflicts**.
- `bec64650`: frozen bank/row lifecycle plus typed HostExps/PLE metadata adapters;
  retire standalone research Cargo manifest/lock and remove its build scratch.
- `7f28f4f1`: reject ambiguous original tensor/record assignments even when layout
  digests differ, preventing an alternative layout from resurrecting a masked ID.

**Feature-lane commits only. Not merged to main, pushed, deployed, tagged, GPU
qualified, or permission to start model work.** Worktree remains active for the
next WP-C round; do not remove it before lead acceptance/closure.

## What actually runs

`crates/memra-tier/src/bank/` implements the frozen `BankedResidency` and
`RowService` traits as a bounded **host-only** service. It imports shared IDs,
layouts, tickets, Completion, lease/proof, domain/hotness/predictor and governor
contracts; no duplicate authoritative contract structs remain.

Catalog validation binds canonical JSON v1 layout identities, immutable tensor
names/artifact, original expert/projection/row IDs, masks, and per-plane expected
checksums. Publication checks actual read bytes, padding and every completion
segment, preserves logical duplicates/order, and keeps failed/refused resources
for explicit retirement. A rejected publication returns its original charge and
backing; the service stores them instead of dropping the failure payload.

One injected frozen governor charges output, resident metadata estimates, bounded
staging and ticket metadata. Cache aliases do not duplicate the physical charge.
Cancellation revokes publication, not lifetime; borrowed views refuse Busy; all
shared-hit consumer tickets must retire before release. Tombstone quota remains
until acknowledge. A partial row-release Busy retains every retry handle.

**Scope boundary:** stage/gather currently performs synchronous bounded host reads;
publication/retirement is ticket-based and separate, but this is not A's asynchronous
worker/TransferEngine. Host completion is never GPU readiness. Device/pinned
allocation requests explicitly refuse. Native A ReadyView/fences/SLRU/GPU and
B production governor integration remain pending.

HostExps bridge and PLE trace details, including exact untouched dispatch sites,
are in **HOSTEXPS-ADAPTER.md**. The unexported engine bridge is compiled only against
an API-shaped CPU fixture; no actual CUDA engine type/build was checked. The
synthetic PLE trace is reproduced by the existing native full/cached host ID
functions in a source-drift-checked test-only capture, then replayed for unchanged
F32/BF16 row expansion. No PLE model or GPU projection ran.

## Verification on the exact source tip

Reproducer: `python3 research/spill-c-20260919/verify-day2.py`.
Machine-readable command/exit/timestamp/source SHA256 receipt:
`raw/day2/verification.json`. Full stdout+stderr: `raw/day2/verify-1.log` through
`verify-7.log`. Logs are captured before inspection; only surplus terminal blank
lines are normalized for whitespace checks. No test command ran over the network.

| Command | Actual result |
|---|---|
| `cargo fmt --all -- --check` | exit 0, no diagnostics |
| `cargo check -p memra-tier --offline --all-targets` | exit 0; `Finished dev profile [unoptimized + debuginfo] target(s) in 0.60s` (Cargo formats `dev` with backticks in the raw log) |
| `cargo test -p memra-tier --offline` | exit 0; **27 bank tests +36 unchanged frozen tests +4 compile-fail doctests**, 0 failures/ignored |
| `cargo clippy -p memra-tier --offline --all-targets -- -D warnings` | exit 0; no warnings, finished in 0.13s |
| `git diff --check` | exit 0, no diagnostics |
| `bash tools/check-flags.sh` | exit 0; `check-flags: runtime literal reads=864`; `check-flags: no uncovered runtime names` |
| `python3 crates/memra-tier/tests/contracts/fixture_reference.py --check` | exit 0; `tier-contract-v1: 8 payload + 3 canonical wire SHA256 pins match` |
| Compare frozen source/tests and root manifests against `259ff819` | `git diff 259ff819 -- crates/memra-tier/src/contracts.rs crates/memra-tier/tests/contracts Cargo.toml Cargo.lock`: empty |
| Real engine/nvcc/GPU/model/performance gates | **not run**, no GPU/nvcc and no permitted reachable rig/artifact bundle |

The exact **lead conformance schedules**, not copied expected results, are path
imported from `tests/contracts/conformance.rs`: `bank_cancel` and `rows_order` both
run on the WP-C implementation. Tests drain their host lifecycle and return all
charges. Additional tests cover native-code qtype pins, multi-plane completeness,
all three epochs, corruption/short reads, failed completion cardinality, cache
pressure, actual byte hashes and duplicate order, masked-ID ambiguity, uniform
proof refusal, release retries, tiny slots, alignment/overflow/tail refusal, and
prefetch that never adds demand heat.

### Development reds / review fixes retained

- Initial `cargo check --all-targets` ran before the planned synthetic fixture file
  existed: exit 101, `couldn't read ... ple-ngram-synthetic.json: No such file or
  directory`. Fixture creation resolved it; this startup diagnostic is in the
  session tool transcript rather than a raw file. It was not a successful check.
- Strict Clippy caught `needless_range_loop` in the **verbatim native test oracle**.
  `raw/day2/clippy-development.log` retains the failure. Added a narrow test-only
  lint allowance to preserve native operation/index order; no production lint
  allowance or numerical rewrite was introduced.
- Code review added retry-safe partial row release, charge retention for resident
  metadata and unacknowledged tombstones, all-segment failure outcomes, and the
  original-ID ambiguity red. Passing intermediate test logs remain in `raw/day2/`.

## Coalescing and CELLS changes

Portable policy is 512-byte requested granularity /one 4KiB slot. This is chosen
from the **existing arithmetic** amplification table, not SSD timing: 264B×48
sparse rows request 24,576/196,608/786,432 bytes at 512B/4KiB/16KiB. Packed adjacency
requests 13,312/16,384/16,384 bytes. Actual backend alignment overrides the portable
policy; no runtime flag or hardware performance default was added.

CELLS.md now records C1-contract/HostExps, C2-contract/synthetic-ngram, C3-policy
CPU results separately from every still-pending native/model/performance cell.
The full non-serving PRO qualification remains blocking; a synthetic trace or a
CPU compile-fail test is not substituted for it. No board numbers changed.

## Access, blockers and budget

One read-only SSH inventory attempt this turn, with BatchMode, ConnectTimeout=8,
ConnectionAttempts=1, returned **255**: `Connection closed by UNKNOWN port 65535`.
`raw/day2/ssh-inventory.log` retains the error. No successful remote inventory,
model load, write, installation, serving-instance action, or second attempt.
No credentials/env files read. No new environment read, CUDA kernel, lock name,
external runtime dependency, paused-model execution or numerical program.

Remaining blockers, in order:
1. **Lead/A:** actual async ObjectStore/TransferEngine + owner ReadyView, native
   pinned/UVA/device allocations and producer/consumer/graph retirement. Host
   lifecycle completion cannot authorize GPU compute.
2. **Lead/B:** same production BudgetGovernor instance, scheduler fairness,
   loader/catalog ownership and measured allocator/RSS overhead. C has no private
   production governor. Current metadata accounting is a conservative estimate,
   not measured allocator consumption.
3. **Lead/C:** engine dependency/export wiring, real immutable HostExps source
   registration, native SLRU adoption, lazy bounded row-catalog/hot-cache integration
   and all untouched dispatch changes listed in HOSTEXPS-ADAPTER.md.
4. **Lead/rig owner:** reachable designated non-serving GPU rig and pinned Hy3/PLE
   artifacts; then queued actual byte/logit/token/serving/performance gates.

Effort: approximately **0.3 agent-hours for this resume** (re-entry included;
wall-clock approximation, not CPU/GPU measurement). WP-C budget remains **8
agent-days**; day-1 agent-hour consumption was not recorded, so no invented
cumulative burn or completion percentage. Day-2 CPU milestone stops here rather
than spending the two-hour cap on unauthorized hardware/runtime work.

Lead-owned documentation fragments: add a TESTING entry for
`cargo test -p memra-tier --test bank` and an INDEX outcome of “frozen-contract
host bank/row lifecycle and CPU metadata adapters; native Hy3/PLE gates pending.”
No FLAGS/KERNELS or performance-board amendment is needed for this CPU slice.

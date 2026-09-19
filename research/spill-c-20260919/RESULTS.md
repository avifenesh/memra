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


# Day-3 CPU integration milestone — partial native patch, not qualification

Repository **avifenesh/memra**, active worktree `/Users/avifen/tiyuvta/wt-spill-c`,
branch **lane/spill-c-20260919**. Exact tested Rust source:
**3e2972fb5a133fdd1c80ec17606820d2c586e47e**. Subsequent receipt/documentation commit
contains no Rust changes. No push, PR, main merge, tag, deployment or GPU execution.

## Commits and isolation

- `0192d08244612bf430e596e4cfdf419ab1a75e64`: requested `--no-ff` merge of
  `lane/spill-integ-20260919` at `98e558dc03a0d999cf38db7aa8f27c1a1a4b12fb`.
  **No conflicts**, no resolution changes; frozen interfaces preserved.
- `b1df0e73335f61e3e82cc8508ffb5aeab0c30123`: ticketed CPU object transfers,
  bounded progress, prediction hooks and integration tests.
- `0e7e388916cb9db1836796e05d9b7f255b0b888c`: cancellation outcomes are Cancelled,
  not misreported as failed reads.
- `3e2972fb5a133fdd1c80ec17606820d2c586e47e`: current priority/deadline/tenant
  forwarded into A's object lease admission; checked allocation refusal releases
  pre-reserved bank charges.

No changes relative to `98e558dc` in engine, root Cargo manifests/lock, frozen
contracts, A substrate or B governor. Only C bank sources/tests and C research
namespace changed after that merge. Existing applied engine/server additions are
inherited from the explicitly requested integration merge, not newly compiled or
qualified by C. No unrelated dirty work was found or staged.

## Delivered CPU behavior

`bank/{residency,rows,types,transport,prediction,mod}.rs` and
`tests/bank/{main,integration}.rs`:

- Stage/gather validate, reserve and enqueue without reading. Publication does
  not hide synchronous reads. Explicit `progress` performs a bounded chunk and
  leaves incomplete output private; producer terminal is never GPU-ready.
- ObjectReader uses **A ExtentStore/ObjectStore → CpuTransfers**. Every accepted
  read ticket is driven, its per-item outcome checked, the host destination
  consumed/dropped, `retired` checked and acknowledge called. Forced expert and
  row misses, scales, duplicate/order output, partial second-chunk corruption,
  and between-chunk cancellation run non-vacuously. No partial bank publishes.
- **B's real `tier::Governor`** is shared across C allocations, A object leases,
  fake pinned-pool backing and transfer queue. No C production governor exists.
  Mandatory headroom survives optional ticket/quota saturation. In particular,
  saturating optional NVMe capacity produces a Demand refusal and an admitted
  MandatoryActive read through the same A adapter, proving priority propagation
  reaches storage rather than stopping at the bank's output allocation.
- RouterTopKHint and NgramLookaheadHint have distinct domains, byte/item/scan
  caps and deduplication. Hints preserve original IDs and never add demand heat
  or mutate router/history; one bank ticket remains reserved for non-prefetch.
- Source pool allocation is startup charged. Bank output/working slot allocation
  refuses capacity rather than silently substituting bytes. Full native RSS and
  eager catalog/source metadata accounting remains a separate qualification task.

## Exact checks actually run

Reproducer: `python3 research/spill-c-20260919/verify-day3.py`.
Machine-readable exact argv/exit/source SHA256/log SHA256 receipts:
`raw/day3/verification.json`. Full output: `raw/day3/verify-1.log` … `verify-13.log`.
No diagnostic parser preceded raw capture. Terminal surplus newlines only were
normalized for whitespace checks.

| Command/check | Actual output/result |
|---|---|
| `cargo fmt --all -- --check` | exit 0; no diagnostics |
| `cargo check -p memra-tier --offline --all-targets` (Mac) | exit 0; `Finished dev profile [unoptimized + debuginfo] target(s) in 0.89s` (raw Cargo log includes backticks around dev) |
| Same check plus `--target x86_64-unknown-linux-gnu` | exit 0; `Finished dev profile [unoptimized + debuginfo] target(s) in 0.91s`; **cross-target type check, not Linux execution/linking** |
| `cargo test -p memra-tier --offline` | exit 0; bank **31**, contracts **36**, peer **10**, placement **6**, storage **26**, compile-fail doctests **4**; **113 total**, zero failed/ignored |
| `cargo clippy -p memra-tier --offline --all-targets -- -D warnings` | exit 0; no warnings; finished in 6.15s |
| `git diff --check`, `git diff --cached --check` | exit 0, no diagnostics |
| `bash tools/check-flags.sh` | exit 0; `check-flags: runtime literal reads=864`; `check-flags: no uncovered runtime names` |
| Frozen canonical fixture reference | exit 0; `tier-contract-v1: 8 payload + 3 canonical wire SHA256 pins match` |
| `bash -n research/spill-c-20260919/rig-cells-c.sh` | exit 0 |
| Runner tests | exit 0; 5090/PRO dry-run branches, repeat invocation, injected exit 23 and non-serving refusal PASS; no GPU commands executed |
| `trace-policy.py --check` | exit 0; six synthetic aggregate table rows reproduced; raw values in verify-12.log |
| `git apply --check …/HY3-DISPATCH-PATCH.diff` | exit 0; patch remains unapplied |

Frozen row conformance assumes immediate publication. Its unchanged schedule is
now run through a **test-only explicit drive wrapper**; production publish does
not secretly drive I/O. No frozen schedule or contract was edited.

## Native patch, trace and runner disposition

- **Task 3 incomplete:** `HY3-DISPATCH-PATCH.diff` is a partial original-mask/
  bounds guard patch only. `PATCH-REVIEW.md` reviews every hunk and every missing
  native callsite, gives exact before/after rig commands and returns **NO-GO** for
  the complete conversion. It does not wire staged/SLRU/grouped dispatch through
  BankedResidency, nor make existing fused kernels take UniformLease.
- No replayable real PLE trace found in the bounded/streaming local research
  search; actual gate summaries are not traces. Existing fixture remains labeled
  **synthetic**. New `fixtures/ple-trace-policy.json` and `TRACE-AUDIT.md` justify
  the portable 512-byte request granularity arithmetically. No SSD/default claim.
- `rig-cells-c.sh` builds outside the canonical lock, refuses non-confirmed/
  occupied rigs, runs tiny PLE first, marks Hy3 PRO-pair scope and records every
  missing adapter cell as BLOCKED. Fresh per-run directories and tee-first JSONL
  retain dry-run success and injected failure logs under `raw/stub-*`. Real mode
  returns **3** while native targets are missing; it cannot pass vacuously.
- Existing `docs/TESTING.md` contains no named Hy3 spill-gate section. Review
  points to actual model loader and kernel-check tests, without inventing a
  registry entry. Kernel-check is GGUF-only; native directory artifacts work in
  run-gen/run-spec and must not be substituted to make kernel-check green.

## Development findings / reds

The initial transport compile used incorrect Epochs field names; fixed to frozen
`state/src_gen/dst_gen`. An integration test needed its explicit ExpertDomain
annotation; Clippy found one collapsible test conditional. Patch generation
initially assumed private rather than pub(crate) grouped function visibility.
Those development commands failed and were corrected; their diagnostics are in
the session transcript, not claimed as successful checks. All final receipts
above ran afterward. Runner's intentional exit-23 red is retained as a raw log.

Review also found that A's CpuTransfers fixes its lease request at construction:
without forwarding current scheduling identity, mandatory storage reads could
be charged as Demand. The C-local ObjectStore decorator fixes that without
changing A/shared files, and the quota-saturation red/green pair tests it.

## Numbered blockers / next owner work

1. **C + lead/native owner:** complete loader source installation, SLRU-backed
   BankedResidency and actual UniformLease-only fused dispatch. The supplied
   patch is deliberately partial, not a ready implementation.
2. **A + native owner:** real asynchronous OS worker and CUDA ReadyView/producer/
   consumer/graph lifetime integration. Current CpuTransfers drive is synchronous;
   cancellation between chunks is not proof of in-flight DMA cancellation.
   Whole-object lookup/lease verification must become bounded/lazy for model-scale
   tables; submitted chunk counters omit that validation traffic.
3. **Rig/artifact owner:** non-serving 5090 fitting fixtures and PRO pair with
   immutable Hy3/PLE artifacts; compile/type-check native changes, implement
   missing GPU targets and run byte/logit/token/serving correctness and balanced
   performance gates. No GPU/nvcc or real Linux runtime check ran on this Mac.
4. **Lead:** install reviewed native module/dependency wiring and TESTING/INDEX
   fragments only with corresponding actual gates. No FLAGS/KERNELS/perf-board
   fragment is needed here: no new env read, kernel or published performance.

Effort this resume: approximately **0.5 agent-hours** at this CPU milestone,
under the requested two-hour cap. WP-C remains an **8 agent-day** budget; day-1
burn was not recorded, so cumulative consumption/completion percentage is unknown.
The active worktree/branch remain for native follow-up, not abandoned or merged.

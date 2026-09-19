# Session D day-2 CPU milestone receipt

Repository **avifenesh/memra**, worktree `wt-spill-d`, branch `lane/spill-d-20260919`.
Final source checked: **`41715fd3cc5ecad97ec6f135cec6436d8e53d5bd`**.
Subsequent receipt-only commit does not alter source, tests, schemas or runner bytes.
**Local committed work only. No push, PR, tag, main integration, deployment, live verification,
CUDA implementation or hardware qualification.** The lane stays open at the day-2 milestone.

## Merge and migration

- `06bbced6`: no-conflict `--no-ff` merge of frozen lead `259ff819`. No resolution discarded
  upstream behavior. Canonical JSON v1 and homogeneous state/source/destination epoch triples
  are preserved without amendments.
- `4cfde7ee`: replaces D's provisional peer schemas with frozen imports/re-exports, sealed
  DeviceOwner handles, unified indexed Completion and common governor charges in CPU fakes;
  migrates PlacementReport/DeviceBudget/RouteBudget to shared versioned metadata. Physical
  replica bytes are a breakdown, not added again. `verify-cpu.sh` now uses the actual workspace.
- `8ed8f390`: collector, schema, placement/topology tools, conformance review, source-drop
  ownership test and scoped direction/context/binary/topology observation expiry.
- `46afe9e1`: bounded raw-log drain including descendants and capture errors.
- `b29d9899`: cell updates and bounded offline receipt generator.
- `45423308`: Rust-2024 placement guard lint fix (no arithmetic change).
- `41715fd3`: monotonic counter checks and byte-delta binding for synthetic telemetry.

Compared directly with the frozen lead: **no diff** in root Cargo.toml/Cargo.lock,
contracts.rs or tests/contracts/. The only shared-file amendments are D's two `pub mod`
lines and its two `[[test]]` entries. No engine, server, CUDA/FFI, flags, numeric program,
performance defaults or other-lane code changed.

## Executed commands and exact result summaries

`day2-checks.jsonl` names exact argv, UTC time, source SHA, exit status, raw gzip path and
uncompressed SHA-256. `raw/day2-check-*.log.gz` retains complete output before parsing.
Reproducer: `python3 -B research/spill-d-20260919/verify-day2.py`.

| Command | Exit / exact result |
|---|---|
| `cargo fmt --all -- --check` | 0; no output |
| `cargo check -p memra-tier --offline --all-targets` | 0; `Finished dev profile` |
| `cargo test -p memra-tier --offline` | 0; `36 passed; 0 failed` shared integration; `10 passed; 0 failed` D peer; `6 passed; 0 failed` D placement; `3 passed; 0 failed` compile-fail doctests; none ignored/filtered |
| `python3 -B -m unittest discover -s crates/memra-tier/tests/battery -p 'test_*.py'` | 0; `Ran 16 tests in 5.128s` / `OK` |
| `python3 -m py_compile tools/tier-battery.py tools/tier-placement.py tools/tier-topology.py` | 0; no output |
| `git diff --check` | 0; no output; staged receipt additions checked separately before commit |
| `bash tools/check-flags.sh` | 0; `check-flags: runtime literal reads=864`; `check-flags: no uncovered runtime names` |
| `python3 -B crates/memra-tier/tests/contracts/fixture_reference.py --check` | 0; `tier-contract-v1: 8 payload + 3 canonical wire SHA256 pins match` |
| `python3 -B tools/tier-battery.py --validate research/spill-d-20260919/day2-dry-run/runs.jsonl` | 0; `BYTE-RECEIPTS MATCH: 22 records / 11 forced pairs; not full battery qualification` |
| `python3 -B tools/tier-battery.py --validate-campaign research/spill-d-20260919/day2-dry-run` | 0; `CPU CAMPAIGN MATCH: 22 runs; synthetic protocol evidence only, NOT GPU qualification` |
| `cargo clippy -p memra-tier --offline --all-targets -- -D warnings` | 0; `Finished dev profile` |

Additional executed CLI cells:

- `python3 -B tools/tier-battery.py --dry-run --out research/spill-d-20260919/day2-dry-run`:
  one OFF/ON control pair before performance, five AB plus five BA pairs interleaved,
  **N=10 per arm**. One injected `ERROR: injected read failure (CPU fixture)` exited 9,
  captured verbatim, excluded from medians. All 22 successful runs have raw logs, state/f32
  logits/u32 tokens, and three virtual 250ms telemetry samples, with file hashes. The fake
  counter delta equals the movement receipt. Clocks/power/temp/VRAM/physical-SSD data remain
  unknown. Timing medians are **synthetic, not measurements**; thermal label is
  `synthetic-no-thermal-measurement`. One canonical lock is held on this Mac, not on a GPU rig.
- `python3 -B tools/tier-placement.py`: `PLACEMENT-TABLES.md` contains both class/grid/frontier
  tables. Equal split first payload overflow N29; unequal N40 at T=1,048,576 with 652 B/record.
  First contexts 1,037,290 / 1,035,194. Every output labels estimates and excluded workspace.
  `python3 -B tools/tier-placement.py --record-bytes 264 --json` produced `placement-264.json`.
  These are opaque arithmetic inputs, never native model/format support.
- `python3 -B tools/tier-topology.py --dry-run crates/memra-tier/tests/battery/topology.fixture.json --out research/spill-d-20260919/topology.dry-run.json`:
  `cpu-fixture: 6 commands; direct P2P remains unqualified`.
- Exactly **one** approved local-development SSH inventory attempt, ConnectTimeout=10,
  BatchMode=yes: exit **255**, stderr exactly `Connection closed by UNKNOWN port 65535`.
  `ssh-attempt.json` keeps UTC/options/verbatim failure, omitting the machine-local alias.
  No remote command execution established; no retry. Per coordinator this expected connection
  failure is **not a day-2 blocker**. No approved PRO-pair address was supplied.
- `command -v nvcc` found no executable. No GPU build or execution was attempted.

## Conformance evidence and gaps

D's own byte-moving fake runs the unchanged generic `peer_cancel` schedule, then checks full
physical retirement/acknowledgement and common-governor release. **No reusable PeerCapacity
schedule exists** in this freeze; we did not invent or claim its execution. D's additional
capacity tests cover the intended cases while the frozen suite runs its own concrete fixture.
`CONFORMANCE-REVIEW.md` reviews all six schedules across eight requested dimensions and
proposes nine extensions. Principal gaps: explicit completion-before-cancel hooks, reusable
capacity/accounting tests, full rejected-item/short vectors, three-epoch and unknown quarantine
schedules, program-identity/missing-plane cases, row byte/dedup assertions and catalog uniform
refusal. Full concrete freeze tests cover many of these, but that does not make them reusable
backend-generic coverage. Lead accepted the gap report for a future freeze revision.

## Observed red checks retained

- Python placement test initially expected 3260 global bytes at T=3; independent explicit
  records `[2,1,3,0] ×652` give **3912**, matching Rust. Corrected the test, not arithmetic.
  Raw failing output: `raw/day2-python-development.log.gz`.
- Extra Clippy check flagged D's inherited nested `if` in placement after moving into the
  frozen Rust-2024 crate. Collapsed the condition; no numeric behavior changed. Full first
  verification attempt including this red result is `day2-attempt1-checks.jsonl` and raw logs.
- Added telemetry delta checks after inspecting the synthetic fixture: previously the final
  fake counter grew to 6 while the movement row said 3. Corrected fixture progression and
  added cross-checks; old synthetic console retained as `raw/day2-dry-run-pre-counter-fix.log.gz`.
  No GPU measurement was involved. Obsolete generated fake bundle directories were scratch,
  replaced with the exact final-source dry fixture; actual test/error logs were retained.
- Final complete check set above is green. Earlier green pre-counter-fix checks remain in
  `day2-attempt2-checks.jsonl`, not presented as final-source evidence.

## CELLS and remaining blockers

CELLS.md now links the CPU collector, telemetry schema, full dry-run/validator, both placement
candidates, topology fixture and failed approved inventory attempt. The official Step FP8 ladder
remains **decode-batch-gate --mode pp/ppspec**; **ppn-gate is GGUF-only / supplementary**.
PP never becomes TP and host bounce never becomes PCIe P2P.

1. **Native integration:** DeviceOwner construction, live route/grant proof, actual allocation
   and CUDA fence/graph retirement adapters remain unimplemented/unqualified. D's fake is test
   code only. A transfer / B production governor / Step-Qwen state consumers still need wiring.
2. **Hardware/model gates:** no CUDA toolchain/GPU/approved PRO address or pinned model access
   for this session. Pair P2P, official Step FP8 PP, Qwen native active KV, integrated four tiers,
   kernel/argmax/spec gates, four-card all-route/fabric pressure and allocation peaks remain
   pending. Failed expected development SSH is not a separate CPU milestone blocker.
3. **Qualification binding:** collector protocol is exercised, but real instrumented counters,
   concurrent-GPU failure capture and serving TTFT/E2E/TPOT/ITL/throughput adapters remain
   pending. Synthetic data cannot fulfill G0–G7 or set a performance/default decision.
4. **Shared fixture follow-up:** lead owns the missing reusable PeerCapacity and other schedule
   extensions. No frozen shared files were amended to work around the gap.

## Hygiene and effort

No new MEMRA environment read: FLAGS/KERNELS fragments are none. No third lock name,
no skip overrides, no serving access, no paused-model/V4.1 code. Created Python caches and
standalone local SSH scratch removed; canonical lock inode deliberately not unlinked. Raw
receipts and synthetic fixture evidence stay in this namespace. Unrelated changes were not
staged. Worktree/branch remain open for lead resume, not a closed or integrated lane.

Day-2 active agent time approximately **0.4 agent-hours**. Day-one actual hours were not in the
handoff; do not invent cumulative consumption. WP-D's **9-agent-day implementation allowance**
remains the program budget; integration-lead overhead belongs to the coordinator, not D.
No hours-per-agent-day conversion or hardware wait cost is inferred.

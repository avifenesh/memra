# Session D day-3 resume — offline tools milestone

Repository **avifenesh/memra**, branch **lane/spill-d-20260919**, worktree `wt-spill-d`.
Checked source: **`fe6a5de772f8cfb9a8f476481514ea45ae34dcae`** (clean at check start).
The following receipt commit adds evidence only; source/test/tool bytes remain unchanged.
**Local lane commits only: no push, PR, tag, main integration, deployment or live verification.**
No GPU, native engine/server compilation, model execution, or hardware qualification occurred.

## Inherited versus finished

Read the entire inherited uncommitted bootstrap, runbook, native-runner tests and dry-run
bundle, plus both modified tool diffs. Nothing was discarded. Stashed all six inherited
paths including untracked files; merged integration **914229ae** using `--no-ff`, then
popped the stash successfully. The temporary stash was dropped by the successful pop.

Local commits:

- **ca974cef827564149aed1c00d749a625534b8602**: merge integration manifest fix. Lead's
  engine memra-tier dependency/storage-bench registration retained; no new shared edits.
- **1420a975**: inherited native subprocess collector, topology/storage inventory and
  Linux bootstrap completed with tests. Added minimum-source ancestry enforcement,
  rejection of unsuccessful nvcc probes and occupied GPU, exact compiler/arch pinning,
  OpenSSL development prerequisite check, failure-source preservation and close-on-exec
  lock wrapper. Expanded stub red tests and CLI exit-status/receipt tests.
- **fe6a5de7**: first-hour runbook, shared registry fragments, bounded verification script,
  original inherited dry run and precommit checks preserved.

Inherited pieces were largely implemented, not empty skeletons. The historical dry-run
bundle predated the inherited script's last edits (missing pp-transport-smoke build target,
installer steps and binary table). It stays unchanged in `day3-bootstrap-dry/`, with a
superseded README; new current-source evidence is `day3-bootstrap-dry-final/`.

Two inherited documentation premises were corrected:

1. Engine dependency/bin wiring is fixed at **914229ae**, confirmed with cargo metadata.
   Server cache types already resolve through `memra_engine::cache`; no direct server
   memra-kv dependency needed. **First native engine + server compile remains blocking**:
   it must compile B's worker field and the registered storage-bench, not only resolve metadata.
2. Linux uncached I/O is implemented, not necessarily Unsupported: the CLI only installs
   `open_uncached` under cfg(macos); Linux reaches `AlignedFile::open(O_DIRECT)` through
   FileBackend. Actual Linux syscall/device/physical-I/O evidence remains **unrun**.

## Exact checks and outputs

Reproduce: `python3 -B research/spill-d-20260919/verify-day3.py --out <new-directory>`.
`day3-final-checks/checks.jsonl` records exact argv, source, clean initial status, UTC,
exit code and uncompressed SHA256; `check-*.log.gz` retain complete output. Independently
rehashed all 14 logs: **14 commands, all exit=0, all hashes match**. Precommit outputs
are separately retained in `day3-precommit-checks/` and not presented as final-source checks.

| Executed check | Exit / exact output summary |
|---|---|
| `cargo fmt --all -- --check` | 0; empty stdout/stderr |
| `cargo check -p memra-tier --target x86_64-unknown-linux-gnu --offline --all-targets` | 0; `Finished dev profile [unoptimized + debuginfo] target(s) in 0.11s` (Cargo quotes dev with backticks in raw) |
| `cargo check -p memra-tier -p memra-kv --offline --all-targets` | 0; `Finished dev profile ... in 0.08s`; existing memra-gguf source.rs:20 `unused import: std::os::fd::AsRawFd` warning preserved, not changed |
| `cargo check -p memra-tier -p memra-kv --target x86_64-unknown-linux-gnu --offline --all-targets` | 0; `Finished dev profile ... in 0.07s` |
| `cargo test -p memra-tier --offline` | 0; bank **27**, contracts **36**, peer **10**, placement **6**, storage **26**, compile-fail doctests **4** passed; all `0 failed; 0 ignored; 0 filtered out` |
| `python3 -B -m unittest discover -s crates/memra-tier/tests/battery -p 'test_*.py'` | 0; `Ran 27 tests in 9.294s` / `OK` |
| `python3 -m py_compile tools/tier-battery.py tools/tier-placement.py tools/tier-topology.py` | 0; no output |
| `bash -n tools/tier-rig-bootstrap.sh` | 0; no output |
| `shellcheck tools/tier-rig-bootstrap.sh` | 0; no output; shellcheck was available and ran |
| `git diff --check` | 0; no output |
| `bash tools/check-flags.sh` | 0; `check-flags: runtime literal reads=864`; `check-flags: no uncovered runtime names` |
| frozen fixture reference `--check` | 0; `tier-contract-v1: 8 payload + 3 canonical wire SHA256 pins match` |
| `python3 -B tools/tier-battery.py --validate-campaign research/spill-d-20260919/day2-dry-run` | 0; `CPU CAMPAIGN MATCH: 22 runs; synthetic protocol evidence only, NOT GPU qualification` |
| offline locked cargo metadata + target/dependency assertions (exact Python argv in JSONL) | 0; `METADATA MATCH: engine tier dependency and all first-hour binary targets registered; NOT native compile` |

The Linux target check is **cross-compilation/type-check evidence only**, not a Linux
executable run, native CUDA build or O_DIRECT execution receipt.

Bootstrap ran through the same stdin shell shape used by SSH, locally with stubs:

```sh
BRANCH=lane/spill-integ-20260919 bash -s -- --dry-run --repo /stub/memra \
  --out research/spill-d-20260919/day3-bootstrap-dry-final < tools/tier-rig-bootstrap.sh
```

Exit **0**, status **dry-run-complete-not-qualified**, **37** stubbed external-command
steps; all raw-log hashes independently matched. `SOURCE.json` binds the actual tool
SHA256s/source commit; BOOTSTRAP's all-ones source commit is the fake remote, not git.
It covers packages, nvcc selection/arch, two 8 GiB full-readback commands and gap, Rust,
branch/source ancestry, 11 inventory commands and all release build commands. No network,
package install, actual sleep, clone, CUDA allocation or native build ran. The canonical
lock is real on this Mac only; it is not a lock held on a GPU rig.

11 new native/bootstrap tests (27 Python total) exercise fake/native argv parity, merged
stderr retention, missing executables/sampler, 250 ms sampler launch/termination, failure
compute snapshots, timeout/unknown cause, malformed RESULT, CLI exit 9 and lock receipt,
non-qualification without RESULT, output collision, main/unset/unknown branch refusal,
low current/max power and NaN, unsupported CUDA architecture, failed nvcc probe despite
version text, old Rust, minimum-source/build failure, both acceptance failures with source
retention, and occupied-device refusal. Stub telemetry is never scored hardware data.

## First-hour order and pending hardware

`RIG-DAY1.md` supplies commands, durations, locks, memory budgets, artifact provenance and
raw destinations. Order: **bootstrap → blocking native engine/server build → A positional
storage-bench → D1 same-device bytes → C tiny PLE → B fitting Qwen baseline**. Existing
self-locking A/B/C runners are referenced by path **if present** after lead integration;
never nest them beneath D's collector lock. Cold installs/builds can consume the whole
first hour; do not skip the compile gate to satisfy the booking estimate.

PRO pair then four-card list stays separate: actual peer/grant/lifetime gates, official
Step FP8 PP ladder, Qwen full 262144 consumed active/prefix state, Hy3/PLE consumers,
integrated pressure/standard exactness, then all 12 directed target routes and shared
fabric/allocation peaks. Pair = four tiers, not four cards; PP is not TP; no NVLink assumed.
No artifact was fetched. No serving instance, paused 0731 oracle or V4.1 code was touched.

## Numbered blockers / lead handoff

1. **First native engine+server build** on a fresh approved dev rig is blocking; this Mac
   has no CUDA compiler/GPU. Rust CPU checks cannot validate B's native worker wiring.
2. **Rig/artifact access:** no rented 5090 yet; lead supplies isolated dev access and
   immutable Qwen/Hy3/PLE/official-Step manifests plus prompt/fit-plan pins. PRO pair and
   four-card target evidence remain required for their cells.
3. **Consumer implementation/qualification:** real A CUDA transfer owner, B active
   materializer/server adapters, C bounded native row consumers and D live peer grants/
   owner/fence adapters remain pending. Existing baseline binaries are not those gates.
4. **Publication/integration:** lead must integrate/push the reviewed branch with D tools
   and A/B/C runners before remote checkout; D neither pushes nor fabricates absent refs.
   Apply `DAY3-SHARED-FRAGMENTS.md` to the existing FLAGS/TESTING rows in the landing change.
5. **Scoring/GO:** native tier/SSD/queue counters, physical I/O and complete serving latency/
   throughput instrumentation plus same-window balanced hardware runs are unrun. G0–G7
   remain pending; the subprocess capture deliberately never emits qualified GPU rows.

## Hygiene / effort

Only D tools/tests/receipt namespace changed beyond the explicit lead merge. Root Cargo/
lib/contracts/A/B/C implementation files were not amended by D. No new flag names or
experimental door; build-configuration registry fragments reuse existing names. No skip
overrides. Inherited artifacts preserved; temporary stash popped; Python scratch/caches
created by this task removed. Canonical shared lock inode deliberately not unlinked.
The worktree remains open for the next program milestone, not abandoned or main-merged.

Resumed active work approximately **0.2 agent-hours**, within the 90-minute cap. Earlier
interrupted-agent hours were not supplied; cumulative day-3 consumption is unknown.
The implementation budget remains **9 agent-days** (lead's separate overhead not charged
here); no invented conversion between hours and owner-scaled agent-days or hardware wait.

# What this harness changed from cell one's, and why

Cell one's harness (`research/toolchain-ab-20260831/harness/`) is the base. The owner's
instruction was to reuse it verbatim and extend only where a cell needs a new env knob.
Every difference below is either a path rebase (this lane runs in its own directory on its
own port) or a named new knob. The measurement itself — prompt, request shape, guard,
decision metric — is unchanged, which is what makes the numbers comparable to cell one's.

## Unchanged

- `digits.txt` — byte-identical. Same banked digits prompt.
- The sealed protocol inside `digits.py`: 512-token streamed completions, **no sampling
  params in any payload**, fresh salt per rep, wall clock including TTFT, tokens from the
  stream's own `usage`, spec receipts from `usage.spec`, 1 smoke (spec-engagement gate) +
  1 discarded warmup + 8 measured reps, one JSON row per stream.

## Path/port rebase (mechanical)

- `digits.py` — 8 lines: `/home/ubuntu/toolchain-ab` -> `/home/ubuntu/perf-chain`, port
  `18620` -> `18640`, prompt path moved under `harness/`. A separate directory, port and
  lock file (`/tmp/memra-pc37.lock`) keep this lane from colliding with anything else on
  the box.
- `stop.sh` — same logic, anchored on this lane's absolute binary path
  (`^/home/ubuntu/perf-chain/bin/memra-server`), drain wait extended 60 -> 90 polls.

## New knobs and gates

### `launch.sh` — one launcher, eight named modes (was two launchers)

Cell one had `launch.sh` (current deploy shape) and `launch-140era.sh` (era + doors). Both
env lists are carried over verbatim as `ERA_BASE`, `DOORS` and `CURRENT_EXTRA`; the modes
compose them:

| mode | = | used by |
|---|---|---|
| `era` | ERA_BASE + DOORS | cell 1 arm O |
| `era-nodoors` | ERA_BASE | cell 1 arm OD, **cell 3's fixed env** |
| `current` | ERA_BASE + vision + CURRENT_EXTRA | cell 2 arm P |
| `current-novision` | ERA_BASE + CURRENT_EXTRA | cell 2 arm PV |
| `fixed-nofiltered` | era-nodoors + `MEMRA_SPEC_GRAPH_FILTERED=0` | cell 3 flag arm |
| `fixed-nochaingraph` | era-nodoors + `MEMRA_MTP_CHAIN_GRAPH=0` | cell 3 flag arm |
| `fixed-nodcw` | era-nodoors + `MEMRA_STEP35_DRAFT_DCW=0` | available, unused |
| `current-nofiltered` | current + `MEMRA_SPEC_GRAPH_FILTERED=0` | available, unused |

The binary marker gates are kept, but the vision-marker gate now applies only to the
`current*` modes: older binaries in the bisect range legitimately predate
`MEMRA_STEP_VISION_DIR`, and cell one's unconditional check would have refused to boot them.

### `boot.sh` — the env axis is proven from `/proc`, not intended

Cell one verified the boot nonce and the exe symlink. Added here:

- The full live `MEMRA_*` environ is banked per boot to `receipts/environ-<arm>.txt`.
- `PC_MODE` is injected into the server env and read back, so a boot cannot silently run a
  different mode than the runner asked for.
- A per-mode **expectation table** is asserted against that live environ: every door the
  cell is about is proven `=VALUE` or proven **absent**. `ENV_FAIL` aborts the boot before
  a single rep runs. Cell 1's whole claim is an env delta, so "the doors were set" had to
  be a receipt rather than an assumption.
- The build receipt's `git log -1` is copied into the boot receipt, so a measured row can be
  traced to a commit without leaving the receipts directory.

### `run-ab.sh` — boot lists, N arms, and a preflight

- Takes a **boot list** (`"4 5"`) instead of a count, so an x5 escalation resumes without
  renumbering banked boots.
- Takes **any number of arms**, so cell 3's staircase and its flag arms all rotate inside
  one interleave rather than being compared across separate runs.
- **PREFLIGHT** validates every arm — binary resolvable (by path or unique sha prefix),
  mode known to `launch.sh` — and banks each arm's md5 + baked fingerprint **before the
  first boot**. This was added after a mistyped sha in a driver script would have aborted
  cell 3 five cycles in, having already burned that card time. Fail at t=0, not at t=17min.

### `build.sh` (new) — one binary per commit, with attribution gates

- Banks `bin/memra-server-<sha12>` per commit, `git log -1` recorded **after** checkout.
- Rejects a build that "finishes" in under 5 s as a failed checkout.
- Refuses only the precise hazard: a live server **running from the checkout's `target/`**.
  This harness always launches a copy out of `bin/`, so a checkout during a measurement is
  safe; the guard asserts that via `/proc/<pid>/exe` instead of refusing on any live runner.
- Caps an overlapped build at `nice -n 19 ionice -c3` with 8 of 48 cargo jobs and records
  `built_under_measurement_overlap=` in the receipt, so the overlap is disclosed per binary
  rather than assumed harmless.

### `chain*.sh`, `build-batch.sh`, `watch-chain2.sh` (new)

Sequencing only: chain the cells and the build batch back to back so the cards do not idle
between them.

## A gate that earned its place mid-lane

The baked-fingerprint check (`bin_fingerprint == sha12`) started as a nicety and caught a
real defect: `bin/memra-server-41b0040e4101` was built from the right tree but labelled
`fp/abc4014151d1`. See the engine-defect section of `RESULTS.md`. Because of it, **arm
identity in this lane binds on binary md5 plus a code-marker test**, and the fingerprint is
treated as a label to be verified rather than trusted.

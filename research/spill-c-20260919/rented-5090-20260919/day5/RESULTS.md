# Day 5 native development cells

Repository: avifenesh/memra. Source: `f68035c4f601091b81121902ae12d9e659eb0d34`
(`source.txt`); the exact additional source is `module-fragment.diff`.
Binary SHA256: `18c171bd8e17de648d8d9f60e431976bf8ce235ce8d364110d1cfc9e7a456a5a`.

## Existing and bounded-row GPU gates

Release gate build succeeded (incremental, 0.05 s); the two known bridge dead-code
warnings remain in `build-gate.log` for the next milestone. Both cells ran via
`tools/tier-battery.py`, `/tmp/memra-5090.lock`, on one rented development RTX 5090.
Each cell captured power cap **400 W**, maximum **600 W**, and checked empty compute
apps inside the lock before launch. Collector before/after compute-app lists are
empty. Raw 250 ms telemetry is retained, not promoted to a throughput benchmark.
Single correctness runs, no controlled thermal/performance claim.

Both receipts end `# verdict\tfailures=0`. Existing rows are identical between the
two receipts after removing the extra row-tier record and summary. Frozen goldens
are adjacent to both receipts, copied without regeneration; SHA256
`4bca9c6a544c0fa09b29d8919ae16936543ff9bed001ec39f96826b8fed43a3e`.
Collector raw-log and telemetry descriptor hashes were independently recomputed
locally after rsync and match.

Verbatim new-arm verdict:

> qwen4exp-gpu-gate PASS [rows-via-tier: BIT-IDENTICAL PLE outputs + convolution state; encodings=2 cases=16 values=6144 tier_calls=16 forced_read_chunks=144 budget_drained=true]

This qualifies only the bounded synthetic native PLE gather slice against its
same-program direct path. It does not qualify model-scale serving, expert dispatch,
NVMe storage, or production. Collector status deliberately remains
`executed-not-qualified`; the explicit gate verdict is the numerical evidence.

Linux bank suite, warning cleanup and final CPU checks follow in separate commits.

## Linux bank suite

At `a5eb77bd`, `cargo test --release -p memra-tier --test bank -j 16`
passed: **46 passed; 0 failed; 0 ignored**. Raw output and exact source are
in `bank-linux/`. This is native Linux CPU evidence, not a GPU cell.

## Native strict clippy and bridge warning scope

Both native release clippy commands passed with `-D warnings`: engine library +
`qwen4exp_gpu_gate`, and tier `--all-targets` (16 build jobs). `clippy-linux/`
contains exact base source + candidate diff and raw logs. The two dead-code
warnings were scoped individually with reason-bearing `allow(dead_code)` on
`HostView` and `map_host_exps`: these remain an uncalled native compile probe,
exercised with API-shaped host fixtures in the bank test target, NOT connected
expert dispatch. No numeric path changed. The nvcc-path build-script notice
remains; it is not a Rust/clippy diagnostic.

## Final source checks and handoff

Final code revision: `082727f3` (exact SHA in `final-linux/source.txt` and
`../../raw/day5-final-cpu/source.txt`, relative to this directory).
The remote candidate file was compared to that committed revision before checkout;
only the two-declaration module fragment remains as the remote source delta.
Strict native engine/gate clippy was rerun successfully at the committed revision;
Linux release bank suite rerun: **46 passed, 0 failed** (`final-linux/`).
No GPU rerun at this later warning-only revision is claimed: GPU binary/source
identity remains the earlier exact receipt above.

Final Mac checks all exit 0; commands, exit statuses and raw outputs are in
`research/spill-c-20260919/raw/day5-final-cpu/`:

| Check | Result / actual scope |
|---|---|
| `cargo fmt --all -- --check` | pass, workspace formatting |
| `cargo check -p memra-tier --offline --all-targets` | pass, macOS type check |
| Same plus `--target x86_64-unknown-linux-gnu` | pass, cross-target check, not execution |
| `cargo test -p memra-tier --offline --no-fail-fast` | 174 passed, 0 failed, including 4 compile-fail doctests |
| `cargo clippy -p memra-tier --offline --all-targets -- -D warnings` | pass |
| `git diff --check` | pass |
| `bash tools/check-flags.sh` | pass, 864 runtime reads covered |

Remaining integration fragment: `mod ple_rows_tier; mod banked_residency;` in
lead-owned engine lib.rs. These declarations were applied only to this lane's
remote scratch checkout, not the local shared source. No dependency/kernel/FFI,
flag/default, numerical program or board values changed in this resume.

No blocker to this requested development milestone. HostExps runtime dispatch,
model-scale source/cache registration, async I/O, NVMe ancestry/storage speed,
full serving/state-restoration and production qualification remain unrun or unwired.
The collector's GPU telemetry remains raw/unvalidated for timing claims. No V4.1
or Engram work was performed.

Effort: approximately **0.7 agent-hours** for this resumed session, below the
requested two-hour stop; estimate includes access, verification and receipt banking.
The WP-C allocation remains **8 agent-days**. Earlier total agent-hours were not
fully recorded, so no fabricated cumulative burn or remaining balance is given.
Lane worktrees remain open for lead integration, not abandoned/completed scratch.

Receipt hygiene: trailing empty lines only were removed from the two final Cargo
test logs for `git diff --check`; all output lines are retained. GPU captures and
their hash-bound raw logs were not rewritten.

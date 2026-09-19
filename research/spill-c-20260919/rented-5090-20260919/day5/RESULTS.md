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

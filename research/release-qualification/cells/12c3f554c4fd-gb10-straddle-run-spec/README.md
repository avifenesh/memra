# run-spec across the deep-split floors on GB10, at 12c3f554c4fd

A scoped extra for the landing of memra#1180, #1194, #1187, #1190 and #1189 (PR memra#1188). It is not a cell record; the gate does not read it.

## Why

memra#1189 gives GB10 a 512-key attention split past 16,384 keys of cache with 4 KV heads and past 32,768 with 2 (`docs/decisions/FA-SPLIT-DEEP-RUNG.md`). The landing's device record reaches that rung only in kernel-check. Its model cells stay below both floors: the calibrated argmax gate runs a 2,048-token board, and run-spec, serve-smoke and the perf A/B use short prompts.

The lane gated the rung under a model on its own heads. This landing's tree adds memra#1180's draft-side change in `spec.rs` on top of the rung's `spec.rs`, a combination no cell had run. These two cells run it.

## What ran

On one GB10, inside the landing lease, right after the device record `12c3f554c4fd-gb10-battery-perf-serving` and with that record's own `run-spec` (`run-spec.sha256` equals the record's `build.binaries` digest). Greedy, 64 tokens, K=1..8 against the plain target, on the lane's two prompts that straddle each floor:

| cell | model | prompt | the cache crosses | result |
|---|---|---|---|---|
| `rs-27b-16364.txt` | Qwen3.8-27B (4 KV heads) | 16,364 tokens | 16,384 at the 21st token | SELF-CONSISTENCY PASS: K=1..8 identical to plain |
| `rs-or-32748.txt` | Ornith-1.5-35B-A3B (2 KV heads) | 32,748 tokens | 32,768 at the 21st token | SELF-CONSISTENCY PASS: K=1..8 identical to plain |

The 27B's acceptance counts at every K equal the lane's cell at its own head (`research/gb10-split-rung-20261110/gates-unit2/rs-27b-16364-764cd39b5b.txt`).

## Files

- `rs-27b-16364.txt`, `rs-or-32748.txt`: the command, the exit code and the tool's output. The prompt token list is left out; the prompts are cut by `research/gb10-split-rung-20261110/tools/cut_prompts.py` at these token counts.
- `run-spec.sha256`: the digest of the binary that ran.

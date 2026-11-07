# A row's own pending prime no longer caps its first speculative burst (memra#1085, 2026-11-07)

Verdict: on the RTX 5090 Laptop the spec tick counted a row's own pending prime when it decided whether the row owed its peers a one-round quantum, so a request whose cold DFlash prime ran as two walker chunks (the grid cut since memra#1083, or the render-stable boundary capture) got a first speculative step of one round right after its prime. Counting only other rows' pending primes gives that step a full burst, with turn latency flat (no speed claim): first burst 0.0/7.0 -> 21.0/39.0 accepted/drafted (median), TTFT 2.430 -> 2.389 s, total 3.426 -> 3.446 s (+0.6%), median of 20 requests per arm over 4 interleaved boots. Greedy text is byte-identical across every boot of both arms.

## Cause

`crates/memra-server/src/worker.rs`, the spec tick: before each row's step,
`peer_quantum = prime_yield_enabled() && active.iter().any(|s| s.prime_service.pending)`. The set
includes the row itself. A cold DFlash prime cut at a capture boundary is two walker chunks: the
first chunk runs and yields (`[prime-yield] count=1 remaining=1`), and on the next tick the row is
still `pending` when its quantum is decided, so after its last chunk runs in that same step,
`PrimeService::decode_target` caps its first speculative step at one round. The receipts of
memra#1083 show it: a prompt that does not split starts with a full burst (`burst=21/38`), a split
one with `burst=1/7` (`research/plain-restore-grid-20261107/`).

The rule (docs/FLAGS.md `MEMRA_PRIME_YIELD`) is about peers: each peer gets one quantum per prime
advance. The prime's own row is not its peer.

## Change

`crate::prime_fairness::owes_peer_quantum(row, pending)`: only another row's pending prime counts.
The spec tick uses it; unit test `a_row_owes_a_quantum_only_to_another_rows_pending_prime`, wiring
anchor `the_spec_tick_peer_quantum_skips_the_rows_own_prime`. The cap bounds cadence (rounds per
tick), never committed tokens, so the bytes are unchanged; the cell checks that.

## Served cell (RTX 5090 Laptop, `cells/`)

`self_quantum_cell.py`: binaries base = 05a8a63503 (main), fix = 9c3a4edd98; one boot per arm,
order base,fix,fix,base,base,fix,fix,base; per boot one warm-up request, then
5 distinct cold chat prompts (2878..2878 prompt tokens, each off the 32-token grid,
no shared prefix), greedy, 96 tokens, thinking off, streamed. Qwen3.8-27B NVFP4 + Q5_K GGUF + DFlash2,
context 8,192, two sessions, prefix cache at its derived size, restore switch at the device default (OFF).

| arm | requests | first burst (accepted/drafted) | TTFT median | total median | total mean |
|---|---:|---|---:|---:|---:|
| base | 20 | 0/7 x20 | 2.430 s | 3.426 s | 3.455 s |
| fix | 20 | 25/39 x4, 21/39 x4, 23/41 x4, 18/41 x4, 20/37 x4 | 2.389 s | 3.446 s | 3.470 s |

Turn latency is flat: fix minus base per adjacent ABBA boot pair (mean of the boot's requests) is
+0.039, -0.012, -0.007, +0.044 s, mean +0.016 s, inside the boot-to-boot drift
(per-request sd 0.153 s base). The base's capped first step accepts 0 of 7 drafts and
the next tick runs the full burst, so the cap costs a wasted verify round and one tick, not
measurable turn time on this shape; TTFT is flat too. The change restores the fairness contract
and drops that wasted round; it is not a speed claim.

Every scored request split its prime (`[prime-yield] count=1`). Checks (`cells.txt`):

```
CELL prompt1-bytes-equal PASS sha ['5f25d6320613c20a']
CELL prompt2-bytes-equal PASS sha ['19081c58ad049d8e']
CELL prompt3-bytes-equal PASS sha ['3e655cee9ed105c4']
CELL prompt4-bytes-equal PASS sha ['8baf7befcad5db71']
CELL prompt5-bytes-equal PASS sha ['1e7a9d4e56a08e6e']
CELL every-prime-split PASS [(1,)]
CELL base-first-burst-capped PASS [[0, 7], [0, 7], [0, 7], [0, 7], [0, 7], [0, 7], [0, 7], [0, 7], [0, 7], [0, 7], [0, 7], [0, 7], [0, 7], [0, 7], [0, 7], [0, 7], [0, 7], [0, 7], [0, 7], [0, 7]]
CELL fix-first-burst-full PASS [[21, 39], [25, 39], [23, 41], [18, 41], [20, 37], [21, 39], [25, 39], [23, 41], [18, 41], [20, 37], [21, 39], [25, 39], [23, 41], [18, 41], [20, 37], [21, 39], [25, 39], [23, 41], [18, 41], [20, 37]]
```

Per-request rows: `runs.jsonl`; medians per prompt: `summary.json`.

## Exactness gate (`pair-cells/`)

`research/plain-restore-grid-20261107/grid_pair_cells.py` on the fix binary (warm boot serves three
turns, one cold boot per later turn; turn 1 is the split cold prime):

| arm | warm env | turn 3 warm / cold chars | bytes | turn 1 sha |
|---|---|---|---|---|
| fix-default | `(device default)` | 371 / 371 | pass | 523ca36af0 |
| fix-on | `MEMRA_DSPARK_PREFIX_RESTORE=1` | 371 / 371 | pass | 523ca36af0 |

Turn 1's sha is the one memra#1083 recorded for every boot (523ca36af0), so the full first burst
leaves the bytes where they were.

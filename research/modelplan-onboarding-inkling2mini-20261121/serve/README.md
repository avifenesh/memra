# Inkling 2 Mini: serve route and batched decode (2026-11-21)

The route is `crates/memra-server/src/inkling2_serve.rs` over `memra_engine::inkling2_gpu`
(per-request `Seq`, `Route::decode_batch`). Card: RTX 5090 Laptop GPU, 24 GB, sm_120a. Each
directory holds the batch script, its log, the receipts and the sha256 of the binaries that ran.

## Batching gate (`ink2-gpu --batch-check`)

B sequences at different positions each decode N teacher-forced tape tokens alone (`forward_seq`),
then again on a schedule: sequence i joins at step i, its prefill runs in chunks of P rows between
its peers' batched decode calls (`decode_batch`), and it leaves after its own tokens. Every decode
row's logits must equal its solo run bit for bit.

| Cell | Class | Schedule | Result | Receipt |
|---|---|---|---|---|
| p1, B=4, N=12 (10 to 12 a sequence), P=7, prompts 6 to 21 | float | 17 batched calls up to 4 rows, 9 interleaved prefill chunks | 45/45 decode rows bit-equal to solo | `s1/batch-float.json` |
| same | declared | same | 45/45 | `s1/batch-declared.json` |
| p2, B=6, N=24 (22 to 24), P=5, prompts 6 to 31 | declared | 32 batched calls up to 6 rows, 27 prefill chunks | 138/138 | `s1/batch-declared-p2.json` |
| p2, B=4, N=16 (14 to 16), P=256, prompts 1000 to 1180 (past the 1024 window: every sliding slot reads a wrapped ring row by row) | declared | 20 batched calls, 19 prefill chunks | 61/61 | `s2/batch-long-declared.json` |
| same | float | same | 61/61 | `s2/batch-long-float.json` |
| red arm `batch-rows-rotated` (each row reads the next row's caches) | declared | as the first row | 43/45 rows differ: rejected | `s2/batch-red.json` |

The kernel changes behind the batch rows (a row descriptor table read by the heads, conv and
attention kernels) leave the one-sequence program as it was: the float p1 score and the declared
p1 score with chunkings 26 vs 1 equal their `gpu-route/b6` and `b8` receipts (`s1/p1-float.json`,
`s1/p1-declared.json`), and the 18 kernel cells pass (`s1/gate.log`).

## Served (`serve_check.py` against memra-server)

`MEMRA_MODELS=ink2=<checkpoint> MEMRA_CTX=8192 MEMRA_MAX_SESSIONS=4`, with `MEMRA_REWRITE_BUNDLE` set to
`verify-serve/` (the inspect bundle; the route checked its family, config and plan hashes and ran
the plan's own program, `s2/server.log`). All checks pass (`s2/serve-check.log`, `s2/api.json`):
chat non-stream and stream, `/v1/completions`, greedy determinism, a tool call parsed by token id
into `tool_calls` (`get_weather {"city":"Paris"}`), `reasoning_effort` 0.5, "high", 0.0 and "none"
served and "0.5" (a string), 1.5 and "ultra" refused, an image part refused, a forced
`tool_choice` refused, `numeric_class: "shipped"` equal to the unnamed request and a class the boot
does not serve refused. At every concurrency, every request's answer equals its c=1 answer
(8/8 at c=2 and c=4).

The stream-vs-non-stream text check compared empty content: at the template's default effort the
model spent the 64-token budget in reasoning. The reasoning text itself is compared by the
concurrency checks.

### Timings (correctness route, not tuned)

8 greedy streamed chat requests per level, 128 output tokens each, prompts of about 20 tokens.
Clamp check exit 2 (unsure) at the start of the lease; the telemetry over the measured window
(`s2/smi.csv`, `judge.py`) shows every loaded sample at P0 and 14001 MHz, so the clocks are full by
the rule, and SW thermal slowdown (0x20) on 827 of 1864 loaded samples at 77 to 89 C, SM
1582 to 2767 MHz. N=8 requests per level, one run, one boot.

| c | TTFT p50 / p95 / p99 ms | TPOT p50 / p95 / p99 ms | ITL p50 / p95 / p99 ms | E2E p50 ms | req/s | output tok/s |
|---|---|---|---|---|---|---|
| 1 | 379 / 381 / 381 | 190 / 192 / 192 | 190 / 193 / 195 | 24,529 | 0.041 | 5.2 |
| 2 | 591 / 717 / 718 | 243 / 244 / 244 | 242 / 248 / 250 | 31,518 | 0.063 | 8.1 |
| 4 | 881 / 1,047 / 1,048 | 255 / 264 / 265 | 250 / 257 / 266 | 33,309 | 0.119 | 15.3 |

A batched decode call of 4 rows costs 1.3 times a call of one, so throughput grows 2.9 times
from c=1 to c=4. These are the route this PR adds on the GPU route of #1397.

### Timings after lane P's decode (memra#1399 merged, `s3b/`)

The same workload on the merged route (`s3b/s3b.sh`, binaries `serve-frozen4`, commit in the
receipt). Clamp check exit 0 at the start; over the window 1456 loaded samples, every one at P0 and
14001 MHz, SW thermal slowdown (0x20) on 1360 at 79 to 90 C, SM 1680 to 2737 MHz. N=8 requests per
level, one run, one boot. `serve_check`: all pass, every request at c=2 and c=4 equal to its c=1
answer (8/8).

| c | TTFT p50 / p95 / p99 ms | TPOT p50 / p95 / p99 ms | ITL p50 / p95 / p99 ms | E2E p50 ms | req/s | output tok/s |
|---|---|---|---|---|---|---|
| 1 | 41 / 42 / 42 | 24.2 / 25.3 / 25.7 | 22.6 / 24.6 / 25.0 | 3,112 | 0.321 | 41.0 |
| 2 | 315 / 363 / 363 | 138.9 / 140.4 / 140.6 | 138.6 / 143.0 / 145.2 | 17,954 | 0.111 | 14.2 |
| 4 | 533 / 701 / 702 | 161.6 / 162.3 / 162.3 | 154.1 / 164.7 / 335.9 | 20,840 | 0.192 | 24.6 |

Lane P's work makes a one-row decode call 7.9 times faster (TPOT 190 to 24 ms); a call of 2 rows
gets 1.7 times faster (243 to 139 ms) and a call of 4 rows 1.6 times (255 to 162 ms). One request
alone now outruns two or four together (41 against 14 and 25 output tok/s): the fast decode GEMV
and the decode graph are one-row programs, and a batched call still runs the multi-row tiles. The
batched decode GEMV is the next piece of lane P's work; until then the route's batching costs
throughput below c=4 on this card.

The merge kept every number the route produces: the batching gates hold (0 rows differing from
solo, p1 B=4 both classes, p2 B=6, prompts of 1000 to 1180 tokens past the window; the red arm
moves 43 of 45, `s3/`), the probe tape and the head rows of every corpus document are bit-equal to
the pre-merge run (`s3/`, `s3b/`), and p1's MoE holds 2736 of 2736 rows against the reference.
`s3/` stopped at the server boot: the split attention's scratch is sized from the load context,
and the serve route's sequences outgrow it (`memra_ink2g_attn` 45025). The fix runs those slots in
the tiled form, the same bits.

## Memory

The boot logs the route's book (`s2/server.log`): weights 17.594 GiB, 4.454 GiB driver free with
the route idle, a 4096-position sequence 0.88 GiB (sliding rings 0.625 GiB, conv 0.007 GiB, the
full-attention caches 64 KiB a position), and no allocator overhead measured at the driver. A
request is admitted when its sequence (prompt plus budget) and a 256 MiB margin fit in driver
free; a request no idle card could hold is refused with the largest session that fits.

## verify serve

The first attempt (`verify-serve/verify-serve.log`) refused for want of a checkpoint band. With the
declared class's band (`../checkpoint-parity/`, decision record
`docs/decisions/INKLING2-DECLARED-CHECKPOINT-PARITY.md`), `verify checkpoint` and `verify serve`
both pass (`../checkpoint-parity/verify/`).

## serve-smoke

`tools/serve-smoke.sh ~/models/inkling2-mini` runs the plain checks 1 to 7 of the pack's row in
`tools/serve-smoke-packs.tsv`: 7 of 7 (`smoke/inkling2.txt`). The default GGUF battery with the
script before and after the change, same binary: the same check lines, 0 failed each
(`smoke/gguf-base.txt`, `smoke/gguf-new.txt`).

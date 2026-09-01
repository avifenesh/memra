# Step35 B=1 eager-parity result

## Verdict

**PROMOTE — keep the B=1 specialization default-on for Step3.5/Step3.7.** The candidate
recovers 90.5% of the performance gap introduced by the b1fix correctness change while
remaining in the batched trunk's numeric class.

The live N=5 interleaved median moved from 81.472 to 85.041 tok/s at c=1, **+4.381%**.
The candidate is 0.447% below the separately receipted pre-b1fix eager median of
85.423 tok/s. Every paired round favored the candidate (from +3.902% to +4.694%), and
c=2/c=4 improved as well.

This promotion does not restore the old eager/fusion chain. It specializes only the
`B=1` state entry inside `step35_decode_batch_layers`: the same attention kernel consumes
the already-contiguous whole `q` row and writes to the already-contiguous whole `attn`
row. No arithmetic kernel, reduction order, numeric format, live dispatch trunk, or B>1
body changes. There is no new flag; the winner is the naked default.

## Anatomy: where the 4.71% went

The B-general attention walk materialized two rows per layer even when `B=1`:

1. copy the whole `q` allocation into `q_row`;
2. run the unchanged `fa_decode_kvmod` kernel;
3. copy the whole `a_row` result back into `attn`.

Step3.7 has 45 layers, so this issued 90 arithmetic-free device-copy launches per token,
plus the temporary row allocations. At `B=1`, the source and destination row ranges are
already the whole tensors. No mask/index generality or fused reduction is required to
remove them.

| anatomy cell | current batched B=1 | specialized B=1 | old eager B=1 | result |
|---|---:|---:|---:|---|
| baseline engine wall, tok/s | 81.8 | — | 86.0 | batched -4.884%, +0.597 ms/token |
| candidate engine wall, tok/s | — | 85.4 | 85.9 | candidate within -0.582% |
| sync-bounded q/a D2D bucket, 16 tokens | 15.0 ms / 6.0% | 0.0 ms / 0.0% | absent | 90 copy launches/token removed |

The wall cells are N=3 medians after one discarded warmup. They were two separate,
fixed-order anatomy blocks, so they establish mechanism rather than the promotion delta.
`MEMRA_BATCH_PHASE=1` synchronizes phase boundaries; its 15.0 ms value is deliberately
inflated and is valid only for rank/share and launch attribution. The final performance
verdict comes from the interleaved live-server block below. Reduced anatomy and full raw
logs are in [`anatomy-baseline/`](raw/anatomy-baseline/) and
[`anatomy-candidate/`](raw/anatomy-candidate/).

## Specialized-entry design

At [`decode_batch.rs`](../../crates/memra-engine/src/decode_batch.rs#L1407), the
`b_n == 1` arm:

- takes the same cache, `k` row, `v` row, SWA offset, physical KV view, and append path as
  iteration zero of the general loop;
- passes the whole `q` allocation directly to the same `fa_decode_kvmod` call with the
  same dimensions, scale, KV encodings, and input bits;
- directs that call's result into the whole `attn` allocation, which is exactly the row
  destination used by the following gate and output projection;
- leaves the operation order before and after attention unchanged; and
- leaves the complete `B>1` loop intact.

The only omitted operations are the two D2D row copies and their diagnostic phase marks.
The old eager path remains unavailable for Step35 PP-N, including through the
`MEMRA_STEP35_BATCH=0` rollback door, so live width changes cannot cross into its former
numeric class.

## Byte-identity and correctness receipts

| gate | receipt |
|---|---|
| frozen c=1 golden, fresh boots | **10/10 PASS**, 0 errors, 0 divergences, 326 bytes each; only SHA-256 `21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de` |
| live c=1 -> B>1 transition | **PASS**; static c=1/c=2/c=4 bytes match, early row emitted before late admission, trace crossed `ready=1 -> ready>=2`, all transition outputs match |
| production-path non-vacuity | decode chunk cap 8; first B>1 walk logged `B=2 layers=[0,22)` |
| `decode-batch-gate`, PP-2, B=1/2/4/8, 24 steps, 2 reps | **ALL GREEN**; 0 differing logit bits, 0 failing arms, split/unsplit exact, B=8 epilogue pass |
| `kernel-check` | **ALL GREEN** against CPU reference |
| Step3.7 `run-gen` | prefill/decode argmax **MATCH**; batched-prime/tokenwise **MATCH** |
| Step3.7 + MTP `run-spec` | K=1..8, **8/8 SELF-CONSISTENCY PASS** against the live B=1 batched target |

The fresh-boot reduction is in [`matrix-c1x10/summary.json`](raw/matrix-c1x10/summary.json),
the stateful width receipt in [`transition/summary.json`](raw/transition/summary.json),
and the standard battery reductions in [`core/summary.json`](raw/gates/core/summary.json)
and [`generation/summary.json`](raw/gates/generation/summary.json). The candidate also
passed local `cargo check -p memra-engine --lib` with CUDA 13.1/sm_120a and the isolated
box1 release build with CUDA 13.2/sm_120a. No formatting command was run.

## Interleaved live A/B

One bounded box1 lock hold ran five paired rounds with a fresh server process for each
arm. Order alternated current/candidate, candidate/current, current/candidate,
candidate/current, current/candidate. Each arm used the same model bytes, prompt, PP-2
placement, context 262144, grouped MoE, prefill tick 2048, decode chunk cap 8, and spec
disabled. There were 40/40 valid measurement points, 80/80 valid request rows, zero
errors, and zero short completions.

| live metric, N=5 median | current batched B=1 | specialized B=1 | delta |
|---|---:|---:|---:|
| c=1 sustained decode, tok/s | 81.472 | 85.041 | **+4.381%** |
| c=2 aggregate decode, tok/s | 113.694 | 119.943 | **+5.496%** |
| c=4 aggregate decode, tok/s | 139.479 | 144.665 | **+3.718%** |
| short eight-token TTFT, ms | 70.297 | 67.844 | **-3.491%** |

The c=1 metric is `(completion_tokens - 1) / (latency - TTFT)` for one 256-token greedy
streaming request. The c=2/c=4 sanity cells use aggregate completion rate for two/four
256-token requests. Short TTFT is measured after one warmup request. Captured thermal
snapshots were 26–36 C; arms alternated without artificial cooldown, and both GPUs kept
their 600 W limits.

The full sample arrays, metric definitions, binary hashes, order, and deterministic
reduction are in [`perf/summary.json`](raw/perf/summary.json); request rows, point rows,
server logs, and per-arm thermal snapshots are alongside it under [`perf/`](raw/perf/).

## Provenance and closeout

- Rig: box1 cloud pair, 2x NVIDIA RTX PRO 6000 Blackwell Server Edition, PP stages 2 on
  devices 0,1.
- Trunk artifact: `Step-3.7-flash-IQ4_XS-00001-of-00003.gguf`, 46,483,327,296 bytes.
- MTP artifact: `Step3.7-flash-mtp-Q8_0.gguf`, 3,707,276,416 bytes.
- Candidate source: `711fbcaaef54491d22488a84d40b7fc35e5a58dd`.
- Candidate `memra-server` SHA-256:
  `43ad098d46bb26d644ba0b742d92f3f014d9287ac72e8a0edb8ebf9dac3ba608`.
- Current-batched server SHA-256, byte-identical to the b1fix performance receipt:
  `6a7c2046eb3197773def91baf012abd629e0b0ced239ec2d38016c93be5ca7e5`.
- Full raw evidence manifest: [`raw/SHA256SUMS`](raw/SHA256SUMS), itself SHA-256
  `20636c29fc90bb241d2dfc6e701cda8938386f4421957627bbcda96dfc5b9896`.

Every GPU block held `/tmp/memra-gpu.lock`; block-start receipts showed no competing
compute applications. The performance improvement is not a published perf-board move,
so no generated board was changed. This lane made no origin push, tag, merge, or release.

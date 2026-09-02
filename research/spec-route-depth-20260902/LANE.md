# spec-route-depth-20260902: the glm5 spec route at depth (attribution, drafter prime, cap)

Lane: `lane/spec-route-depth-20260902`, cut from a freshly fetched `origin/main` at
`41c28fb72` (includes PR #93). No GPU in this lane; the 2x B200 pair runs the bins between
its rounds. Every number below that is not marked "this lane" is the pair's receipt as
delivered to this lane.

## 1. The measurement this lane serves

2x B200, GLM-5.3-Flash NVFP4, PP-2 resident 124, all B200 doors + the pipelined prime, one
256,756-token prompt, 256 tokens, vendor sampling:

| route | TTFT | decode |
|---|---|---|
| plain | 69.08 s (3,717 tok/s prefill) | 31.13 tok/s steady |
| DFlash2 spec, K=3, cold-long | 152.95-160.64 s | 15.4 / 32.1 / 15.4 tok/s across three boots (bimodal) |

512k: spec 10.7 vs plain 28.3 tok/s; 1M: 6.2 vs 27.0; one posture. At 66 tokens the spec
route is 1.48x plain with TTFT parity (PR #93).

Coordinator attribution after this lane opened: most of the TTFT gap is the pipelined
prime arm declining when the hc tap sink is armed (another lane owns that); the drafter's
OWN prime is about 24 s at 256k (this lane, section 3). The decode bimodality is spec-only
(this lane, section 2).

## 2. Attribution: what one boot at 4k / 42k / 128k / 256k tells

`MEMRA_SPEC_PROF=1` (PR #93) now prints three line shapes per served glm5 spec request:

1. `[spec-prof]` (one line, after the first burst) gains, at the end:
   `draft_prime: arm=eager|chunked rows= chunks= h2d= feat= kv=` (the drafter prime split:
   host to device movement, the fc feature GEMM, the 5-layer k/v ingest; on the eager arm
   this is round 1's ingest, on the chunked arm it is the per-chunk ingest inside session
   creation), `sink_alloc` (the eager host tap sink allocation), `tap_dtoh` (the eager arm's
   synchronous tap DtoHs inside the target prime, a share of `prime`), `draft_kv_mb`, and
   `free_mb_before` / `free_mb_after` per device (free device memory around the session's
   allocations: the graph-launch headroom guard and every pool-growth path key on it, so a
   per-boot bimodality shows here first).
2. `[spec-prof-rounds]` after every burst, for the first 64 rounds of the session:
   `k=[..]` drafts that entered verify (after the confidence gate), `j=[..]` accepted,
   `wall`/`draft`/`verify`/`accept`/`rest` ms per round (under the trace's drains),
   `seq=[..]` verify rows that took the PER-ROW mixer arm (0 = every layer batched; a
   non-zero count at depth names the slow path by itself), `ctx0`.
3. `[spec-prof-summary]` once, when the log fills or the session ends: k/j means, accept
   rate, tokens per round, wall mean/min/median/max, `slow_rounds(>1.5x med)` (the
   within-boot bimodality count), verify mean, `seq_rows_total`.

Reading the bimodality from one boot per depth:

| symptom | what the lines say |
|---|---|
| accept rate collapsing on this prose | `[spec-prof-summary] accept=` and `j=[..]` fall with depth while `wall` per round stays flat; tokens per round drops toward 1 |
| verify at t=4 hitting a slow MLA/DSA path at depth | `verify` per round grows with `ctx0` far faster than the plain per-token cost; `seq` non-zero means the per-row arm ran |
| drafter KV / VRAM pressure per boot | `free_mb_after` differs between the fast and slow boots; `draft_kv_mb` is the spec-only allocation (2 x 5 x (ctx+8) x n_kv x head_dim x 4 B, about 40 KB per row at the pinned geometry: 10 GB at 256k) |
| within-boot bimodality (rounds alternating fast/slow) | `slow_rounds(>1.5x med)` non-zero with a wide `wall min/max` |

Box invocation (the section 1 spec boot plus the profile; one boot, one request per depth,
`stream: true`, vendor sampling, 256 tokens):

```
MEMRA_SPEC_PROF=1 MEMRA_GLM5_SPEC=1 MEMRA_GLM5_DFLASH=<dflash2 dir> MEMRA_SPEC_PMIN=0.7 \
  MEMRA_GLM5_VERIFY_BATCH=1 MEMRA_SPEC_GATE_LOW=2 MEMRA_SPEC_GATE_HIGH=4 \
  <the pair's serve command> 2>&1 | tee spec-depth.log
# then one request each at 4k / 42k / 128k / 256k prompt tokens, and:
grep -E '^\[spec-prof(-rounds|-summary)?\]|^\[glm5-spec\] route=' spec-depth.log
```

Add `MEMRA_GLM5_DRAFT_PRIME_V2=1` to the same boot (or a second boot, interleaved) to read
the chunked arm's `draft_prime: arm=chunked ...` beside the eager arm's.

## 3. The drafter prime, and the chunked arm (`MEMRA_GLM5_DRAFT_PRIME_V2=1`, default OFF)

What the eager arm does at 256k (file:line on this branch): `HcTapSink::new` allocates a
`[256756, 5 x 4096]` f32 host Vec (21 GB, `glm_spec.rs` session creation); the target prime
fills it through five synchronous pageable DtoHs per prime chunk (`glm5_hc_tap`, the host
branch); round 1 uploads it back in 256-row synchronous pageable HtoD chunks and runs
`ctx_features` (fc, `[5 x 4096 -> 4096]`) + `ingest_ctx` (5 layers of k/v projection,
k-norm, rope, two copies) per chunk (`glm5_dflash_round_drafts`). Note for the record: the
ingest is projection-only; the drafter's attention runs at round time over the cached ctx
KV, so "a 5-layer full-attention prefill of the prompt" is not what the eager arm pays. The
cost is data movement (2 x 21 GB pageable) and 1,003 small GEMM chunks.

What the chunked arm does: iterates the engine's own `hyper_prime_ranges` (4096 rows;
PP-aware, the same schedule the whole-prompt entry walks internally), arms a DEVICE-staged
chunk sink per range (`HcTapSink::new_device_staged_at`), calls `prime_cache` on the range
with `queued_after = plen - end` (exactly the whole-prompt entry's per-range call, so the
trunk program is unchanged), then drains the five device slots into a pinned cacheable
buffer, interleaves on the host into the fc layout, uploads ASYNC from pinned, and ingests
at the chunk width. Host transient: two chunk-sized pinned buffers. The drafter KV is
allocated before the prime; round 1 finds nothing pending.

Numeric class: the drafter KV rows are the same GEMM class at a different M. Rig gate 15
(`gpu_dflash_chunked_drafter_prime_kv_matches_eager_ingest`) asserts KV bit-identity at
equal chunking and served-tape identity vs plain decode on both arms at every chunking, and
reports the KV max-abs-diff and the acceptance delta under a forced multi-chunk prime
(`MEMRA_PRIME_CHUNK=16` over the 24-token fixture). Drafts only, never output.

Default OFF by the new-flags law; the box A/B (eager vs chunked `draft_prime` buckets,
interleaved boots) flips it. Not touched: the B200 pipelined prime arm (its "declines when
the hc tap sink is armed" is the other lane's fix; note that the chunked arm arms
`device_stage` sinks per chunk, which that fix should accept too).

Named follow-up: a device-resident tap path with no host bounce (2D device copies per
slot; cross-stage slots need a peer path or the pinned bounce).

## 4. The cap: `MEMRA_SPEC_MAX_PROMPT=<tokens>` (default unset = unlimited)

A route-policy door at admission: a glm5 spec-eligible request whose prompt is longer than
the cap serves plain, `reason=prompt-above-spec-max` on its route line, one
`[glm5-spec] MEMRA_SPEC_MAX_PROMPT=` line per process at the first capped request. Composed
after every correctness exclusion (`glm5_admits && !glm5_capped`), so the other reasons keep
precedence. Unlimited until the section 2 boot places the crossover; the fleet caps that day.
glm5 route only (dspark and qwen keep their own admission; follow-up).

## 5. Gates

This lane (no GPU): `cargo fmt --check`, `git diff --check`, clippy `-D warnings` at
`MEMRA_CUDA_ARCH=120a` and `100a`, `tools/check-flags.sh`, `tools/check-public-boundary.py`,
worker tests (`spec_max_prompt_cap_parses_and_caps_strictly_above`,
`spec_max_prompt_cap_is_wired_after_the_route_predicate`, the glm5 wiring and step-OOM
tests), `spec_phase` unit tests. Rig: gate 15 plus gates 1-14 of
`glm5_dflash_session_gpu` (`NVIDIA_TF32_OVERRIDE=0 flock /tmp/memra-5090.lock cargo test
-p memra-engine --test glm5_dflash_session_gpu -- --ignored --test-threads=1`).

## 6. Boot A (2026-09-03): what the lines said, and the fix it forced

Pair, main + PR #101 build, spec route, `MEMRA_SPEC_PROF=1`, W8 + main doors, no pipelined
prime, one request per rung, 256 tokens, vendor sampling. `[spec-prof-summary]` as
delivered (raw per-round and drafter-prime lines: darklanes
`research/glm5-b200-20260902/box/specdepth/specdepth-a-prof.txt`):

| rung | K | rounds | k_mean | j_mean | accept | tok/round | wall mean / min / med / max (ms) | slow_rounds | draft_mean | verify_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| 4k (6,920) | 3 | 64 | 2.23 | 1.41 | 0.629 | 2.41 | 58.8 / 45.7 / 62.8 / 156.3 | 1 | 8.9 | 48.8 |
| 42k | 3 | 64 | 1.80 | 1.05 | 0.583 | 2.05 | 66.7 / - / 59.7 / 630.2 | - | 16.3 | 49.3 |
| 128k | 3 | 64 | 1.66 | 0.97 | 0.585 | 1.97 | 126.1 / - / 50.5 / 4525.9 | - | 77.3 | 47.7 |
| 256k | 3 | 64 | 1.83 | 1.19 | 0.650 | 2.19 | 200.6 / - / 63.7 / 8925.2 | - | 146.1 | 53.6 |

Reading (coordinator + this lane):

1. The steady state is about 2.2 tokens per 55-64 ms round at every depth (35-40 tok/s):
   the per-round verify at t=4 (~50 ms) is the ceiling, so at 256k the spec route is at
   parity with plain (43 tok/s with the new DSA door), not 1.5x above it. Accept rate does
   not collapse with depth (0.58-0.65). `seq` rows: the summary's `seq_rows_total` is in
   the raw file; the batched arm is the served one.
2. ONE round per request costs 0.63 / 4.5 / 8.9 s at 42k / 128k / 256k, linear at about
   35 us per prompt token: `draft_mean x 64` reproduces it (146.1 x 64 = 9.35 s at 256k,
   77.3 x 64 = 4.9 s at 128k), so it is round 0's `draft` bucket, i.e. the eager ingest.
   Under the round-cadence door (`MEMRA_SPEC_FIRST_TOKEN_EAGER`, default ON since #93)
   `glm5_spec_session_burst_inner` emits the anchor BEFORE round 0 runs, so the ingest
   lands INSIDE decode: 256 tokens over 8.9 s + 255 rounds at 37 tok/s = 16 tok/s, the
   "15.4" mode; without the stall 37 tok/s, the "32.1" mode. Under the burst-cadence arm
   it sat inside TTFT. That is the bimodality.

Fix (this lane, same PR): the eager arm ingests the prompt AT SESSION CREATION, before the
session and its anchor reach the worker (`glm5_dflash_ingest_rows`, shared with round 1);
`MEMRA_GLM5_DRAFT_PRIME_LAZY=1` restores the round-1 placement. Same rows, same GEMMs,
same KV bytes; only WHEN moves (second-token latency -> TTFT). Rig gate 15 asserts the
creation-time coverage. The chunked arm (`MEMRA_GLM5_DRAFT_PRIME_V2`, boot B) already
ingests at creation and makes the ingest itself cheaper.

Open levers named by the coordinator: (b) verify cost at t=4 is the depth ceiling; the
DSA door (PR #104, int2) keys its single-pass kernel at t_q=4 and its scorer at all t,
`verify_mean` under `MEMRA_B200_DSA_DECODE=1` is boot B's twin; (c) the K=5 re-price for a
rank-trimmed DFlash2 head (#103) is the other lever on tok/round. Follow-up here: overlap
the drafter ingest with the next trunk chunk on a drafter stream so it costs neither TTFT
nor decode (needs a second stream plus event fences; the chunked arm is the shape it
attaches to).

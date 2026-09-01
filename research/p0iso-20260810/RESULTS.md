# P0 isolation: served bytes depend on live decode-width history

## Verdict

**Trial-blocking serving bug.** The darktrain2 byte divergence is reproduced without a
trainer, without another CUDA context, and without prefix fanout. The isolated mechanism
is an admission-timed transition between two intentionally different floating-point
decode classes on the Step3.7 PP-2 live path:

- a chunk of `B=1` takes the eager/fused per-stage trunk;
- a Step3.7 chunk of `B>1` takes `step35_decode_batch_layers`;
- the scheduler is free to move a session from the first class to the second when more
  rows become ready.

The same prompt, `temperature=0`, seed 3407, model bytes, and fresh-boot state therefore
produce three stable completion classes solely from decode-width history. The trainer in
the predecessor cell was a timing perturbator, not a necessary cause. H3 was deliberately
not run: the prescribed ladder allowed it only if H1/H2 were clean, and H2 failed
decisively with no co-tenant.

## Reproduction matrix

"Divergent" below means different from the predecessor's 326-byte golden completion. A
cell is one fresh server boot; the measured prompt was never used as a warmup. Every block
held `/tmp/memra-gpu.lock` for its full lifetime.

| condition | controlled change | cells | cells divergent | non-golden requests | first positive `ready` | result |
|---|---|---:|---:|---:|---|---|
| `same` | c=8 barrier, dedup on | 20 | 20/20 | 20/160 | `1`: 20 | every cell 7 golden + 1 transition |
| `stagger` | c=8, arrivals spread 0-200 ms | 20 | 20/20 | 20/160 | `1`: 20 | every cell 7 golden + 1 transition |
| `dedup-off` | c=8 barrier, `MEMRA_PREFIX_DEDUP=0` | 20 | 19/20 | 19/160 | `1`: 19, `2`: 1 | 19 cells 7+1; direct-`B=2` cell 8/8 golden |
| `h2-c2` | c=2 barrier | 10 | 9/10 | 9/20 | `1`: 9, `2`: 1 | `ready=1` cells split 1+1; direct-`B=2` cell 2/2 golden |
| `h2-first-late` | c=8, client index 0 delayed 100 ms | 10 | 7/10 | 7/80 | `1`: 7, `2`: 3 | divergence followed the new admission-rank-0 row; delayed index 0 was 10/10 golden |
| `h2-c1` | c=1 | 10 | 10/10 | 10/10 | `1`: 10 | all ten returned the stable all-solo class |

Predecessor anchors were train-absent 8/8 golden and trainer-running 7/8 golden plus one
transition-class completion. This lane ran 90 fresh boots and 590 successful requests:
505 golden, 75 transition-class, and 10 all-solo. There were no HTTP errors, CUDA/server
failure signatures, or prefix-cache hits.

## Three output classes, exact map

| live decode history | completion SHA-256 | bytes | requests | cell-level result |
|---|---|---:|---:|---|
| batched from first decode | `21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de` | 326 | 505 | all rows golden in all 5 such cells |
| one solo decode, then `B>=2` | `7a5032f2d723e3cf9ef788fdc9d4067fe2eb909157189b666430b7997a56961f` | 310 | 75 | exactly the admission-rank-0 row in all 75 such cells |
| `B=1` for the full stream | `d35be2307889b24ec1ba4361eb22fdc6ceabda65864df261bd66c08f37f192c1` | 326 | 10 | the only row in all 10 c=1 cells |

The mapping has no exceptions across the 90 cells:

- 75 cells began `ready=1` and later reached `B>1`; each had exactly one 310-byte
  transition result, always on admission rank 0.
- 5 cells began at `ready=2`; all 34 requests in those cells were golden.
- 10 cells stayed at `ready=1`; all returned the third, all-solo hash.

The predecessor's "earlier first token" clue is explained by the solo tick. In the c=8
transition cells, the transition row's first token preceded its peers' median by
91.24-91.98 ms (condition medians 91.43-91.53 ms). At c=2 the lead was 23.10-23.28 ms.

## Hypothesis disposition

### H1 fanout/dedup: rejected

Across all 590 requests the server recorded 590 prefix-cache misses, zero hits, and no
fanout event. The live chat path is checkpoint-capturing; fanout eligibility explicitly
requires `ckpt_at.is_none()` at
[`worker.rs:5337-5355`](../../crates/memra-server/src/worker.rs#L5337). Disabling prefix
dedup still produced the same output classes, and its one clean boot was exactly the boot
that began directly at `ready=2`.

Fresh-boot nondeterminism is also rejected: each decode-history class produced one stable
hash across every boot.

### H2 batch grouping: confirmed

The c=2 control is paired evidence: 9 `ready=1 -> ready=2` cells diverged and the sole
direct-`ready=2` cell was clean. The delayed-index control moved the divergence away from
client index 0 and onto whichever request became admission rank 0. This excludes a fixed
client/request identity and makes the live grouping transition causal.

The existing "B=2..8 is per-row exact" contract does not cover this live transition. It
compares rows within the batched numeric class. The default PP-2 server also has a distinct
`B=1` eager/fused class, and the scheduler crosses that class boundary.

### H3 co-tenant CUDA context: not reached

No trainer or other CUDA context was present in any lane cell. Because the exact
predecessor divergence reproduced 75 times under scheduler-only controls, the conditional
H3 block would not further isolate this P0 and was not run.

## Code receipt: the live class boundary

1. The worker builds `ready` from whatever sessions have completed prefill in the current
   tick, then groups those rows up to the model's cap. It carries no per-session numeric
   class affinity: [`worker.rs:3657-3694`](../../crates/memra-server/src/worker.rs#L3657)
   and [`worker.rs:5883-5897`](../../crates/memra-server/src/worker.rs#L5883).
2. The resulting chunk is passed directly to
   `decode_step_batch_sampled_lean_masked`:
   [`worker.rs:3734-3764`](../../crates/memra-server/src/worker.rs#L3734).
3. On PP-N, `b_n == 1` enables `b1_stage_fast`; Step3.7 `B>1` instead enables
   `step35_batched`. The source explicitly describes the former as the eager/fused trunk
   and the latter as `step35_decode_batch_layers`, with an accepted FP-class gap:
   [`decode_batch.rs:760-802`](../../crates/memra-engine/src/decode_batch.rs#L760).
   The actual per-stage branch is visible at
   [`decode_batch.rs:804-865`](../../crates/memra-engine/src/decode_batch.rs#L804).
4. The standing Step3.7 B2 geometry gate masks this live boundary by launching the server
   with `MEMRA_SERVE_B1FAST=0`, so its c=1 reference also runs the batched body:
   [`step35-b2-geometry-gate.sh:85-99`](../../tools/step35-b2-geometry-gate.sh#L85).
   The engine gate does the same for its within-config comparison at
   [`decode_batch_gate.rs:279-297`](../../crates/memra-engine/src/bin/decode_batch_gate.rs#L279).

Mechanism name: **PP-N Step3.7 eager-`B=1` to batched-`B>1` numeric-class transition**.

## Severity

This is a **trial-blocking serving bug**, not a co-location-only artifact. A treatment that
changes admission timing can change model bytes even when the treatment never overlaps a
model kernel. That makes served-output comparisons under the darktrain QoS experiment
causally uninterpretable.

This receipt does not show CUDA-context corruption, bad model bytes, or request failure.
It shows a reproducibility contract failure at the live scheduler/engine boundary: fixed
greedy inputs are load-history-dependent.

## Fix shape

The fix must give Step3.7 PP-N one numeric class across every live width.

1. **Fail closed first:** exclude Step3.7/Step3.5 from `b1_stage_fast` so `B=1` also runs
   the stage-scoped Step35 batched trunk. `MEMRA_SERVE_B1FAST=0` already exercises that
   shape in the standing gate; the code change should be a model-specific correctness
   default, not a new permanent user flag. Re-run the target-rig correctness and throughput
   gates because the source documents a real B=1 performance incentive for the eager
   fusion chain.
2. **Performance-preserving endpoint:** make the Step35 eager fusion chain bit-identical to
   the batched trunk (or add an exact B=1 twin), then re-enable it only after a stateful
   transition gate is green.
3. **Close the gate gap:** run the PP-2 Step3.7 gate with live defaults and compare complete
   token/logit streams for `B=1` only, batched from the first decode at B=2..8, and explicit
   `B=1 -> B=2..8` transitions. Remove the `MEMRA_SERVE_B1FAST=0` masking pin from the
   served-byte assertion once one numeric class is the default.

Pinning a session to whichever class it first encountered would remove the third
transition class but would still make c=1 and batched requests differ. A longer batching
hold has the same problem. Neither is a correctness fix.

No runtime fix was implemented in this isolation lane.

## Evidence and provenance

- Reduced matrix: [`raw/reproduction-matrix.json`](raw/reproduction-matrix.json)
- One validated row per cell: [`raw/cell-analysis.jsonl`](raw/cell-analysis.jsonl)
- All 590 completion hashes and request identities:
  [`raw/completion-hashes.jsonl`](raw/completion-hashes.jsonl)
- All 921 raw-evidence file checksums: [`raw/SHA256SUMS`](raw/SHA256SUMS)
- Reproducer: [`qos_probe.py`](qos_probe.py) and [`run-box1.sh`](run-box1.sh)
- Deterministic reducer (validates response bytes against their hashes, expected counts,
  admission ranks, scheduler history, prefix counters, and error signatures):
  [`analyze.py`](analyze.py)
- Per-condition raw directories: [`same`](raw/same), [`stagger`](raw/stagger),
  [`dedup-off`](raw/dedup-off), [`h2-c2`](raw/h2-c2),
  [`h2-first-late`](raw/h2-first-late), and [`h2-c1`](raw/h2-c1)

All six blocks used remote source
`188154299064a42b67fc8eb1f41757cf6237300d`, server binary SHA-256
`e7e6515e9f47030a7137ba9fdf7c40d43f0764d02699b38959f134ee0ace65b3`, probe
SHA-256 `6c9e7386e3304deb6b625db1e7bd5089b3f0cf4844c198b17d7173e5c0082e9d`, and the
fixed golden hash above. The model was
`Step-3.7-flash-IQ4_XS-00001-of-00003.gguf` plus
`Step3.7-flash-mtp-Q8_0.gguf`, served PP-2 on devices 0,1 with context 262144,
grouped MoE, and prefill tick 2048.

# Full-token segmented replay: component checkpoint

Status: model-free component source only. No full-model dispatch exists yet.
No model exactness, model graph coverage, device sampling, or throughput result
is claimed. This checkpoint is for independent review before integration.

Base: `2d0271fb3b6021c2c9f1b3b36d2598614a10284f` (fresh origin/main).
The existing issue #4 claim remains with its continuing controller. No open
full-token replay PR existed when this branch was opened. Existing partial
graph owners and probe guards are preserved.

## Implementation map

| Owned file | Required integration |
| --- | --- |
| `src/dsv4_gpu.rs` | At `decode_step_tp_ep`, retain the walk mutex and request workspace. Allocate controls and capture both rank forward segments without executing either alone. Snapshot and restore capture-time host metadata. Replace host compressor address/cadence, indexer `nbs`/`kks`/`slots`, and commit `slot_rows` with live controls. Restore both transient planes on refusal. |
| `src/dsv4_graph.rs` | New capture-only lifecycle; current `capture_layer` pre-drains, launches immediately, and post-drains, which cannot represent independent paired rank capture. Graph executables must die before request buffers and AR state. Drain both ranks on submission failure. |
| `cu/dsv4_replay_control.cuh`, CUDA implementation and `src/dsv4_ffi.rs` | One stable input upload per rank per token. Same-device IF bodies for C4/C128 emission. Dynamic append and store addresses. Keep compressor pool/attention/selector loop bounds and reduction order exact. No per-node host parameter update loop. |
| `src/dsv4_ep.rs` | Retain the actual one-shot AR signal allocation, device epoch updates, start/end barriers and sticky refusal words across graph launches. Capture must not count as successful execution. |
| `src/dsv4_grouped.rs` | Require the existing device route and mirror-validation-off program. Preserve unsupported EP/compressor guards in the older probes. Scope admission before any capture-side mutation. |
| `src/dsv4_sampler.rs`, `cu/dsv4_sampler.cu` | Split enqueue from result drain. Feed the existing f64 position-keyed uniform from stable device storage; it is currently a by-value launch argument. Preserve the sampler numeric class. |

Paths above are under `crates/memra-engine/`. The component-stage header is
not wired into the runtime build or a scored arm.

The first segment owns embedding, all 43 complete layers and both ARs/layer
on each rank. The host reads both refusal words after both graphs are enqueued.
Only `[0, 0]` permits the second segment: successful ring commit on both ranks,
head and device sampling on the existing head rank. Token/EOS readback remains
the feedback barrier for the next token. There must be no per-layer eager
fallback, recapture, or silent fallback from the scored arm.

Existing success commit (`commit_verify_dev_plane`) ends with a stream drain;
its enqueue portion must be separated without moving the refusal decision.
Compressor checkpoints mutate pending payloads and host block counts before
that decision. A capture records GPU writes but executes Rust metadata changes;
restore those changes before first launch, then update metadata for each replay.
On refusal restore pending payloads and high-water marks and quarantine retry.
Unused emitted rows beyond the restored high-water mark follow the existing
rollback contract; do not broaden live cache state to include them.

Admission is plain short-context, one token at a time, device caches, attention
TP2 plus expert-ID EP, device sampler plus diet, split-K off. Refuse DSpark,
host C4, incompatible numeric programs and unsupported topology/shapes.
The future runtime door defaults off and must be decided by 2026-09-22.

## Component contract and limits

`tools/dsv4-full-token-control-gate.cu` compiles the actual `cu/tp_ar.cu`
source into a standalone executable. It creates two rank-local forward graphs,
each containing live control, a pending-state fixture snapshot, two conditional
bodies and 86 actual AR launches. Each rank has a separate commit fixture graph.
It never loads model weights. Integer payload fixtures do not implement model
compressor pooling, logits, or sampling.

The positive cell changes token/position/uniform bits on every replay across
513 positions, C4/C128 emissions and four 128-slot ring wraps. Every partial sum,
live scalar, pending/ring payload, block count and device AR epoch is checked.
The negative cells inject a sticky refusal independently on either rank at
127/255/511, require neither committed plane/readback to advance, restore both
transient fixtures, and reject retry. Each negative cell now replays a real
prefix, so saved C4 counts are nonzero and saved C128 counts are nonzero at
255/511. A reset-to-zero rollback cannot pass those cells.

The first source review found two exceptional-path bugs in the initial component:
rank destruction could free peer memory before the other rank drained, and sum
validation preceded refusal handling even though a timed-out AR leaves sums
unwritten. A `Pair` owner now drains both ranks before either member destructor
releases resources, also during stack unwinding. Failed drains are reported and
retain both allocations until process exit. Refusal handling immediately follows
both word reads; it restores only submitted ranks and quarantines both before
looking at any control or sum output.

Focused cells launch the actual AR kernel without its peer on each rank, with
poisoned sums and nonzero high-water marks. These use separate two-join fault
graphs with a 5,000,000-cycle bound; the positive 86-join graphs keep the original
2,000,000,000-cycle bound. A separate host submission exception after rank A
launch and before rank B enqueue tests stack unwinding: both drains must precede
either free, and A's device word must report the actual 40043 start timeout.
This is an injected host submission failure, not an invalid CUDA call suppressed
from sanitizer reports. There are no sanitizer suppressions.

Eight-token cells check all elements and all per-block epochs at the default
production AR shapes: 4096 floats/1 block and 24576 floats/48 blocks. The latter alternates attention
and expert geometry on the same signal allocation: block 0 advances 86 epochs
per token, other blocks advance 43. Uniform
payloads vary all 64 bits. This remains a payload freshness check, not sampling
qualification. No full-model refusal or numeric qualification is implied.

Build remotely (no local rig gates):

```sh
nvcc -t 2 -std=c++17 -O2 -fmad=false -arch=sm_120a \
  tools/dsv4-full-token-control-gate.cu -o "$CARGO_TARGET_DIR/dsv4-full-token-control-gate"
```

Use the controller-assigned pair slot with fd9 on `/tmp/memra-gpu.lock`.
Remote build and component receipts live in the private companion ops lane.
No result is inferred from installed headers or a successful compile.

## Required next gates

1. Run the model-free component on the assigned pair, inspect failures and
   CUDA sanitizer results before integrating the runtime.
2. Review the complete capture integration at a stable source checkpoint.
3. On the same loaded model prove changing-token cache/hidden identity on both
   ranks across C4/C128/ring wrap, AR epochs, both-rank refusal before commit,
   actual sampling-uniform freshness and complete model graph coverage.
4. Only then run one same-load 20-row ABBA, five rows per block, 256 prime and
   256 sampled outputs. Measure the complete `sample_plus_forward_envelope`,
   including control upload, refusal reads, commit, head, sample and readback.
5. Stop on correctness failure. Flat/negative full replay removes the entire
   diagnostic arm in this lane. A win goes back to the controller for a full
   qualification decision, without merge or automatic scope expansion.

CUDA API source: [NVIDIA CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html),
conditional node requirements and capture restrictions, read 2026-09-08.
Conditional bodies are single-device. Capture cannot synchronize a captured
stream. The installed CUDA 13.1 header uses the six-argument `cudaGraphAddNode`
form with edge data; the initial five-argument documentation example did not
compile. The component explicitly supplies null edge data.

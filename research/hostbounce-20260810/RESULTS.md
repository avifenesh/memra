# Host-staged PP boundary fallback — results

Date: 2026-08-10 | Branch: `lane/cx-hostbounce` | Base: `3f8ca2ef` |
Tested source: `650a2aecf5d6798f57108b705d71862924c41829`

## Verdict

**PASS as an opt-in host-class fallback.** `MEMRA_PP_HOST_BOUNCE=1` restores
byte-correct PP-2 serving on the Vast 2x RTX PRO 6000 host whose peer-copy API reports
success but does not preserve bytes. The healthy box1 default remains the original
`cudaMemcpyPeerAsync` transport and passed the full batched PP-2 exactness gate with zero
differing bits.

The flag stays default OFF. Enable it on this Vast host class while its peer path remains
untrustworthy; do not infer peer correctness from `cuDeviceCanAccessPeer=1` or a bandwidth
number. The fallback is an operational containment, not a diagnosis of the driver,
container, IOMMU, firmware, or PCIe fault.

At the final receipt, the Vast server was still running as PID 16734 with sharded
`MEMRA_PP_DEVICES=0,1`, `MEMRA_PP_HOST_BOUNCE=1`, and no `MEMRA_PP_SHARD=0` override.
`GET /readyz` returned ready, the worker was idle with zero Xid warnings, and the current
server log had no illegal-address, CUDA error, panic, segmentation-fault, or OOM match.

## Failure baseline

The pre-change serving repro for `Say exactly: hello world` returned HTTP success but
generated sixteen consecutive BOS tokens and no content. That exact response is preserved
in `raw/vast-pre/corrupt-repro.log`.

The preceding `research/p2pvast-20260810/RESULTS.md` receipt, read from the sibling
`wt-cx-p2pvast` lane, establishes the transport failure:

- Vast topology was `NODE` with `DmaRemapPeerMmio=1` and no intervening host reboot.
- Both directions reported `cuDeviceCanAccessPeer=1`.
- A synchronized 16 KiB `cudaMemcpyPeerAsync` check returned success with 16,320 of
  16,384 bytes wrong; memra's four boundary roundtrips and NVIDIA `simpleP2P` also failed.
- The bandwidth-only path printed an apparent roughly 110 GB/s peer number even though the
  destination bytes were wrong. The official host-staged matrix was roughly 34 GB/s.

The first host-bounce live serve attempt also exposed a separate mapped peer read:
Step-3.7's model-level `rope_freqs.weight` existed only on the primary device but full
attention layers on both stages dereferenced it. The resulting illegal-address receipt is
`raw/vast/server-illegal-address.log`. The final change uploads one immutable copy to every
distinct PP device and selects the stage-local copy.

## Fallback design

`PpNRt` captures an exact `MEMRA_PP_HOST_BOUNCE=1` setting at initialization and selects
one transport per stage boundary:

- same-device boundary: unchanged local D2D;
- healthy cross-device default: unchanged `cudaMemcpyPeerAsync`;
- opted-in broken-peer boundary: pinned D2H on the source stage's stream, source event,
  destination-stream wait, then pinned H2D into the destination slot.

The H2D is followed by the existing destination-local D2D into the stage-owned working
buffer. The existing RX event remains the write-after-read guard before that pinned slot
can be reused. No peer pointer is formed in the host-bounce branch, and peer-context
enables plus default-pool peer grants are bypassed.

Two page-locked, portable host slots are allocated once for each cross-device boundary
after the authoritative GGUF geometry is known. Capacity is computed as:

`PRIME_CHUNK_MAX_TOKENS * n_embd * sizeof(f32)`

Step-3.7 supplies `n_embd=4096` and the prime cap is 4,096 tokens, so each slot is
67,108,864 bytes (64 MiB), or 128 MiB for PP-2's one boundary. Every operation transfers
only its exact payload prefix: a one-row decode is 4,096 f32 values, or 16 KiB, while the
largest prime chunk can use the full slot. The width is not hard-coded.

### Cross-device path audit

| PP-2 serve path | Host-bounce treatment |
|---|---|
| Prime, pipelined prime, and Step-3.7 batched prime | Their `rt.tx` / `rt.tx_pipelined` / `rt.rx` activation handoffs use the selected transport. Invalid or unsplit remote-weight walks fail closed. |
| Eager decode and batched decode | The one-row or `[B,n_embd]` activation handoff goes through the same central transport. |
| Spec verify | The host-input PP verify split's `[T,n_embd]` boundary uses the same `PpNRt` transport. Device-resident stream-mode verify still peer-reads shared primary-device token, position, and embedding inputs, so it returns an explicit error under host bounce; serving spec admission is disabled instead of silently crossing that path. |
| Position state | There is no `pos_d` broadcast copy in the Step-3.7 path: every stage uploads its own position vector from host memory on its own stream. Gemma4's remaining shared-position peer read is explicitly refused under host bounce. |
| Step-3.7 RoPE factors | `rope_freqs.weight` is now copied once to each distinct PP device and resolved through the current stage's engine. This closes the mapped peer read found by the first live attempt. |
| Prefix and affinity cache snapshots | These issue device copies through the primary engine rather than the activation boundary. Prefix snapshots and plain-affinity checkpoints are disabled under host bounce. |
| Weight loading and output | Sharded weights are already uploaded from host through their owning stage. `MEMRA_PP_SHARD=0` is refused, and the primary engine must be the last/head stage so no returned hidden state or head weight is peer-read. |

Thus every explicit runtime peer memcpy in normal PP execution remains centralized in
`pp.rs` and is unreachable when the fallback is active. Paths whose hazard is a mapped
peer read or an out-of-band cache copy are shut or localized, not mislabeled as bounced.

## Verification

### Local 5090 compile and tests

The final source passed:

- `cargo check -p memra-engine -p memra-server`;
- `cargo build -p memra-server`;
- `cargo test --workspace`.

The engine unit set was 54 passed, zero failed, one pre-existing CUDA-only test ignored.
The three new unit tests cover default transport selection, Step-3.7's 64 MiB geometry,
and invalid/overflowing capacities. Every remaining workspace suite passed; the only
other ignored test was its existing hardware-only fixture. Raw logs are under `raw/local/`.

### Vast transport and content restoration

Rig: 2x RTX PRO 6000 Blackwell Max-Q Workstation Edition, CUDA 13.1 build, driver/container
stack recorded in `raw/vast/provenance.txt`. The final server binary SHA-256 is
`8d84431595109fc82c03fd8576b91f03dc1d9ede0548a9782a5b56ad26ee39ed`.

The two-device transport smoke explicitly skipped its peer arm under the flag, initialized
the 64 MiB geometry-sized slots, and made four alternating 1 MiB patterned roundtrips.
Every roundtrip reported `bytediff=0` and the smoke ended PASS. The startup log confirms
`HOST-STAGED pinned D2H -> H2D` and that peer access/grants were bypassed.

Content gates:

- `Say exactly: hello world` returned HTTP 200 with content exactly `hello world`.
- The b1fix-class prompt, temperature 0, seed 3407, c=1, max 64, sequential x3 produced
  one SHA-256 on all three requests:
  `b3260d8e22e1151df311c75f690ce59db37d81305dcd6fe048e7f7126cd2acb7`.
  All three bodies were identical coherent English rather than BOS garbage.
- The matching `Say OK.` short probe also produced one coherent hash across 3/3 requests:
  `ff8831822fbe209f6c00687fca83dde8c7a2d2556c11c01799cfc55464ff3b90`.
- The single 4k request consumed 4,107 prompt tokens and returned coherent text
  (`Got it, let's tackle this.`), not a repeated special token.

### Vast latency receipt

All runs were c=1, greedy, unclocked, on the running fixed arm. Decode cadence excludes
the first token: `(completion_tokens - 1) / (latency - TTFT)`. N is stated per row.

| Shape | N | Prompt / max output tokens | TTFT | Decode cadence | Errors |
|---|---:|---:|---:|---:|---:|
| Matching `Say OK.` short | 3 | 15 / 32 | 0.185 s p50 | 68.98 tok/s p50 | 0 |
| Fixed load-harness short | 3 | 228 / 8 | 0.656 s p50 | short sample only | 0 |
| Fixed load-harness sustained | 3 | 228 / 256 | 0.655 s p50 | 73.80 tok/s p50 | 0 |
| 4k cold single | 1 | 4,107 / 8 | 7.465 s | not scored | 0 |

Thermal regime:

- matching short N=3 began idle at 24-25 C / P8 and ended at 28 C / P1;
- the 228-token short/decode block began at 28-29 C / P8 and ended at 35-36 C / P1;
- the single 4k run began at 26-27 C / P8 and ended at 32/35 C / P1.

Comparison with the prior peer-path receipts is directional, not a controlled A/B: those
responses crossed a transport now proven byte-corrupt, and the fallback also closes
prefix/affinity/spec peer-read doors. On the exactly matching short shape, TTFT was
0.185 s versus the earlier N=3 0.184 s receipt (+0.5%, effectively flat); decode was
68.98 versus roughly 74.5 tok/s (-7.4%). Against the later fixed-binary single receipt
of 0.218 s / 70.4 tok/s, the new decode number is -2.0%; its lower TTFT is not evidence
that bouncing is faster. The 4k single was 7.465 s versus the latest corrupt-path
6.652 s, a +0.813 s / +12.2% cost.

The mechanism is consistent with a latency tax: the official matrix showed roughly
34 GB/s host staging versus an apparent roughly 110 GB/s peer path, while the decode
boundary is only 16 KiB and now pays two DMA submissions plus the event dependency.
At that size fixed latency dominates bulk bandwidth. The observed end-to-end decode
deltas are small enough that cross-run clock/thermal and prompt effects prevent a more
precise attribution; the 4k prime exposes the clearer transfer-volume cost.

### box1 default-peer no-regression

box1 was the designated 2x RTX PRO 6000 Server Edition verification host: PIX topology,
driver 595.71.05, CUDA 13.2. The source and binary hashes are pinned in
`raw/box1/driver.log`. The gate acquired `/tmp/memra-gpu.lock` with both GPUs idle and
explicitly removed `MEMRA_PP_HOST_BOUNCE` and `MEMRA_PP_SHARD` from its environment.

The startup receipt selected the original `cudaMemcpyPeerAsync` path with peer and pool
grants. `decode-batch-gate` then ran PP-2, 24 steps, two split repetitions at each
`B=1,2,4,8`, plus the unsplit `ppncache` control at every width. Every arm reported
`BIT-IDENTICAL (0 differing bits)`, the epilogue passed, the final verdict was
`ALL GREEN`, and the script exited 0 before releasing the lock. GPUs went from 27 C idle
to 39 C after the gate.

This proves the refactor did not change bytes on the healthy default peer transport.

## Recommendation and auto-detect proposal

Keep `MEMRA_PP_HOST_BOUNCE` as an explicit host-class flag now:

- ON for this Vast instance/container while a byte-integrity probe fails;
- OFF by default on box1 and other healthy PIX/NVLink hosts;
- never enable the unsharded PP rollback or re-open the documented peer-read doors with it.

A future boot-time auto-detect should be a correctness gate, not a topology heuristic:

1. Before model weights or the long-lived PP runtime are initialized, test every directed
   adjacent stage-device pair with the exact production `cudaMemcpyPeerAsync` API.
2. Allocate a 1 MiB source, destination, and host oracle; fill the source with a
   deterministic non-zero pattern and the destination with a distinct sentinel.
3. Copy, synchronize, D2H the destination, and compare every byte. Repeat in the reverse
   direction. Treat any API error or any mismatch as failure.
4. On failure, select host bounce for the whole server process before serving and emit one
   loud structured log containing device pair/direction, driver and runtime versions,
   topology, `CanAccessPeer`, mismatch count/first offsets, and the selected fallback.
5. Run the peer probe in a short-lived launcher/helper process using the same container,
   libraries, and `CUDA_VISIBLE_DEVICES`. This avoids leaving peer enables or suspect
   mappings in the serving process after a failed test.
6. Re-run once per boot/container/driver-library change. Never switch transport midway
   through a loaded model, and never accept bandwidth without a byte comparison.

This design catches the exact Vast failure mode: API success, capability bit set, fast
apparent timing, wrong destination bytes. It is a proposal only; this lane does not
implement auto-detection.

## Raw evidence index

- `raw/vast-pre/`: repeated-BOS corruption repro before the fallback.
- `raw/local/`: final local check, build, focused tests, and full workspace tests.
- `raw/vast/`: builds, transport smokes, initial mapped-read failure, fixed server logs,
  exact-output rows, performance rows, GPU snapshots, final environment/process/ready
  receipt, binary/model provenance, and SHA-256 manifest.
- `raw/box1/`: locked default-peer gate driver, exactness output, build log, thermals,
  source/binary provenance, and SHA-256 manifest.

No published perf board, runtime default, tag, release, or origin ref changed in this lane.

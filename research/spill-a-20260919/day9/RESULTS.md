# WP-A day 9 — native canonical v1.3 bindings

Repository: **avifenesh/memra**, lane `lane/spill-a-20260919`.
Integration replay was merged/pushed as `12122467a`; no main merge, tag or serving
deployment is part of this checkpoint.

## Implementation

- Independent source/destination graph and CUDA-consumer retention records.
- Frozen trait `retire_source` forwards to the native implementation.
- Taken pinned destinations share one backing/charge with the ticket until
  acknowledgement, then remain resident under the destination owner's charge.
- Mutable shared-host reuse refuses `Busy`; immutable pre-ack H2D reuse remains
  supported. Device release/hand-back refuses live ticket ownership before a
  blocking stream synchronization.
- The native gate calls the unchanged canonical `device_hand_back` and
  `transfer_source_retirement` functions. Both fixtures explicitly observe a
  genuinely pending producer, held by a CUDA callback, before entering the
  schedules. Graph lifetime objects are real retention pins, not CUDA graph
  execution captures. Unknown observation is injected, not a physical context loss.

See [V13-BINDING.md](../V13-BINDING.md) for the exact step mapping and ownership
contract. No frozen schedule, CUDA kernel, numeric program, dependency or
environment read was changed.

## Native execution history

Target shape: **one RTX PRO 6000 Blackwell 96 GB, 600 W**. All GPU cells use
`tools/tier-battery.py --rig pro-single` and `/tmp/memra-gpu.lock`; two cells run
back-to-back in one collector invocation. The runner retries canonical lock
contention every 60 seconds for at most 60 minutes, never a competing lock.

1. `4751b48e0`: native build passed; two lock refusals preceded execution.
   Both canonical functions printed their verdicts, but the additional early-drop
   fixture failed because it omitted the mandatory D2H producer fence:
   `Rejected { ... producer_fence: None }, error: NotReady }`.
   This is a failed complete campaign, not an all-green receipt. No roundtrips
   ran. Raw receipts are retained under `native/initial/`.
2. `188fd9f72`: native build passed. The strengthened held-producer source
   schedule passed; the held-producer hand-back fixture then blocked while its
   foreign owner was being constructed. cudarc `new_stream` synchronizes a newly
   wrapped primary context, conflicting with the deliberately closed producer
   latch. The collector retained the timeout. Foreign owner construction was
   moved before the latch in `9ce5e3fac`; immutable/mutable reuse regression
   coverage was extended in `1adf2be3d`. Raw receipts are retained under
   `native/held-producer-attempt/`.

3. **Final campaign, source `1adf2be3d9f8ea82596dbc5917407e35d597c939`: PASS.**
   Native release build exit 0; collector/driver, conformance and roundtrip exit 0.
   `native/final/attempt-00/` contains the successful full campaign. No lock retry
   was needed for this final invocation. Binary SHA-256 before/after:
   `3eae930dab5504c89bc5f7f18473c9d5762b4f4bceed87a5d7dd139ef4782be4`.

### Canonical verdicts — verbatim

```text
PASS v1.3 transfer_source_retirement native CUDA
PASS v1.3 device_hand_back native CUDA
PASS native governor zero after controlled drain
```

The same conformance invocation also passes all existing v1/v1.1/v1.2 cases and
an extra early-destination-drop check. Its full 11 verdict lines live in
`native/final/attempt-00/conformance.log`; these labels are printed only after
the canonical functions return successfully.

All **six N=1 exact roundtrips** passed: **4 KiB, 64 KiB, 1 MiB, 16 MiB, 64 MiB,
256 MiB**. Each row retains matching independent expected/actual hashes and
`byte_exact=true source_freed_host_live=true handback_no_copy=true governor_zero=true`
in `native/final/attempt-00/roundtrip.log`. The pre-ack immutable host reuse path
remains exercised; no roundtrip is replaced by a format or numerical fallback.

The final collector captured **56 samples at 250 ms**, **600/600 W** cap/max,
**35–37 °C**, and empty compute snapshots before/after. This is a short
correctness window (starting in P8), not a thermal-steady-state performance run.
`hardware-after.log` directly records the RTX PRO 6000 Blackwell Server Edition
and 97887 MiB total memory. Collector disposition remains
`executed-not-qualified`, `qualification: false`: native development ownership
correctness, not production/serving/default qualification.

Offline replay verified **all 86 remote file hashes** across the final and two
failed campaigns. The final campaign has 27 hashed files. Six replay/tamper tests
passed, including missing canonical verdict, mismatched roundtrip hash, manifest
omission and incomplete telemetry handling. See `replay.json`, `hash-audit.json`
and `mac/replay-tests.log`. Frozen v1.3 SHA-256 remains
`95c016abf65c3d627461c6b8c66993829d69eafe56b4195da8e1d53d34855ead`;
`git diff 12122467a 1adf2be3d -- crates/memra-tier/src/conformance/revision_v13.rs`
is empty.

## CPU / static checks

- `cargo test -p memra-tier --offline --no-fail-fast`: **196 passed**, including
  both unchanged CPU fake-backend v1.3 schedules and default-Unsupported coverage.
- Mac and Linux-target `memra-tier` + `memra-kv` checks: exit 0.
- CPU tier/KV all-targets clippy `-D warnings`: exit 0.
- Workspace fmt, explicit native-file rustfmt, flags census and diff check: exit 0.
- `DOCS_RS=1 cargo check --target x86_64-unknown-linux-gnu -p memra-engine
  --bin tier-transfer-gate --offline`: exit 0. This is a type check with docs stubs,
  not a native CUDA build.
- A broader Mac engine/gate check actually ran and failed on the existing
  Linux-only `cpu_experts.rs`/`spill_pread.rs` libc APIs (`cpu_set_t`, `CPU_SET`,
  `sched_setaffinity`, `posix_fadvise`, `POSIX_FADV_*`, `O_DIRECT`),
  not the changed files. `mac/engine-check.log` retains the full diagnostics.
  Mac engine portability is not claimed; the separate Linux/native checks are
  authoritative for this Linux CUDA backend.

No full-workspace/native clippy or serving/model-scale battery was run here.

## I/O decision input

[IO-BASELINE.md](../IO-BASELINE.md) now gives explicit illustrative +5% O_DIRECT
cold targets from the day-8 pread rows: **1276.539 MiB/s at 1 MiB** and
**3502.406 MiB/s at 16 MiB** (64 MiB pass; approximately 50.135/18.273 ms).
These are arithmetic decision inputs, not io_uring measurements. A candidate must
beat fresh same-box controls in five AB plus five BA pairs, then meet the composed
pipeline, tail-latency and headroom gates. **io_uring remains DEFERRED**.

No new storage measurements were made. Storage ancestry remains unproven; no
NVMe, spill-speed, cross-box timing, serving/default or production claim is made.

## Handoff

No contract gap requiring a frozen change remains for these two native schedules.
The source checkpoint is pushed; final receipts/docs are a subsequent lane commit.
The active worktree/branch are retained for lead integration, not merged or released.
All A native tmux jobs ended; owned temporary bundles and replay directories were
removed. This session used approximately **0.4 agent-hours**, below the three-hour
stop limit. Broader serving/model-scale qualification and io_uring admission remain
outside this completed day-9 correctness checkpoint.

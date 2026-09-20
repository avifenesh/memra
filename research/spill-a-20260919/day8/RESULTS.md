# WP-A day 8 — target-card transfer and block-device I/O

## Revision and disposition

Repository: **avifenesh/memra**, lane `lane/spill-a-20260919`.
Integration merge: `e4f7e633b27fc2caf85d553d6e0659449af1b3b2`, pushed before
execution. Both native binaries were built at that revision with nvcc 13.2 and
`MEMRA_CUDA_ARCH=120a`. Batch runner source: `7eba2ebb8`. Raw batch receipt commit:
`ed4a7d6d5`. Binary SHA-256 values, identical before/after:

- `tier-transfer-gate`: `769b589e17b36072bd871f8b946ce9345492e3a90714775fa52fe6e8c2304795`
- `storage-bench`: `43bf1e5d78af16343524b357884867f3410f97f1777173f80f9f7023610dc618`

**Existing native schedules and byte roundtrips PASS; full canonical v1.3 remains
HELD.** Storage and pread cells below are executed development evidence, not
native-serving qualification or a runtime-default/performance decision. No
runtime, numerical program, CUDA kernel, dependency, or environment read changed.

Target shape: **one RTX PRO 6000 Blackwell 96 GB**, observed cap/max **600/600 W**.
All cells ran via `tools/tier-battery.py --rig pro-single` under the canonical
`/tmp/memra-gpu.lock`. Compute snapshots were empty before/after each successful
collector. GPU telemetry was requested at 250 ms; short individual preliminary
storage cells can have zero samples and are not thermally qualified. The final
combined batch has its own full-window telemetry. No cross-machine timings are
compared and no production claim is made.

## Native conformance — verbatim

```
PASS v1 transfer_cancel native CUDA
PASS v1.1 transfer_complete_cancel native CUDA
PASS v1.1 transfer_lifetime native events + injected observation loss + graph retention
PASS v1.1 transfer_zero_accept Unsupported NVMe preserves owned input
PASS v1.1 acceptance exhaustive native mixed batch; rejected sibling blocks publication
PASS v1.2 transfer_completion_bytes native CUDA; stale epochs, ready publication, take once, authentic consumer fence
PASS additive source retirement Busy while source consumer bound; host destination survives source release
PASS native governor zero after controlled drain
```

`native/conformance-retry1/command.log` retains these exact strings. The first
attempt refused lock contention without running the gate. Collector verdict:
`executed-not-qualified`, `qualification: false`.

### Roundtrips

All six sizes passed once: **4 KiB, 64 KiB, 1 MiB, 16 MiB, 64 MiB, 256 MiB**.
`native/roundtrip/command.log` retains both expected/actual SHA-256 values and the
verbatim `byte_exact=true source_freed_host_live=true handback_no_copy=true
governor_zero=true` verdicts. This tests `retire_source` and original-pointer
`take_device` in the existing additive schedules. N=1 correctness, no medians.

### Why canonical v1.3 is still HELD

The integrated binary imports `memra_tier::conformance`, but its body does not call
`device_hand_back` or `transfer_source_retirement` from the frozen
`conformance/revision_v13.rs`. Existing additive tests are not renamed to those
new schedules. Static inspection also identifies a concrete binding limitation:
`CudaTransfers::pin_graph` retains one ticket-wide graph pin, and `retire_source`
refuses while that pin remains. The canonical source-retirement schedule requires
source retirement while an independent destination graph lifetime stays live.
The D2H implementation also retains a taken host destination as a whole-ticket
consumer until that lease drops; the frozen schedule checks destination lifetime
after acknowledgement. A faithful binding must resolve these lifetime surfaces,
not hide them with fixture flags or drop the destination prematurely.

Required next: design and qualify native producer/unknown/source-consumer/source-
graph/destination-consumer/destination-graph bindings and original allocation
hand-back against the unchanged shared schedules. The pending native bindings
are a blocker to **all-v1.3 PASS**, not a failure of the observed byte roundtrips.
No shared contract or runtime behavior was changed to force a green result.

## Block-device storage cells

Storage class, exactly: **block-device ext4 (virtio; NVMe ancestry provider-claimed, not proven)**.

`native/storage-batch/filesystem.log` records `stat -f`, `findmnt`, the
`/dev/vda1` ext4 mount and `/sys/block/vda/device` virtio path. This upgrades these
cells from overlay-only evidence to a real block device. It proves neither NVMe
ancestry nor media placement. The collector's coarse `overlay-unproven` enum
combines overlay and unproven ancestry; its raw metadata is preserved unchanged.
The precise observed class above is the interpretation of the filesystem probes.

Final batch: `native/storage-batch/attempt-04`, eight subcells, all
`status: byte-exact`, `backend_actual: linux-o-direct-read-write`, `fallbacks: 0`.
Each restore reads the object produced by the preceding roundtrip; checksums
match. `io_bytes` counts backend I/O, not physical media bytes.

| Valid bytes | Padded payload bytes | Roundtrip I/O bytes | Restore I/O bytes | Verdict |
| ---: | ---: | ---: | ---: | --- |
| 264 | 4096 | 81920 | 40960 | Both byte-exact |
| 4097 | 8192 | 102400 | 49152 | Both byte-exact |
| 1048576 | 1048576 | 5304320 | 2129920 | Both byte-exact |
| 4194568 | 4198400 | 21168128 | 8495104 | Both byte-exact |

Earlier separate 264/4097-byte cells also passed; preliminary lock refusals are
retained under `native/storage/`. By lead ruling, all eight final subcells and the
pread probe share one collector lock/receipt; the entire batch was synced and
pushed afterwards. The runner waited only on lock refusal, every 60 seconds with
a 60-minute maximum; attempt 04 succeeded after four refused attempts. No competing
campaign was interrupted and no alternative lock was introduced.

## Positioned-read decision input

All eight N=1 rows passed exact SHA-256 and explicit `mincore` cache preconditions.
See [IO-BASELINE.md](../IO-BASELINE.md) for the full numeric table, method and
unchanged io_uring admission criterion. On a single 64 MiB read pass:

| Chunk | Buffered cold/warm, MiB/s | O_DIRECT cold/warm, MiB/s |
| --- | ---: | ---: |
| 1 MiB | 978.294 / 13676.184 | 1215.751 / 1283.325 |
| 16 MiB | 2064.930 / 12798.828 | 3335.625 / 4189.292 |

**N=1 development plumbing, fixed order, no medians, not spill speed.** Host cache
state is measured, storage-controller cache state is not. The criterion remains
≥5% end-to-end pipeline improvement with ≤2% serving-tail regression and consistent
five-AB/five-BA evidence, plus correctness and measured route/SSD headroom. This
syscall probe does not replace a bounded-worker/native-consumer baseline.
**io_uring remains DEFERRED**; no new door or dependency was admitted.

## Verification and remaining work

- Native release engine/gate/storage build actually ran: exit 0, `native/build/`.
- Mac fmt and explicit native rustfmt: exit 0.
- Mac + Linux-target checks, `memra-tier` + `memra-kv`, offline: exit 0.
- `cargo test -p memra-tier --offline --no-fail-fast`: **196 passed**.
- CPU tier/KV all-targets clippy `-D warnings`: exit 0. This is not full native
  engine/workspace clippy; that broader scope was not rerun in day 8.
- Diff/flags checks: exit 0; push hooks ran without overrides.
- Five offline replay/tamper tests: PASS; full `verify_day8.py` replay: PASS.
- 152 remote receipt file SHA-256 values matched after rsync; native binaries
  were unchanged. See `native/remote-hashes.json`, `native/local-hashes.json`,
  and `receipt-replay.json`. Hash match does not promote qualification.

No main merge, release tag or serving deployment was performed. Canonical v1.3
native bindings, composed I/O/native-consumer pipeline and scored io_uring/default
admission remain pending. The active lane/worktrees are retained for that handoff;
owned temporary bundles and storage fixtures are removed after receipt sync.

This session used approximately **1.1 agent-hours**, below the 3-hour stop limit.
It is the **day-8 checkpoint of the original seven-day lane**, not a claim that
all work fit seven calendar days. Earlier sessions' cumulative agent-hours were
not reconstructed or invented.

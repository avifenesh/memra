# WP-A v1.3 native binding

The frozen schedules in `crates/memra-tier/src/conformance/revision_v13.rs`
are unchanged. `tier-transfer-gate` calls **both canonical functions directly**:
`v1::transfer_source_retirement` and `v1::device_hand_back`. See
[day9/RESULTS.md](day9/RESULTS.md) for executed revisions/verdicts, including failed
attempts; this design description alone is not a qualification receipt.

## Native retention model

Each transfer entry has separate source and destination retention records:
independent graph-pin sets and recorded CUDA consumer events. The legacy
`pin_graph` retains both sides. `pin_source_graph` and `pin_destination_graph`
retain only their named side; new source retention after source retirement is
refused. Owner code must drop graph pins only after its actual graph use retires.

`retire_source` requires observed DMA completion, source graph retirement,
source consumer event completion and absence of an overlapping live H2D consumer
binding on the D2H source. It does **not** require destination graph retirement.
The frozen `TransferEngine::retire_source` method now forwards to this native
implementation rather than inheriting the default `Unsupported`.

`retire` still requires both sides' use lifetimes to finish, including the exact
recorded destination consumer fence after publication. Destination residency is
not itself an unfinished consumer. A taken pinned destination and its ticket
share one sealed allocation/charge; the ticket retains its reference through
acknowledgement. This both allows a destination to outlive acknowledgement and
prevents early caller-drop from freeing bytes retained by a graph. There is no
second allocation, copy or second governor charge.

A taken pinned lease is read-only while shared: `write` and reuse as a mutable
D2H destination fail `Busy` without losing the owned input. Immutable H2D source
reuse remains supported before the earlier D2H acknowledgement (the existing
active tier gate uses this ordering). After H2D source retirement, the earlier
D2H ticket can still retain the pinned charge until its acknowledgement. A sole
remaining destination owner may write again after acknowledgement.

Device release/hand-back first rejects live ticket ownership without synchronizing
the stream. Otherwise a supposed producer-pending `Busy` check would silently
wait for the producer—or deadlock an owner-controlled producer. Final release
still synchronizes and uses the sealed registry's retained-lease/binding checks.
`take_device` returns the original pooled allocation; typed VMM ownership remains
`take_plane`, with pooled-only `take_device` refusing VMM as `Unsupported`.

## Canonical fixture mapping

| Frozen step | Native action/evidence |
| --- | --- |
| Initial pending producer | A native CUDA host callback holds the owner stream; `poll` explicitly observes `producer_done == false` before entering each schedule |
| Unknown | Inject observation loss, retaining owned backing and charges |
| Producer | Release the callback latch and synchronize the real owner stream |
| Recover | Reobserve the actual recorded copy event; source schedule takes and checks the pinned destination |
| SourceConsumer | Complete the authentic consumer event on a published H2D ticket binding the same source; record/reobserve the source-side event |
| SourceGraph | Drop only the source graph pin; source retirement/release succeeds while destination graph remains pinned |
| DestinationConsumer | Check the pinned bytes, record and observe the actual destination consumer fence; retirement still refuses the destination graph |
| DestinationGraph | Drop destination graph pin, retire with the recorded fence |
| Post-ack destination check | Exact pinned bytes and original pinned governor charge remain live; releasing the destination drains that charge |
| HandBack Consumer / Graph | Record/reobserve actual consumer fence; graph blocks retirement until dropped; retire/ack then take original allocation |
| HandBack returned once | Original device pointer and exact bytes match; registry/charge drain; second take fails |

The callback never calls CUDA, and its RAII latch releases on unwinding. Foreign
owner fixture construction is done **before** holding the producer: cudarc's
`new_stream` may implicitly synchronize a newly wrapped primary context.

Graph pins in this substrate fixture are real retention objects, not instantiated
CUDA graph-execution receipts. Observation loss is injected; this is not physical
context-loss recovery evidence. The schedules qualify ownership/control behavior
on the recorded target-card development rig, not a serving deployment or a
performance/default decision.

## Additional regression coverage

- All existing v1/v1.1/v1.2 schedules remain in the same gate invocation.
- Early drop of a taken host destination keeps backing and pinned charge until
  graph retirement **and acknowledgement**.
- Canonical source retirement checks write refusal while shared and successful
  write after acknowledgement.
- Six exact D2H→H2D roundtrips retain immutable pre-ack host reuse and original
  device-pointer hand-back; every size drains the full governor.
- CPU fake backends still run the same unchanged canonical schedules in
  `cargo test -p memra-tier`.

No contract gap requiring a frozen-schedule change has been identified. Execution
results, failures and remaining qualification limits are recorded separately.

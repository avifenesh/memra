# DFlash retained-prefix admission

Issue: #372. Base: `72aa777c3763a21281fb9c1c55b2499f2c14d1ee` (v0.136.0).

The admission plan now distinguishes native restore from a validated DFlash restore.
The DFlash binding carries the destination context capacity and either full cover or
an eligible suffix of at least `PRIME_MIN_T`. The leased source must retain its exact
token boundary, target identity, valid draft-tail storage and finite boundary logits.
Consumption revalidates this binding and the live DFlash route before target copying.

Only cold prefill workspace is replaced. Source residency is never credited as free;
full destination context KV, draft KV, fixed residual and reserve remain charged.
Full-cover DFlash restores retain a minimum-prime workspace allowance for conversion
and boundary sampling. Alternate donors are suppressed after a plan commits.

A failed planned conversion returns a rate-limit error before cold prime. The failure
barrier releases the serving lease, and the caller releases the admission lease on
both success and failure. There is no new runtime environment variable.

## Verdict

DRAFT: the restore contract passes, but the required 128k cold/1024-output plus warm
continuation does not pass on the candidate. No merge or capacity claim.

The unchanged v0.136.0 public control completed the cold request and then returned
HTTP 429 on the warm continuation. Three candidate trials failed during cold decode,
before any retained-prefix admission plan, at 166, 668 and 523 generated tokens.
The logs show a later shared FA-pool generation, including output growth from
50,380,800 to 100,761,600 elements, with old generations retained. Fresh entropy is
used for omitted seeds, so the passing main control does not establish a paired
regression result. The cold failure needs investigation before this lane can satisfy
its headline gate. Closed #373 is not revived or included here.

## Receipts

All commands ran on the authorized non-production RTX 5090 with
`MEMRA_CUDA_ARCH=120a`, `CARGO_TARGET_DIR=/root/target` and the canonical
`/tmp/memra-gpu.lock`. No build, test or gate ran on the local rig. Pushed with
`MEMRA_SKIP_PERF_CI=1` and local hooks disabled. Hosted CI still gates integration.

Full hashes, source hashes, request/profile/log hashes, counters and excerpts are in
[qualification.json](qualification.json), approximately 103 KiB. Binary abbreviations
in the table refer to these SHA-256 hashes:

- Main control: `d9ac87e69ee32fb309b46775bec75d4a1e1533b6ff2170cae3040727f4be6636`
- Final candidate: `84e8fdf5b5d807fceb139895fb81d8b48a83e8292a975f93a76fc3c8f18b3065`
- Fault overlay: `208b76117d1edf9e5b7f26dfa4a9cdb3934d53e986ebc5f9e6e4fed639ba9a02`
- Pressure readback overlay: `1433500890e2787914cc3aceb318bbe1a3caae6e4f8559261dd5a87de2ca4b10`

| Binary | Cell | Before / after | Log evidence |
| --- | --- | --- | --- |
| `d9ac87e6` | Recorded 128k cold then warm | Cold: 131074 prompt + 1024 output; warm: HTTP 429, 7081 MB charged vs 6908 MB attainable | `effective free 2106MB -> 6908MB` |
| `84e8fdf5` | Required candidate headline | Cold failed; warm not reached | `step OOM NOT parked ... generated 523, streamed 541` |
| `208b7611` | Full cover | Cost 2283 -> 1152 MB; reserve 1720 MB unchanged; 32708 cached | `DSPARK restore: 32708 of 32708 prompt tokens` |
| `208b7611` | Boundary / suffix | Cost 2288 -> 1204 MB; reserve unchanged; 32672 cached, suffix 121 | `DSPARK restore: 32672 of 32793 prompt tokens` |
| `208b7611` | Missing tail, short tail, invalid source, missing logits, lost source, short suffix | Seven fault cells, including consumed carrier below: HTTP 429; admitted remains 1; no OOM/replay | `admitted=false ... source_pins=0 pinned_bytes=0` |
| `208b7611` | Consumed carrier | HTTP 429; pool used returns to 23517179796 bytes; admitted remains 1 | `dspark restore declined (injected failure after carrier consumption); cold admission required` |
| `84e8fdf5` | Old-storage oracle and greedy cold/restore tapes | All four tap/feature/position/logit sets and both 64-token outputs byte-identical; 32672 cached | `DSPARK restore: 32672 of 32708 prompt tokens` |
| `14335008` | Co-located pressure | Before restore: 24729231252 pool-used bytes; after: 26047464532; 32672 cached | `active=0 parked_dspark=1 pinned_bytes=0 pool_used=26047464532` |

The full-cover, boundary, fault and pressure cells use a **diagnostic source overlay**
that forces a valid affordable plan on a 32k fixture. It does not alter charges,
reserves, model precision or the DFlash program. Faults mutate the actual leased
source after commitment or consume the real carrier before returning an error.
These cells establish the failure contract; they do not establish 128k admission.
Pressure uses `MEMRA_MAX_SESSIONS=4`, full vision residency and a second parked DFlash
session holding KV. No topology capacity claim is assigned.

The lost-source injection explicitly releases the admission pin before removing its
entry, so the caller correctly reports `admission_lease_released=false` with zero
remaining pins. Other fault arms release it in the caller. Metrics snapshots can be
stale after a rejected request because no new active tick runs; direct result-log
pool readbacks are authoritative for those failures. Successful pressure retirement
also reports zero pinned bytes with one parked session still resident.

## Software gates

- `cargo fmt --all -- --check`: passed.
- `cargo clippy --release --all-targets -- -D warnings`: passed.
- `cargo test --release -p memra-server`: 636 passed, 2 ignored.
- `cargo test --release -p memra-engine dflash`: 56 library tests passed, no failures.
- Explicit GPU plan matrix: full cover, boundary, missing/short/corrupt storage tail,
  tail position, missing/nonfinite logits, short suffix, source position/layout/key,
  lost source and capacity checks passed; each fixture ended with zero pins and bytes.
- Reinstating cold fallthrough after consumed-carrier failure and accepting a 15-token
  suffix produced two observed test failures (exit 101). Restoring the guards made
  both tests pass. This is mutation evidence for the former unsafe fallthrough and
  suffix policy, not a claim that every existing source validator failed on main.

## Reproduction and disposition

`replay.py` replays immutable request templates in an isolated server under the GPU
lock. `inject_faults.py` applies only to a disposable diagnostic checkout, never the
shipping source. It adds no runtime environment variable. `bank.py` retains compact
receipts instead of copying multi-MB logs. The public binary rejects the private
`MEMRA_REQUEST_LEDGER` setting, so the public replay omits only that wrapper setting.
All specified model, context, cache, reserve, vision and sampling settings remain.

No release or fleet action is part of this lane. If merged later, it ships in the next
batched release; fleet exposure is deferred by the owner. Issue #372 remains open.

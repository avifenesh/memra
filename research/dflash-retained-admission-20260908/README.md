# DFlash retained-prefix admission

Issue: #372. Initial base: `72aa777c3763a21281fb9c1c55b2499f2c14d1ee` (v0.136.0).
Final integration base: `dcfeab7c738912a150ebbfea277112724bb99de4`.

The admission plan now distinguishes native restore from a validated DFlash restore.
The DFlash binding carries the destination context capacity and either full cover or
an eligible suffix of at least `PRIME_MIN_T`. The leased source must retain its exact
token boundary, target identity, valid draft-tail storage and finite boundary logits.
Planning applies the same live DFlash load, sampling and hit-shape decision as
consumption. A known route veto leaves the full charge and existing defer behavior
intact. Consumption revalidates the binding and route before target copying.

Only cold prefill workspace is replaced. Source residency is never credited as free;
full destination context KV, draft KV, fixed residual and reserve remain charged.
Full-cover DFlash restores retain a minimum-prime workspace allowance for conversion
and boundary sampling. Alternate donors are suppressed after a plan commits.

A failed planned conversion returns a rate-limit error before cold prime. The failure
barrier releases the serving lease, and the caller releases the admission lease on
both success and failure. There is no new runtime environment variable.

## Verdict

PASS: the exact recorded 128k cold/1024-output request followed by its warm
continuation now admits with 131040 cached tokens, DFlash engaged and zero OOM/retries.
The unchanged main control returns HTTP 429 on the same warm request. No general
capacity claim is assigned.

[final-qualification.json](final-qualification.json) revalidates the composed source,
including the early route-policy guard, on current main. It records all software,
headline, fault, oracle and pressure gates.
[sequence-check.json](sequence-check.json) records the earlier exact-pair control and
candidate. Both sets use exactly the two specified requests,
with no sampling fields and no extra harness calibration request. Main and candidate
both produce 1024 cold output tokens; the candidate then produces 1024 warm output
tokens. Source hashes match the code used for the fault and oracle qualification.

The initial copied harness inserted a separate 1050-token calibration request before
the pair. Three candidate trials in that longer sequence failed during cold decode,
before warm admission, at 166, 668 and 523 generated tokens. Those receipts remain in
qualification.json. Their logs show later FA-pool growth, including output growth from
50,380,800 to 100,761,600 elements with old generations retained. Omitted seeds use
fresh entropy, so this is not a controlled causal attribution to the calibration
request. The longer sequence remains unqualified. Closed #373 stays closed.

## Receipts

All commands ran on the authorized non-production RTX 5090 with
`MEMRA_CUDA_ARCH=120a`, `CARGO_TARGET_DIR=/root/target` and the canonical
`/tmp/memra-gpu.lock`. No build, test or gate ran on the local rig. Pushed with
`MEMRA_SKIP_PERF_CI=1` and local hooks disabled. Hosted CI still gates integration.

Full hashes, source hashes, request/profile/log hashes, counters and excerpts are in
[final-qualification.json](final-qualification.json), approximately 54 KiB. Historical
trials remain in [qualification.json](qualification.json) and
[sequence-check.json](sequence-check.json). The final binary SHA-256 hashes are:

- Candidate: `be87c098583d6dfe246e6c8500f55da64df80a815301de34f2720bd48f89a44e`
- Fault/pressure overlay: `f7147c3b9d5c01ef31d600eea9b2f827dbd5d550d0994b9738e14a75ad08bd0f`
- Earlier exact-pair main control: `f287b51ed5d7458b66595df94d1598e656ad6cf49af95f66a995c965ef5aea16`

| Binary | Cell | Before / after | Log evidence |
| --- | --- | --- | --- |
| `f287b51e` | Exact 128k cold then warm, main | Cold: 131074 prompt + 1024 output; warm: HTTP 429, 7081 MB base charge plus draft/reserve vs 6934 MB attainable | `effective free 2132MB -> 6934MB` |
| `be87c098` | Exact 128k cold then warm, candidate | Both output 1024; warm cached 131040; cost including draft 7250 -> 4129 MB; reserve 1720 MB unchanged | `restored=131040 suffix=72 cost 7250MB -> 4129MB; reserve 1720MB unchanged; cold fallback forbidden` |
| `84e8fdf5` | Longer calibration-prefixed sequence | Cold failed; warm not reached | `step OOM NOT parked ... generated 523, streamed 541` |
| `f7147c3b` | Full cover | Cost 2283 -> 1152 MB; reserve 1720 MB unchanged; 32708 cached | `DSPARK restore: 32708 of 32708 prompt tokens` |
| `f7147c3b` | Boundary / suffix | Cost 2288 -> 1204 MB; reserve unchanged; 32672 cached, suffix 121 | `DSPARK restore: 32672 of 32793 prompt tokens` |
| `f7147c3b` | Missing tail, short tail, invalid source, missing logits, lost source, short suffix | Seven fault cells, including consumed carrier below: HTTP 429; admitted remains 1; no OOM/replay | `admitted=false ... source_pins=0 pinned_bytes=0` |
| `f7147c3b` | Consumed carrier | HTTP 429; pool used returns to 23517179796 bytes; admitted remains 1 | `dspark restore declined (injected failure after carrier consumption); cold admission required` |
| `be87c098` | Old-storage oracle and greedy cold/restore tapes | All four tap/feature/position/logit sets and both 64-token outputs byte-identical; 32672 cached | `DSPARK restore: 32672 of 32708 prompt tokens` |
| `f7147c3b` | Co-located pressure | Before restore: 24728514452 pool-used bytes; after: 26046747732; 32672 cached | `active=0 parked_dspark=1 pinned_bytes=0 pool_used=26046747732` |

The full-cover, boundary, fault and pressure cells use a **diagnostic source overlay**
that forces a valid affordable plan on a 32k fixture. It does not alter charges,
reserves, model precision or the DFlash program. Faults mutate the actual leased
source after commitment or consume the real carrier before returning an error.
These cells establish the failure contract. The independent exact-pair receipt
establishes the requested 128k admission result.
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
- `cargo test --release -p memra-server`: 648 passed, 4 ignored (includes current-main host-cache tests).
- `cargo test --release -p memra-engine dflash`: 56 library tests passed, no failures.
- Explicit GPU plan matrix: full cover, boundary, missing/short/corrupt storage tail,
  tail position, missing/nonfinite logits, short suffix, source position/layout/key,
  lost source and capacity checks passed; each fixture ended with zero pins and bytes.
- A new route-policy regression rejects pinned/gate-off peer load, the DFlash
  mid-band veto, disabled routes and unsupported hit/decode shapes before a discount.
  Reinstating metadata-only nomination produced an observed failure on the composed
  base; restoring the guard passed.
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

## Cleanup

The remote worktree and all owned scratch phases (2.5 GiB, 1.8 GiB and 1.9 GiB) were removed
after banking. All 34
recorded server PIDs were absent, no scratch handles remained, the GPU was empty and
`flock -n /tmp/memra-gpu.lock true` passed. The reference capacity lane, unrelated
remote checkout changes and shared Cargo cache were preserved. See [cleanup.json](cleanup.json).

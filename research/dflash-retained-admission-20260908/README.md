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

## Qualification status

Pending final receipt banking. All commands run on the authorized non-production RTX
5090 with `MEMRA_CUDA_ARCH=120a`, `CARGO_TARGET_DIR=/root/target` and the canonical
`/tmp/memra-gpu.lock`. No build, test or gate ran on the local rig.

- Remote format, clippy and server suite passed; server: 636 passed, 2 ignored.
- DFlash engine suite passed. The explicitly enabled GPU plan fault matrix passed.
- All four historical tap, draft-feature, position and logit oracle sets match exactly.
- Both historical greedy cold/restore outputs match exactly; restore cached 32672 tokens.
- The recorded historical 128k control cold request completed 1024 output tokens, then
  its warm continuation returned HTTP 429.
- Two candidate attempts failed during the cold request before warm admission, at 166
  and 668 generated tokens, when the shared FA pool grew. This is a blocking receipt,
  not a passing 128k qualification. The historical control binary contains the former
  FA rounding experiment; current main does not. Closed #373 is not revived here.

`replay.py` replays immutable request templates in an isolated server under the GPU lock.
`inject_faults.py` is a diagnostic source overlay, never part of the shipping source.
It forces valid affordable plans on small fixtures and corrupts committed sources or
consumes a carrier to exercise the real failure paths. It adds no shipping flag.

No release or fleet action is part of this lane. Ships in the next batched release
only after merge; fleet exposure is deferred by the owner.

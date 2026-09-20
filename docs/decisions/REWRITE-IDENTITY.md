# Bind rewrite admission to loaded bytes and the running program

Decision: 2026-09-20, issue #542.

Strict rewrite admission uses v2 receipts. The loader supplies the opened artifact digest,
the running executable digest, and the loaded numerical-program digest independently of the
bundle. The validator compares all three plus the serialized plan. Receipt and lock hashes
still protect bundle integrity, but cannot provide these trusted runtime identities.

The old policy accepted a same-plan bundle with an unrelated artifact label and arbitrary
implementation hash. The fail-before reproduction and CPU validation are recorded in
`research/modelplan-onboarding-rewrite-identity-20260920/`.

Rejected: treating a label, geometry, artifact.lock hash, or source revision as the running
implementation identity. Also rejected: promoting an unbound synthetic fixture receipt or
transferring a gate executable's receipt to a different serving executable. Unsupported
composite sources must refuse strict qualification until every contributing artifact is bound.

Unbundled execution remains explicitly legacy and unqualified. Configuring a bundle requests
strict admission, and an installation failure cannot retain legacy or previously granted
permissions. No hardware default or model support state is promoted by this change. Native
GPU qualification remains required before integration.

Native follow-up: the first Qwen3-0.6B probe compared fresh-F32-KV `forward_last` with
quantized-cache eager decode and correctly failed (max_abs3.1681318). The existing standing
`run-gen` quantized-cache verify-prefill control matched decode at1.907e-6. These are distinct
programs, so `forward-fresh-kv` now has a distinct manifest/receipt and cannot borrow eager
admission. The native gate uses independent verify-prefill and tokenwise executions of the
same cached-KV class; tolerances are unchanged. Fresh-KV outputs remain diagnostic evidence
and require their own parity receipt before strict admission. No shared attention math changed.

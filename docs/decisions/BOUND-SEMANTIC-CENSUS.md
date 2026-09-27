# Compiler-owned tensor interpretation

2026-09-20, issue #541. Implementation status: access/compiler foundation; dense and hybrid
loader migration remains pending.

The compiler owns the complete tensor binding, including physical names, transforms, shape and
storage contracts, auxiliaries and output-head ownership. Runtime access must consume that binding
instead of resolving a second set of checkpoint names. Keep the bundle immutable, borrow the
opened artifact, and report materialization errors separately from optional native-representation
absence.

Reject output-head inference from missing tensors: it makes an incomplete untied checkpoint look
like a valid tied checkpoint. Use explicit checkpoint configuration, then the model pack's declared
default. Also reject integer storage for a generic model weight. Inspection adopts these checks in
this increment; production-loader enforcement is a later part of the same issue.

The binding digest covers interpretation, and must be composed with #542's opened-artifact digest
before becoming trusted runtime identity. A binding digest alone cannot attest weight values.

Evidence: [CPU fixtures, validation and remaining scope](../../research/modelplan-onboarding-bound-census-20260920/README.md).

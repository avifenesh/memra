# Gemma root ABI review repair

This is a separate repair after immutable draft commit
`1254b644b2f4fae94c193ef407882b38e0fd8711` and root commit
`3a0f1ed71c804410498c46a2420617fbb380ede6`. Carver's root review requires changes for
`ARCH541-ROOT-01` and `ARCH541-ROOT-02`. The draft review remains an incomplete checkpoint,
not source GO. Both reports are preserved under `reviewed_input`.

## Repair

The runtime no longer requires a bindable GGUF artifact schema to obtain the engine aliases
for Gemma's registered HF parallel-MoE program. A private alias-only compiler reuses the common
plan traversal and adds the parallel program's semantic operand names. It returns names and
IDs, not a physical tensor contract. Public artifact construction retains the original refusal
for a parallel-MoE GGUF schema, and all reads still use the validated physical binding.

Gemma's dense-spelled FFN aliases resolve to its parallel shared branch. Fused gate/up and down
bank aliases preserve the encoded bank and shapes expected by the existing hybrid loader.
Router input and per-expert output scales resolve to their independent semantic roles before
generic codec-scale handling; a router quantization macro cannot shadow the router input vector.
Ordinary folded codec auxiliaries retain their owner-based resolution.

The dense loader's known optional router request resolves to an absent semantic role and returns
None. The regression also follows the next unconditional `inp_gate.weight` presence probe;
that absent per-layer embedding gate now has a semantic role too. This adds no E4B artifact
schema or config support. Unsupported configs, undeclared gate rows, unknown names, noncanonical
layer spellings, and out-of-plan requests still refuse. No raw-source fallback is introduced.

## CPU evidence

Carver's two preserved probe programs were copied into the owner's existing target namespace.
Their code and Cargo.lock are byte-identical; only Cargo.toml's dependency path points at this
checkout. The same probes fail with exit101 before the repair and pass with exit0 afterward:

- MoE: complete source preflight/census passes; callback changes from false to true.
- Dense: raw optional scale is None; prepared result changes from an unknown-owner error to None.

All four generated fixture files remain byte-identical to Carver's originals after the runs.
Three new regression tests exercise the complete registered Gemma source, actual hybrid-loader
operand names, byte/shape checks against distinct physical payloads, shared aliases, fused-bank
selection, explicit scale vectors beside a distinct quantization macro, dense absence, and
positive/refusal boundaries. A source wrapper panics on legacy name lookup.

Affected existing controls also pass: ordinary runtime4, composite runtime2, and draft12.
Strict GGUF all-target Clippy, Linux DOCS_RS engine lib/test Clippy, formatting and whitespace
checks pass. The engine check is a typecheck, not native execution. No broad frozen root suite
was rerun. Existing targets were reused with incremental compilation disabled.

`evidence.json` pins four changed source files, all raw logs and the preserved before/after CPU
executables. The preservation receipt verifies the remaining919 tracked crate files against1254,
including all engine/#542/worker/LRU, placement, source/config, canonical consumer, composite and
draft implementation files. Source review and finding closure remain pending with Carver.

No additional acceptance surface, remote job, native/model/default promotion, main intake,
merge, rental or topup is part of this repair. Parent coordination resumes the original Carver
task for finding closure and completion of the checkpointed1254 review.

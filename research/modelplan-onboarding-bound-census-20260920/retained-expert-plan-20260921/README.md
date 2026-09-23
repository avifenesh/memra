# Typed retained-expert program and reference baseline

`MoeMlpPlan` can now declare a strictly increasing set of retained original router IDs. The router
and selection-bias axes remain the original expert count; stored weight banks contain only retained
rows, in that declared order. Missing, duplicate, out-of-range and too-small sets fail. No pruned
weight row is fabricated. Existing unpruned plans leave the field absent from Debug serialization;
a direct comparison against frozen 82a28b22 proves identical pretty-plan and execution-manifest
hashes for unpruned Qwen MoE and Hy3 with MTP.

The reference executor masks pruned logits before softmax's maximum/denominator and before top-k,
keeps selection bias separate from output weights, rejects forced token-hash routes to pruned IDs,
and translates selected original IDs to compact bank rows. A hand-derived one-row MLP control
selects original ID 3 from a two-row bank and pins its actual result. Full uniform banks with pruned
rows fail shape validation. Parallel routed-output-scale topology explicitly refuses this operation
until its own tensor and reference program is implemented.

HF per-expert and GGUF per-expert groups retain original IDs in physical names. Missing retained
members and extra pruned weights fail the tensor census. GGUF groups preserve StackExperts as a
structural transform. The Step HF stacked-bank schema keeps router/bias axes at the original count
and weight axes at the retained count; this is a typed schema control, not checkpoint qualification.

`RetainedExpertRouting` is an explicit operation with no optimized execution surface admitted.
The generated execution-surface table includes its empty row, and every optimized rewrite manifest
is tested to carry the blocker. No source-adapter activation occurs here: current overlays/masks
still refuse before binding. Their complete physical provenance, shadowed/retained inventory,
identity and runtime original-ID mapping must be integrated and reviewed next.

Validation: all six new plan/contract/reference controls pass; the existing parallel-MoE test also
pins refusal. GGUF 368 pass/2 ignored; CLI13, Step1, inspector7, external3 and doctest2 pass.
The full reference library run has 68 pass and one Qwen3.5 numerical-bit failure. That exact failure
reproduces on read-only frozen 82a28b22 with identical actual and expected bit arrays; both raw failures
are preserved. The suite is not labeled green. This matches existing [issue #548](https://github.com/avifenesh/memra/issues/548):
macOS ARM64/Rust 1.97.1/debug, seven of eight entries differ, maximum 12 ULP and
8.940696716308594e-8 absolute difference. No GPU tolerance inference follows. Reference binary tests pass (two tests); no reference
doctests exist. All-target GGUF/reference/CLI Clippy, Linux engine/server lib/bin/test Clippy with
DOCS_RS=1 documentation stubs, formatting and whitespace checks pass. Initial test-placement and
GGUF-group-transform failures are retained separately, as are the generator output, old/new
serialization comparison, and baseline source manifest.

Independent review passed on exact `2e7ce065aea400aa512dc5b587ff2c776344b639`.
The reviewer independently ran the focused controls, parallel-topology refusal, registry/document
checks and current serialization comparison; lossless report and proofs are under `review/`. No GPU, root activation, model support, native rewrite or performance
qualification is claimed. The existing Step native vision implementation remains untouched. Final
#542 dependency intake still waits for the coordinator's accepted ref.

# Tool choice, 2026-10-03

Owner: lane A, issue #530. Branch `lane/rig-toolchoice-530-20261003`.

Question: do required/named requests constrain function names and arguments, and does
parallel=false limit generation to one call, through the actual streaming worker?

Comparison: retained baseline ELF `6ca4ea6a08606c459605ac65f9b40f458d928a63c9fc9af3afae5a347c6a4514`.
Its reviewed source binding is `bef76711297eaf86fe8aef51c8fb2d8e8d968b6b` plus
`research/cache-metrics-20261002/review-fixes/source.diff` (SHA256
`dfb654e642962b5c1c5455c6bc00ea6403402daabdf57343a90fad914f3f23bb`).
All 1,110 retained compiled inputs match main `42e9ed7447a19aa27d6be88b3341296b9ba16661`.
The baseline binary and source archive were hash checked before reuse.

Implementation: existing llguidance masks and completion, using template-specific call
frames around schema-checked JSON arguments. Unconstrained auto/none retain their existing
rendering and parsers. Qwen permits multiple constrained calls where parallel is allowed;
parallel=false ends after one. Gemma's existing single-call handoff refuses explicit true.
Unimplemented DSML, HY3 and GLM constrained-tool dialects refuse requests by name.
No CUDA, numerical operation, model pack, default policy or support-state change.

Budget: owner-authorized rig-only lane A530; $0 rentals and paid compute. Cached artifacts
only. Heavy builds and native cells use the shared rig.py broker, one heavy job total,
4 CPU/32 GiB/no swap, compiler jobs2, private target. GPU cells hold canonical FD9.
Correctness cells are bounded by 1,200 seconds, context4096 and per-request token caps.
No performance tuning, production access or owner accelerator lanes.

Checks: existing changed-input planner with affected/omitted reasons; server CPU contracts
for wire selection, schemas, template refusals, single-call masks, quoted delimiters,
Responses translation and fresh independent parsers. Native streams on cached Qwen3.5-9B
Q8_0 and Gemma4-12B Q4_0 must witness declared name/schema, terminal call count, argument
stream assembly and actual grammar masking. Baseline/candidate auto and tool-none are
mandatory regressions. Native identity binds source, binary, model, request, hardware,
context and helper hashes. Missing mask witness, wrong name/schema, duplicate calls or
altered auto/none must fail the assertion map; a build or planner result is not a native pass.

Decision: stop after these acceptance edges pass and reviewed PR/CI/merge/cleanup finish.
Diagnose concrete failures before any repeated cell. Preserve failed raw records.
A local fixture result does not establish model-family qualification.

Status: bounded native acceptance passed. See RESULTS.md and the hash-bound native-v3 receipt. Review, full selected CI and manager merge remain.

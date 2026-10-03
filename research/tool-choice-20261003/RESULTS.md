# Tool-choice acceptance, 2026-10-03

The two cached-artifact native streams passed the bounded #530 acceptance.
Required and named calls obey the declared function/schema, and parallel=false
completes exactly one call. Argument strings containing both dialect closing markers
remain valid data. Ordinary auto/none generation and token/cache counts match the
retained baseline. Raw elapsed timing is preserved and excluded only from cross-time
identity; changed token counts, cache counts or payload still fail.

## Source and execution

- Candidate source: `a19a907d7402b2a5378c1cb1aeab22037fce2775`.
- Candidate server SHA256: `5a1e23aad8e6ada860d704f97764a406a77cd2c1da7e6553007f664dc4b5ce65`.
- Baseline server SHA256: `6ca4ea6a08606c459605ac65f9b40f458d928a63c9fc9af3afae5a347c6a4514`.
- Controlled native job: `c4c62f315a23`; canonical FD9 lock held, exit0,35.52 seconds.
- Hardware: NVIDIA GeForce RTX5090 Laptop GPU,24463MiB,driver595.91.07.
- Requests: context4096,max_tokens512,greedy temperature0,seed530,explicit reasoning_effort=none.
  These are correctness fixtures, not vendor-default or performance measurements.
- Fresh baseline/candidate boots per artifact; prefix/reuse disabled only for this diagnostic.
  Server PIDs and telemetry were owned; final GPU compute list empty.
- Every source/helper input checked before and after the cell; exact executable retained privately.

The current receipt carrier adds documentation, immutable receipts and CPU replay
registration only. All compiled runtime inputs match the tested source. The raw
frozen-input manifest and source-binding.json retain the exact custody details.

## Assertions

All15 native behavior edges and8 coherent red controls passed through the existing
validation_coverage contract. Required/named/single-auto observed34 masked steps on
Qwen and38 on Gemma, valid declared arguments,one call,tool_calls termination and
positive usage. Missing masks,wrong names,wrong schemas,duplicate calls,changed auto,
changed none,changed context and wrong sent policy each fail the receiver.

Unknown names and invalid schemas return400 with tool_choice named. Gemma explicit
parallel=true returns400 with parallel_tool_calls named; its catalog capability is false.
Qwen advertises parallel capability. The unimplemented DSML/HY3/GLM constrained
routes retain explicit CPU-tested refusals. Fresh parser tests preserve initial modes,
JSON argument mode and independent consumed state, including choice-row integration.

Artifact hashes and exact request/response/SSE traces are in the receipt. Source-bound
helper snapshots and the independent expected context allow offline CI admission of
all23 edges. The optimizer reports qualification=false; it is not a model-support gate.

CPU validation:1090 server tests passed;28 declared manual/hardware ignores remain
listed in the private static census. Strict same-feature-program release Clippy,
formatting,flags checks,16 receiver/replay controls,94 validation-framework checks and the
14-package registry passed. The committed actual-receipt replay checks both native-v3 and native-v4.
Native-v3 remains evidence for its own earlier source only. The conservative full CI fallback remains required before merge.

Earlier port-preflight and timing-oracle failures are retained privately. The timing
replay exposed null frontend error.param values; those were repaired,recompiled and
proved in this successful fresh cell. Failed records were not rewritten as new runs.

No CUDA/kernel math,model numerical program,model support-state record,serving pin
or runtime-default promotion changed. Evidence applies to these artifacts,binary,
requests and hardware; it does not establish model-family qualification.

publicity: skipped: maintenance release

qwen artifact SHA256: `0825505bda37933f5856fd0751273b3bdf7224961d81dad9c4fcc1d47d49210c`;bytes 9527501696.

gemma artifact SHA256: `93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b`;bytes 6975879296.

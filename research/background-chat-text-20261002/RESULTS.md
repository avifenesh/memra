# Background chat and text delivery

Memra #914 adds opt-in background jobs for chat and text completions. Results keep the submitting API's fields, token counts and errors. Tenant ownership is checked on every poll and cancel. The switch remains OFF, retention remains 900 seconds and accounted storage remains 64 MiB. The broader #550 policy contract stays open.

Stored output now reserves capacity before buffering and terminal settlement, including Responses. A result the store cannot retain settles unbilled. Stock publication hides the result during settlement and protects it from concurrent collection, cancellation and writes. Unsupported backends fail visibly. Accounting is approximate; this is not an exact process RSS limit.

Final runtime source: `39b3962f4bca577a45d32911f90bc73576a68486`.
Observer binary: `296d90731d4d4ccf11277d687c3b9742ad1b016dc0f14704cdb8e390d06a8194`.
Server binary: `07c42e0e5d358b56e2117dbb0179ad0cf4582adfbf870ad88a01a7f8023f0672`.
Hardware: NVIDIA GeForce RTX 5090 Laptop GPU, UUID `GPU-1a3cbffc-29df-926c-df5c-29b4c210ef5d`, driver 595.91.07, 24463 MiB. All GPU cells held the inherited canonical FD9 lease; clean compute-process checks followed shutdown. No rentals or model downloads.

CPU: 17 focused background contracts, 1084 full server tests, nine Python collector controls, 94 validation-planner controls, strict release all-target Clippy, binary builds, formatting/diff/flags passed. The full suite retained 28 declared hardware/manual ignores. The source-bound composite map keeps mandatory regressions outside reduction. The selector's conservative full fallback for its own changed test input is retained in `receipts/coverage/affected-plan.json`.

NVFP4-MTP artifact `52c9cceb190055e0591a9a30c21f7200572eaf3ff1c59f6e9a1eda838a8f39de`, context 8192:
- Fresh OFF/ON boots passed chat, native-text and OpenAI-text sync/background identity, including native token ids, tenant 404, key rotation, repeated reads and terminal conflicts.
- Bare chat/text omitted decoder fields. Loaded metadata hash `a02a22ff5d95c0dedcd18305c9903a7173260a60d28d6a3b67ffc9e57a7911e3` declared both vendor arms and preserved the artifact/template default. Actual post-listener bursts resolved T1/P.95/K20/min0 with penalties enabled. Presence/repetition values are fixture declarations; the trace directly observes the listed sampler fields and penalty enablement.
- Cold/warm text and chat restored exactly 480 prompt tokens, preserved 256 generated tokens and original text. Speculative execution after cache restoration was observed in 71 text rounds and 100 chat rounds. Cached cancellation retained 34 text tokens and 32 chat tokens with explicit errors and one partial callback. Following synchronous requests matched the original reference. These are bounded generation tapes, not answer-quality or conversation-history claims.
- Eight terminal callbacks in the short cell and eight in the cache cell; no late duplicates or unsettled receipts.

Q8_0 artifact `0825505bda37933f5856fd0751273b3bdf7224961d81dad9c4fcc1d47d49210c`, context 32768: Chat background retained 18,001 tokens and the exact 600-string schema result after 303.290 seconds, equal to the independent streaming reference and its usage. Native cancellation preserved 32 output tokens with an explicit error and one partial callback. Foreign running polls/cancels returned 404, and repeated rotated-key reads returned the identical result. Text background retained all 24,000 generated tokens after 412.057 seconds, equal to the independent streaming reference and worker-truth usage. It reports `incomplete`/`MaxNew` for the declared token budget rather than claiming a completed answer. Native cancellation retained 32 actual tokens with an explicit error and one partial callback. This evidence does not transfer across artifacts or promote model support, defaults, performance or native release qualification.

Failed attempts remain visible. The old-order storage red control produced a billable completion and failed the new unbilled assertion. Build failures retain their logs and source bindings. Short v1 passed its executed checks but mislabelled the manifest context as 32768; its raw environments show 8192. It is superseded for context acceptance by short v2. Private originals and public whitespace-normalized stdout hashes are preserved separately.

All four final native cells settled 22 requests exactly once. Long Chat/Text each had two complete callbacks and one cancel-partial callback. The worker itself reports more than 90 seconds, independently of observer timing. Both source-bound composite coverage admissions passed (28 short/cache edges; 25 long edges). Missing-edge and wrong-source-contract records were refused.

Public compiler transcripts redact transitive TLS dependency labels required by the boundary guard. Original bytes remain private with their hashes. This changes no test assertion or measurement.

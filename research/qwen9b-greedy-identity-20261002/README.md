# Short cold Qwen greedy identity (#918)

The cold sub-16-token speculative prime used the GDN batched T=1 target step. Plain HTTP prefill uses eager `decode_step` below the same 16-token floor. These produce different cache state and logits. This was a prefill mismatch, before draft verification: K=1 and rowwise verification reproduced it.

The synchronous and cooperative speculative primes now use eager `decode_step_h` for this short cold case. Longer tokenwise overrides, segmented/carried primes, generated-token target steps, verifier math, sampler defaults and tolerances are unchanged. The `run-spec` short GDN oracle now follows the actual plain HTTP prefill program.

## Native evidence

All jobs use one broker-controlled RTX 5090 Laptop GPU with the canonical lease and owned-process cleanup. No support-state promotion or target-pair qualification is implied.

- Source: `a9ee8e59b9edcb3754fc1f03d89396f2d295abc0` plus the exact `fix-build/source.diff`; its SHA-256 is `acf04b8a1ca268ba94523ed52e5cde67dbacc11b6ee855947705c813398bb684`. Binary hashes are in `fix-build/binaries.sha256`.
- Qwen3.5-9B NVFP4-MTP GGUF: `52c9cceb190055e0591a9a30c21f7200572eaf3ff1c59f6e9a1eda838a8f39de`. Model inspection/config verification pass; plan, census and rewrite eligibility are in `inspection/`. Native rewrite qualification remains a separate obligation.
- Qwen3.6-35B-A3B IQ4_XS MTP GGUF: `df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf`, pinned revision `5bc3e238d916f48a861bac2f8a1990a0e9b7e98d` as recorded in the companion #777 receipt.

`before/` retains the unchanged-binary failure. The exact raw prompt is “Count from one to twenty in words, separated by commas.” (12 tokens), temperature 0, max_tokens 16, fresh native-format streams. Default MTP, K=1 and rowwise verification each differ from plain. An eight-token prefix also differs; lengths 1, 15, 16, 17 and 32 match. The job exits 1.

`after/` retains the same four fresh process arms and requests. All 24 text comparisons pass. All 21 blocking prompt-ID comparisons also have exact token-array equality; the three primary streams compare text because native SSE does not expose a terminal token array. `run-spec` on the same raw 12-token prompt passes K=1 through K=8 for 64 generated tokens with positive acceptance at every K. The job exits 0.

`controls/` has eight passing gates: `run-gen`, short and longer `run-spec` K=1..8, and `spec-on-cache-hit` for each named artifact. The cache gate includes cold/restore equality, sampled engagement, stable prefix accounting and growth. These are local regression controls, not a performance claim. All eight selected CPU schedule tests pass in `fix-build/prime-tests.log`.

The initial probe-build job compiled successfully but its inspect invocation used an invalid pack name, `qwen35_dense`. Its error is retained in `model-inspect.log`; the corrected canonical `qwen35` invocation passes in `inspect-valid.log` and `config-verify.log`.

# Background Responses endpoint acceptance, 2026-10-02

**PASS for the recorded local API scope.** One background request returned its full
600-item JSON result after 246.917 seconds and 18,001 output tokens. A second request
was cancelled after producing 244 tokens. Its retained text is a prefix of the original
result. All prompt, cached and output token counts match their terminal callbacks.
All five accepted requests across both boots produced exactly one terminal callback.

The change scopes job storage and cancellation by authenticated tenant and adds
`ServerWiring::with_job_store`. The stock store reserves enough room for a terminal
failure record even when its output budget is full. The background switch remains
OFF by default; retention and total capacity remain 900 seconds and 64 MiB.

## Source and conditions

- Runtime build: `95c7b7cba910c58fc218df06b862310f79f6139e`.
- Executed tree: `5c9cb225f5edab533f44e4a93d969ff7a302b0de`. The only changes after
  the build were the CPU skip-census parser/tests and the Python port preflight.
  Compiled source, Cargo manifests/lock and build configuration were unchanged.
- Binary SHA256: `78da2bdcd21248de9740ce753c81eb41b4458ac0dc3819f4818bd499ffdbb840`.
- Model: `qwen3.5-9b-judge-q8_0.gguf`, SHA256
  `0825505bda37933f5856fd0751273b3bdf7224961d81dad9c4fcc1d47d49210c`.
- One NVIDIA GeForce RTX 5090 Laptop GPU, 24,463 MiB, driver 595.91.07.
- Context 32,768; greedy sampling; reasoning disabled; prefix cache disabled.
  The long request has a 24,000-token output budget and an exact 600-string JSON schema.
- Separate fresh OFF and ON processes, in one exclusive GPU cell. The ON process runs
  short synchronous/background identity controls before the long completion and cancel.
  The process group was limited to four CPU cores and 32 GiB host RAM, with no swap.

This is one functional acceptance run for these bytes and hardware. It does not change
any model support state or qualify another target.

## Independently checked properties

| Property | Assertion and evidence |
| --- | --- |
| Default-OFF control | Synchronous response completes; background request returns 400 naming `background`. `raw/attempt2/off/summary.json`, `http.jsonl`. |
| Tenant isolation | A foreign key gets 404 for the running job's poll and cancel, and for the cancelled result. CPU tests also cover a forged internal key, key rotation within a tenant, missing auth and terminal foreign cancellation. |
| Delivery and deadline | Queued acknowledgement arrives within 30 seconds; terminal callback arrives after 246.917 seconds; status is completed. Duration ends at the observed callback, not the later poll. |
| Original result | The full array equals all 600 requested strings. Repeated polling returns the same terminal envelope. Short synchronous/background output and usage agree. |
| Cancellation | The second job reports cancelled with 244 tokens. Its text matches the original output prefix; a second cancel returns 409. The worker log records its channel closing after 244 generated tokens. |
| Accounting | Exactly one callback for each of five accepted requests; long completion uses `complete`, cancellation uses `cancel_partial`; every usage field agrees. No unsettled receipt is logged. |
| Bounded storage and TTL | CPU regressions pass for the exactly-full 256-byte reservation, terminal failure fallback, stale writes, terminal immutability and TTL-to-404. No native storage-cap claim. |

Run `python3 research/background-responses-20261002/verify.py` to recheck the retained
accounting and partial-output assertions. The native collector is
`tools/background-responses-gate.py`, using the prebuilt
`background_accounting_gate` example with `--external-lock 9`. Raw requests, responses,
callbacks and server logs are under `raw/attempt2/`.

## Failures retained

Both local full server suites compiled and ran: 1,036 passed, one failed, 27 ignored.
The sole failure in both debug and release was the wall-clock batching assertion
`dsv4_serve::c4_host_budget_tests::one_workspace_keeps_jittered_lanes_in_full_batches`.
All background API regressions passed. Full raw CPU output is retained. The DSV4
coalescer implementation was unchanged; its scheduling-sensitive test remains under
separate investigation. GitHub server-tests passed at `5d7191f4df`.

The first native attempt passed the OFF control but refused the second boot during
its port preflight. No listener remained afterward. The collector now permits TCP
address reuse after shutdown, and the passing pair uses distinct loopback ports.
The first attempt's output and failure are retained under `raw/attempt1/`.

The CI skip-census failure was a parser defect: it counted a brace inside a Rust string
and invented a nested module name. The shared fix passes 18 tests and verifies all
61 declared rows without changing the manifest or skip budget. Revuto is capped at
two prior rounds; it did not review these updates. The author self-review is on #910.

#550 remains open. Chat/text completion background delivery is split into #914,
and the owner decision on retention/cap defaults remains open. Shared storage does
not transfer live cancellation between processes; routing and recovery remain the
deployment's responsibility.

Local model-directory prefixes and the device UUID are normalized in these public
copies. The device identity is retained as a SHA256. No request text, result, usage,
error or timing was changed by normalization. Original queue logs and native receipts
remain in the private dated receipt archive.

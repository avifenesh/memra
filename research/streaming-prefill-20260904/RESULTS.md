# Streaming prefill heartbeat

The first-token peek previously held headers until generation started, so the SSE
keepalive was constructed too late to protect a long prefill.

The opt-in extended streaming path commits after an operator-selected interval, starts
the existing five-second SSE comments, and keeps enforcing the original first-token
deadline. A subsequent timeout is rendered in each API's error grammar and settles the
zero-debit `deadline_exceeded` outcome. Ordinary deadlines retain pre-header HTTP 408.
Admission still precedes this commit: the gate covers admitted prefill, not unbounded queues.

## Verification

- 605 server unit tests passed, including the real five-second keepalive and immediate
  cancellation after a client disconnect during silent prefill.
- Clippy on all memra-server targets passed with warnings denied.
- Flags census passed: 854 runtime reads, zero undocumented flags.
- Fresh-process configuration test passed: streaming defaults to 300,000 ms when configured,
  non-streaming stays at 90,000 ms, and the production peek commits before a generated token.
- `tools/local-ci.sh` passed: kernel cells, graph lifetime/corruption detection, HTTP smoke,
  64 concurrent streams, cache accounting, sampled cache/spec checks, 386 CPU engine tests,
  and all three GPU engine tests. Missing optional model artifacts were reported as skips.
- Manual `tests::prefill_proxy_fixture` through an isolated Cloudflare named tunnel:
  both arms delayed their first synthetic token by 150 seconds. The held-header arm
  returned 524 after 125.032717 seconds. The early-commit arm returned 200, headers at
  16.238256 seconds, completed at 151.227611 seconds with the token and `[DONE]`.
  The fixture uses the actual prefill bridge and SSE serializer, with synthetic events.
  This verifies transport behavior and is not a model performance measurement.

## Deployment

Set `MEMRA_STREAM_TTFT_MS_MAX` to the measured model's required first-token window and
`MEMRA_SSE_PREFILL_COMMIT_MS` below every upstream header timeout, with room for the
five-second heartbeat interval. Both default to the existing behavior when absent.
An extended streaming maximum without a positive earlier commit threshold fails boot.
Keep ordinary non-streaming limits and proxy deadlines independently bounded.

No production deployment has been qualified by this change yet. Before enabling it,
check the consumer's stream-error handling, admission queue, advertised metadata, and
real model request through the full route. Client TTFT measurements must ignore comments.

Publicity: skipped — maintenance fix.

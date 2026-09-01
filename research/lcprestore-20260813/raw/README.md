# Raw evidence index

The campaign outputs are the unmodified, tee-first files copied back from the isolated box1 run.
`post-run-verification.log` is the exact stdout of the separate tenant-clean probe after the runner
had exited.

## `attempt1-qwen3-unsupported/`

Run window: 2026-08-12 23:34-23:42 UTC. Source commit:
`2581dd31c0814cb7d99e0872bc5e2c5d77af5fe5`.

The official Qwen3-4B dense artifact could not load through the pinned server's hybrid-only model
surface. No request ran. `orchestrator.log`, `exactness/control-server.log`, and `gpu/cleanup.log`
carry the fatal text and 0 MiB cleanup receipt.

## `attempt2-gemma-exactness-fail/`

Run window: 2026-08-12 23:49-23:55 UTC. Source commit:
`0ab3c23658b4949b4ea33a492ef5601ce53c185b`.

- `build/`: source commit, dirty-state receipt, exact model hashes, model metadata/config, cargo
  tests, and release builds.
- `exactness/requests.jsonl`: raw request records including returned UTF-8 bytes as base64,
  cached-token counts, TTFT, output hashes, namespace, physical card, and UUID. The last row is the
  reducer summary with the verbatim failure set.
- `exactness/candidate-server.log`: split source/restored hashes and partial-hit/refusal logs.
- `exactness/control-server.log`: feature-off server log.
- `gpu/`: 250 ms and 1 s telemetry, VM state, ownership preflights, 0 MiB cleanup, and the separate
  post-run GPU/port/lock receipt.
- `orchestrator.log`: one combined tee of the complete run in execution order.

The exactness failure terminated the runner before performance, mixed-serve, or standard battery
directories could receive outputs.

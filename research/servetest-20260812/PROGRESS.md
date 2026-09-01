# cx-servetest progress

Last updated: 2026-08-12T15:37:14+03:00

## Scope

Run the owner-funded two-day public Qwen3.6 serve test on Vast instance `47529373`. The endpoint
now serves both qualified listings from one native `memra-server` process:
`qwen/qwen3.6-27b` and `qwen/qwen3.6-35b-a3b`. Both retain the sold shape: tenant concurrency
cap 4, budget-sized prefix cache, OpenAI compatibility, plain decode, and exact usage accounting.

The deployed binaries remain pinned to `b227063f896d261a27922c4aa814b89802080c1f`. That revision's
runtime is source-identical to the `ac6ef049b` runtime used by the pair requalification merged at
`09900dcaa`; this lane is based on that evidence merge. As of this update, `origin/main` has
advanced to `8e2549d1` with runtime changes. Those later bytes were not introduced into the live
two-day test.

Secrets stay outside the repository. This lane records only secret locations and fingerprints,
never key material.

## Status

- [x] Dedicated worktree and branch confirmed.
- [x] Required progress ledger created before deployment work.
- [x] Build and gate the pinned Q27 runtime on the Vast RTX PRO 6000.
- [x] Launch authenticated Q27 through trusted TLS with cap 4 and hourly monitoring.
- [x] Verify the deployed binary's native multi-model registry.
- [x] SCP the Q35 IQ4_XS GGUF from local storage and match size and SHA-256 on both hosts.
- [x] Pass the requested on-box Q35 `run-gen` prefill and batched-prime argmax checks.
- [x] Pass a co-resident two-model canary without changing the production listener.
- [x] Promote the native Q27 + Q35 process on `127.0.0.1:8002` with rollback protection.
- [x] Pass the public Q35 21-check protocol/accounting battery through `api.tiyuvta.ai`.
- [x] Pass a manual two-model monitor run: 12/12 per model, 24/24 total, zero errors.
- [x] Confirm the first real cron-fired two-model run: 12/12 per model, zero errors.
- [ ] Accumulate the full observation window through at least 2026-08-14T08:28:55Z.

## Receipts

- Primary OpenAI base URL: `https://api.tiyuvta.ai/v1`.
- Fallback OpenAI base URL: `https://<relay-host>/v1`.
- Models: `qwen/qwen3.6-27b`, `qwen/qwen3.6-35b-a3b`.
- Local bearer-key handoff:
  `/home/avifenesh/.local/state/memra/cx-servetest-47529373/api-key` (mode `0600`, outside the
  repository).
- Original Q27 correctness: `raw/gates-20260812T082332Z/`.
- Q35 transfer identity: `raw/q35-artifact-20260812T120356Z/`.
- Q35 on-box argmax: `raw/q35-run-gen-20260812T120437Z/`.
- Passing pair canary: `raw/pair-canary-20260812T120724Z/`.
- Passing production cutover and public Q35 battery:
  `raw/pair-cutover-20260812T121353Z/`.
- Manual two-model monitor: `raw/hourly/20260812T121437285712Z/`.
- First cron-fired two-model monitor: `raw/hourly/20260812T123701636462Z/`.
- Full report: `RESULTS.md`.

## Live processes

- Vast tmux `cx-servetest-server`: supervised native Q27 + Q35 server; pair ready since
  2026-08-12T12:14:12Z.
- Vast tmux `cf-tunnel`: primary `api.tiyuvta.ai` ingress; pane PID remained unchanged across
  the pair cutover.
- Vast tmux `cx-servetest-relay`: sslip.io fallback ingress; pane PID remained unchanged across
  the pair cutover.
- Vast `/usr/sbin/cron`: two-model public monitor at minute 37 of every hour.
- Revengent Caddy 2.11.4: sslip.io TLS terminator, enabled and active.

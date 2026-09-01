# cx-priceset progress

## Objective

Apply the owner's 2026-08-12 pricing order: list both served Qwen3.6 models at
approximately 5% below their live canonical OpenRouter market prices and
propagate the approved prices across the live feed, repository metadata,
BitRouter PR, connection packet, prior submissions, and OpenModels.

## Success criteria

- Re-check the canonical OpenRouter prices live and preserve the current
  cached-input/input ratio.
- Update both models on Vast 47529373, set truthful OpenRouter readiness, and
  perform a supervised cutover with the public endpoint gap measured.
- Verify the public OpenRouter and OpenModels schema responses after restart.
- Keep repository metadata and connection records aligned with the live box.
- Update BitRouter PR #814 through its fork branch only.
- Send corrected-price amendments to Surplus and Onlist through the same
  channels and complete the OpenModels submission with the capped key.
- Commit every intended repository change without committing credentials.

## Status

- [x] Lane inbox, repository instructions, branch, remotes, and clean worktree verified.
- [x] Live effective prices verified at 2026-08-12T14:07Z and approved listed
  prices calculated: Q27 $0.28/$2.69; Q35-A3B $0.12/$1.03 per million.
- [x] Serve-box metadata updated; supervised cutover recovered on PID 22646 with
  a measured 20.072-second public/loopback gap and unchanged ingress panes.
- [x] Repository metadata, live launcher, OpenRouter packet, focused tests, and
  sanitized cutover evidence synchronized with the proven public state.
- [x] BitRouter PR fork branch updated to `6e4729e23756` with the approved input,
  cached-input, and output prices; all four local registry gates passed.
- [x] Surplus and Onlist amendments sent from `hello@tiyuvta.ai`; Cloudflare
  reported each recipient delivered with no permanent bounce.
- [x] OpenModels Community application `8d3b4c56-ec06-4910-97fa-8da54428ccdc`
  is live with capped replacement key and applied USD snapshot
  `3491a728-b46e-431f-9040-bffb29a34e66`.
- [x] The OpenModels currency code gap was isolated in commit `5ffcfb5a8`, built
  on Vast without rustup, canaried, and deployed under the existing supervisor.
  The second cutover observed a 20.563-second public 502 gap and recovered on
  PID 27479; relay and tunnel pane PIDs remained unchanged.
- [x] The off-host restore receipt is published on `origin/main` via
  `c1f630e72`; OpenRouter's technical blocker list is empty.
- [x] Final repository verification passed: focused Rust tests, perf-board drift,
  JSON receipts, launcher syntax, exact BitRouter mirror, live process/feed, and
  both authenticated model probes are green. Intended changes are ready to commit.
- [ ] Owner rotates the exposed Cloudflare account-owned Email Sending token;
  its permission-bounded self-roll returned HTTP 403 and did not change it.

## Guardrails

- No merge, tag, or push to memra origin; the BitRouter fork push is the sole
  allowed push.
- No rustup, credential commits, verifier bypass, or unsupervised live restart.
- Preserve the relay/tunnel and record the public endpoint interruption in seconds.

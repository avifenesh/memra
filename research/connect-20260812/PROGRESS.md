# cx-connect progress — 2026-08-12

## Mission

Prepare and send every currently sendable provider submission in the owner's specified order,
stage endpoint- or owner-signature-gated actions precisely, and retain a confirmation ledger in
`SUBMISSIONS.md`.

## Fixed inputs

- Source ranking: `research/routersweep-20260812/REPORT.md`, superseded for qualification state by
  `research/requal-20260812/RESULTS.md` from main at `09900dcaa`
- Models: `qwen3.6-27b` and `qwen3.6-35b-a3b`, both qualified at concurrency 4 per model
- Public wallet address: `<payout-wallet>`
- Endpoint dependency: `lane/cx-servetest`; use only its evidenced final HTTPS hostname
- Price posture: owner-authorized introductory prices approximately 5% below the canonical live
  weighted market after clean rounding: Q27 `$0.28/$0.07/$2.69` and Q35-A3B
  `$0.12/$0.03/$1.03` per million input/cached-input/output tokens
- Signing posture: wallet signatures and SIWE are owner-only; never request or handle keys

## Lane state

- Branch: `lane/cx-connect`
- Base at start: `b227063f896d261a27922c4aa814b89802080c1f`
- Worktree was clean at start.
- Endpoint status at start: not yet published by the servetest lane (its worktree was still at the
  shared base commit when checked). It later published `https://<relay-host>/v1`, with
  root readiness at `https://<relay-host>/readyz`; the current canonical front door is
  `https://api.tiyuvta.ai/v1` with readiness at `https://api.tiyuvta.ai/readyz`.

## Checklist

- [x] Create the required lane checkpoint before authoring submission artifacts.
- [x] Reconcile router-sweep ranking, biz-direction requirements, sell-gate capacity, and test evidence.
- [x] Verify current provider documentation, forms, contacts, and canonical pricing online.
- [x] Open the BitRouter registry PR against the live custom domain, then amend it after Q35 passed
      its endpoint gate.
- [x] Send or stage Surplus Intelligence, Swan, and OpenRouter submissions.
- [x] Send the 18-channel inquiry wave wherever a reachable submission surface exists.
- [x] Start the destination/timestamp/payload/confirmation ledger in `SUBMISSIONS.md`.
- [x] Commit all intended artifacts without merging, tagging, pushing, or formatting unrelated files.
- [x] Propagate the owner-approved introductory prices to the live feeds, BitRouter fork, current
      packets, Surplus/Onlist amendments, and OpenModels.

## Activity log

- 2026-08-12: initialized the submission lane and captured the endpoint/signature gates.
- 2026-08-12: verified the live BitRouter registry shape, OpenRouter application contract, Swan
  signup flow, Surplus SIWE/Base contract, current tiyuvta policies, and the owner's pinned
  initial `$0.285/$2.816` Q27 snapshot, later superseded by the same-metric refresh below.
- 2026-08-12: Swan account created; Proxygate supplier inquiry accepted by its official contact
  form; Coaxiom waitlist accepted. The in-app Browser and user Chrome session were unavailable,
  and the configured mail account could not authenticate because its app-password entry is absent
  from the password store.
- 2026-08-12: servetest remains pre-endpoint; no endpoint-gated form, PR, or allowlist request has
  been sent with a guessed hostname.
- 2026-08-12: the current Foza signup client targets a registration route that returns HTTP 404;
  no Foza account or submission was created. The BitRouter staged manifest passed the current
  upstream registry validator.
- 2026-08-12: an isolated workspace browser exposed Inpherio's public Operator signup; a
  **Sell compute** account was accepted for `hello@tiyuvta.ai` and is pending the owner's emailed
  six-digit verification code. No node software or endpoint credential was attached.
- 2026-08-12: rechecked OpenRouter's live canonical model and endpoint APIs. Their static route and
  per-host list prices are not the completed-day traffic-weighted effective-pricing metric used by
  the owner-directed modelpick snapshot, so every endpoint-gated send requires a final same-metric
  recheck immediately before sending.
- 2026-08-12: steering at main `09900dcaa` restored Q35-A3B to the sellable set. Reworked the full
  packet for the qualified Q27/Q35-A3B pair using the frozen requalification evidence. The
  initial Q35-A3B snapshot was `$0.125/$1.065`; both model pairs remained subject to a final
  same-metric recheck.
- 2026-08-12: sent a Proxygate addendum through an isolated workspace browser. Its official form
  displayed `Message sent! We'll be in touch soon.`; the addendum supplied both qualified model
  envelopes and both price pairs without inserting a URL that had not yet been checked in this
  lane. The immediately following servetest check found the Q27 origin published and healthy.
- 2026-08-12: the servetest lane published a trusted HTTPS Q27 origin. Direct live checks returned
  readiness 200, authenticated `/v1/models` with exact `qwen/qwen3.6-27b`, and a public Provider
  Monitor schema-2.4 document with a 262,144-token context. All three surfaces currently expose
  Q27 only. The exact URL has replaced every placeholder, but BitRouter and OpenRouter remain
  correctly blocked on Q35-A3B live exposure; Surplus is now blocked only on outbound mail and the
  later owner SIWE step.
- 2026-08-12: created an OpenModels account for `hello@tiyuvta.ai` and reached its Community
  application. Stopped before submission because the server does not expose OpenModels' required
  priced/readiness feed: `schema=openmodels` returns HTTP 400 and their docs explicitly reject an
  OpenAI/OpenRouter-shaped substitute. The account credential is in the system keyring.
- 2026-08-12: inspected DGrid's direct provider application and binding 2026-05-11 Provider Terms.
  Its no-training/no-derivation covenant conflicts with tiyuvta's published trial policy, so the
  agreement was not falsely checked and no endpoint key was submitted.
- 2026-08-12: sent Proxygate the exact live Q27 API/readiness/catalog receipt. The form displayed
  its success confirmation; the message explicitly withheld Q35-A3B activation pending public
  exposure and probes and did not transmit an API key through the public contact form.
- 2026-08-12: refreshed OpenRouter's live `shape=v7` effective-pricing feed. Raw weighted pairs
  moved to Q27 `0.2859364714500493 / 2.7503328569812213` and Q35-A3B
  `0.12452658842312533 / 1.0865610445375953`; all unsent packets now use three-decimal parity
  `$0.286/$2.750` and `$0.125/$1.087`. Sent Proxygate a correction because its earlier pair
  addendum carried the superseded snapshot; the official form displayed its success confirmation.
- 2026-08-12: reconciled the catalog's 262,144-token model context with servetest's
  `MEMRA_CTX=8192`: the flag is the fallback session-cache context, not the finite-request cap. A
  public single-run request with 9,010 prompt tokens plus one output token succeeded with exact
  usage, and its repeat reported all 9,010 prompt tokens cached. The full 262K boundary remains a
  pre-listing depth gate rather than a claimed measured capacity.
- 2026-08-12: final pre-commit live refresh moved the effective Q27 pair to
  `0.28662468703642124 / 2.7514636380151267` and Q35-A3B to
  `0.12459691579398467 / 1.0870212751932313`. Every unsent packet and the BitRouter manifest now
  use three-decimal parity `$0.287/$2.751` and `$0.125/$1.087`. Proxygate's 12:08 correction is
  retained as a time-stamped send receipt at the values current when it was submitted; repeated
  millipoint feed drift is handled by rechecking at authenticated onboarding rather than spamming
  the public contact form.
- 2026-08-12 12:18+03:00: a fresh authenticated Q27 request with reasoning explicitly disabled
  returned `OK`, `finish_reason: stop`, and exact `17 / 2 / 19` prompt/completion/total usage. The
  final public catalog recheck still exposed Q27 only. The staged final-price BitRouter manifest
  passed upstream `registry validate` (`52` canonical models, `51` providers).
- 2026-08-12 14:41+03:00: opened BitRouter PR #814 from the fork with an active Q27-only manifest
  at `https://api.tiyuvta.ai/v1`. Upstream helper tests (33), registry validation/build, and the
  generated-state check passed locally. The PR is mergeable; the title-policy action itself failed
  with `Resource not accessible by integration`, while other CI continued.
- 2026-08-12 15:20+03:00: BitRouter's completed substantive CI was green across formatting,
  three-platform clippy, dist, docs/doctests, feature isolation, MSRV, hygiene, macOS tests, and
  Windows tests. Ubuntu tests remained in progress. The only red job was still the fork-token label
  action, not title syntax or provider code.
- 2026-08-12 14:43+03:00: Gmail sent the Surplus domain-allowlist request for
  `api.tiyuvta.ai`; it carries only the public wallet address and leaves SIWE signing to the owner.
- 2026-08-12 14:56+03:00: Swan's resend endpoint accepted the exact
  `hello@tiyuvta.ai` payload with HTTP 200. No message appeared in Gmail `in:anywhere`, so Gmail
  sent a delivery escalation to Swan's published business contact. Inpherio separately delivered
  its welcome mail, and a fresh authenticated login proved the Operator account active with no
  remaining email-verification requirement.
- 2026-08-12 15:16+03:00: final Gmail `in:anywhere` Swan recheck still found only the lane's sent
  escalation and no incoming verification or support response.
- 2026-08-12 15:06+03:00: sent Onlist's published contact a Q27 provider application after reaching
  the genuine site over clean Vast egress. The self-service Turnstile was not automated. A 15:08
  correction replaced the accidental unpushed-lane evidence URL with the existing public `main`
  qualification receipt; both sends received Gmail's `Message sent` confirmation, and the unused
  local draft credential was cleared.
- 2026-08-12 before 14:56+03:00: issued a separate rate-1 OpenModels tenant key on Vast, retaining only
  its SHA-256 fingerprint in the repo, and proved it with an authenticated exact-usage Q27 smoke.
  Added and unit-tested a native OpenModels models-feed adapter locally. The application remains
  correctly unsubmitted until the concurrent servetest lane releases the public process and
  truthful price/`is_ready` metadata can be deployed and probed.
- 2026-08-12 15:09+03:00: refreshed the exact OpenRouter 15-field Q27 packet against the live custom
  domain, effective legal pages, and operational status worker. Remaining gates are the off-host
  restore drill, public price/`is_ready` metadata, and owner confirmation/submit authorization.
- 2026-08-12 15:14+03:00: the servetest lane cut Q35-A3B into the same public process. Readiness,
  the OpenAI catalog, and the schema-2.4 Provider Monitor feed list both exact ids. The Q35 public
  battery finished 21/21 with zero failures and exact accounting; summary SHA-256
  `77f3d70ed792503f71777a5e5aa0b4d235ab927f3f5fa4a13643756b8a3aa2de`.
- 2026-08-12 15:24+03:00: amended BitRouter PR #814 to the live pair at fork head `d71e6dc`, reran
  all four local registry gates successfully, and pushed only the BitRouter fork branch. GitHub
  marked the latest-head workflows `action_required` with no jobs; maintainer approval is required
  to run the fork workflows.
- 2026-08-12 15:26+03:00: Gmail sent Surplus the live-Q35 allowlist update with both models, the
  shared base, current pair prices, and the passing public receipt.
- 2026-08-12 15:27+03:00: Gmail sent Onlist the live-Q35 application update with the same pair
  facts and receipt. The exact 15-field OpenRouter packet and owner checklist now cover both models;
  restore, public truthful metadata, and explicit owner authorization remain gates.
- 2026-08-12 17:24+03:00: deployed the owner-approved price triples and `is_ready=true` for both
  models under the existing Vast server supervisor. Public recovery took 20.072 seconds; the
  server, relay, and Cloudflare tunnel pane identities remained unchanged. Both public schema
  feeds and both authenticated model probes passed.
- 2026-08-12 17:38+03:00: pushed BitRouter pricing commit `6e4729e23756` only to the fork branch.
  The exact current manifest passed 33 helper tests, 52-model/51-provider validation, registry
  build, and generated-state check; PR #814 remained open and mergeable at the final recheck.
- 2026-08-12 17:46+03:00: OpenModels accepted Community application
  `8d3b4c56-ec06-4910-97fa-8da54428ccdc` as `live`. The first snapshot exposed a currency-contract
  gap: omitting the field made OpenModels interpret source micro-prices as CNY. The server fix is
  isolated with focused tests in `5ffcfb5a8`.
- 2026-08-12 18:02+03:00: rotated the accidentally exposed OpenModels rate-1 key, updated the live
  provider profile, disabled the predecessor through the hot-reloading keyring, and re-proved both
  models with the replacement. Only fingerprints are retained.
- 2026-08-12 18:04+03:00: cut the tested USD-currency binary into the same supervised server pane.
  Raw availability recorded 85 consecutive 502 samples and recovery: 20.563 seconds from first
  failure to first 200, with 20.752 seconds signal-to-ready. Relay/tunnel panes were untouched.
- 2026-08-12 18:06+03:00: OpenModels accepted corrected snapshot
  `3491a728-b46e-431f-9040-bffb29a34e66` as `applied`; both public Tiyuvta routes became active
  with exact unconverted USD prompt/cache/completion prices.
- 2026-08-12 18:09–18:10+03:00: Cloudflare Email Sending reported the corrected-price amendments
  delivered to Surplus and Onlist with no permanent bounce. The scoped sender token was then found
  exposed in private diagnostic output; its self-roll returned HTTP 403 because it lacks Account
  API Tokens Write. Rotation is recorded as an owner action; the value is not in the repository.
- 2026-08-12 final recheck: the frozen prices remain 5.1–7.3% below current canonical weighted
  market values. The off-host restore receipt `c1f630e72` is published on `origin/main`, closing
  OpenRouter's last technical blocker; form submission remains outside this pricing lane.

## cx-orsubmit lane

- 2026-08-12 18:46+03:00: started the owner-authorized OpenRouter submission from dedicated
  worktree `wt-cx-orsubmit`, branch `lane/cx-orsubmit`, at `584ed0af05`. The task is limited to
  the live application and its local evidence records: no memra-origin push, no secret capture,
  and no credential material in repository files or logs.
- [x] Revalidate the public application inputs and inspect the current OpenRouter form in Chrome.
- [x] Submit the two-model provider application and retain a redacted confirmation screenshot.
- [x] Update `SUBMISSIONS.md` and `OWNER-ACTIONS.md` with the submission receipt and follow-up hook.
- 2026-08-12 18:50+03:00: both exact models passed fresh 21/21 public protocol/accounting gates
  with zero failures and exact client/engine usage reconciliation. The live OpenRouter-schema feed
  retained both price triples, `is_ready:true`, Ontario/Canada datacenters, concurrency 4, text,
  streaming, tools, and structured outputs; legal and status surfaces returned HTTP 200.
- 2026-08-12 18:51+03:00: refreshed OpenRouter's canonical `shape=v7` weighted metric. Q27 was
  `$0.3035585/$2.8920730` and Q35-A3B `$0.1301004/$1.0851909` per million input/output tokens,
  keeping the owner-authorized list prices 5.1–7.8% below current market.
- 2026-08-12 18:53+03:00: submitted the exact 15-field application in the signed-in Chrome
  session. The live HubSpot form returned `Thanks for submitting the form.` without a CAPTCHA or
  confirmation id. A redacted screenshot and the fresh gate receipts are retained under
  `raw/openrouter-submission-20260812T1553Z/`; the next hook is OpenRouter's Slack Connect
  invitation to `hello@tiyuvta.ai`.

## cx-accounts lane residue salvaged before worktree cleanup (2026-08-13)

The cx-accounts worker lane was stood down from browser work mid-run (owner: workers fumble
interactive flows) and its single commit was never merged. Its non-duplicated findings:

- A dedicated rate-1 Inpherio upstream tenant key was generated on the Vast serve box. The first
  generated key could not reach Secret Service from its shell and was **immediately revoked**; the
  replacement passed authenticated catalog and chat probes. Only its fingerprint is recorded:
  `4f94a0e06d980000e3c4d987357aa5cc97a8143276c9cf17bcff430fe773de31`. The key remains unused
  because Inpherio has no BYO-URL endpoint path (see OWNER-ACTIONS §8) — it is available if the
  owner ever authorizes that channel, otherwise it should be revoked at cleanup.
- Inpherio's only node-registration flow installs Ollama, Python, and the Inpherio connector on
  the serving host — confirming the §8 blocker from the worker's independent read of the console.
- Onlist's form validation requires a logo upload; the worker could not attach a file (Chrome
  extension lacked file-URL access). The owner completed the Onlist form himself, so this is moot,
  but it is why a worker-driven Onlist submission was never confirmed.

No other content from that lane was unique; its coldfix/connect file states are superseded by the
merged versions on main.

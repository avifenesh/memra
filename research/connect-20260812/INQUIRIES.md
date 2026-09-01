# Eighteen-channel supplier inquiry wave

This file is the send packet and disposition record for the router-sweep inquiry wave. It does not
turn an unavailable mail session, an endpoint-gated application, or a buyer waitlist into a claimed
submission. The public wallet below is an address only; no private key or delegated signature is
part of any packet.

## Shared supplier packet

Subject: `Independent Qwen3.6 27B + 35B-A3B endpoints — provider intake and commercial terms`

```text
Hi __PROVIDER__ team,

__PERSONALIZED_LINE__

We have qualified OpenAI-compatible routes for the exact canonical models qwen3.6-27b and
qwen3.6-35b-a3b and would like to join your provider network. Both are live at the evidenced public
origin and have passed their public protocol/accounting probes.
Both passed 40/40 required qualification cells, standard/serial exactness, and exact cached-token
reconciliation at concurrency 4 per model on the concurrent 2x RTX PRO 6000 pair. Q27's
100-request arm measured
144.462 completion tok/s mixed throughput, 18.660 ms all-traffic TTFT p50, and 21.565 ms cache-hit
TTFT p95. Q35-A3B's 100-request arm measured 394.157 completion tok/s, 7.678 ms all-traffic TTFT
p50, and 101.203 ms cache-hit TTFT p95. Concurrent pair-window throughput was 288.923 completion
tok/s median across five runs.

Both routes support streaming, tools, structured output, bounded prefix caching, request ids, and
exact prompt/completion/cached-token accounting. Our owner-authorized introductory Q27 list price
is $0.28 per 1M input tokens, $0.07 per 1M cached input tokens, and $2.69 per 1M output tokens.
Q35-A3B is offered at $0.12 per 1M input tokens, $0.03 per 1M cached input tokens, and $1.03 per
1M output tokens. Cache reads are accounted separately at 25% of the ordinary input rate. The
public settlement address, where an EVM/Base rail applies, is
<payout-wallet>. Any wallet signature remains owner-only.

Before integration, please confirm:

1. Do you currently need each exact model in the final endpoint region, and what seven-day
   routed-token volume do you expect for each?
2. What provider percentage/take, settlement currency and rail, payout cadence or Net-N, and
   minimum withdrawal apply?
3. What reserve, hold, failed-payment, dispute, clawback, and tax/invoice rules apply?
4. How are cached prompt tokens billed?
5. What endpoint, key, health-probe, uptime, latency, error-rate, and data-retention SLA applies?

Public evidence:
https://github.com/avifenesh/memra/blob/main/research/requal-20260812/RESULTS.md
https://github.com/avifenesh/memra/blob/main/docs/SERVING.md
https://tiyuvta.ai/privacy/
https://tiyuvta.ai/terms/

Regards,
Avi Fenesh
https://tiyuvta.ai/
hello@tiyuvta.ai
```

The pair API base is `https://api.tiyuvta.ai/v1`. A dedicated revocable key is added only to
a provider's authenticated application after its external probe gate. The effective policy is
content-free: no durable prompt/completion bodies and no training/evaluation use. Re-run the same OpenRouter `shape=v7`
effective-pricing query before sending any currently staged packet. The prices above are the
owner-frozen introductory rates; a recheck should confirm that they remain under market rather
than silently mutate an already-propagated quote.

## Per-channel routing and personalized line

### 1. Proxygate — SENT

- Surface: <https://proxygate.ai/contact>
- Confirmation: official form server action returned `{"success":true}` at
  `2026-08-12 11:05:49+03:00`.
- Personalized line sent: `Your Own-Endpoint Listing is a direct fit. Can settlement use the
  owner-controlled Base/EVM address below, or is a separate owner-held Solana wallet mandatory?`
- The message asked for the shared commercial terms and acknowledged the documented 95% seller
  share / 5% seller fee. It said the HTTPS test endpoint was coming up; no hostname was invented.
- Q35 addendum: the official form displayed `Message sent! We'll be in touch soon.` at
  `2026-08-12 11:46+03:00`. It added exact `qwen3.6-35b-a3b`, both qualified c=4 envelopes, both
  owner-pinned price pairs, and asked Proxygate to evaluate the pair under the first inquiry's
  terms. It again said the public HTTPS origins were still coming up and asked whether an
  owner-controlled Base/EVM address is acceptable or a separate owner-held Solana wallet is
  required.
- Endpoint receipt: the same official form displayed `Message sent! We'll be in touch soon.` at
  `2026-08-12 12:01+03:00`. It supplied the exact Q27 API base, readiness, and public catalog URLs;
  cited the passing 21-check protocol/accounting gate; offered a dedicated key through an
  authenticated onboarding channel; and explicitly told Proxygate not to activate Q35-A3B until
  it appears at the public origin and passes its own probes.
- Price correction: the form displayed the same success confirmation at
  `2026-08-12 12:08+03:00`. The message supplied the raw live effective-pricing values and current
  three-decimal Q27 `$0.286/$2.750` and Q35-A3B `$0.125/$1.087` list rates, explicitly superseded
  the earlier `$0.285/$2.816` and `$0.125/$1.065` snapshot, retained no cache discount, and repeated
  the Q35-A3B not-live boundary.

### 2. OpenModels — LIVE; USD SNAPSHOT APPLIED

- Surface: <https://openmodels.market/providers/apply>
- Account: created and signed in as `hello@tiyuvta.ai` at `2026-08-12 11:57+03:00`; credential is
  held only in the system keyring under `service=cx-connect`, `provider=openmodels`,
  `account=hello@tiyuvta.ai`.
- Upstream credential: the original capped key was rotated after accidental diagnostic exposure
  and is disabled in the live keyring. The replacement remains lane `interactive`, rate limit `1`;
  only SHA-256 fingerprint
  `8123e0e8e632ba18be4e05c9d395abfc5480f6a7c25532bd0e3a53dea4e4db73` is recorded. Both model
  probes returned HTTP 200 after revocation. Plaintext remains only in the serving secret store and
  local system keyring.
- Personalized line: `Can the Community validation path admit exact Q27 and Q35-A3B before a
  separate Verified review, and what per-model traffic, provider deduction, and remittance terms
  apply?`
- Completion receipt: Community application `8d3b4c56-ec06-4910-97fa-8da54428ccdc` is `live`.
  After the feed was corrected to declare USD, snapshot
  `3491a728-b46e-431f-9040-bffb29a34e66` was accepted with status `applied`. The provider-specific
  public routes are active and retain exact USD source prices: Q27 `280000/70000/2690000` and Q35
  `120000/30000/1030000` micro-units for prompt/cache-read/completion.

### 3. Inpherio — OPERATOR ACCOUNT ACTIVE

- Surface: <https://inpherio.co.uk/register>; contact `support@inpherio.co.uk`
- Confirmation: Inpherio delivered `Welcome to Inpherio` at 14:07. A fresh login request using the
  keyring credential returned HTTP 200 for `hello@tiyuvta.ai`, role `operator`,
  `email_verification_required:false`, `twofa_required:false`, and an access token. The token was
  discarded. The password remains only in the system keyring under `service=cx-connect`,
  `provider=inpherio`, `account=hello@tiyuvta.ai`.
- Personalized line: `Can existing OpenAI-compatible Q27 and Q35-A3B endpoints enroll without
  installing the node agent, and is there current warm-node demand for either exact model?`
- Remaining gate: the account is active, so no owner code-entry step remains. The signup form has
  no inquiry field; the commercial question is still staged for support. Do not install the node
  agent/updater or attach an endpoint until its source and outbound destinations are reviewed.

### 4. Foza — FAILED, CURRENT SIGNUP API RETURNS 404

- Surface attempted: <https://foza.ai/register>
- Personalized line staged: `Can admin approve exact Q27 and Q35-A3B and reconcile the homepage's
  120M/day claim with the much smaller public API counters, while confirming payout rail and
  cadence?`
- Receipt: the current signup JavaScript targets `POST https://api.foza.ai/api/auth/register`,
  which returned HTTP 404 `Cannot POST /api/auth/register`. Four plausible alternate public paths
  also returned 404. No account or provider submission was created. The generated signup
  credential is retained in the system keyring; it was transmitted over HTTPS only to the site's
  current 404 registration route, which did not accept an account.

### 5. IDLE Protocol — BLOCKED, EMAIL AUTH UNAVAILABLE

- Contact: `protocol@earnidle.com`
- Personalized line: `Can the partner API admit externally hosted Q27 and Q35-A3B routes, and must
  provider settlement use Solana or can you support the Base/EVM address below?`
- Note: current public terms describe Solana USDC. The EVM address was not mislabeled as a Solana
  wallet; any additional wallet and signature are an owner decision.

### 6. Scalattice — BLOCKED, EMAIL AUTH UNAVAILABLE

- Contact: `support@scalattice.com`; program: <https://scalattice.com/providers/>
- Personalized line: `Can your curated catalog qualify exact qwen3.6-27b and qwen3.6-35b-a3b
  before any provider-agent install, and what exact majority share, currency, payout cadence, and
  reserve apply?`

### 7. Onlist — PROVIDER APPLICATION SENT BY PUBLISHED EMAIL

- Surface: <https://onlist.io/provider/apply>
- Clean-path receipt: local DNS/TLS was intercepted (`*.prod.bzq.securingsam.com`), but a Vast
  egress reached the genuine Cloudflare-backed Onlist site over valid TLS. The self-service signup
  requires Cloudflare Turnstile; it was not automated and no credentials were submitted there.
- Email submission: the Q27 provider packet was sent to Onlist's published contact
  `contact@onlist.io` at 15:06. Gmail displayed `Message sent`. A 15:08 correction supplied the public
  `main` qualification receipt and also displayed `Message sent`. The packet asks for listing fee,
  settlement rail/currency, Net-N, minimum, reserve/holds, disputes, cache billing, probes, and
  SLA. A live-Q35 follow-up with both exact ids, the shared base, `$0.126/$1.087` Q35 price, and
  the passing 21-check receipt was sent at 15:27; Gmail displayed `Message sent`. The unused local
  draft credential was cleared. At `18:10:01+03:00`, Cloudflare Email Sending reported a pricing
  amendment delivered to the same address with no permanent bounce. It explicitly superseded the
  parity prices with Q27 `$0.28/$0.07/$2.69` and Q35-A3B `$0.12/$0.03/$1.03` per million
  input/cached-input/output tokens and made no endpoint, model-id, capacity, or key change.

### 8. DGrid — NOT SUBMITTED; PRIOR POLICY CONFLICT CLOSED, FRESH REVIEW REQUIRED

- Surface: <https://dgrid.ai/provider>
- Personalized line: `We can register Custom OpenAI endpoints for Q27 and Q35-A3B; please clarify
  whether provider earnings settle in USD or DGAI, whether staking is required, and all
  take/cadence terms.`
- Historical gate: the earlier review found DGrid's no-training covenant incompatible with the
  then-published trial policy, so no terms box was checked and no API key was submitted. Tiyuvta's
  policy is now content-free/no-training, closing that specific conflict. DGrid was not part of the
  newest submit wave; its current agreement and required Telegram/Discord/X contact still need a
  fresh review before any binding submission.

### 9. RouteMarket — STAGED, ACCOUNT AND HUMAN VERIFICATION REQUIRED

- Surface: <https://routemarket.ai/supplier>
- Personalized line: `Do you accept a sole operator offering the exact Q27/Q35-A3B pair, and what
  final-region, rate-card, take, hold, and Net-N terms apply?`
- Send gate: the supplier page redirects to login; registration includes human verification and
  pair-capacity fields. No CAPTCHA or account gate was bypassed, and Q35-A3B cannot be marked live.

### 10. BatchRouter — NOT SUBMITTED, REQUIRED ELIGIBILITY CLAIM IS NOT TRUE

- Surface: <https://batchrouter.com/providers/apply>
- Personalized line staged: `Does a sole operator whose business registration is still pending
  qualify, and is there actual asynchronous-batch demand for exact qwen3.6-27b or
  qwen3.6-35b-a3b?`
- Reason: the current form requires the applicant to affirm it is a registered organization,
  data center, or professional edge provider. The public tiyuvta terms say registration is pending,
  so the checkbox was not falsely affirmed. The async queue/control-plane requirements also need a
  separate capacity specification beyond the synchronous c=4 measurement.

### 11. Requesty — BLOCKED, MANUAL BUSINESS EMAIL

- Contact: `sales@requesty.ai`
- Personalized line: `Are you onboarding independent upstreams, is exact Q27/Q35-A3B capacity
  useful, and what supplier review, per-model demand, take, Net-N, cache-billing, and SLA terms
  apply?`
- Note: the available booking path would reserve owner time; no meeting was scheduled without the
  owner's calendar authorization.

### 12. Martian — BLOCKED, MANUAL PARTNER EMAIL

- Contacts: `contact@withmartian.com`, `support@withmartian.com`
- Personalized line: `Does the current 283-model/46-provider gateway admit independent OpenAI
  endpoints, and is exact Q27/Q35-A3B supply useful in the final live region?`

### 13. Hugging Face Inference Providers — Q27 INQUIRY ALREADY SENT; PAIR FOLLOW-UP BLOCKED

- Program: <https://huggingface.co/docs/inference-providers/en/register-as-a-provider>
- Personalized follow-up line: `Can tiyuvta enter partner review for exact Q27 and Q35-A3B, and
  what company tier, per-model route traffic, provider remittance, request-id billing lookup, and
  retention terms apply?`
- Receipt boundary: the merged router-sweep report records the partnership inquiry as already sent.
  This lane could not inspect Sent mail because the configured mail credential is unavailable, so
  it neither invents a confirmation id nor sends a duplicate. The earlier inquiry predates the
  Q35 requalification; the two-model follow-up above remains blocked on mail authentication.

### 14. OrcaRouter — BLOCKED, EMAIL AUTH UNAVAILABLE

- Contact: `support@orcarouter.ai`
- Personalized line: `Will your fixed curated upstream catalog onboard independent
  OpenAI-compatible Q27 and Q35-A3B routes, and if so what admission, per-model traffic, take, and
  remittance contract applies?`

### 15. Novita AI — BLOCKED, MANUAL SUPPLY EMAIL

- Contact: `gpu@novita.ai`; supplier surface: <https://novita.ai/>
- Personalized line: `Do you purchase a managed inference endpoint or only bare GPU capacity, and
  what SKU/region minimum, rate, take, SLA, and settlement terms apply?`
- Accuracy boundary: the public servetest origin now exposes managed Q27 and Q35-A3B inference.
  The requalification evidence covers both models on a concurrent 2x
  RTX PRO 6000 pair, but neither the servetest nor verification rig is represented as available
  bare-GPU inventory.

### 16. vXAI — STAGED, PRIVATE PROVIDER ADMISSION

- Surface: <https://www.vxai.io/>
- Personalized line: `How can independent OpenAI-compatible Q27 and Q35-A3B endpoints enter
  provider-role review, and are a corporate payout account and registered entity mandatory?`
- Send gate: provider dashboards exist, but no public supplier form or independent-domain contact
  path was found. Do not create a buyer account and call it a provider application.

### 17. Coaxiom Exchange — SENT, WAITLIST ONLY

- Surface: `POST https://coaxiom.io/api/subscribe`
- Confirmation: `{"ok":true,"email":"hello@tiyuvta.ai","list":"newsletter",` followed by
  `"You're on the waitlist. We'll reach out when your spot opens."` at
  `2026-08-12 11:06+03:00`.
- Personalized line staged for reply: `Please confirm that founding-seller 0% fees and weekly
  Stripe payout are contractual, and share exact Q27/Q35-A3B demand plus production probes/SLA.`
- Boundary: the endpoint accepted only an email and source tag, so no supplier claims or questions
  were silently attached to the waitlist request.

### 18. Atlora — BLOCKED, SUPPLIER EMAIL UNAVAILABLE

- Contact: `hello@atlora.com`; site: <https://atlora.com/>
- Personalized line: `Is the early preview accepting independent endpoint suppliers, rather than
  only buyer use cases, and what Q27/Q35-A3B traffic, take, payout, and SLA terms apply?`
- Boundary: the visible waitlist asks for buyer use case and model preferences. It was not falsely
  submitted as a supplier application.

## Delivery constraints observed

- During the first wave the in-app Browser, user Chrome session, and configured CLI mail credential
  were unavailable. Clean isolated workspaces handled the reachable browser actions. On resume,
  the signed-in Gmail session became available and sent only the messages with explicit receipts
  recorded above; remaining email-only channels stay staged where no send receipt exists.
- Curl/API delivery was used only where the provider returned an unambiguous positive result.
- No endpoint hostname was fabricated. No endpoint API key, wallet signature, seed phrase, or
  private key was transmitted through a public contact surface.

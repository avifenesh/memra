# Surplus Intelligence allowlist and owner-only SIWE handoff

State: **UNDER-MARKET PRICE AMENDMENT DELIVERED; SIWE OWNER-ONLY**.

Gmail sent `Seller domain allowlist request — api.tiyuvta.ai` to
`support@surplusintelligence.ai` at 14:43 and displayed `Message sent`.

The initial Q27 request included:

- hostname `api.tiyuvta.ai`, API base `https://api.tiyuvta.ai/v1`, and readiness
  `https://api.tiyuvta.ai/readyz`;
- canonical model `qwen3.6-27b`, current Ontario/Canada location, concurrency 4, and early 429;
- streaming, tools, structured output, bounded cache, exact usage, and request ids;
- `$0.291/M` input and `$2.763/M` output, with no cache discount claimed;
- public wallet `<payout-wallet>` for Base settlement;
- public qualification evidence, privacy, and terms; and
- an explicit boundary that Q35-A3B would be added only after it was live and publicly probed.

After that gate closed, Gmail sent `Q35-A3B now live at api.tiyuvta.ai — allowlist update` at
15:26 and displayed `Message sent`. The update supplied exact `qwen/qwen3.6-35b-a3b`, the shared
base/readiness URLs, Ontario concurrency-4 and early-429 posture, `$0.126/M` input and `$1.087/M`
output, no cache discount, and the passing 21-check public receipt.

Those two paragraphs are timestamped records of the earlier messages, not the current offer.
At `2026-08-12 18:09:35+03:00`, Cloudflare Email Sending accepted a plain-text amendment from
`hello@tiyuvta.ai` to the same published support address and reported it delivered with no
permanent bounce. The amendment explicitly superseded the parity quotes with:

- Q27: `$0.28/M` input, `$0.07/M` cached input, and `$2.69/M` output;
- Q35-A3B: `$0.12/M` input, `$0.03/M` cached input, and `$1.03/M` output.

It also stated that the API base, model ids, capacity, and access-key posture did not change.

## Exact owner steps after allowlist confirmation

1. Open <https://www.surplusintelligence.ai/sell> in the owner's normal browser and connect the
   account ending `0D5F`.
2. If prompted for a network, select Base and verify chain id `8453`. Never paste or disclose a
   seed phrase or private key.
3. Inspect the SIWE preview: domain must be `surplusintelligence.ai`, address must match the public
   address above, URI must be HTTPS on that domain, chain id must be `8453`, and nonce/timestamps
   must be present. Sign only that login message. Reject any token approval or transfer.
4. Create the Q27 and Q35-A3B offers, using new dedicated capped provider keys. Recheck each
   canonical price immediately before accepting the listing. Do not expose the servetest primary key.
5. Record each offer id, accepted price, payout rail/cadence, minimum, reserve/holds, and settlement
   receipt in this lane; never record a signature or private key.

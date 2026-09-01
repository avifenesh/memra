# Swan Inference provider signup and delivery escalation

State: **ACCOUNT EXISTS; RESEND ACCEPTED; VERIFICATION MAIL STILL UNDELIVERED**.

- Email: `hello@tiyuvta.ai`
- Account id: `2065cd1b-8085-4fcf-9f68-0f47e619ae3a`
- Signup response: `Account created! Please check your email to verify your address before signing
  in.`
- Exact resend payload: `{"email":"hello@tiyuvta.ai"}`
- Resend receipt at 14:56:05: HTTP 200,
  `If an unverified account with this email exists, a verification link has been sent.`
- Delivery check: Gmail `in:anywhere` still returned no incoming Swan message at the 15:16
  recheck; the only match was this lane's sent escalation.
- Escalation: Gmail sent `Swan Inference verification email not delivered — provider account
  2065cd1b` to `business@swanchain.io` at 14:56 and displayed `Message sent`.
- Credential: held only in the system keyring; never committed or printed. No wallet was attached
  and no signature was requested.

The escalation quotes the exact email, account id, resend endpoint/payload/result, and missing-mail
check, and asks Swan to retrigger delivery, identify any block, or activate the account. Official
support references checked for this handoff are <https://docs.swanchain.io/resource/links> and
<https://discord.gg/DM5xBUnvt9>.

After verification, propose Q27 and Q35-A3B at `https://api.tiyuvta.ai/v1`. Obtain written
acceptance of both exact models and written payout/take/collateral/slashing/worker/telemetry terms
before attaching capacity.

Owner action: watch spam and `hello@tiyuvta.ai` for Swan's link or support response, verify it is an
official Swan destination, and complete email verification. Any later wallet message must be
inspected and signed personally; reject token approvals/transfers and never disclose wallet keys.

# Pricing cutover receipt

The owner-authorized pair pricing was deployed on Vast 47529373 through the existing
`cx-servetest-server` supervisor. The old process remained live while the replacement binary,
metadata, and launcher passed a co-resident loopback canary.

- Source revision: `d2fba620031920032b253b700443af5ef1ec7866`.
- Deployed binary SHA-256: `871c70e26f3b47f3b1b21a57b1f927b3385dd811f67939525b361cbd907365f4`.
- Metadata SHA-256: `f70b554bcf8861e8b3f9cc1e57c9281e973da35b2c2fff7bd73e9bffba6e5a7e`.
- First canary attempt stopped before model load because the candidate metadata had not yet been
  installed at the launcher's final path. Production returned HTTP 200 for all 60 samples.
- The corrected canary loaded both models, exposed both strict schemas, and passed one-token
  authenticated smokes for each model while production remained ready.
- The supervised production cutover replaced PID 15511 with PID 22646. The measured interval
  from first non-200 to joint loopback/public recovery was 20.072 seconds.
- The `cf-tunnel` and `cx-servetest-relay` tmux pane PIDs remained 16631 and 12512.
- Fresh safe public probes for both models passed the OpenRouter 2.4 schema and matching
  streaming/non-streaming output checks. The downloaded schema SHA-256 was
  `c5ec05a453e262c9c1fd9041ca2624e48b8681ed48df9e73ab5a3642e00675d0`.

No API key or metrics token is present in this receipt.

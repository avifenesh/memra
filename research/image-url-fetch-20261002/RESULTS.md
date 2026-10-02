# Image URL endpoint acceptance, 2026-10-02

**PASS on the existing Qwen3.8-27B vision path.** Fresh OFF/ON server processes
exercised real image generation and the bounded HTTP fetch path. Red and blue images
produced the correct distinct color answers; fetched and inline inputs returned the
same text. Chat Completions and Responses both accepted fetched images. The switch
remains default OFF.

The initially requested Gemma12B pair has a separate, explicit implementation gap:
its supplied projector is `gemma4uv`, while the native loader accepts `gemma4v`.
The plan compiler deliberately represents Unified12B as text-only. No projector
substitution, loader relaxation or Gemma12B vision claim was made. The complete failed
boot is retained under `raw/gemma12-blocked/`. The supported Qwen alternative was
selected from cached assets and the existing vision path; no model code changed.

## Identity and conditions

- Runtime build: `a6138b306c82d5416de95fe0f82583315eb19ef9`.
- Executed tree: `d25f12b9526d95f647174a7ae71095497a6f32e4`. Main's intervening changes
  added research receipts and a standalone probe bin, leaving serving/library code
  unchanged.
- Binary SHA256: `8fc0e5e4eef6e08bc1628457f424d850e581a53173c788f4e52c1c1fc6916743`.
- Qwen3.8-27B NVFP4+Q5_K GGUF SHA256:
  `1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a`.
- Qwen vision `outside.safetensors` SHA256:
  `bdbee7e072110ccddcf126a7f0c44f9f364bd00b07203dae244aa292a2e81ba5`.
- NVIDIA GeForce RTX 5090 Laptop GPU, 24,463 MiB, driver 595.91.07.
  Context 2,048, greedy requests, reasoning disabled, prefix cache disabled. Worker
  logs show K=0 for the image sessions. This is one functional cell, not a performance
  study or a model support-state promotion.
- Local fixture and API listeners used separate loopback ports. Only the exact fixture
  IP was allowlisted. Two fresh server processes ran under one exclusive GPU lease.

## Independently asserted properties

| Property | Observed verdict |
| --- | --- |
| Default-OFF behavior | Inline vision succeeds; both APIs refuse remote images; the fixture receives no request. |
| Actual fetch bytes | The explicitly executed production-code fixture test returns exact red/blue PNG bytes as data URIs, including five redirects, and preserves content metadata. One test executed, no ignored/filter-only success. |
| Native vision | Correct red and blue answers, with exact inline/fetched text equality. Chat and Responses both generate from the fetched image. |
| Literal and DNS safety | Non-allowlisted public/private literals and a hostname resolving to loopback return `image_url_blocked` before the fixture receives a request. |
| Redirect safety | Redirects to a private literal, blocked DNS target and non-HTTP scheme return `image_url_blocked`. Five follows succeed; six refuse. |
| Mixed image count | Eight inline images plus a URL refuse before any network fetch. |
| Per-image bytes | Oversized Content-Length and chunked growth past 12 MiB both return `image_url_too_large`. |
| Expanded request bytes | Both APIs refuse a received body below 192 MiB when the fetched base64 would exceed that same budget. Recorded wire bodies are 200,274,263 and 200,274,271 bytes. |
| Failure mapping | Unsupported MIME, non-success HTTP status and a truncated body return `image_url_unreachable`. |
| Deadlines | Per-image refusal at 10.004 s, caller's one-second deadline at 1.003 s, and the multi-image pass at 20.002 s. Every failure is a named 400. |

The first passing ON arm contains 24 HTTP cases, including six successful native image generations.
The OFF arm adds one successful inline generation and two remote refusals. All raw
responses, elapsed times, fixture hits, server logs and fixture pixels are retained.
`verify.py` rechecks the persisted verdicts without starting a model.

## CPU and review evidence

The full release server suite passed 1,039 tests with zero failures and 27 declared
hardware/manual ignores before the final redirect classification correction. The
final correction passed all 17 focused image-fetch/body tests. The dedicated live
fetch-byte test then executed explicitly in the GPU cell and passed. The pure
address classifier passed all 15 tests; CI registers a floor of 15 and refuses skips
or ignored cases. Actual chunked JSON byte counting and JSON content-type rejection
are covered by CPU tests.

The first Qwen attempt passed OFF controls and exposed a real classification bug:
reqwest can leave a non-HTTP redirect as a 302 before invoking custom policy. The
server previously called that unreachable. It now checks the returned Location and
reports blocked; the complete fixture passed afterward. That failed attempt remains
under `raw/qwen-attempt1/`.

The change also fixes the five-redirect boundary, mixed inline/remote counting,
base64 expansion against actual received bytes, and caller-deadline enforcement.
Fetching now holds the existing vision preprocessing permit, and Responses uses the
same preflight as Chat. These are API changes; model and projector implementations
remain untouched.

Gemma12B requires a native encoder-free Unified vision front end, its tensor/program
contract, and its own qualification. A factored Gemma tower or this Qwen result cannot
satisfy that separate requirement. #533's generic image-fetch behavior is validated
by the recorded Qwen vision cell. Merge remains subject to exact-head CI and review.

Local artifact-directory prefixes and the device UUID are normalized in public
copies. Device identity remains hash-bound. Request/result bytes, errors, counts and
timings are unchanged. Original queue and native receipts remain in the private dated
archive.

## Mandatory vendor-default requests

A final fresh OFF/ON cell at collector source
`b6d1fe685b5451720765932c22b6027ff1cc7821` repeated the complete fixture on the same
runtime binary and added bare requests on both APIs. Their exact saved JSON contains
only `model` plus `messages` or `input`: no token budget, temperature, sampling,
reasoning or other decode field. Both completed with the correct red image answer.
The final ON arm has 26 HTTP cases and eight successful native generations; all
prior controls passed again. The OFF arm remains one inline generation and two
remote refusals. Evidence: `raw/qwen-vendor-pass/`, including the two request files,
HTTP responses and server path-engagement log.

## Combined background and image admission

The integrated background/image tree at `a35091775e18ef290d5329af5ac649fcebfe012d`
passed the release server suite: 1,053 passed, zero failures, 28 declared ignores.
The address classifier passed 15 tests with zero ignores. The release binary has
SHA256 `e9f4db9f3665cbb0461e8731506aae8b68359ea699280e01d4393b8504445e31`.
Collector-only source `7c314dcfaaea39c48b05c902694898433a762bcc` then repeated the
fresh OFF/ON native fixture with the background Responses switch also enabled.

All earlier image controls and both bare requests passed again. The ON arm adds
a remote-image background request that returned queued, then completed with the
same red answer as the inline control. A blocked literal and a one-second caller
fetch deadline both returned named 400s before any queued envelope. The deadline
refused at 1.001 seconds. This confirms that bounded image preflight and background
job delivery compose in the shared admission handler.

The combined cell contains ten successful native image generations across OFF/ON
and 33 ON HTTP requests including four background polls. The exact production
fetch-byte fixture ran and passed once. Evidence: `raw/qwen-combined/`,
`raw/cpu-combined.log`, and `combined-provenance.json`. The earlier long-duration
background and usage-accounting receipt remains separate and unchanged.

# Background delivery is always available (2026-11-09)

**Chosen:** `background: true` on `POST /v1/responses`, `POST /v1/chat/completions` and
`POST /v1/completions` is always accepted. `MEMRA_BACKGROUND_RESPONSES` is deleted, with the
env read, the two refusals (`responses_api::translate` and `background_jobs::validate`), the
door-OFF tests, the env lock those tests needed, and the OFF boots of the two background gates.
The boot refuses the retired name. The contract is unchanged: `docs/SERVING.md`, "Background
delivery for long non-streaming requests", and [COMPLETE-RESULT-PATH-V1.md](COMPLETE-RESULT-PATH-V1.md).

**Rejected:**

- Deleting the feature. It has two PASS receipts and no negative one, it closes the
  complete-result gap of memra#550, and a request without `background: true` never reaches it.
  Removing it would drop a measured capability to save code that already passes its gates.
- Keeping the switch with a new decide-by. It is not a rollback seam with a user: no launcher or
  deployment tool sets it, and only the background gates turned it on. It is not hardware
  config either, since background delivery reads the same worker stream as a synchronous
  request on every card. Its only effect was a named 400, and its env read forced a shared
  lock on 21 tests. Two test fixes on 2026-11-08 (`69eff3e360`, `e75dfb012d`) were about that
  lock.

**Measurement that settled it:**

- `research/background-responses-20261002/RESULTS.md`, Responses on Qwen3.5-9B Q8_0 and one
  RTX 5090 Laptop GPU, fresh OFF and ON boots. A 600-item JSON result completed after
  246.917 s and 18,001 tokens. A cancel kept its 244-token prefix, and a second cancel got 409.
  Five accepted requests each settled one terminal callback with matching usage. Foreign
  tenants got 404.
- `research/background-chat-text-20261002/RESULTS.md`, chat and text on two pinned Qwen3.5-9B
  artifacts (NVFP4-MTP at context 8192, Q8_0 at 32768). Background output matched the
  synchronous or streaming reference, native token ids included. On Q8_0, chat finished after
  303.290 s and text after 412.057 s. Cache-restored and cancelled cells kept their partials.
  All four native cells settled 22 requests exactly once.

These are functional acceptance runs for the named artifacts and card. They change no model
support state and qualify no other target.

**Still owed (memra#550):** the owner's call on the stock store's retention (900 s) and
capacity (64 MiB). Both stay runtime parameters, `MEMRA_BACKGROUND_JOB_TTL_SECS` and
`MEMRA_BACKGROUND_JOB_MAX_BYTES`.

**What would change it:** a receipt where a background job changes a request's output or
accounting against its synchronous reference, or leaks across tenants.

# Native and OpenAI completion JSON

Revuto identified that the cache gate normalized native prompt/cache usage but
omitted native `n_tokens`. The Prometheus emitted-gap check therefore could not
use that supported response format. `response_usage` now maps `n_tokens` to
`completion_tokens`, preserves OpenAI usage/extensions, and refuses a missing
native count. Three CPU tests cover those cases.

Two fresh processes on the same native binary each pass **all 50** cache and
Prometheus assertions. One explicitly uses `MEMRA_COMPAT=native`, the other
`MEMRA_COMPAT=openai`; both use completion and exclusive operator credentials.
`native/raw.jsonl` retains flat `n_tokens` responses and their normalized usage;
`openai/raw.jsonl` retains the original usage object. Each scrape passes real
promtool. Both process cleanup receipts are clear. `results.json` and the
resource-job result are exit 0.

The binary is unchanged from the earlier metrics run. `identity.json` retains
its source/hash, artifact identity, and the corrected collector commit. This
checks the response-format edge identified in review; it does not change the
separate negative plain/MTP control tracked in
[memra#918](https://github.com/avifenesh/memra/issues/918).

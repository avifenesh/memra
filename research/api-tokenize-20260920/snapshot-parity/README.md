# HTTP/worker tokenizer snapshot parity

TC557-TOKENIZE-01: the HTTP map opened tokenizer files before the worker did. Replacing
`tokenizer_config.json` between these reads could make token inspection include a BOS
that ordinary generation did not use, or select a different chat template.

The worker supervisor now retains one tokenizer object per configured model source.
HF, GGUF, repack/override and DSv4 loader paths use that snapshot; the HTTP map receives
the same `Arc<Tokenizer>` objects after successful model loading. Worker respawns reuse
them. A changed source for the same alias or an incomplete ready snapshot refuses.
No new numerical qualification or model support is claimed.

CPU reproduction and regression commands:

```sh
python3 -B research/api-tokenize-20260920/run-snapshot-tests.py --red --out NEW_RED_DIR
python3 -B research/api-tokenize-20260920/run-snapshot-tests.py --out NEW_GREEN_DIR
```

The red harness reproduces the two independent production tokenizer opens without a
CUDA boot and requires the token-ID parity assertion to fail for the named reason.
The green harness imports the production snapshot module and requires three passing,
unfiltered tests: actual BOS/template replacement with shared HTTP/worker/respawn
objects, failed/incomplete load refusal, and conflicting alias-source refusal.
The server suite also checks the HTTP handoff and the existing token endpoint behavior.

Raw logs and source hashes are retained below. The first green fixture used an
unrecognized template marker and therefore rendered unchanged ChatML; its failed
assertion is preserved. The final fixture uses the supported Qwen thinking-tail
marker and proves both BOS and template behavior change on an independent reopen.
This was a fixture correction; the snapshot implementation did not change.

The engine/server library, binary and test cross-target clippy check passed using
`DOCS_RS=1 MEMRA_MMQ_ARCHIVE_HASH=host-check-placeholder`; it is type checking only.
Full Linux/server and native serving composition checks remain separate gates.

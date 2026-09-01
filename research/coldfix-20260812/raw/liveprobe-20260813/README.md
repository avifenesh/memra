# Live-box restored-hit probe — 2026-08-13 (orchestrator, in response to cachesize attempt 7)

## Why
cx-cachesize attempt 7 (`research/cachesize-20260813/raw/attempt7-cache-hit-eos/`) established a
REPEATABLE Q27 (dense) restored-cache-hit early-EOS: a full 4,860-token hit returned HTTP 200 with
`finish_reason=stop` after 11 of 60 tokens, and attempt 3 produced the SAME 11-token text hash at a
different budget. It also falsified the earlier "batched working-set seeding" explanation — attempt 7
used the corrected sequential seed path. That raised an immediate question the orchestrator had to
answer before anything else: **is the LIVE endpoint exposed?**

## Probe (live api.tiyuvta.ai, v0.81.2 binary, production config)
Three distinct ~5.6k-token prefixes, each sent twice: first as a cold MISS, then as a restored HIT
of the same prefix, `max_tokens=60`, `temperature=0`, capped OpenModels tenant key (never the
servetest primary).

| prefix | miss | restored hit |
|---|---|---|
| 1 | `length` 60 tok, cached 0, prompt 5,623 | `length` 60 tok, **cached 5,623** |
| 2 | `length` 60 tok, cached 0, prompt 5,623 | `length` 60 tok, **cached 5,623** |
| 3 | `length` 60 tok, cached 0, prompt 5,623 | `length` 60 tok, **cached 5,623** |

## Verdict
**Live production is NOT reproducing the defect on this shape: 3/3 restored hits completed all 60
requested tokens with `finish_reason=length` and full-prefix cache attribution.** This is a
negative probe on the shipping configuration, not a proof of absence — attempt 7's failure appeared
at a specific budget (8,192 MiB) and working-key position (`prefix_id=87`) inside a 96-key working
set, which this 3-prefix probe does not recreate. The serve box runs `MEMRA_PREFIX_CACHE_MB=4096`.

No emergency cutover or cache disable was taken on the strength of a probe this narrow; the
root-cause lane (cx-eosclass) owns the mechanism, and its steering now carries attempt 7 as the
primary reproduction because it is deterministic and has a passing control (the same key MISSED
cleanly at 1,024 and 4,096 MiB and completed 60 tokens both times).

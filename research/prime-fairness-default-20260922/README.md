# Prime fairness default (memra#521): the decision cell

Verdict: `MEMRA_PRIME_YIELD` ON by default. On the 9B NVFP4 MTP GGUF, one 131,072-id cold prime beside three peers, three interleaved reps per rig: peers' p95 first event 49.64 s OFF to 1.87 s ON on the local RTX 5090 and 14.51 s to 0.53 s on a rented RTX PRO 6000; `tick_max_ms` 51199 to 2309 and 16338 to 870; the long prime's first event +1.68 s and +0.93 s; every request's greedy bytes identical across all 12 boots.

Decision record: `docs/decisions/PRIME-FAIRNESS-DEFAULT.md`. Door history and earlier receipts:
`research/prefill-fairness-20260908/`.

## The gate (`tools/prime-fairness-gate.py`)

One boot per `MEMRA_PRIME_YIELD` arm on the 9B NVFP4 MTP GGUF's spec route with the concurrency
demotion pinned off (`MEMRA_SPEC_GATE_LOW=64 HIGH=65`), `MEMRA_MAX_SESSIONS=4`, `MEMRA_TICK_TRACE=1`.
Greedy `prompt_ids` streams on `/v1/completions`: a seeded 4,096-id prompt, then at t=0 a 131,072-id
cold prime, at +2 s and +3 s two cold 2,048-id peers, at +3 s more the seeded prompt again (a cache
hit). Verdicts: bytes identical across arms; on the yielding arm every peer's first event within 8 s
and the peers' p95 at most half the other arm's; `/health` `tick_max_ms` on the yielding arm at most
6,000 ms; every request finished; `[prime-walk] supported=true yield_door=true` and `[prime-yield]`
lines present on the yielding boot. `--reps 3` interleaves the arms by boot (OFF/ON, ON/OFF, OFF/ON).

## Receipts

| rig (3 reps, arms interleaved by boot) | peers p95 first event OFF / ON (s) | peers max OFF / ON (s) | `tick_max_ms` OFF / ON | long prime first event OFF / ON (s) | bytes identical | yields ON |
|---|---|---|---|---|---|---|
| local RTX 5090 (9950X host) | 49.64 / 1.87 | 49.64 / 1.87 | 51199 / 2309 | 50.77 / 52.45 | yes (every request one sha across 6 boots) | 189 |
| rented RTX PRO 6000 Blackwell WS (Core Ultra 9 285K host, driver 595.71.05) | 14.51 / 0.53 | 14.51 / 0.53 | 16338 / 870 | 16.33 / 17.26 | yes (every request one sha across 6 boots) | 189 |

| receipt | what |
|---|---|
| `raw/gate-5090-9b-run1/` | the first run, BEFORE the route pin: PASS on mechanism (peers 42.3/39.7 s OFF to 1.38/6.22 s ON, `tick_max_ms` 46098 to 2207) but bytes differed for one peer, see the findings below |
| `raw/gate-5090-9b-run2/` | route pinned, one rep: PASS, bytes identical |
| `raw/gate-5090-9b-reps3/` | the 5090 decision cell, three interleaved reps: PASS |
| `raw/pro6000/gate-pro6000-9b-run1/`, `raw/pro6000/gate-pro6000-9b-reps3/` | the PRO 6000 decision cell (one rep, then three): PASS; `raw/pro6000/box-identity.txt` names the card, driver, CUDA and host; the binary there was built from this lane's commit 5a9fd041 on the box |
| `raw/spec-vs-plain-probe/` | the peer prompt solo on the spec route, solo on the plain route, and as four concurrent plain copies: all identical |
| `raw/local-ci/local-ci.log` | the full battery on the final tree with the new stage inside it |

The rented box passed the acceptance gate before any weight was staged (idle 15.5 W at 180 MHz,
cpu-loop 0.76 s, 188 GB RAM, 200 GB disk, IPv4 precedence set) and was destroyed after the receipts
were pulled. Vast offer 51720139, instance 52037661, 1.85 $/h, about 40 minutes.

## Findings beside the decision

- **Route by concurrency, not the yield, moved a peer's bytes.** In `raw/gate-5090-9b-run1/` the
  non-yielding arm admitted the peers while the long prime held the worker, so the spec gate
  demoted them to plain decode (`K=0 source=concurrency`, wave 4); the yielding arm admitted them
  at `active=2` and they took `K=3 source=cold-long`. Peer `peer-cold-b` then produced different
  greedy bytes on the two arms (`...</div>\n</body>...` versus `...</div>\n</div>...`) on a
  synthetic near-tie prompt. The same prompt solo on the spec route, solo on the plain route and as
  four concurrent plain copies produced identical bytes (`raw/spec-vs-plain-probe/`), so neither
  spec-versus-plain nor batched-versus-solo alone reproduces it; the divergent run had the peer
  primed under `prefill_tick` beside a spec long request in the same ticks. Cause unknown; the raw
  cell JSONs and both server logs are kept for the repro. The gate pins the route
  (`MEMRA_SPEC_GATE_LOW=64 HIGH=65`) so it measures the yield; with the pin every one of the 12
  decision boots agreed byte for byte. Filed as its own issue.
- **Synthetic-id prompts make the model's first token EOS sometimes** (`peer-cold-a` finishes with
  no text). The gate therefore measures the first choice event (a token or an immediate finish) as
  "first event" and keeps `first_token_s` separately; bytes are compared on whatever text arrived.

# Latest-main integration

Runtime source `b0fa08d93526969c759b0448c49db9bc3dd2c238` includes main through
#377's retained DFlash admission and #325's GLM host cache. The `crates` tree is
`2a2347f5c29e41dd8a1d861aac8551dec8acd9d9`. Later lane commits only add harnesses
and receipts. The composed server binary is SHA-256
`997515ac0ba866b308e9514c6b49d70acda70016f3d05953557c7610fa6a0d0e`.

CPU checks on the authorized box, nice 19: fmt, release clippy all targets,
61 targeted engine tests and 652 server tests pass. Four existing manual/GPU
fixtures are ignored. Source, build and test logs are in `cpu/composed/`.

## Capacity check: Qwen does not pass

One additional interleaved OFF/ON boot per route used the original offered-load
shape. This is an integration check, not an extension of the three-boot
decision medians, because the binary changed.

| Route | Small p95 OFF / ON, seconds | Long TTFT OFF / ON | Long total OFF / ON | Completion |
|---|---|---|---|---|
| Ornith | 15.122 / 5.744 | 24.364 / 25.841 | 28.839 / 32.720 | 21/21 each arm |
| Qwen | 60.854 / not eligible | 68.122 / 73.088 | 78.023 / 81.438 | OFF 21/21; ON 18 complete, 1 token-capped, 2 client-ceiling rejects |

The Qwen ON server admitted 19 requests and completed/drained all 19, with zero
OOM or retry. The harness's stricter completion gate excludes a small request
that hit 2,048 output tokens, and two small arrivals were refused by the
client's in-flight ceiling before HTTP submission. The 66.827-second p95 over
received small responses is not an eligible 20-request offered-load statistic.
The failed cell is retained in `gpu/composed/qwen-r1-on.json`; it was not rerun
to select a favorable sample. This strengthens the admission/peer-throughput
limitation in [RESULTS.md](RESULTS.md), and prevents a Qwen capacity/SLO claim.

An earlier Ornith ON integration boot stopped during seeding when a sampled
seed hit its 2,048-token cap with reasoning only. No scored arrivals had begun.
That attempt is retained as `ornith-seed-incomplete` in the private archive;
the fresh boot used identical settings and a new nonce. Neither failure is
silently folded into the eligible decision medians.

## Correctness and sampled cache twin: pass

The durable controller completed all remaining cells and wrote `CAMPAIGN_PASS`.
After the app-server restart, the saved bytes and full boundary-oracle line
multisets were checked again: `BYTE_AND_BOUNDARY_GATES_PASS`. Existing completed
cells were banked, not relaunched.

| Gate | Ornith | Qwen |
|---|---|---|
| c1 four-turn OFF/ON bytes | 4/4 equal | 4/4 equal |
| c2 long/small versus own c1 bytes | 2/2 equal | 2/2 equal |
| Pair yields, LOW/HIGH=64/65 | 250 | 129 |
| Pair route counts | 2 MTP, 0 plain | 2 DFlash, 0 plain |
| Small c1 / c2 TTFT, seconds | 0.226 / 0.248 | 0.550 / 1.185 |
| Logits and actual boundary captures | equal | equal |

DFlash tap, feature and position hashes also match. See
[gpu/composed/byte-boundary.json](gpu/composed/byte-boundary.json).

The vendor-default sampled cache twin ran eight turns OFF and eight ON on each
model: 32/32 complete streams, every turn on the model's speculative route,
zero OOM, zero admission deferrals, and cached tokens on all 28 follow-ups.
Sampling fields and raised spec gates were absent from those requests/profiles.
Outputs are not compared across sampled timing, and no greedy or token-capped
output is used to manufacture a throughput statistic.

These correctness results do not turn the retained Qwen capacity failure into
a pass. No Qwen launcher or shared engine default is flipped.

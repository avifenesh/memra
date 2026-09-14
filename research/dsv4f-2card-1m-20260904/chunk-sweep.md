# Matrix/EP chunk discovery

2026-09-06 UTC. The full-model matrix/EP gate has passed its pinned scope;
the next question is useful per-launch work, not whether the cards merely fit.
This discovery gate changes chunk width within the same matrix/EP program.
It introduces no runtime flag and changes no default.

`dsv4_chunk_sweep_gate` loads the house artifact once on two devices with native
FP8-QAT matrix MoE and expert-ID EP. It uses the frozen real-source corpus
`f6e175a6f2588953568746fec0cd43fcd046405f74b5c71ce071fe7f37238ded`, prefixed
`Review this inference engine source:\n\n`, and records the exact token hash.
The initial count is 9900. No source repetition or padding is allowed.

Order: 32, 64, 128, 256, 512, 32. The final 32 repeats the initial control.
Each case starts from a new decode/draft state and independently allocates
its requested transient width. The timed section is chunked engine prefill
including required DSpark prime, ending only after both streams drain. Model
loading, tokenization, allocation and verification/hashing are outside that
section. This is not HTTP TTFT or output-token throughput.

After prefill, every width must match the baseline's complete final logits,
all live cache classes, DSpark rings, 16 vendor-shape sampled continuation
tokens, round/confidence data, and final cache/rings. Both EP and device-route
counters must prove engagement. Masked pending-score negative infinities are
allowed; unrelated non-finite values fail. Generation is an untimed exactness
check, never an output-throughput denominator.

Timings are individually labelled discovery rows. They do not establish a
hardware default, balanced performance, concurrency fairness, broad checkpoint
quality or serving readiness. Any promising width needs a same-window balanced
comparison and real endpoint/continuation/admission checks before promotion.
The private controller offers the 9.9K and 64K cases, serializes on the canonical
pair lock and stops only its own child if another GPU workload appears.

The current [Sarathi-Serve paper](https://arxiv.org/abs/2403.02310) supports
chunking as part of a latency-aware schedule, not a universal optimal chunk
size or a DSV4/SM120 speed prediction. This gate measures our fixed EP topology;
it does not implement stall-free cross-request scheduling. No external engine
or kernel dependency was added.

Local checks: prompt-count bounds test, strict gate clippy, release build,
formatting, whitespace and runtime-flag census pass.
Initial binary SHA256:
`585a9189cc86f562e1a16de8bb9add9f63d3b6ffc15273c16d80dc94eacca8db`.
The target run is pending; staged/running is not a result.

## Completed 9.9K discovery

The initial binary completed status 0 on the PRO pair at 2026-09-06 03:40:18
UTC. Prompt token SHA256:
`7ee3433913da75a2bbd60c2e0937bba1d1bf05cc9ceaa032e83033460037cc2f`.
All six cases had the same complete-state/output signature
`97d0856833def8fc724cef3b8ee0ee16d00c4f30cc0b85dcbcdcee082e50b381`.

| Chunk | Prefill seconds | Input tokens/s |
|---|---:|---:|
| 32, first | 62.974815 | 157.206 |
| 64 | 59.263502 | 167.051 |
| 128 | 54.757049 | 180.799 |
| 256 | 51.694605 | 191.509 |
| 512 | 50.411365 | 196.384 |
| 32, repeat | 62.859513 | 157.494 |

These are discovery rows. Width 512 is a candidate: 1.248x the throughput of
the mean width-32 time in this run, not a default or HTTP-latency claim. Width
256 is close and may matter for memory/latency tradeoffs in later serving work.
The original default remains unchanged.

## Balanced follow-up, frozen before execution

The gate now accepts `compare WIDTH` after the prompt count. For 512 it runs
`32,512,512,32` three times: six measurements per arm, equal exposure to both
orders, fresh state each time and the same exactness checks for every case.
UTC start/end milliseconds bracket each prefill for correlation with the
controller's 250 ms GPU telemetry. Watts/clocks are metadata, not a bottleneck
attribution. Output generation remains outside the timed section.

Count bounds and balanced schedule tests, strict clippy and release build pass.
Comparison binary SHA256:
`2314414bfcafe05b8af2dd05cf9d21d54d4b9186cfd8cfdf7301f981f3aedb42`.
The comparison is not yet complete. Even a measured winner still needs wider
context, endpoint, admission and concurrent scheduling/fairness qualification.

The 9.9K ABBAx3 comparison subsequently completed at 2026-09-06 04:10:08 UTC,
status 0. All twelve cases retained the same signature. Independent audit of
schedule, inputs, engagement, intervals and process/telemetry records gives:

| Chunk | N | Median prefill seconds | Median input tokens/s | Range, seconds |
|---|---:|---:|---:|---:|
| 32 | 6 | 62.839335 | 157.5446 | 62.813646–62.936561 |
| 512 | 6 | 50.413537 | 196.3758 | 50.316004–50.494093 |

Median-time ratio is 1.24648x, a 24.65% engine-prefill rate gain. The six
adjacent paired ratios range from 1.24467x to 1.24871x. Only one GPU process
was observed. Phase-correlated telemetry has a maximum observation gap of
348 ms; observed SM clocks span 2220–2340 MHz and temperatures 33–67 C.
Power limits were 300 W throughout; that is metadata, not power attribution.
No claim of HTTP TTFT, output TPS, 64K/1M behavior or concurrency follows from
this fixed-input engine measurement. Defaults remain unchanged.

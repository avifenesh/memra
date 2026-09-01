# cx-probehitch results

Date: 2026-08-12

Base: `09900dcaa2011beefa3c316442740f693a76468c`

Verdict: **PASS**

## Result

The 4096-token / 64 MiB runtime peer re-probe no longer runs on an interactive scheduler
boundary. Each probe width now owns an independent staggered copy-count deadline. The maximum
width is idle-only from its first run; any smaller width whose measured complete owner-thread
cost exceeds 5 ms becomes idle-only on later cycles. An overdue idle-only width does not block a
later cheap deadline, and late work advances to the first future cycle rather than replaying a
catch-up burst.

A native re-probe failure now one-way latches native P2P off, allocates the existing pinned
host-bounce slots, exercises a real D2H/event/H2D row transfer and readback at every cross-device
boundary, and only then publishes the live host-bounce transport. The worker continues admitted
plain requests, disables new spec/dual-active and unsafe cross-device snapshot paths, evicts
unpinned prefix entries, logs `SECURITY RED`, and exposes
`peer_probe_degraded_to_host_bounce` in operator metrics. Failure to arm or validate bounce
staging remains the only panic path; the process never returns to native peer transport without a
restart.

## Real PP-2 hit-TTFT receipt

The final measurement used the steered `sbox-2card` pair: 2 x NVIDIA RTX PRO 6000 Blackwell
Server Edition, PIX topology, CUDA 13.2 / sm_120a, and the three pinned Step-3.7-Flash IQ4_XS
GGUF shards. The successful window held `flock /tmp/memra-gpu.lock` continuously and began with
both GPUs idle at 0 MiB, P8, 180 MHz, and 26 C. Active snapshots stayed at 31--33 C; observed SM
clocks were about 2407--2415 MHz on device 0 and 2325 MHz on device 1. Clocks were not locked on
this remote pair, so the evidence is the required alternating same-lock-hold comparison, not a
cross-run absolute-clock comparison.

Protocol: five whole-server pairs, order alternated by pair, spec off, PP-2 devices `0,1`, one
warm-up followed by one measured streaming request per arm. Every measured request had 519
prompt tokens, 513 cached tokens, eight generated tokens, and the same output SHA-256
`dbf29017ca508c93f8b4117b220141ab05fbe44109ae1c9da05526334fe90aab`.
The measurement-only source patch, retained in `raw/sbox/source.diff`, arms only the maximum rung
after a request is already queued. The `before` arm makes the old inline scheduling decision;
the `after` arm makes the production busy-boundary decision and drains the still-pending rung
after the request at idle. The trigger is not part of the committed runtime.

| Arm | N | TTFT p50 | TTFT p95 | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before: max rung inline | 5 | 518.521 ms | 518.906 ms | 517.725 ms | 518.906 ms |
| After: max rung deferred | 5 | 53.629 ms | 53.801 ms | 53.532 ms | 53.801 ms |

Paired `before - after` improvement: median **464.731 ms**, range 464.096--465.320 ms.
All five before logs ran the max probe inline in 463.868--465.131 ms. All five after logs returned
from the busy check in 0.000 ms and then completed the same probe at idle in
464.377--465.705 ms. Thus integrity coverage moved in time; it was not removed. All ten metric
snapshots report one completed runtime re-probe, zero failures, and a healthy (non-degraded)
transport.

The earlier CPU-only scheduler proxy remains supporting evidence, not the scored result. Across
nine injected cycles it measured all-inline owner-boundary p95 431.364 ms, new busy-boundary p95
2.070 ms, and deferred idle-drain p95 431.434 ms (`raw/owner-stall.jsonl`).

## Correctness and availability gates

- Focused scheduling policy: 3 passed; failover latch: 1 passed; worker mismatch/continuity: 1
  passed; operator-only metrics: 1 passed.
- Final `cargo test -p memra-engine -p memra-server`: engine library 81 passed / one CUDA-only
  ignored; server 197 passed; no failures (`raw/cargo-test-engine-server-final.log`).
- Local RTX 5090 `tools/local-ci.sh`: GREEN. Preflight found no concurrent compute application,
  held `/tmp/gpu5090.lock`, and `nvidia-smi` accepted the required 210--1200 MHz graphics-clock
  cap. `kernel-check` was `ALL GREEN (106 cells, 1 skipped)`; Qwen 35B `run-spec` K=1..8 passed;
  `run-gen` argmax matched; graph canary, batch gates, serve smoke, c=64 stress, and acceptance
  gate all passed (`raw/local-ci.log`). The existing non-fatal FLAGS warning names unrelated
  `MEMRA_MOESD_*` literals.
- The real PP-2 A/B completed `rc=0`; all ten arm logs are free of panic, failure, and
  `SECURITY RED`. Raw model/prompt/binary hashes, clock/thermal snapshots, client rows, server
  logs, metrics, and the exact source delta are under `raw/sbox/`.
- The first remote receipt shakeout stopped at a compile error in its temporary diagnostic string,
  before any model boot. Its raw driver/build logs are retained; the corrected final run is
  timestamped `20260812T123536Z`.

No merge, tag, push, board update, formatting sweep, hook bypass, rustup operation, or nsys
artifact was used.

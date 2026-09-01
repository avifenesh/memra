# Box1 validation at the real 262,144-token serving context

Lane `lane/cx-val256`, runtime base `4f80a088`, validated 2026-08-09 on box1 with two RTX PRO
6000 GPUs. This lane changes no runtime code. It tests the merged plain-affinity and request-shaped
admission hardening under the owned PP-2 serving shape:

```text
MEMRA_CTX=262144
MEMRA_PP_STAGES=2
MEMRA_PP_DEVICES=0,1
policy: K=0 source=pp2-placement
```

## Verdict

**FAIL as a hardening-wave validation gate. Do not promote these receipts into a merge, tag, or
unqualified capacity/$-per-token claim.**

| block | N | result |
|---|---:|---|
| 1. Affinity, deep rewritten history | target N=3 per arm; 1 ON arm complete, first OFF arm partial | **FAIL / INCOMPLETE**: rewinds fired and the available ON receipt was flat, but OFF hit a captured allocation OOM before one pair completed; the requested slope and three-replay determinism gates were not reached. |
| 2. Mixed-context admission | N=1 per ordering | **FAIL**: 8k, 128k, and 256k requests were all charged as 262,144-token requests. Reclaim ordering, clean completion, and zero step-OOM passed; the inverse small-request arm was over-gated. |
| 3. Requested-128k capacity | N=1 | **RECEIPT COMPLETE**: the first defer occurred with 1 active session. This is current behavior for a requested-128k request charged at the 262k cost, not correctly scaled 128k capacity. |

Per instruction, both failures were captured and no runtime fix, alternate runtime setting, or
retry was attempted.

## Block 1 — plain affinity beyond 32k

Raw verdict: [block1-failure-summary.json](raw/block1-affinity-20260809T150220Z/block1-failure-summary.json).
The final recorded workload passed its geometry gate: all 8 sequential turns used rewritten
history and ranged from 37,823 to 47,175 prompt tokens. The server selected PP-2 plain decode
(`K=0`).

The completed affinity-ON replay reported 8 `plain_affinity_rewinds` and 8
`continuation_pool_hits`. After the cold first turn, the cached boundary advanced with the
conversation while resumed TTFT stayed in a narrow band:

| turn | prompt tokens | cached tokens | ON TTFT, seconds |
|---:|---:|---:|---:|
| 1 | 39,159 | 37,817 | 4.131 |
| 2 | 40,495 | 39,153 | 4.146 |
| 3 | 41,831 | 40,489 | 4.162 |
| 4 | 43,167 | 41,825 | 4.176 |
| 5 | 44,503 | 43,161 | 4.187 |
| 6 | 45,839 | 44,497 | 4.224 |
| 7 | 47,175 | 45,833 | 4.244 |

The record server and the one completed replay server produced matching `text_sha256` values on
all 8 sequential turns. That is a positive two-server observation, but it is not promoted to the
requested three-replay determinism result.

The first affinity-OFF replay had zero cached tokens and completed only turns 0–2, with TTFT
101.988, 105.790, and 109.930 seconds. Turn 3 then failed with the captured client/server text:

> `cache alloc failed: DriverError(CUDA_ERROR_OUT_OF_MEMORY, "out of memory")`

At capture, `nvidia-smi` compute-app memory was 87,530 MiB on GPU 0 and 97,036 MiB on GPU 1.
Final OFF metrics reported 3 parked continuation entries, 0 admission VRAM defers, and 0 step-OOM
parks. Because the failure happened inside the first A/B pair, there is no N=3 median or accepted
ON-vs-OFF slope. The N=1 ON direction and three partial OFF points above are retained only as
diagnostic evidence.

Thermal regime: continuous one-second sampling under one exclusive lock, including model load and
all attempted arms. GPU 0 ranged 26–52 C and GPU 1 ranged 27–55 C. Raw request JSONL, responses,
metrics, server logs, failure scan, and GPU samples are under
[`raw/block1-affinity-20260809T150220Z/`](raw/block1-affinity-20260809T150220Z/).

Two pre-verdict calibration receipts are deliberately excluded from the result. The first was
deep enough but had `history_rewritten=false` on all turns; the second reached 67,913–73,681
tokens but still produced no visible assistant history to rewrite. Their raw logs remain under
`raw/block1-affinity-20260809T143906Z-pilot-aborted/` and
`raw/block1-affinity-20260809T145304Z-reasoning-calibration-aborted/`.

## Block 2 — admission at true 256k

Raw verdict: [admission-summary.json](raw/block2-admission-20260809T152512Z/admission-summary.json).
Each ordering is a single run, not a median. Both ran under one shared exclusive lock with
continuous one-second GPU sampling.

| ordering | requested sequence | completion | first reclaim before first defer | observed request charge | defer polling events | step-OOM |
|---|---|---:|---|---|---:|---:|
| forward | 8k calibrator; two 256k parks; c=4 128k burst | 7/7 | yes: 2 parked sessions evicted, effective free 9,346 -> 31,727 MB; reclaim line 36, defer line 37 | only `ctx=262144`: 83,520 B/token, 21,894 MB | 423 | 0 |
| inverse | one 256k first; c=4 8k burst | 5/5 | yes: 1 parked session evicted, effective free 20,536 -> 31,727 MB; reclaim line 30, defer line 31 | only `ctx=262144`: 83,520 B/token, 21,894 MB | 423 | 0 |

The reclaim-on-defer ordering check passed in both arms: eviction and the effective-free re-read
preceded the first `VRAM defer`. All requests completed, and both captured failure scans were
empty.

The request-shaped charge check failed. No 8,192- or 131,072-token cost appeared in either server
log; every request used the 21,894 MB 262k cost plus a 21,894 MB reserve. The inverse c=4 8k burst
therefore serialized at 0.350, 1.551, 2.751, and 3.948 seconds TTFB. The forward c=4 requested-128k
burst similarly serialized at 0.321, 1.518, 2.718, and 3.916 seconds. The 423 values are admission
polling events, not 423 unique requests.

Thermal regime: N=1 per order under one continuous lock; GPU 0 ranged 26–36 C and GPU 1 ranged
27–37 C. Raw request JSONL, ordered admission lines, metrics, server logs, empty failure scans, and
GPU samples are under
[`raw/block2-admission-20260809T152512Z/`](raw/block2-admission-20260809T152512Z/).

## Block 3 — one honest capacity row

Raw verdict: [capacity-summary.json](raw/block3-capacity-20260809T153030Z/capacity-summary.json).

| N | server context | request `max_ctx` | effective charged context | offered concurrency | active before first defer | peak GPU 0 | peak GPU 1 | combined peak used | completed | step-OOM |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 262,144 | 131,072 | 262,144 | 24 | **1** | 66,705 MiB used / 30,548 MiB free | 77,683 MiB used / 19,570 MiB free | 144,388 MiB | 24/24 | 0 |

The first defer saw 1 active session and 31,727 MB effective free, below the 21,894 MB request
cost plus 21,894 MB reserve. Final metrics reported 19,573 defer polling events. All 24 streams
completed; wall times ranged 1.277–29.021 seconds, and the failure scan captured no CUDA, OOM, or
panic line.

This row answers what admission actually did, but the Block 2 failure is load-bearing context:
the requests advertised 128k while admission reserved the 262k amount. It may be carried into a
capacity table only as `requested 128k / effective charge 262k`; it is not evidence that the
request-shaped 128k estimator admits only one session.

Thermal regime: N=1, continuous one-second sampling under one exclusive lock. At the simultaneous
peak sample the GPUs were 34 C and 35 C; maximum temperatures anywhere in the block were 40 C and
41 C. Raw JSONL, the 19,573-event metric, emitted ordered defer lines, server log, empty failure
scan, and GPU samples are under
[`raw/block3-capacity-20260809T153030Z/`](raw/block3-capacity-20260809T153030Z/).

## Run identity and evidence discipline

- The IQ4_XS three-shard model and Q8_0 MTP draft were staged byte-identically on box1 local NVMe
  under `/opt/dl-image/nvme/models/step-3.7-flash`. The artifact-manifest SHA-256 is
  `4c22bdce378de2c365cdcbf3ce6dcf94d9dd690b0058e5fb01e3fb71a5b29312`; individual hashes are in
  [artifact-sha256.txt](raw/block1-affinity-20260809T150220Z/artifact-sha256.txt).
- Blocks 2 and 3 used server binary SHA-256
  `34ede9f390f8fe3792007dd0d7e8b4560adbc510a1f2b0cb81b6e30737d48022`.
- Recorded server commits are `5b6c28f0` for Block 1, `d88fde38` for Block 2, and `61bd1cb4` for
  Block 3. Every descendant change from runtime base `4f80a088` is confined to
  `research/val256-20260809/`; runtime source stayed unchanged.
- Each block held `/tmp/memra-gpu.lock` once. Long runs were detached, and raw output was tee'd
  before analysis. No pipe was used as the sole copy of stderr.
- `~/.lanectl/inbox/cx-val256.md` was absent at start and at every required pre-block check.
- No origin push, merge, tag, perf-board edit, `rustup`, `nsys`, runtime fix, or serving-default
  change was made.

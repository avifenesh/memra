# 27B beside Step — controlled A/B protocol

Date: 2026-08-11

Status: **PENDING box1 execution**. This document contains no new GPU result and makes no
pricing, business, deployment, merge, or release decision.

## 1. Question and controlled object

Measure the cost and productive capacity of serving the Qwen3.8-27B release target beside the
resident Step-3.7-Flash daily model on the designated 2x RTX PRO 6000 96GB serving pair:

1. How much does an idle resident Q27 change Step latency, throughput, and memory headroom?
2. How much does active Q27 traffic change Step TTFT, prefill, and decode?
3. What input/output throughput does the Q27 slot produce while Step is idle and while both
   services are active?
4. Do those measurements give the owner enough inputs to compare co-residency with a separate
   box at the pinned market rates?

The scored object is an exact model artifact, not the family name. Execution MUST stop until the
Qwen3.8-27B target exists locally and its source revision, quantized target, optional draft,
chat template, context cap, and full SHA-256 hashes are frozen. The current 15--16 GiB Qwen3.6
NVFP4 weight size is only a planning proxy; it is not a Qwen3.8 memory receipt. Never attach a
Qwen3.6 draft to Qwen3.8 unless the model-specific correctness battery proves that exact pairing.

## 2. Arms, processes, cards, and memory envelope

### Arm A — Step alone

- One `memra-server` process on port `8002`.
- The process sees physical GPUs 0 and 1 through `CUDA_VISIBLE_DEVICES=0,1`.
- Step remains PP-2 with `MEMRA_PP_STAGES=2` and `MEMRA_PP_DEVICES=0,1`.
- Model is the pinned Step-3.7-Flash IQ4_XS trunk plus its pinned external Q8_0 draft, with the
  exact serving configuration from the accepted baseline receipt.
- No Q27 process exists. No unrelated compute process may use either GPU.

The accepted box1 [raw load-plan receipt](../serve-ready-20260808/raw/server-w1-20260808T160419Z-head.log)
records 45.72 GB experts + 3.92 GB trunk = **49.64 GB on physical card 0** before that card's Step
KV/cache/scratch, and 55.35 + 3.92 = **59.27 GB on card 1**. The Q3 inventory measured Step resident
and serving with 50,672 MiB free on card 0 and 40,332 MiB free on card 1, on 97,887 MiB cards.
Record both the load-plan line and post-warmup NVML values; do not replace measured free memory
with subtraction alone.

### Arm B — Step plus Qwen3.8-27B

- Keep the Arm-A Step process byte-for-byte and environment-for-environment unchanged.
- Start a second independent `memra-server` on port `8003`.
- The Q27 process uses `CUDA_VISIBLE_DEVICES=0`, so its logical device 0 is physical card 0.
- Pin `MEMRA_CTX=32768` for the first campaign, a separate model alias, its own key/port, and
  VRAM-aware admission. Pin speculative mode and K only after the exact Qwen3.8 artifact/draft
  battery; if no validated draft exists, use plain decode consistently in every Q27 cell.
- `B-idle` means both processes and both models are resident but Q27 receives no request.
- `B-active` means both are resident and the Q27 load generator has host-monotonic overlap with
  every scored Step request. The Step process still occupies both GPUs; this is intentionally
  **not** an exclusive device partition. Physical card 0 is the controlled contention point.

Q3's Qwen3.6 planning envelope is 15--16 GiB weights + 8--10 GiB serving KV/cache/scratch =
**24--26 GiB total**. Against the measured 50,672 MiB card-0 free value, that projects:

```text
low envelope:  50,672 MiB - (24 * 1,024 MiB) = 26,096 MiB = 25.48 GiB free
high envelope: 50,672 MiB - (26 * 1,024 MiB) = 24,048 MiB = 23.48 GiB free
```

That is a fit hypothesis, not permission to score. Capture Qwen3.8 file bytes, post-load used/free
memory, post-warmup used/free memory, peak used memory in every load cell, and the final free-memory
floor. A captured allocation failure is `OOM`; a process death without the driver's OOM text is
`died, cause unknown — repro needed`.

### Launch manifests

Resolve every placeholder before execution and retain the expanded, secret-redacted environment
in raw evidence. The expected topology is:

```text
# Arm A and the unchanged Step half of Arm B
CUDA_VISIBLE_DEVICES=0,1
MEMRA_ADDR=0.0.0.0:8002
MEMRA_CTX=262144
MEMRA_MODELS=stepfun/step-3.7-flash=<STEP_IQ4_XS>+<STEP_MTP_Q8>
MEMRA_MOE_GROUPED=1
MEMRA_PP_STAGES=2
MEMRA_PP_DEVICES=0,1
MEMRA_PREFIX_CACHE_MB=<PINNED_BASELINE_VALUE>
MEMRA_PREFILL_TICK=<PINNED_BASELINE_VALUE>

# Additional Arm-B process
CUDA_VISIBLE_DEVICES=0
MEMRA_ADDR=0.0.0.0:8003
MEMRA_CTX=32768
MEMRA_MODELS=qwen38=<QWEN38_27B_TARGET>[+<VALIDATED_QWEN38_DRAFT>]
MEMRA_PREFIX_CACHE_MB=<PINNED_Q27_VALUE>
MEMRA_SERVE_SPEC=<PINNED_0_OR_1>
MEMRA_SPEC_K=<PINNED_IF_SPEC>
```

Host-bounce/P2P flags are machine facts, not arm variables. Freeze them from the target host's
accepted service manifest and keep them identical between A and B. A baseline from a different
PRO 6000 SKU, power limit, peer path, or host-bounce regime does not stand.

## 3. Serving-surface seam — current source verdict

**Verdict: memra is already multi-model per process. Arm B does not require model-registry or
request-routing implementation. The scored two-process topology requires zero serving-code LOC.**

Verified seams:

- `crates/memra-server/src/main.rs:1-6` describes different models on one endpoint, owned by one
  GPU worker.
- `crates/memra-server/src/main.rs:37-41` defines comma-separated `MEMRA_MODELS` entries.
- `crates/memra-server/src/main.rs:1785-1849` parses multiple named model paths and per-model
  drafts; the fallback itself is a two-model pair.
- `crates/memra-server/src/main.rs:1601-1633` parses the plan and blocks until every model has
  loaded.
- `crates/memra-server/src/worker.rs:2716-2763` creates one `Engine` and iterates over the model
  plan; `crates/memra-server/src/worker.rs:2883-2933` stores each loaded model and publishes its
  capabilities by name.
- `crates/memra-server/src/worker.rs:212-214` carries the model id on every generation request;
  `crates/memra-server/src/worker.rs:4727-4751` validates it against the loaded map before queueing.
- `crates/memra-server/src/main.rs:2072-2077` and `crates/memra-server/src/main.rs:2415-2421`
  enumerate every loaded alias on `/models` and `/v1/models`; routes are registered at
  `crates/memra-server/src/main.rs:1664-1670`.

This is startup-resident multi-model service, not hot swapping. Models load once per server process
(`crates/memra-server/src/worker.rs:410-412`); no request-time load/unload or model-switch route was
found in the inspected surface.

The exact heterogeneous placement wanted here is still better represented by two processes.
`CUDA_VISIBLE_DEVICES`, `MEMRA_PP_STAGES`, and `MEMRA_PP_DEVICES` are process-wide; one in-process
model plan also shares the single CUDA-owner thread and scheduler described at
`crates/memra-server/src/worker.rs:1-16`. A same-process control can load and route two aliases,
but it does not provide the requested Step-PP-2/Q27-card-0 process isolation. Making placement and
worker ownership per model would be a medium-to-large loader/scheduler refactor across the config
parser, `worker::spawn`/`run`, engine ownership, health, and tests—not a missing hash-map patch and
not required for this A/B.

## 4. Prior receipts and the Arm-A adoption gate

The private Q3 economy note and Trial Plan V2 supply the question, market-rate inputs, and candidate
topology. The repository receipts below remain the authority for measurement claims. These are
historical anchors, not cells produced by this protocol:

- [`research/serve-ready-20260808/RESULTS.md`](../serve-ready-20260808/RESULTS.md) records box1
  Step PP-2 through the HTTP surface: short TTFT 0.595 s p50 (228 tokens, N=8), 4k TTFT 6.052 s
  p50 (4,107 tokens, N=5), and decode c=1/2/4/8 of 88.5/118.2/146.1/166.5 aggregate tok/s
  (N=3/rung). Its implied solo cold prefill is 679 tok/s.
- [`research/concprefill-20260808/RESULTS.md`](../concprefill-20260808/RESULTS.md) records the
  loaded prefill class: 568.2/575.3/577.6 aggregate tok/s for one/two/four simultaneous 4k primes
  with four decode streams live, and 580.5 tok/s for the four-prime control (N=3/cell).
- `ECONOMY-20260810.md` Q3 cites +18.7% Step TTFT from
  [`research/darktrain2-20260810/RESULTS.md`](../darktrain2-20260810/RESULTS.md). That source says
  the intended N=3 interleave stopped after two of nine cells: +18.706% came from one running-train
  cell and one of eight outputs differed. It is a sourced proxy, not a valid Q27 conclusion.
- The later direct [`research/27bab-20260810/RESULTS.md`](../27bab-20260810/RESULTS.md) Qwen3.6
  campaign found resident-idle neutrality but severe active contention on a different Vast Max-Q
  receipt: +290.2% Step short TTFT, -68.7% Step c=1 decode, and +19.7% Step 4k TTFT at N=5. It
  justifies separating `B-idle` from `B-active`; it does not populate a Qwen3.8 cell.

Do **not** re-measure Arm A when its receipts stand. They stand only when all of the following are
identical and recorded: physical host and GPU SKU/serials/power limits, P2P or host-bounce regime,
runtime commit and binary hash, Step target/draft hashes, full Step environment, prompts and
rendered token counts, cache state, concurrency/max-token shape, clock/thermal regime, and raw
receipt availability. Preserve the receipt's real N; never relabel N=3 as N=5.

If any item differs, the receipt is a historical comparator only and the executor MUST collect a
fresh Arm A in the same N=5 interleaved block as B. This is not an optional favorable rerun: the
denominator changed. If Arm A is adopted, record `baseline_origin`, original N, and why every
adoption check passed.

## 5. Frozen provenance and preflight

The future GPU lane MUST create a raw directory before starting either server and retain:

1. UTC timestamp, target host identity, GPU UUIDs/SKUs, driver, CUDA toolchain, power limit,
   `nvidia-smi topo -m`, P2P receipt, disk path, and free disk.
2. Runtime `git rev-parse HEAD`, clean/dirty status, binary path and SHA-256. Build outside this
   docs lane only if the target binary does not already match the pinned runtime.
3. Step trunk/draft and Qwen3.8 target/draft file sizes, source revisions, full SHA-256 values,
   manifests, and prompt/template hashes.
4. Secret-redacted process environments, command lines, PIDs, ports, `/health`, `/readyz`,
   `/v1/models`, and `/metrics` before and after every block.
5. One content-sanity request per model and a fixed output hash or exact token oracle. A wrong
   alias must return `model_not_found`; each endpoint must list only its intended process-local
   alias in the scored two-process shape.
6. The target-rig correctness battery appropriate to both exact artifacts: `kernel-check` all
   green, `run-gen` argmax match, and `run-spec` K=1..8 self-consistency for every model/draft path
   that uses speculation. Do not score after a red or vacuous gate.

Run under one exclusive `/tmp/memra-gpu.lock` hold. Capture stdout and stderr to raw logs before
parsing; never pipe a live run directly into a parser. Retain exit status separately. Scan only the
raw file for `CUDA_ERROR_OUT_OF_MEMORY`, `out of memory`, CUDA errors, Xids, illegal address,
panic, request errors, and server death.

## 6. Clock, thermal, and cleanliness contract

- Use the target host's production clock/power regime. If it is unlocked, report it as
  **unlocked production clocks**; do not call it clock-locked.
- Keep that regime unchanged for the entire block. Record a continuous 500 ms trace of SM/memory
  clocks, power, temperature, utilization, and used/free memory for both cards. Report N and
  min/median/max temperature and clocks with every median.
- Use same-window interleaving. Cross-run and cross-day results are not decision-quality paired
  evidence. The only allowed exception is the explicitly requested Arm-A receipt adoption in §4;
  keep it labeled historical with its original N and complete adoption proof.
- `window_clean=true` is allowed only for Arm A with no unrelated GPU compute process.
- `B-idle` and `B-active` MUST record `window_clean=false` and
  `controlled_co_resident=qwen38:{idle|active}`. Co-residency is the experiment, not contamination
  to hide. List every allowed PID; an unexpected PID excludes the cell.

## 7. Measurement schedule

Use the same prompt bytes, rendered token counts, sampling parameters, completion length, cache
policy, and request client in every compared cell. Use a unique cold `cache_salt` for TTFT/prefill
cells. For steady chat, pin exactly the intended persistent namespace per worker; unbounded fresh
namespaces measure allocator exhaustion, not steady c-load.

Each newly scored comparison is N=5 per condition. Use the frozen order
`A,B,B,A,A,B,B,A,A,B` for two-condition blocks. For the residency question, B is `B-idle`; for the
active question, B is `B-active`. Start one excluded warmup per fresh server state. Never rerun a
completed favorable arm because its paired arm failed; retain and mark the incomplete block
excluded, then restart the entire pair only after fixing the cause.

### 7.1 Step-facing cells

For each condition:

- TTFT: exact 228-token short fixture and exact 4,107-token rendered 4k fixture; streaming first
  visible content; p50 and nearest-rank p99 (with N=5, p99 is the maximum and must be labeled so).
- Decode: c=1/2/4/8, one barrier burst per replicate, exactly c requests, 128 completion tokens,
  aggregate tok/s and per-stream tok/s.
- Solo prefill: one cold 4,107-token request, 8 completion tokens; report server prime tok/s and
  the client-implied `prompt_tokens / TTFT` value separately.
- Loaded prefill: reuse the concurrent-prefill receipt shape—four background Step decode streams
  plus one/two/four simultaneous cold 4k primes; report aggregate prompt tok/s and background
  inter-token p95. The ~580 reference is a loaded class, not a solo number.

In `B-active`, hold Q27 at a steady c=2 chat load with exactly two persistent namespaces. Require
the background interval to begin before and end after every Step request. Record Q27 output tok/s
during the same interval; a load generator that silently empties makes the Step cell invalid.

### 7.2 Q27 productive-capacity and reverse-interference cells

With both models resident, first characterize Q27 while Step is idle, then while Step is active:

- Q27 TTFT on the same short and 4k prompt classes, with its own observed rendered token counts.
- Q27 decode c=1/2/4/8, 128 completion tokens, N=5/rung.
- Q27 solo and loaded prefill, using the same definitions as Step.
- Reverse overlap: issue a Step 4k prime across each Q27 c=1 replicate and a sustained Step c=4
  decode window across each Q27 c=1/2/4/8 replicate. Require host-monotonic overlap assertions.

Report input and output tok/s separately. Do not turn speculative acceptance into billable output;
only response usage completion tokens enter productive throughput.

The existing immutable client
[`research/27bab-20260810/measure.py`](../27bab-20260810/measure.py) already writes per-request
JSONL, summary JSON, first-token/latency/decode fields, usage, acceptance deltas, output hashes, and
monotonic intervals. Reuse or wrap it; do not hand-copy results. Parse the completed raw logs only.

## 8. Pending result matrices

Every value below awaits the exact Qwen3.8 artifact on the target box. Historical values in §4 are
references only. N=5 labels apply to newly executed cells; an adopted Arm-A value must identify its
source and original N in the cell rather than being relabeled N=5.

### Step cost

| Step metric | Arm A | B-idle | B-active (Q27 c=2) | A→B-active delta |
|---|---|---|---|---|
| TTFT short p50/p99, N=5 | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution |
| TTFT 4k p50/p99, N=5 | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution |
| decode c=1 aggregate/per-stream, N=5 | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution |
| decode c=2 aggregate/per-stream, N=5 | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution |
| decode c=4 aggregate/per-stream, N=5 | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution |
| decode c=8 aggregate/per-stream, N=5 | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution |
| solo prefill server/client tok/s, N=5 | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution |
| loaded prefill aggregate tok/s, N=5 | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution |
| loaded background inter-token p95, N=5 | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution |

### Q27 slot value and reverse interference

| Q27 metric | Step idle | Step active | Idle→active delta |
|---|---|---|---|
| TTFT short p50/p99, N=5 | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution |
| TTFT 4k p50/p99, N=5 | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution |
| decode c=1 aggregate/per-stream, N=5 | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution |
| decode c=2 aggregate/per-stream, N=5 | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution |
| decode c=4 aggregate/per-stream, N=5 | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution |
| decode c=8 aggregate/per-stream, N=5 | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution |
| solo prefill input tok/s, N=5 | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution |
| loaded prefill input tok/s, N=5 | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution |
| output acceptance and errors | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution |

### Memory and health

| Receipt | Arm A | B-idle | B-active | Delta/peak |
|---|---|---|---|---|
| card 0 used/free MiB | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution |
| card 1 used/free MiB | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution |
| per-process used MiB | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution |
| request errors/OOM parks/Xids | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution |
| `window_clean` and allowed PIDs | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution | PENDING box1 execution |

Latency interference is `(B - A) / A`; throughput interference is `(B - A) / A`. Report signs,
absolute differences, all five samples, median, range, and the thermal regime. Never summarize a
single observation as a median.

## 9. Economics-shaped decision rule — no verdict here

Use the pinned market rates only as inputs:

```text
p_in  = $0.20 / 1,000,000 input tokens
p_out = $1.15 / 1,000,000 output tokens
```

For model `m` over one second, with measured sustainable input/output rates and owner-supplied
utilizations:

```text
gross_m = in_tok_s_m  * p_in  * input_utilization_m
        + out_tok_s_m * p_out * output_utilization_m
```

Compute Step's co-residency interference tax at matched demand, never from latency alone:

```text
step_tax = max(0, gross_step_A - gross_step_B_active)
q27_gain = gross_q27_B_active
co_resident_increment = q27_gain - step_tax - incremental_co_resident_cost
```

The minimum pass condition is `q27_gain > step_tax` plus all correctness/reliability gates. If the
interference tax on the paying Step service equals or exceeds the marginal Q27 gross, co-residency
fails. This document does not choose utilization or a Step regression bound.

To compare with a separate box, the owner supplies the separate-box cost and measured Q27 rates:

```text
co_resident_net = q27_gain_co - step_tax - incremental_co_resident_cost
separate_net    = q27_gain_separate - separate_box_cost

co-residency is worth the slot only if:
co_resident_net >= separate_net
```

Report short/4k TTFT and decode/prefill regressions beside this equation as QoS constraints. The
Q3 suggestion of a <=5% Step TTFT-p50 regression is prior guidance, not a decision made here; the
owner must approve any bound after seeing the exact Qwen3.8 receipts.

## 10. Completion and stop conditions

The future execution is complete only when every newly executed cell has raw N=5 evidence, every
adopted Arm-A cell has its original N and complete adoption proof, every compared interval has the
required overlap/cleanliness receipt, all model outputs and gates are correct, and the result report
states the exact arm manifest, N, clock/thermal regime, and hashes.

Stop without a verdict on any of these: missing or mismatched artifact hash; unvalidated Qwen3.8
draft; changed Step manifest; unexpected process; lost stderr; failed overlap assertion; wrong or
empty output; request error; OOM/CUDA/Xid/panic; thermal/clock regime change; or an Arm-A adoption
claim that cannot prove every gate in §4. Retain failed raw evidence and quote the captured cause.

The owner—not this protocol—decides whether to allocate a serving slot, move Q27 to a separate
box, set a QoS threshold, publish pricing, begin a trial, reserve hardware, merge, tag, or release.

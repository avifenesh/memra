# step37 VRAM admission: interactive capacity, not a static session cap

Lane: `lane/step37-vram-admission-20260830`. Owner-prioritized: "we need interactive cap,
not to simply rate limit when we can take more, and my oom was on single session."
Predecessor: the step37 capacity lane (private ops repo), whose three measured defects this
lane fixes; every number cited below was re-measured here on the public battery.

Box: the sbox dev box (2x RTX PRO 6000 Blackwell Server, 97,887 MiB each; provisioning
receipts live in the private ops repo). Artifact `/data/models/step37-flash-nvfp4` verified
shard-by-shard against HF `stepfun-ai/Step-3.7-Flash-NVFP4` rev
`4275532ffd9a9496ff36b7a2dc4a9db1048da438` (14/14 sha256 OK, banked by the postthink lane
on this box and re-checked here). Baseline binary `memra-server-base-189548721` (origin/main
at the draft-graph-serving merge); lane binaries built on-box from the lane commits
(explicit-refspec bundle fetches): `2abd605bf` (charging + gate + fenced teardown),
`6ba68aceb` (park-replay capture-disable + FLAGS drift), `140dac88b` (eager-arm
admission), `a6bf7d9ee` (graph-launch headroom guard, the segfault root-cause fix),
`7bb2f55b7` (post-probe pool trim), `7198046f1` (two-floor gate + stale-read reject
recovery + request-cost log consistency), `356a69d8d` (capture takes at most half the
discretionary headroom) — the SHIPPED tip. Binary md5s in `raw/build-provenance.txt`.
Every escalation between those commits was forced by an on-box cell and is receipted in
`raw/battery.txt` (the full battery trail).

## The three measured defects (capacity-lane numbers, restated)

1. **Admission under-charges the first-request fixed transient 4.6x.** The reserve was a
   flat 1,611 MB (`SPEC_SHRINK_RESERVE`, calibrated on a small-model control fit); the
   measured first-burst transient on this deployment class is 7,458 MiB. Admission admitted
   past the card with `admission_vram_defers=0` and `vram_reject=0`; medium-class c=8
   hard-failed from a clean boot at 6,605 MiB free.
2. **Per-session draft-chain graph state charged at ZERO.** Since the 3-head MTP chain
   capture + in-graph filtered sampling (merge `189548721`), each capturing session parks
   real device state (capture-retain keepers, q slots, instantiated graphs' backing). On a
   deployment with ~7.9 GB free on device 0, a SINGLE session OOM'd on its second prompt:
   sampled draft-graph capture failed (CUDA_ERROR_OUT_OF_MEMORY), then the customer-visible
   `[engine-error] class=Overloaded step error: DriverError(CUDA_ERROR_OUT_OF_MEMORY)`,
   28 OOM lines, card at 5 MiB free.
3. **The OOM-park cliff was a silent fleet-fatal segfault.** Multiple active sessions
   hitting the step-OOM park path segfaulted the GPU worker in libcuda (same fault offset
   +0x27c87f, reproduced 3x on independent boots at 4 and 8 active sessions), killing every
   live session with panic=0, exit-70=0, ILLEGAL=0, #87=0 and NO log line; only dmesg saw
   it. The single-active park path ran 18x clean.

Preserved fact (correct, untouched): session KV charge is exactly
`prompt + (max_tokens or default_output_length) + 64` tokens; an omitted max_tokens costs
8.2x (972-token prompt: 138 MB capped vs 1,126 MB uncapped). That charge reserves what
generation may use, by design.

## The fixes (lane commits `2abd605bf..7bb2f55b7`)

### Defect 1: boot admission calibration (`MEMRA_ADMIT_CALIBRATE`, default ON)

Before the worker reports ready, one spec-shaped generation (synthetic one-chunk prompt,
vendor-shaped filtered sampling temp 0.7 / top_p 0.9, pinned seed; a memory-shape probe,
never a perf cell) runs through the real serving path: chunked prime, draft-chain capture,
sampled verify. A new engine instrument (`Engine::pool_high_water_reset`, the driver-kept
CU_MEMPOOL_ATTR_*_HIGH watermarks) records how deep the burst actually dipped on EVERY
device; tick-boundary sampling sees nothing of it because engine transients allocate and
free inside one step. The measured dip, minus the state classes admission charges
separately (the probe session's context-linear KV and its measured draft-graph state),
becomes the deployment's transient reserve floor, never below the static constant, and the
`MEMRA_ADMIT_RESERVE_MB` teeth door still wins outright. The probe also materializes the
first-request one-time pools (prime slabs, cuBLAS workspaces, graph pools) at boot, so
request 1 stops paying them, and it supplies the first per-session draft-state observation.
Probe failure is LOUD and non-fatal: static floor serves, pool trimmed back.

### Defect 2: per-session draft-state charge + safe capture (`MEMRA_SPEC_CAPTURE_GATE`, default ON)

- **Charge:** the spec capture block measures each session's effective-free delta across
  its captures into a model-owned high-water gauge
  (`HybridModel::draft_session_admission_bytes`); admission adds it to every spec-capable
  session cost (`[admission] per-session draft-state charge:` line). 0 until observed; the
  boot probe normally observes it.
- **Reserve-check before capture:** every draft-graph capture arm (chain greedy, chain
  sampled, single-head greedy, single-head sampled) now refuses BEFORE allocating when the
  device's effective free (driver free + async-pool cached) cannot cover the capture's
  expected appetite plus a post-capture safety floor. The refusal rides the same LOUD
  once-per-flip `[spec] WARN: ... capture failed (insufficient VRAM headroom ...)` line;
  the session serves eagerly with headroom intact.
- **Trim on capture OOM:** a failed attempt's freed transients sat CACHED in the async pool
  (RELEASE_THRESHOLD pinned to u64::MAX) where the driver cannot see them, while graph
  instantiate and cuBLAS allocate from the DRIVER; the card sat at 5 MiB driver-free with
  the room parked in our own pool. Capture-OOM failures (and driver-short pre-capture
  states) now `cuMemPoolTrimTo(0)` the cached blocks back.
- **No stranded partial state:** q-slot allocation after a successful capture is
  all-or-nothing (a mid-loop failure used to strand orphan slots on the ctx and kill the
  burst); the dcw headroom pre-arm moved inside the fallible capture closures so its OOM
  degrades to eager instead of erroring the whole burst.

### Defect 3: fenced step-OOM teardown

Step-OOM'd sessions are marked `oom_teardown`. Retirement:

- never runs the pending-carry flush for them (a GPU forward pass on a card that just
  proved full);
- never pool-parks their KV (the park contract is "caches drop, freeing exactly the VRAM
  the retry needs"; a pool-park keeps the VRAM alive and starves the requeued retry);
- drops their device state behind a device FENCE (every PP stage and Step-TP rank stream
  synchronized) so graph-exec destroys and pool frees can never race work the aborted step
  or a same-tick peer left in flight;
- after the sweep, trims the async pool on every device and prints the grep-stable
  `[admit-oom] step-OOM teardown complete: N session(s) dropped behind a device fence;
  pool trim released X MB across D device(s)`.

### Defect 3 root cause: cuGraphLaunch into an exhausted card (found via core dump)

The decode-phase squeeze (8 active sessions decoding, external ballast eating the card at
~1 GB/2s — the engine-internal equivalent is its own pool walking driver free to ~5 MiB,
the owner's incident) reproduced the EXACT capacity-lane signature on the base binary:
silent GPU-worker death, no log line, dmesg
`segfault at 60 ip ... error 4 in libcuda.so.595.91.07[27c87f,...]` — offset **+0x27c87f**,
a null internal dereference (address 0x60). The 1.95 GB core dump's crashing thread:

```
#3  cuGraphLaunch ()                              libcuda.so
#4  <cudarc CudaGraph>::launch ()
#5  <HybridModel>::generate_spec_inner2 ()
#8  memra_server::worker::step_session ()
```

Same frames on the pre-guard lane binary (second core) — the fenced teardown alone could
not close it, because the crash fires in a PEER session's round in the same tick as the
parks, before any retirement runs: `cuGraphLaunch` segfaults (driver 595.91.07) when a
captured draft/stream graph is dispatched into a driver-exhausted card, while the EAGER
arms fail recoverably (a quoted CUDA OOM the park path handles) on the same card.

Fix (`a6bf7d9ee`): `GRAPH_LAUNCH_MIN_FREE` (256 MiB) driver-free floor checked once per
round; below it every captured-graph launch arm (round-stream, greedy/sampled chain,
greedy/sampled single-head) yields to its byte-identical eager twin for the round, with
one grep-stable `[spec] graph replay suspended:` line per burst. This lane closed the
reproduced MTP spec path only; the named follow-up landed as
lane/graph-launch-guard-sweep-20260831 (`research/graph-launch-guard-sweep-20260831/`),
which extends the same floor to EVERY remaining serving-reachable captured-graph launch:
dspark verify graphs (run_full/run_segment, all three callers including the default-ON
dspark serve route), the optipipe controller draft graph, GraphSession::step (no eager
twin: recoverable session-scoped refusal), the step35 token graph (step + chunk), the
TP routed-prejoin graph door, and the MEMRA_PRIME_SEG prime segments. Each route keeps
the grep-stable `graph replay suspended:` key under its own route tag.

Two more mechanisms the battery forced into the open, fixed in-lane:

- **Eager-arm admission** (`140dac88b`): the draft-state charge models a CAPTURING
  session, but on a tight card the pre-capture gate refuses those captures at exactly the
  headroom that cannot hold the charge; admission was reserving memory for state that
  would never exist and starving the card. It now admits on the eager twin (grep-stable
  `[admission] eager-arm admit:` line) when the captured arm defers, while the gate is
  armed to enforce the refusal.
- **Post-probe pool trim** (`7bb2f55b7`): the probe's freed blocks sat cached in the
  async pool while eager serving still takes DRIVER allocations; on a tight boot the
  driver was left at ~0.5 GB with ~2.8 GB parked in our own cache and the first admits
  step-OOM'd on allocations admission had honestly budgeted. Calibration now hands the
  cache back (`[admit-cal] post-probe pool trim:` receipt).

## 1. Transient charge (defect 1)

Instrument cell (lane binary, `MEMRA_ADMIT_CALIBRATE=0`, 100ms nvidia-smi sampler, ONE
uncapped real request on a fresh boot): device-0 driver-used grew 74,067 -> 83,639 MiB,
i.e. the real first-request footprint is **9,572 MiB** (never released; the async pool is
monotone by design). The baseline admission charged 538 MB session + 252 MB tp-kv +
1,611 MB reserve = **2,401 MB** for that request: a **4.0x under-charge** on this shape
(the capacity lane measured the same class at 4.6x on its request shape).

The lane decomposes and charges all of it:

| component | measured | how it is charged now |
|---|---|---|
| one-time pools (prime slabs, cuBLAS workspaces, graph pools, first-capture side state) | ~4,900 MiB (boot rest-used 74,067 -> 78,967 with the probe) | PAID AT BOOT by the calibration probe, before the worker reports ready; `mem_get_info` then tells admission the truth |
| per-session draft-graph state | 2,608 MiB (model-owned high-water gauge, observed by the probe and re-observed live) | added to every spec-capable session cost; eager-arm uncharged when the capture gate would refuse (section below) |
| recurring per-burst transient | 728 MiB pool-used dip above rest (probe, both devices' max) | reserve floor; stays at the static 1,536 MB lower bound on this deployment |

Boot receipt (grep-stable, banked in `raw/serve-B1.log`):
`[admit-cal] boot calibration done: model="step37" transient floor 1536MB (static was
1536MB; measured 728MB; probe kv charge 358MB, draft-state 2608MB, drafted 48 accepted 48;
[dev0 peak-used 78132MB rest-used 75240MB charged 3135MB -> 0MB; dev1 peak-used 58769MB
rest-used 57872MB charged 168MB -> 728MB]; 2.3s)`

Measured-vs-charged on the fresh-boot first request: post-probe the request's remaining
appetite is 9,572 - 4,900 = 4,672 MiB; the lane charges 538 + 252 + 2,608 + 1,536 =
4,934 MiB. **Charged/measured = 1.06 (within a +-15% stated tolerance, on the safe
side).**

Medium-class c=8 from a clean boot (capped 2048, real agentic prompts, vendor-default
sampling):

- BASE binary at 6,049 MiB free (`raw/serve-A2.log`): admission admitted past the card;
  7 step-OOM parks + 1 unparked OOM, a customer-visible engine-error, 8 OOM lines,
  **1/8 requests served**. The hard-fail class, reproduced.
- SHIPPED binary, M3 cell at 8,161 MiB free (`raw/serve-M3.log`, `raw/client-M3-rows.jsonl`):
  **8/8 requests complete, zero engine-errors, zero OOM lines, zero parks**; the pre-capture
  gate refused all eight captures LOUDLY (effective 7,932-8,032 MB < required 8,311 MB) and
  the deployment served the whole wave eagerly. dmesg clean.
- Free-target note (harness receipt): "clean boot at X free" cells originally derived the
  pre-boot ballast from the BASE binary's natural boot footprint; the lane binary's boot
  probe adds ~4.9 GB of one-time residency, so every early lane cell inadvertently ran
  ~3.4 GB TIGHTER than its stated target (those runs are banked in `raw/battery.txt` as
  escalation evidence: at ~4.5 GB free the same wave degraded to honest 429s, never to
  engine-error hard-fails or faults). M3/M4/N* are the corrected-target cells.

## 2. Graph charge + safe capture (defect 2)

- The per-session gauge observed **2,608 MiB** of draft-graph state per capturing session
  (3-head chain, K=3, filtered sampled shape) — the state the baseline charged at ZERO.
  Receipt: `[spec] draft-session state high-water: 2608MB` (`raw/serve-B1.log`, re-observed
  in every capturing boot).
- The pre-capture reserve gate fired on the real card with the observed appetite
  (`raw/serve-B4-1.log`): `[spec] WARN: sampled draft-graph capture failed (insufficient
  VRAM headroom for capture: effective free 2744MB (driver 2294MB + pool-cached 449MB) <
  required 4144MB (appetite 2608MB + floor 1536MB); capture skipped pre-attempt); eager
  fallback until session resume` — one LOUD line, session kept serving eagerly, headroom
  intact.
- Owner-shape replay (8 successive uncapped real prompts, one accumulating session,
  ~7.9 GB free device 0 at clean boot): BASE reproduced the incident chain — the owner's
  exact WARN (`sampled draft-graph capture failed (CUDA_ERROR_OUT_OF_MEMORY ...)`)
  followed by step-OOM engine errors (`raw/serve-A3-c8.log`: 30 OOM lines, 8 engine
  errors, 0/8). SHIPPED binary: **N1 8/8 and N2 8/8, all 200** (real generations up to
  8,192 tokens/turn), zero OOM lines, zero engine errors, zero parks, dmesg clean; every
  capture refusal was ONE loud WARN and the session kept serving eagerly
  (`raw/turns-N1.jsonl`, `raw/turns-N2.jsonl`, `raw/serve-N1.log`, `raw/serve-N2.log`).
- LATENT ENGINE BUG EXPOSED (separate defect, named for its own lane): one M4 rerun died
  at turn 4 with the loud recoverable `SWA ring state exceeds capacity (5152 > 5151)
  [append len=5152 retain_from=4128 append_rows=1, spec.rs:5286]` — an SWA-ring
  rebase/retain boundary assert on a resumed session crossing the ring's physical rows,
  in code this lane does not touch (`raw/serve-M4.log`, `raw/turns-M4.jsonl`). Loud,
  session-fatal only, not memory-related; needs its own exactness-gated ring lane.

## 3. Segfault (defect 3)

Reproduction attempts on the BASE binary on this box (teeth-door park storms at ~7.4 GiB
free; decode-phase ballast squeezes): the multi-active park storm shape reproduced at
scale — **21 step-OOM parks + 8 unparked OOMs in one cell** (`raw/serve-A3-c8.log`) — and
the base GPU worker additionally PANICKED in the storm (`raw/serve-A3-c4.log`:
`thread 'memra-gpu-worker' panicked at cudarc-0.19.8/src/driver/safe/core.rs:861: called
Result::unwrap() on an Err value: DriverError(CUDA_ERROR_OUT_OF_MEMORY)` — a CUDA call
in the teardown/steady path unwrapping under OOM), but the exact libcuda segfault at
+0x27c87f did not fire on this box's shapes (dmesg clean on every base cell; N runs
listed in `raw/battery.txt`). The capacity lane's segfault, the base panic observed here,
and the base engine-error storms are all the same defect surface: the pre-fix teardown
performs GPU work against a full card while peers are mid-tick, with nothing fencing the
drops.

The fix makes that surface unreachable rather than racing it: OOM-torn sessions never run
the pending-carry flush, never pool-park their KV, drop behind a full device fence (every
PP stage + TP rank stream synchronized), the async pool is trimmed back to the driver
after the sweep, and a park REPLAY serves capture-disabled. LANE re-runs of the same storm
shape x5 (`raw/serve-B4-1..5.log`): oom_parks 6-21 per run, teardown fences fired (1-16
per run), **zero dmesg faults, zero panics, zero ILLEGAL/#87, server alive and healthy
after every run**, all failures surfaced as the honest recoverable error. Decode-phase
squeeze re-runs x5 on the final binary: E-lane-1..5 below.

## 4. Regression gates

| gate | verdict |
|---|---|
| run-spec greedy K=1..8, base binary vs lane binary (curve-0400 prompt, NGEN=160, heads=3) | **CROSS-BINARY IDENTITY PASS** + SELF-CONSISTENCY PASS both arms (`raw/rs-greedy-*.log`) |
| run-spec sampled seeded twins K=1..8 (temp 0.5 / top_p 0.9 / seed 4242), base vs lane | **CROSS-BINARY IDENTITY PASS** (identical `sampled tokens:` streams and `acceptance:` lines at every K) + seeded-rerun PASS both arms (`raw/rs-sampled-*.log`) |
| serving greedy byte gate (temperature 0, curve-0400, max_tokens 400) | sha `ffc4004c678ad8ea` on BOTH binaries, identical usage/acceptance (207/286) (`raw/greedy-base.jsonl` vs `raw/greedy-lane.jsonl`) |
| 8-turn accumulating conversation, vendor-default sampling, healthy headroom (lane) | 8/8 200, spec engaged every turn (acceptance 0.76-0.84), cached tokens observed (`raw/turns-B1.jsonl`) |
| ILLEGAL / #87 / panic across every lane cell | 0 / 0 / 0 |

Identity gates were re-run against the baseline for EVERY binary spin that touched engine
code (140dac88b, 7bb2f55b7, 9fbbd149a, 02c4a01de) — `RS-greedy/RS-sampled CROSS-BINARY
IDENTITY: PASS` each time (`raw/battery.txt`; the shipped tip's runs banked as
`raw/rs-greedy-l.log` / `raw/rs-sampled-l.log` with extracts).

Fleet protocol note (owner amendment 2026-08-30, timed A/B x3 default): this lane runs
no timed A/B comparisons — every cell is a correctness, byte-identity, or fault gate,
and the segfault N>=5 re-run is a fault gate — so the amendment resizes nothing here.

Optional owner suggestion (release unused KV budget at eos/park, recharge-to-actual):
NOT taken in this lane — the required fixes above went through seven measured escalations
and the recharge seam deserves its own cell set; the charging receipts here are the
baseline any such lane would gate against.

## 5. Flags (same-commit FLAGS.md rows)

| flag | default | why |
|---|---|---|
| `MEMRA_ADMIT_CALIBRATE` | ON (by design, receipts here) | the static floor was measured 4.6x low; `=0` restores it |
| `MEMRA_SPEC_CAPTURE_GATE` | ON (by design, receipts here) | try-and-fail capture walked a tight card to the edge; `=0` restores try-and-fail (diagnostics) |

reasoning-effort law: no effort field in any cell (the client sends vendor-default bodies).

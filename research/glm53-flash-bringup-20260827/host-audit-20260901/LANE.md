# HOST-MICRO AUDIT — the non-GPU per-token wall, engine-wide

Lane `lane/glm5-host-audit` (memra), base `origin/lane/glm53-flash-bringup` @ `216ffd114`.
**Scope is ENGINE-WIDE, not glm5's**: the audit walks the SHARED decode hot path every served
family rides (the worker tick, the engine decode step, the allocator/sync sites, the sampler)
plus the places each family's walk diverges. glm5 is the measurement vehicle because it is the
heaviest available decode shape and the only one with a free bench box; every fix that lands is
family-agnostic by construction, and §PROD-APPLICABILITY says which prod stack may adopt what.

Question the owner set: prefetching, cache-friendliness, context switches and locks on the decode
hot loop have never been audited, and ~10 ms/token (~27% of the serving round) is unattributed
non-GPU wall. Where does it go, and does placing the worker thread on the host recover any of it?

**THE HEADLINE, and it reframes the brief.** The unattributed host wall is **CALL-COUNT bound,
not locality bound.** The decode tick makes thousands of host-side calls per token — driver calls,
`getenv`s, hash probes, heap allocations, and on one live prod request shape an O(vocab) sort —
and those costs are paid wherever the thread runs. CPU pinning addresses migration and L3 reuse,
which is a second-order effect layered on top; the code audit predicted the pinning arms would
land small, and **the box measured them at zero** (§B1: +0.021% / -0.009%, inside a 0.04% noise
floor, with the pin's 12x-narrower mask confirmed by kernel readback). The levers are the
call-count reductions. The single largest finding is not a lock
or a cache miss at all: it is a **default that routes an entire live prod request shape through an
O(n_vocab) host sampler** (§A5), and it is a one-line fleet config change plus a fix that landed
in this lane.

Nothing in §A is measured — it is source-derived, with file:line for every row and counts stated
as arithmetic against the serving trunk. §B0 and §B1 ARE measured: the Box B window ran
2026-08-31T21:12-21:48Z (11 boots, greedy + vendor-default sampled on every arm) and **every
intervention arm came back NULL**. The estimated microsecond bands in §A remain arithmetic against
banked box constants and are labelled as such throughout; the only measured wall numbers in this
lane are §B1's tok/s rows.

---

## A. THE CODE AUDIT — per-token host cost by class

Trunk for the counts: glm5_next, 45 layers (42 MoE), `n_used=8`, mHC streams=4 ⇒ 90 hc sites/token.
Default flags, T=1 (the latency-sensitive serving width). At B=1 one scheduler tick == one token
(`worker.rs:10206`), so every per-tick cost below is a per-token cost at that width.

### A1. Locks taken per token

| lock | site | count/token | shared with | verdict |
|---|---|---|---|---|
| **`std::env::var` → std's process-global `ENV_LOCK`** + a `CString` malloc + a linear `environ` scan | ~30 non-memoized gate fns; in the tick `worker.rs:12418`, `:12427`, `:12948`, `:18290`; per-layer `lib.rs:13786` (`MEMRA_FAST`, 121 call sites), `:17401`, `:12942`, `:24670-24671`, `:1229/1232`; per-hc-site `hyper.rs:318`; per-MoE-layer `hybrid_forward.rs:374`, `:390`, `:9001-9004`, `lib.rs:1657`, `:1860`, `:3790` | **~600–1,200** | **every thread in the process, including all tokio workers** | **THE ONLY GENUINELY CONTENDED LOCK ON THE PATH.** A cross-core cache-line ping-pong that scales *against* concurrency. Est. 200 µs – 1.2 ms/token |
| `Engine::func(name)` — `Mutex<HashMap<String, CudaFunction>>` + SipHash of a 15–30 byte kernel name + `Arc` clone/drop | `lib.rs:2440` (field `:1030`), 478 launcher call sites | **~1,760–2,100** (once per `launch_builder` launch) | nothing — one GPU worker thread, invariant stated at `lib.rs:1006` | **UNCONTENDED BY DESIGN.** Est. 180–300 µs/token. The clearest removal candidate, but removal is a 478-site type change ⇒ named follow-up, not an in-lane fix |
| `Mutex<Metrics>` publish | `worker.rs:13634` | **1 per 32 tokens** + 1/retire | tokio HTTP threads (`lib.rs:2027, 2192, 5185, 6029, 14253, 14300, 16725`) | Correctly throttled, and the throttle is documented at `worker.rs:13614` (it prices the unthrottled form at ~1.7 ms/token). Not a finding. But see A2 for what runs *inside* the critical section |
| `AtomicU64` RMW — `health.beat_busy()` → `beat_ms.swap(Release)` | `health.rs:220-224`; call `worker.rs:10303` | 1 | `/health`, `/readyz` (loads only) | Fine. Note the comment at `worker.rs:10299` claims "no syscall" and the path also reads the clock (`health.rs:102`) — documentation drift, worth a one-word fix |
| `router_stage.lock()` + pinned DtoH ×2 + **blocking `synchronize()`** | `lib.rs:6152`/`:6165`, reached from `hybrid_forward.rs:9089` | **42** | nothing | Uncontended, but the *sync* is the cost — see A3 |
| `moe_cache`, `verify_ws`, `hyper_decode_ws`, `fa_part_pool`, `w8_*`, `capture_keep` | `lib.rs:967, 1063, 10035, 1024, 978/983` | 0–42, mostly 0 on the serving config | nothing | All single-worker-thread by the same invariant. Uncontended by design; none hot enough to be worth removing |
| `Gpu::stream()` / `Gpu::blas()` — thread-local `RefCell` borrow + `Arc` clone (2 atomic RMW with the drop) | `memra-runtime/src/lib.rs:193`, `:201`; 719 static sites | ~2,000+ | nothing (thread-local) | Est. 50–100 µs/token. The by-value contract is deliberate (nested-override stability), so this needs care, not a blind change |
| tokio `mpsc::send(Event::Token)` → receiver wake | `worker.rs:17204-17210` | **1** | **the tokio SSE task** | The one real cross-thread wake per token: at B=1 the SSE task is parked in `rx.recv()`, so it goes `AtomicWaker::wake` → remote schedule → **eventfd `write(2)`**. ~1–3 µs + a context switch. Structurally hard to remove: the SSE-cadence law at `worker.rs:18555-18558` forbids coalescing ids |
| `s.tx.is_closed()` cancel sweep | `worker.rs:11369` | 1 per active session | tokio HTTP task | Reads a word the receiver mutates ⇒ one cache-line bounce per token per session. Correct and cheap |
| **the device allocator** | `lib.rs:10696` (`alloc_uninit`), `:10025` (`zeros`) → `cuMemAllocAsync`/`cuMemFreeAsync` | ~2,358 alloc + ~2,358 free | — | **NOT a Rust lock at all.** The cost is the driver-call count, which is exactly what `SCRATCH_ALLOC_CALLS` (`lib.rs:1134`) measures. Est. ~4.3 ms/token at the box's banked ~1.06 µs/driver call |

`OnceLock` reads are ~free acquire-loads and are **not** findings; the codebase's dominant idiom is
already correct. The nine `env::var` sites in the tick are the exception, not the rule.

### A2. Host data structures touched per token

| item | site | size / count | verdict |
|---|---|---|---|
| **`Vec<(u32,f32)>` over the FULL vocab in the host sampler** | `memra-sampling/src/lib.rs:309-313` | **n_vocab × 8 B ≈ 1.2 MB** | Above glibc's initial `M_MMAP_THRESHOLD` ⇒ an `mmap`/`munmap` pair on the first tokens, plus a full n_vocab write pass. Only on the host-sample path — see A5 |
| **the penalty pass hashed once PER CANDIDATE** | `memra-sampling/src/lib.rs:411-415` | **n_vocab SipHash probes/token** | **FIXED IN THIS LANE** (§C1): the loop was inverted the expensive way round; `penalty_counts` has ≤ `PEN_WINDOW_MAX` entries |
| **full n_vocab `sort_unstable_by` for top-k** | `memra-sampling/src/lib.rs:338-341` | ~2.6 M comparisons at 152k vocab | Named follow-up: `select_nth_unstable_by` is the shape, but it changes which of equal-logit candidates survive at the k boundary ⇒ **numeric-class, needs its own gate**, not an in-lane fix |
| `active[i].last_logits = rows[k].clone()` | **`worker.rs:12890`** | n_vocab × 4 B ≈ 608 KB | Gratuitous: `rows` is an owned `Vec<Vec<f32>>` returned by value, so `std::mem::take` removes the alloc+memcpy. Free when `MEMRA_SERVE_LEANLOGITS` (default ON) empties the source; **not** free for host-sample rows |
| logits `Vec<f32>[n_vocab]` freshly allocated per token by the D2H | `hybrid_forward.rs:2737-2738` via `lib.rs:9946` | ~0.6 MB | The only per-token host buffer > 512 KB on the shared engine path, and it is a **pageable** D2H that could ride the `PinnedStage` pattern `router_stage` already uses (`lib.rs:1264-1270`) |
| mHC glue allocations | `hyper.rs:404-408`, `:667` | ~630 device allocs/token | `MEMRA_HC_DECODE_WS` (default OFF) already removes these; receipt 76.0 → 48.3 allocs/step |
| MoE sequential per-expert loop | `hybrid_forward.rs:9769-9808` | **6 launches + ~6 allocs × 8 experts × 42 layers ≈ 2,016 each** | The single biggest already-known lever: `MEMRA_MOE_FUSED_EPI` (default OFF) takes ~49 launches/token-layer to 4 |
| `e.htod(&vec![1.0f32; t])` — re-uploading a CONSTANT | `hybrid_forward.rs:10883` | 42 allocs + 42 pageable HtoD | `MEMRA_GLM5_HTOD_DIET` (door H, default OFF) replaces it with a resident buffer |
| `si.iter().map(...).collect()` + `sw.to_vec()` out of the pinned router stage | `lib.rs:6166` | 84 Vecs/token, 8 elems each | Pure copy-out of bytes the pinned buffer already holds |
| ~16 tick-scoped Vec/VecDeque allocations | `worker.rs:12809, 12810, 12811, 12820, 12837, 12850, 12863, 12751, 12765, 11355, 10399, 11361`, `group_chunks` `:18265-18281` | 16 allocs/tick | Hoistable to persistent scratch + `clear()`. Est. 1.5–4 µs/token. Includes `model_name.clone()` at `:12811`, a String alloc purely to satisfy the borrow checker |
| `decode_bytes_special` returns a fresh `Vec<u8>` per token | `memra-tokenizer/src/lib.rs:1067-1068` | 1 malloc+free/token, 1–8 B payload | Immediately extended into `s.decoded_bytes` and dropped; a `decode_bytes_into` variant removes it |
| **`contains_stop_string` re-scans the WHOLE completion every token** | `worker.rs:17190-17201`; calls `:17918, 17978, 18941` | **O(len(decoded) × n_stops) per token ⇒ O(n²) per request** | Paid only when the client sent `stop` — which agent clients (buyer profile 2) routinely do. Est. 5–15 µs/token at 2k generated. Fix: scan only the `max_stop_len + new_bytes` tail |
| two full percentile sorts under the `Metrics` lock | `memra-lanes/src/lib.rs:95-104`; calls `worker.rs:13641-13642` | allocates ≤16,384 f32 and fully sorts it, **twice** | ~2.2 ms on the tick it lands, ~70 µs/token amortized by the 1/32 throttle. Also inside that critical section: 4 HashMap clones and `mem_get_info()` (`worker.rs:13692`), a CUDA driver call that can contend the driver lock with the tick's own launches |
| 3–5 SipHash-of-model-name `HashMap`/`HashSet` probes | `worker.rs:17860, 17941, 18500, 19209, 19401, 12812, 18267, 12716, 12758` | 3–5/token | Flattenable: resolve a model index once per session at admit and carry it on the `Session` |
| **cuBLASLt descriptor churn per mHC GEMV** | `hyper.rs:359` → `lib.rs:21437`/`:21464`/`:21496` → cudarc `cublaslt/safe.rs:322-418` | **90 calls/token**, each ~15 host API calls + a `cublasLtMatmulAlgoGetHeuristic` | **The largest single item invisible to BOTH prior censuses** — it is one launch and one alloc, so neither the 2,125-launch nor the 2,358-alloc receipt sees it. Est. 0.9–1.8 ms/token, on a GEMV whose n is **24** (`hyper.rs:82`). The repo already priced this exact pathology at 40 calls/step as "~10% of the H100 q35 decode step" (`hybrid_forward.rs:10852-10855`); there are 90 still on it |
| per-layer enum dispatch + `Vec<HybridLayer>` walk | `hybrid_forward.rs:1826-1859` | 45 | Enum, not `Vec<Box<dyn>>` — already cheap. **Not** a finding |

**THP verdict: the arm's precondition is NOT met, so it is not run.** Two independent lines of
evidence. (1) Source: the largest per-token host object on the shared decode loop is the ~0.6 MB
logits Vec; the weight/expert mmaps are file-backed `MAP_SHARED` (`memmap2`, `crates/memra-gguf/src/lib.rs:535`,
`safetensors.rs:259`, `source.rs:673`), which THP does not cover, and they additionally carry
`MADV_RANDOM` by default (`source.rs:55-64`, `MEMRA_MOE_MMAP_ADVICE`). (2) Measured on Box B
(§B0): `AnonHugePages = 0 kB` against `Anonymous = 2.57 GB` and `Rss = 337 GB`, of which
`FilePmdMapped = 110 GB` — i.e. the huge-page win on the artifact is **already being taken by file
PMD mappings**, and there is only ~2.5 GB of anonymous memory for a THP arm to act on. THP is
worth a lane only for the `MEMRA_ST_PINNED` posture (a measured **171 GB** anonymous pinned slab,
`model.rs:2464-2465` / `docs/FLAGS.md` `MEMRA_ST_PINNED` row) — a different serving config, and it
carries an open question with no receipt either way: whether khugepaged collapse is legal on
`cuMemHostAlloc`-registered pages. Named follow-up, not this lane's arm.

### A3. Syscalls per token beyond the CUDA driver

| syscall class | count/token | detail |
|---|---|---|
| **`getenv` + `ENV_LOCK` + malloc** | **~600–1,200** | A1 row 1. This is the syscall-shaped finding, not `clock_gettime` |
| `clock_gettime` (vDSO on a TSC clocksource) | **7** | `worker.rs:10221` and `:10326` (the *same* function taking a fresh `Instant::now()` twice per tick), `health.rs:102` via `beat_busy`, `worker.rs:12699`, `:12951`, `:13107`, and `memra-lanes/src/lib.rs:71` taking its own instead of the caller's. ~175 ns/token on TSC — **but ~10 µs/token if the box's clocksource is `hpet`/`acpi_pm`/`xen`**. One `let now = Instant::now()` threaded through the tick collapses 7 → 2. **No clock storm exists in the engine**: every per-layer `Instant::now` is behind `timing.then(..)` / `MEMRA_STEP_TP_TIMING` (`decode.rs:846,849`; `hybrid_forward.rs:20894,20948`), and the unguarded ones sit in `t >= 16` prefill arms. Clean |
| `write(2)` to **unbuffered** stderr | 1 per spec burst | `worker.rs:19314` `[dspark-acc]`, `:19469` `[glm5-acc]`, `:19118` `[gspec-acc]`, `:18711` `[spec-acc]` — the only **unflagged** writes on the serving hot path. They are deliberate (they are the spec-engagement receipts the deploy gate greps, per the never-serve-greedy law), but `io::stderr()` is unbuffered, so each is a `ReentrantLock` + an immediate write that **blocks** if stderr is a pipe/journald socket with a slow reader. Their cost scales *inversely* with acceptance: worse acceptance ⇒ shorter bursts ⇒ more writes per token |
| `mmap`/`munmap` | 1–2 total, not per token | The ~1.2 MB sampler Vec crosses glibc's initial 128 KB threshold, but glibc's *dynamic* threshold ratchets up on the first free of an mmap'd chunk, so steady state serves it from the main arena. **Assumption, not a receipt** — one `strace -c -e mmap,munmap` settles it (§B1 cell 3) |
| `futex` | **0 on the shared T=1 path** | Every condvar/channel belongs to an off-by-default tier: `SpecPipeSync` (`spec.rs:1869-1873`, needs `MEMRA_SPEC_PIPE`), `spill_pread` (`:314,366,1218`, needs `MEMRA_SPILL_IO != mmap`), `cpu_experts` (`:653,681,764`, needs `MEMRA_CPU_EXPERT_LIB`). The one real futex is the tokio token wake (A1) |
| `sleep`/`yield`/`park`/file IO | 0 | `recv_timeout` (`worker.rs:10284`) is idle-only; the `try_recv` polls are empty-channel atomic reads, not spin loops |
| **`cuCtxGetCurrent`** (driver, but neither census counted it) | **~4,100–4,500** | cudarc calls `ctx.bind_to_thread()` at the top of every launch (`launch.rs:212`) *and* every alloc (`core.rs:1534→1538`); the impl (`core.rs:350`) makes a real driver call. Pure overhead in a single-context decode. Est. 260–440 µs/token. Fixable only in vendored cudarc |
| cudarc per-launch heap churn | ~6,000–8,000 heap ops | `launch.rs:63-72` builds three `Vec::new()`; `args` takes ~7.8 pushes on average ⇒ 2 mallocs + 1 realloc memcpy + 1 free per launch. Est. 120–250 µs/token |

### A4. Thread topology at serve time, and the affinity answer

**~197 threads, of which exactly ZERO had a CPU affinity mask set by this codebase.**

| threads | `comm` (15-byte truncation) | spawn site | default | contends with the worker? |
|---|---|---|---|---|
| **192** (= `available_parallelism()`) | `tokio-runtime-w` | `lib.rs:4311`, a bare `#[tokio::main]` — **the only runtime construction site in the workspace; no `worker_threads`, no `Builder`, anywhere** | always | **YES — this is the central topology finding.** 192 unpinned tokio workers + 1 unpinned GPU worker on 192 logical CPUs lets CFS migrate the worker across any of the 12 CCXs at any tick, and every JSON body and SSE frame is an L3-pollution event against the worker's working set |
| 1 | `memra-gpu-worke` | `worker.rs:20104-20105` | always | — (it *is* the worker) |
| 1 | `memra-gpu-watch` | `health.rs:508-509` | always (`MEMRA_GPU_WATCH != 0`) | small but periodic: a `fork`/`exec` of `nvidia-smi` every 60 s plus a 100 ms poll while the child lives |
| 1 | `memra-xid-tail` | `health.rs:624-625` | always | blocked on `/dev/kmsg`; negligible |
| 1 | `memra-sd-watchd` | `health.rs:743-744` | only under systemd `WatchdogSec=` | negligible |
| 1/model + 1/compile | `memra-constrain` (**both names collide at 15 bytes**) | `constrained.rs:227-228`, `:255-256` | always (idle) / per constrained request | **YES** — `llguidance` grammar compilation is real unbounded CPU work, completely unpinned. The sharpest per-request CPU spike in the process |
| 1 transient, **unnamed** | inherits `memra-gpu-worke` | `decode_batch.rs:1386`, `hybrid_forward.rs:3342`, one in `spec.rs` | per dual-PP decode step / prime chunk | Rust scoped threads are real `clone()` calls, not a pool. **They inherit the parent's mask**, which is the argument for pinning the worker thread rather than the process |
| 3+ | `cuda-EvtHandlr`, `cuda000…` | CUDA driver | always | Unmanaged by us; measured at exactly 10 voluntary switches/s each = a 100 ms poll timer, not per-token work |
| 0 | `memra-cpu-execu`, `memra-moe-prefe`, `memra-spill-*`, `dsv4-serve-*`, `memra-bg-runner` | — | need `MEMRA_CPU_EXPERT_LIB` / `MEMRA_SPILL_IO != mmap` / a dsv4 ckpt / `MEMRA_BG_JOB` | absent in a stock GPU serve |

**Affinity, definitively: none, anywhere on the serving path.** The whole tree contains one
`sched_setaffinity` (`cpu_experts.rs:571`, reached only via `MEMRA_CPU_EXPERT_LIB`) and one
`pthread_setaffinity_np` (`tools/memra_cpu_experts.cpp:2355`, in the companion `.so`). **Zero**
`sched_setscheduler` / `SCHED_FIFO` / `setpriority` / `nice()` calls; **zero** `numa`/`numactl`
references. `libc = "0.2"` is already a dependency of both `memra-server` and `memra-engine`, so
the fix needed **no Cargo change**. Two latent defects found in passing: `pin_current_thread`
**discards the `sched_setaffinity` return value** (`cpu_experts.rs:571`) — a failed pin is
indistinguishable from a successful one — and `cpu_experts.rs:588-589` documents that a global
`GOMP_CPU_AFFINITY` silently overrides per-thread affinity for OMP teams.

Corrections to the brief's premises, both measured: the GPU worker is **not** the only
`memra-gpu-worke` in `/proc` (unnamed children inherit the name — Box B showed 4), and there is
**no NIC/GPU-local CCD to prefer**: Box B is **NPS1**, one NUMA node, and `nvidia-smi topo -m`
reports `CPU Affinity 0-191 / NUMA Affinity 0` for all four cards. So the intervention is "confine
to ONE CCX for L3 reuse", never "pick the CCD near the GPU".

### A5. THE LARGEST FINDING: a default routes a live prod request shape through an O(n_vocab) host sampler

This is not a locality question and not a lock. It is a configuration/algorithm interaction, and
every link in the chain is verified in source and against the live fleet config:

1. `sampler_config` arms penalties from **any** non-neutral coefficient — `lib.rs:3746-3747`:
   `let penalties_on = frequency_penalty != 0.0 || presence_penalty != 0.0 || repetition_penalty != 1.0;`
   and then `penalty_last_n: if penalties_on { PEN_WINDOW_MAX } else { 0 }`.
2. `devsample_meta` **refuses the device sampler** for any penalized config unless
   `MEMRA_SERVE_DEVPENALTY=1` — `worker.rs:18037-18043`. Refused ⇒ meta `None` ⇒ no mask staged ⇒
   full `[n_vocab]` D2H (the `MEMRA_SERVE_LEANLOGITS` saving is forfeited) ⇒ host sample.
3. The host sample, per token: a ~1.2 MB `Vec<(u32,f32)>` over the full vocab
   (`memra-sampling/src/lib.rs:309-313`), **n_vocab hash probes** in the penalty pass
   (`:411-415`), and — because `top_k > 0` — a **full n_vocab `sort_unstable_by`** before the
   truncate to k (`:338-341`).
4. **The fleet runs `SERVE_DEVPENALTY=0` on every box**: `ops/serving/box-box12.env:48` (q38, the
   primary billing stack), `box-step37.env:27`, `box-orn.env:25` — each with a note explaining
   that 0 is the live truth and that the repo's older default of 1 was an install-box state leak.
5. **A served model's own vendor-recommended arm arms it.** `deploy/q38/models.toml`
   `non_thinking_sampling` carries `presence_penalty = 1.5` with `top_k = 20` (lines 88-91),
   quoted from Qwen's "API Usage Tip". The thinking arm does not (`presence_penalty = 0.0`,
   `repetition_penalty = 1.0` ⇒ `penalties_on` false), and step37 and ornith declare no penalties.

**So: every token of every q38 NON-THINKING request on box12 is sampled by an O(n_vocab) host
path that hashes ~152k times and sorts ~152k candidates to keep 20.** That is a multi-millisecond
per-token host cost on the primary billing model's cheap/fast request shape — the exact shape the
personal-agent buyer profile uses. It is consistent in magnitude with the ~10 ms the owner asked
about, and it is invisible to a GPU-side census by construction.

Two independent, already-available responses, in order of blast radius:
- **The lane fixed the algorithm** (§C1): the penalty pass is now O(distinct penalized ids)
  instead of O(n_vocab), bit-identical, gated. This makes the fallback cheap **whatever** the flag
  says, so it needs no fleet change to help.
- **The flag is the bigger win and is not this lane's to flip**: `MEMRA_SERVE_DEVPENALTY=1` moves
  the whole shape onto the device sampler. Note that q38's own launcher already *defaults* to 1
  (`deploy/q38/q38_serve_launch.sh:54`, calling it a "reversible transfer canary") and only the
  per-box state pins 0 — so this is aligning live state with the launcher's intent, not inventing
  a posture. It still needs its own A/B with a byte-identity gate and a sampled-default probe
  before any flip: see §PROD-APPLICABILITY.

### A6. The spec/dspark round — where the host cost DIVERGES from plain

q38 serves dspark today; glm5 serves plain (spec is NO-FLIP at every K, `VERDICT:glm53:loop-port-no-flip`).
A spec round drafts K then verifies, so per-round costs amortize over the accepted tokens
(1.907/round at K=3 on the banked 3way receipt) — and then the round machinery gives much of it back.

| cost class | plain | spec greedy | **spec sampled (= the product)** | note |
|---|---|---|---|---|
| **`verify_ws_on()` — TWO uncached `env::var` + one `Mutex` per pooled buffer take/recycle** (`lib.rs:1936-1941`; six call sites `lib.rs:10056, 10077, 10093, 10107, 10116, 10125`) | 0 (the t=1 walk uses the separate `MEMRA_HC_DECODE_WS`) | **~3,444 environ scans + ~1,722 Mutex round-trips/round** (`kda.rs:409-591` ×34, `hybrid_forward.rs:13022-13170` ×42, `lib.rs:20907` ×~148, `lib.rs:6597` ×42) | same | **THE #1 SPEC-PATH FINDING.** Est. 0.7–2.4 ms/round = 0.36–1.3 ms/accepted token. Door W flipped **default ON 2026-08-31**, so this rides every spec boot on the fleet as of last week |
| MoE router drain + `router_stage` lock + 2 host Vecs | 42/**token** | 42/round ≈ 22/token | ≈22/token | spec amortizes the biggest shared cost — this is the real reason a round is cheaper than K plain steps |
| accept/draft device drains | 0 | +9–13/round | **+22–31/round** | `glm_spec.rs:2403` reads `th`,`z`,`mx` as **three separate single-scalar `dtoh`, i.e. three full stream drains for three f32**; repeated at `:2250` and `:2369`. `Engine::dtoh_pair` (`lib.rs:9954-9963`) exists and is unused here. **The greedy path got the loop-port batching (`glm_spec.rs:1909-1922`); the sampled twin did not** — the instrument was optimized and the product was not, which is the never-serve-greedy law biting in reverse |
| KDA ssm snapshot | 0 | **34 × `clone_dtod` = `alloc_zeros` + `memcpy_dtod`** (`lib.rs:5579-5586`, calls `glm_spec.rs:614, 684`) | same | **136 MiB of memset per round that is immediately overwritten.** One-line fix to `alloc_uninit`; `lib.rs:10683-10684` already prices memsets at ~6.5% of decode GPU time |
| `hyper::pre_exact` loops over t **on the host** | n/a | 2 sites × 45 layers × t=4 = **360 launches/round** (`hyper.rs:381-389`) | same | the explicit host-loop-over-K; ~0.8 ms/round of launch issue at the box's 2.216 µs/launch |
| host embed gather | 1/token (`model.rs:1334-1344`) | **2/round**, 12 rows, ~14 heap allocs | same | **`embed_gather_device` exists (`lib.rs:10456+`) and the qwen path uses it (`decode.rs:813, 1134, 1903`); no glm5 path does.** A per-family gap, not a spec one |
| FR-Spec d2t map re-upload | 0 | 0 | **1 × 128 KB+ pageable HtoD per REJECTED round** (`glm_spec.rs:2306`) | should be a load-time resident buffer |
| MLA `len_d` mirrors | — | **22 pageable 4-byte `memcpy_htod`/round** (`glm_spec.rs:1102`) + 1 that bypasses the door (`:1188`) | same | `lib.rs:10488-10489` calls this primitive "poison mid-round" in as many words. Door H (`MEMRA_GLM5_HTOD_DIET`) fixes 22 of the 23 and is default OFF |
| per-round rebuilds | — | `Glm5VerifyCkpt`'s 5 Vecs (`glm_spec.rs:484-497`), a **327 KB zeroed `HcTapSink.rows`** (`memra-kv/src/lib.rs:2205-2225` via `glm_spec.rs:1876`), `taps.clone()`, nested `Vec<Option<Vec<..>>>` stashes (`glm_spec.rs:378, 384`) | same | all session-shaped, all trivially persistent. ~30–60 µs/round |
| `glm5_tap_drain` | — | **5 separate full drains where 1 would do** (`glm_spec.rs:838`, loop `823-843`) | same | the port's own stated goal was to "kill the five in-walk DtoHs"; it moved them post-walk and kept them as five |
| eager `format!` in the `#87` sentinel guard | — | K+2 `String` allocs/round (`glm_spec.rs:1806-1809, 1971-1974, 2340, 2377`; guard takes `&str`, `spec.rs:860-874`) | same | cosmetic, but on the innermost loop; a closure fixes it |
| worker-side per-round bookkeeping | n/a | **0** — glm5/dspark/gemma bursts pass no `on_commit` (`glm_spec.rs:1608-1615`, `dflash.rs:4391-4399`, `gemma_spec.rs:1923-1931`) | 0 | only the qwen MTP route flushes per round (`worker.rs:18583-18619`) |
| `contains_stop_string` | per **token** | per **burst** (`worker.rs:19523`) | per burst | spec is *cheaper* here |
| the O(n_vocab) host sampler (§A5) | **yes** when penalized/constrained | **never** — excluded at admission (`glm_spec.rs:1443-1452`) | never | so on q38 the penalized non-thinking shape is *also* the shape that cannot use dspark |

Two spec-side clock notes, both correcting a suspicion rather than confirming it: `Instant::now()` is a
vDSO read at ~20–25 ns and there are only 1 (`glm_spec.rs`) / 7 (`dflash.rs`) / 25 (`spec.rs`) of them —
**there is no clock storm.** What *is* expensive is the `spec_phase.rs:97-103` clock, which
**synchronizes both streams** before reading — correctly gated behind `MEMRA_SPEC_TRACE` (`phase` is an
`Option`, `glm_spec.rs:1667-1668`), 0 by default. And `SpecTelemetryWindow` does not carry per-round
cost: the per-burst telemetry is `worker.rs:18557, 18705-18709`.

One contradiction between two independent passes, resolved by reading the code: cudarc's
`event_tracking` was flagged as a possible ~700–1,000 extra `cuEventCreate`/`Destroy` per round. It is
**disabled** at `lib.rs:2165`, so that cost does not exist. Recorded because "a plausible large finding
that turns out to be already fixed" is exactly what a second pass is for.

### A7. Per-family columns, and the multi-card thread question answered

Family routing is **not** an arch enum in the decode loop; it is derived from the compiled
`ModelPlan` on every call. The fork is `decode.rs:740-771`: `hyper.is_some()` → glm5;
`is_gemma4_e4b()`; `uses_gemma_program()`; `pp::pp_cuts()` → `decode_step_h_ppn` (step37 PP, glm5
PP); `uses_sliding_gated_moe_program()` → the step35 graph door; else the generic walk (q38).

**Q4, the highest-value question, answered definitively: EVERY family's multi-card decode walk is
driven by ONE host thread issuing to N devices sequentially. There is no per-token thread, no
barrier, no condvar and no join in any decode path.** Cross-device ordering is entirely CUDA events
plus context push/pop on that single thread — step37 TP-2 at `tp.rs:11990-11999` and
`hybrid_forward.rs:22405-22411`, glm5 PP-3 and step37 PP-2 as a plain sequential stage loop
(`hybrid_forward.rs:2672-2696`, `decode.rs:1405-1431`, `pp.rs:2480-2485/2588-2647/2680-2705`), and
`pp.rs:2770` states the invariant in as many words: *"All primary consumers of the previous round's
stage-allocated buffers are enqueued by then (single host thread)."* A sweep for
`thread::spawn|thread::scope|Barrier|Condvar` across `tp.rs` (13,992 lines), `pp.rs`, `hyper.rs`,
`kda.rs`, `mla.rs`, `glm5_tp.rs`, `moe_cache.rs`, `ep_map.rs` returns **nothing**.

Exactly two places do go multi-threaded, both outside single-sequence decode: **pipelined prefill**
(`hybrid_forward.rs:3342-3381`, one scoped spawn+join per prompt chunk — and its comment is the
canonical statement of why decode is host-bound: *"Step's MoE router readback synchronizes once per
layer. Two CUDA streams on one host thread therefore serialize even if the calls are ordered as a
pipeline"*), and the **batched dual-wave PP arm** (`decode_batch.rs:1382-1392`, one spawn+join per
batch tick).

**So there is nothing to spread across cores. Pinning that one worker thread — and living with the
driver's own threads inheriting the mask — is the whole placement lever**, which is exactly what
C2 implements and what arms (i)/(i-b) price.

Y = pays it; a number is the count per token.

| host-cost class | q38 (dense GDN + DFlash2) | step37 TP-2 | step37 PP-2 | glm5 PP-3 | gemma (batched, prod) | embed/rerank |
|---|---|---|---|---|---|---|
| `fn_cache` mutex + name hash per launch | Y | Y | Y | Y (~2,125) | Y | prime only |
| `Gpu::stream()` Arc clone per launch | Y | Y | Y | Y | Y | prime only |
| driver alloc per scratch buffer | Y | Y | Y | Y (2,358) | Y | prime only |
| plan `trunk_operations()` **fresh Vec + linear scan**, per token | 2-3 | **~49** | 2-3 | 1-2 | 1 | - |
| the same walk **per layer** ⇒ O(n_layers²)/token | verify arm | Y (`decode.rs:3576`) | - | - | **Y (`decode_batch.rs:2691-2695`, inside a live `assert!`)** | - |
| uncached `env::var` in the per-layer body | 3-4/layer | **~15-17/layer** | 3-4/layer | ~9/layer | 3-4/layer | - |
| per-layer MoE **host router readback + full sync** | **— (dense, no router)** | 42 if `DEV_ROUTER=0` (the compiled default) | 42 | **42, and no T=1 escape** | — (device top-k) | - |
| **total host syncs per token** | **1** (+3-5 per dspark round) | **1** | ~43 | **43** | 1 per batch step | 1 per prime chunk |
| cross-layer add+norm+q8 fusion available | Y | Y | Y | **structurally NO** (the hc residual replaces the serial one, so the fused arms refuse `Mixer::Mla`/`Mixer::Kda` — `decode.rs:904-907`) | Y | n/a |
| multi-card CUDA context push/pop per token | - | **~1,150** | 6 | 6 | dual arm | - |
| PP boundary tx/rx slot mutex + events; fresh `cuEventCreate`/token | - | - | 4 + 1 | 4 + 1 | dual arm | - |
| N per-stage `htod_i32(pos)` instead of 1 (the per-stage-pos law) | - | - | 2 | 3 | - | - |
| host pointer/descriptor table rebuilt per token/round | verify only | rows arms | - | - | **Y (`batch_layer_ctx`, `decode_batch.rs:2221-2325`)** | - |
| `verify_ws` pool: 2 uncached env + 1 mutex per buffer take AND recycle | - | - | - | **Y (~2,700 + ~2,700 on the verify shape)** | - | - |
| model-global mutex held across a WHOLE burst | **Y** (`dspark_vgraphs`, `dflash.rs:4468`, taken before the door check) | - | - | same pool via `dflash.rs:3093` | - | - |
| draft/verify selection computed **on the host** | **Y** — 3 blocking D2H then a host walk (`dflash.rs:1927-1939`) + an `nd × n_embd` D2H and host dot per confidence slot (`dflash.rs:4615-4631`, `:337-352`) | MTP verify | " | `MEMRA_GLM5_DFLASH` arm | - | - |
| `format!("g:{name}")` String heap alloc per fa launch | - | - | - | - | **Y** (`lib.rs:2419-2438`; fires by default, `gkv_on()`/`wkv_on()` both ON) | - |
| per-sequence (B) host loop for append/attention | - | - | - | B `pos_d` H2D | **Y — B × n_layers** fa calls, each 1 pool lock + 1 uncached env read + 3 memsets (`decode_batch.rs:3651-3700`) | - |
| OS thread spawn+join per tick | - | - | dual arm | - | dual arm | - |
| the O(n_vocab) host sampler (§A5) | **Y — the non-thinking arm, today** | - | - | - | - | - |
| **enters the per-token decode loop at all** | Y | Y | Y | Y | Y | **NO** — prefill-only, `max_new: 0` (`embed_api.rs:136`), 1 sync per prime chunk, N sequential admitted requests, hard-pinned to `Lane::Harvest` |

**SHARED — one fix helps every family**: the `fn_cache` mutex+hash per launch, `Gpu::stream()` Arc
churn, unpooled scratch allocation, the always-synchronizing `dtoh*` helpers (there is **no**
non-blocking `dtoh` in the engine), the un-memoized `trunk_operations()` predicates, and the
uncached `env::var` reads in per-layer bodies. All scale with launches per token.

**FAMILY-SPECIFIC — needs its own column and its own arm**: glm5's and step37's 42 router syncs
(with a door that closes it on step37 but **not** on glm5 at T=1 — `MEMRA_MOE_VROWS_DEV_TABLES`
requires `t >= 2`, `hybrid_forward.rs:9043-9071`), step37 TP's ~1,150 context push/pops and ~700
door env reads, glm5's structural exclusion from norm fusion, q38's host-side draft selection plus
the burst-long model-global mutex, gemma's `format!`-per-launch and B × n_layers attention loop,
and the PP event/slot tax paid only by multi-stage shapes.

Two premise corrections worth banking: **q38 is GDN + DENSE MLP, not MoE** (`docs/MODELS.md:82`,
`docs/FLAGS.md:623`), so it has no router and no per-layer MoE host sync — its host cost is almost
entirely the dspark round. And **dense gemma4 does not use `gemma4_decode_step_h`**: the worker
carves it out of the eager-only class (`worker.rs:9655-9662`, `:17808`) and routes it to
`gemma4_decode_batch` with `MEMRA_GEMMA4_BATCH` default ON.

---

## B. THE MEASURED AUDIT — Box B

Box B = the 4-card glm53 lane box, **AMD EPYC 9654 96-Core (192 logical), 4× RTX PRO 6000 WS
600 W**. Identity is recorded in the private ops repo, never here.

### B0. Host capability probe and the topology receipt — DONE (2026-08-31)

**Read from sysfs, not assumed:**
- **12 L3 domains** (`cache/index3/shared_cpu_list`), each 8 cores + their 8 SMT siblings:
  domain *k* = cpus `{8k…8k+7} ∪ {96+8k…96+8k+7}`. Confirms 12 CCDs.
- **ONE NUMA node** (NPS1): `node0/cpulist = 0-191`. `nvidia-smi topo -m`: all four GPUs
  `CPU Affinity 0-191`, `NUMA Affinity 0`. **There is no GPU-local CCD to choose.**
- THP mode: `always [madvise] never` — `madvise`, so only madvise'd regions get anon THP, and
  nothing in `memra-server`/`memra-engine` madvises.

**WHAT THIS BOX CANNOT MEASURE — measured, not taken from the image's word:**

| tool | state | consequence |
|---|---|---|
| `perf stat` / `perf trace` / any PMU counter | **UNAVAILABLE.** `perf` is on PATH but the kernel-matched `linux-tools-6.14.0-37-generic` is absent, and `kernel.perf_event_paranoid = 4` (≥3 disables all access) in an unprivileged container | **cache-misses, L3/LLC-misses and the exact `cpu-migrations` counter cannot be measured on Box B.** The brief's PMU rows are not deliverable here. Migrations are recovered by SAMPLING `/proc/<tid>/stat` field 39 (a strict LOWER BOUND, labelled as such in every row); cache-miss rates have no substitute and are reported as NOT MEASURED |
| `chrt` (SCHED_FIFO) | **DENIED** — `Operation not permitted` (no `CAP_SYS_NICE`) | Arm (ii) as briefed cannot run |
| `nice -n -10` | **DENIED** — `Permission denied` | ditto; positive nice works |
| `strace` | **WORKS** (ptrace permitted) | the futex/mmap census is real, labelled INSTRUMENTED |
| `taskset` / `sched_setaffinity` | **WORKS** | arm (i) is real |

Arm (ii) is therefore **respecified** rather than dropped, to the two levers that ARE available
and that the code audit says matter more anyway: **`TOKIO_WORKER_THREADS=N`** (the 192-thread
runtime is the actual preemption source, and tokio honours the env var with no code change) and a
**positive-nice control** on co-tenant noise. A `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` arm rides
along as the zero-code anon-THP probe.

**Opportunistic procfs receipt** (45 s, 0.05 s sampling, `host-sampler.py`) — labelled clearly:
this is **another lane's 1M-battery boot during its prime phase, NOT my decode cell**. It was
taken read-only (no ptrace, no signals, no requests) because it costs the box nothing and answers
the topology question for free:

| thread | busy% | stime/45 s | vol/s | **nonvol/s** | distinct cpus | **distinct CCX** | sampled CCX crossings | SMT hops |
|---|---|---|---|---|---|---|---|---|
| `memra-gpu-worke` (primary) | **51.3** | **23.11 s** | 0.33 | **5.58** | 11 | **3** | 3 | 12 |
| `memra-gpu-worke` ×3 (per-device helpers) | 0.0 | 0.0 | 1.1–2.0 | 0.0 | 1–2 | 1 | 0 | 0 |
| `cuda-EvtHandlr` ×3 | 0.0 | 0.01–0.02 | **9.97** | 0.0–0.04 | 1–2 | 1 | 0 | 0 |
| `memra-server` (main) | 0.0 | 0.0 | 0.09 | 0.02 | 2 | 1 | 0 | 1 |

Four things this settles:
1. **The worker is unpinned and it DOES migrate**: 11 CPUs across **3 L3 domains** in 45 s. Arm (i)
   has something real to act on.
2. **Its CPU is essentially ALL system time** (23.11 s of 23.1 s busy). The worker's host cost is
   syscall/driver-call dominated — which is exactly what §A3 predicts and is the strongest
   independent corroboration that the wall is call-count bound, not cache bound.
3. **5.58 involuntary preemptions/second** with near-zero voluntary switches: the worker is
   CPU-bound and being preempted by the 184-thread runtime, not blocking. (A CUDA sync that
   spin-waits shows exactly this signature, so the two cannot be separated without PMU access.)
4. `AnonHugePages = 0` / `Anonymous = 2.57 GB` / `Rss = 337 GB` / `FilePmdMapped = 110 GB` —
   the THP arm's precondition fails (§A2).

`/proc/<pid>/status` **must not** be used for context switches: it reports the MAIN thread only,
which read `voluntary=114 / nonvoluntary=1` for a process whose worker was taking ~250 involuntary
switches per minute. The sampler reads `/proc/<pid>/task/<tid>/status` per thread for this reason,
and the trap is recorded in its header.

### B1. Intervention arms — RUN, and **every arm is NULL**. The brief's hypothesis is refuted.

Window taken 2026-08-31T21:12Z after bankfix's DONE line, first claim exercised, released clean.
Binary: lane `b0ff87b25`, real 4m45s compile (not a stale no-op), 3 `[worker-affinity]` strings
verified in the binary. 3-card PP ship recipe, real 190.7 GB NVFP4 artifact, cards 0-2, port 18700.
**11 boots.** Greedy = the instrument, vendor-default sampled = the product, both on every arm,
`reasoning_effort` pinned, prime-once then 3 steady reps. Receipts:
`box-receipts/box-window-20260901.tgz`.

| arm | greedy tok/s | Δ vs base | sampled tok/s | Δ vs base | verdict |
|---|---|---|---|---|---|
| base (n=4 boots) | **34.3171** | — | **39.8937** | — | — |
| `MEMRA_WORKER_AFFINITY=ccx` (n=3 boots) | 34.3242 | **+0.021%** | 39.8902 | **−0.009%** | **NULL** |
| `TOKIO_WORKER_THREADS=16` (n=1) | 34.3126 | −0.013% | 39.8920 | −0.004% | **NULL** |
| composed: `ccx` + `TOKIO_WORKER_THREADS=16` (n=1) | 34.3153 | −0.005% | 39.8852 | −0.021% | **NULL** |

Within-arm rel spread across the 3 reps was **0.010–0.082%** and arm-to-arm boot spread
**0.026–0.054%**, so this is a *high-confidence* null, not an underpowered one: the ccx effect is
**0.50× the pooled spread on greedy and 0.23× on sampled** — smaller than the noise floor. Nothing
crossed the 0.5% "finding" bar, let alone the 2% "lever" bar. No ×5 escalation triggered (every
spread was far under 0.5%).

**BYTE IDENTITY, and this closes the gate the rig could not**: across all 11 boots there is exactly
**ONE greedy sha (`a04810ad8d0fd43d`) and ONE sampled sha (`f8f13e305768bba1`)**. So
`MEMRA_WORKER_AFFINITY` and `TOKIO_WORKER_THREADS` are proven non-numeric on **glm5** too — on the
real artifact at the ship recipe — not just on the two rig families.

**THE PIN DEMONSTRABLY TOOK EFFECT, which is what makes the null trustworthy.** The ON boots' own
kernel readback: `[worker-affinity] engaged request=40-47,136-143 effective=40-47,136-143 cpus=16
l3_domains=1` — the `ccx` form self-resolved on the real 12-CCD host to exactly one CCX (8 cores +
8 SMT siblings), narrowing 192 CPUs to 16 in one L3 domain. This is not "we set a flag and saw no
change"; it is "the kernel confirms a 12× narrower mask and the tokens are bit-identical and the
throughput does not move."

**AND HERE IS WHY IT IS NULL — a correction to my own §B0 evidence.** A per-thread procfs sample
taken *during the timed decode reps* (12 receipts, `sched-v*.json`) shows the worker does **not
migrate during decode at all, in either arm**:

| arm | n | busy% | stime | distinct cpus | **distinct CCX** | **sampled CCX crossings** | vol/s | nonvol/s |
|---|---|---|---|---|---|---|---|---|
| base | 6 | **98.8** | 5.55 s | **1.0** | **1.0** | **0.0** | 0.09 | 4.89 |
| ccx | 6 | **98.9** | 5.55 s | **1.0** | **1.0** | **0.0** | 0.00 | 5.74 |

The OFF arm was **already** on one CPU in one CCX. CFS's wake-affinity parks a 98.8%-busy thread
and leaves it there, so **there was nothing for the pin to fix.** The migration in §B0 (11 CPUs,
3 CCXs) was sampled during another lane's **1M PRIME** phase at 51% busy — a different workload
phase with a different scheduling signature — **not during decode.** My own §B0 paragraph
generalized a prime-phase observation to the decode loop, and the window disproved it. Recorded
rather than quietly corrected, because "the worker migrates" was the premise the whole pinning
brief rested on.

Two things this *does* confirm, both third-time-independently: the decode worker is
**CPU-SATURATED** (98.8%, one core), and **essentially all of it is SYSTEM time** (5.55 s of
~5.6 s busy). That is the call-count-bound headline measured a third way — the worker is not
waiting on the GPU and not thrashing cache; it is executing syscalls and driver calls back to back.

THP, third independent confirmation that its precondition is absent: on this 3-card ship recipe
`AnonHugePages = 0` with `Anonymous = 2.13 GB` against `Rss = 169.7 GB`, and `FilePmdMapped = 0`
(the 4-card 1M posture had 110 GB of file PMD pages; this one has none). Either way there is ~2 GB
of anonymous memory to act on and no anon THP in play. The `GLIBC_TUNABLES` arm was dropped as
unable to matter at that size rather than run for form.

**What the window did NOT get**: the bounded `strace -c` futex/mmap census. Its cell aborted on a
`set -e` interaction in the tid-selection line after the rows had already landed, and by then the
arms were the better use of the remaining window. It is the one cheap item left and it is the only
way to settle the glibc `M_MMAP_THRESHOLD` assumption in §A3; ~10 minutes on any later window.

### B1-prior. The plan as briefed, and how it changed once the box answered

**Box B is held**: the 1M-depth re-price window is in flight (marker up, cards 0-3) with a bankfix
window queued behind it. This lane is **registered in `/root/BOX-QUEUE.md`** (2026-08-31T19:34Z)
as a short window behind both, with the capability findings above written into the queue entry so
no later lane re-discovers that perf is unavailable. No timed cell has been run and **no arm delta
is claimed.**

Harness, banked and ready in this directory:
- `host-sampler.py` — procfs-only per-thread scheduling receipts (self-tested on the rig, then run
  against a live `memra-server` on Box B).
- `decode-rows.py` — prime-once then N steady reps, greedy (instrument) **and** vendor-default
  sampled (the product), `reasoning_effort` pinned, sha16 identity per rep, sampler running
  concurrently with each timed rep so the scheduling receipt and the tok/s row describe the same
  tokens.
- `run-window.sh` — boot/stop with a **PID + `/proc/<pid>/exe`-verified** stop (never `pkill`,
  never a basename kill), a boot nonce and a listener-pid check for arm identity (a 200 on
  `/readyz` proves a listener, not which server), and an engagement assert that fails a boot
  claiming the ON arm without the `[worker-affinity] engaged … effective=…` readback.

Arms, one boot each, ×3 interleaved (×5 on >0.5% within-arm spread or an arm within 2× pooled
spread of baseline), greedy + the vendor-default sampled twin:

| arm | change | rationale | expectation from the code audit |
|---|---|---|---|
| baseline | ship 3-card recipe, no change | — | — |
| (i) CCX pin | `MEMRA_WORKER_AFFINITY=ccx` | removes the measured 3-CCX migration and confines the inherited driver-helper threads to one L3 | **small** — the wall is call-count bound; >0.5% is a finding, >2% a lever |
| (i-b) narrow pin | `MEMRA_WORKER_AFFINITY=<one core + sibling>` | separates "stop migrating" from "keep an L3 to yourself"; also the negative control for over-constraining a worker whose driver helpers inherit the mask | may REGRESS — that is a result, not a failure |
| (ii) runtime cap | `TOKIO_WORKER_THREADS=16` (and 32) | the 184-thread runtime is the preemption source; not a `MEMRA_*` name, so no FLAGS row is owed | plausibly the larger of the two placement effects |
| (ii-b) composed | (i) + (ii) | pin the worker AND stop 184 threads from being schedulable everywhere | |
| (iii) anon THP | `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` | zero-code anon-THP probe | **near-zero on this config** (§A2) — run only as the cheap disproof, and DROP the madvise-code arm: its precondition is not met |
| census | bounded `strace -c -f` on the primary worker tid | prices the futex/mmap/getenv classes directly | numbers labelled INSTRUMENTED (observer effect); settles the glibc-`M_MMAP_THRESHOLD` assumption in §A3 |

---

## C. WHAT LANDED

### C1. `apply_penalties_dense` — the O(n_vocab) hash storm in the host sampler, removed

`crates/memra-sampling/src/lib.rs`. The penalty pass did **one `HashMap<u32,u32>` SipHash probe
per candidate** — one per vocabulary entry, ~152k per token on the Qwen class — when
`penalty_counts` holds at most `PEN_WINDOW_MAX` distinct ids. The loop was inverted the expensive
way round. It now walks `penalty_counts` and indexes straight into the dense, index-aligned
candidate row (which is what the single call site always passes, built from
`logits.iter().enumerate()` immediately above).

- **No flag.** House doctrine: a flag exists only as a runtime parameter, machine config, a
  rollback seam, a diagnostic, or a blocked door. A bit-identical algorithmic inversion is none of
  those; it is a winner, and winners are defaults.
- **The old scan form is KEPT as the oracle** (`apply_penalties_scan_reference`, `#[cfg(test)]`)
  rather than deleted, so the replacement has something to be proven against.
- **Gate: `dense_penalties_match_the_scan_reference_bitwise`** — 24 cases (6 coefficient triples ×
  4 window sizes) compared by `f32::to_bits`, not `==`, because `==` would call two NaNs equal and
  `-0.0` equal to `0.0`, and the presence subtraction can produce exactly `-0.0`. Inputs include
  negative logits (the repeat rule branches on sign), repeated tokens (frequency count > 1), and
  ids at both ends of the row that were never generated and must be untouched. Plus
  `the_sampled_path_penalizes_the_token_it_generated`, which exercises the index-alignment
  precondition through the real `sample()` path so a future caller that filters before penalizing
  cannot quietly pass. **18/18 green.**
- Bit-identity is by construction, not by luck: the same set of entries, the same per-entry
  arithmetic, each entry touched exactly once in both forms, so no float re-association is
  possible.

This is beyond the brief's named mechanical fixes and is called out as such. It is in because it
is ~10 lines, bit-identical with a randomized-equivalence gate, and sits on the live prod path of
the primary billing model (§A5).

### C2. `MEMRA_WORKER_AFFINITY` — the pinning seam, default OFF by design

`crates/memra-server/src/affinity.rs` (new), applied at the top of `worker::run`
(`crates/memra-server/src/worker.rs`), declared in `crates/memra-server/src/lib.rs`, row in
`docs/FLAGS.md` §2 (Machine-specific config) — **in the same commit as the read**, which
`tools/check-flags.sh` requires (755 runtime reads, 0 uncovered).

Design decisions, each with its reason stated (the new-flags law):
- **Default OFF.** A mask is machine-specific: the right one on a 12-CCD EPYC is wrong on box12's
  Core Ultra 9 285K and wrong again inside a narrow cpuset — the exact `card-keyed-defaults-need-
  full-pins` shape. And pinning strictly *reduces* the worker's available CPU. Default-ON needs
  per-host-class receipts.
- **Applied in `worker::run`, before `Engine::new`**, not in `spawn`'s prologue: it runs *on* the
  worker thread (so `sched_setaffinity(0, …)` needs no TID plumbing), and a mask installed before
  the CUDA context exists is inherited by the driver's helper threads and by the scoped threads
  the dual-active PP arm spawns per step — which is the point on a big-core box, and a hazard on a
  narrow mask, which is why arm (i-b) exists.
- **A malformed or unsatisfiable value FAILS THE BOOT** through the worker's load verdict, rather
  than warning and serving unpinned. An operator who types `8-15,1O4-111` (letter O) would
  otherwise get an unpinned server whose numbers are filed under the pinned arm — the
  exception-that-absorbs-the-regression shape. `ccx:N` past the end of this host's map, and a cpu
  absent from sysfs, are refusals too, never a silent clamp to something the operator did not name.
- **The receipt is a READBACK, never the request.** `sched_setaffinity` can succeed and still leave
  a *narrower* mask (an outer cpuset clamps it), and it can fail with EPERM leaving the thread
  exactly where it was; both are invisible to a caller that trusts its own request. The announce
  prints `effective=` from `sched_getaffinity` after the call and appends
  `CLAMPED-BY-OUTER-CPUSET` on a mismatch.
- **Both arms announce** (`engaged …` / `off`), following the `[gpu-watch] disabled` precedent: an
  arm whose identity is the *absence* of a line is indistinguishable from an arm whose announce
  regressed.
- **Topology from sysfs, never arithmetic.** `ccx`/`ccx:N` resolve against
  `cache/index3/shared_cpu_list`; a container that hides the cache tree gets a refusal telling it
  to use an explicit list, not a silent fallback to "the whole host".
- Unit tests: OFF makes **no syscall at all** (the OFF arm must be the shipped syscall trace, not
  "the same mask, set explicitly"), `ccx` forms resolve and refuse out of range, malformed values
  never degrade to OFF, absent cpus are refused, `apply` reports the kernel readback, and `SelfCcx`
  is never a silent no-op.

### C3. NOT done, and why

- **`Engine::func`'s Mutex + String-keyed HashMap** (~1,760–2,100 probes/token) is proven
  uncontended-by-design — one GPU worker thread, invariant at `lib.rs:1006` — which is exactly the
  brief's condition for an in-lane lock removal. It is still a **named follow-up**, because
  removing it is not mechanical: a `[CudaFunction; N]` indexed by a `KernelId` enum touches 478
  call sites, and the lock is not the cost anyway (the SipHash of the name is). A wide change is
  blessed when it brings good with it — but it belongs in its own lane with its own launch-count
  identity gate, not smuggled into a host audit.
- **The `top_k` full-vocab sort** is numeric-class: `select_nth_unstable_by` changes which of
  equal-logit candidates survive at the k boundary. Own lane, own gate.
- **THP madvise arm**: precondition not met (§A2). Dropped with receipts rather than run for form.

### C4. NAMED FOLLOW-UPS, ranked, each with its measured justification and its blocker

Every row is engine-wide unless the family column says otherwise. The estimates are arithmetic
against the banked box constants (2.216 µs/eager launch, ~1.06 µs/driver call,
`decode-diet-20260831/LANE.md:7`), which were measured on a 2-card glm5 shape — they do not
transfer to q38/step37/gemma without their own census.

| # | follow-up | est. | why it is not in this lane | first cell |
|---|---|---|---|---|
| 1 | **`verify_ws_on()` — 2 uncached env reads + 1 mutex per pooled buffer take AND recycle**, ~1,722 sites/round on the glm5 verify walk (`lib.rs:1936-1941`) | **0.7–2.4 ms/round** = 0.36–1.3 ms/accepted token, and door W went **default ON 2026-08-31**, so it is live fleet-wide as of last week | Not mechanical: the door's documented contract is *"Read per call — the rollback seam"*, and `glm5_spec_session_gpu.rs:1057/1075/1085` **does flip `MEMRA_GLM5_VERIFY_WS` mid-process** to drive both arms in one binary. A `OnceLock` would break that gate | **Design is already checked out**: the gate flips between BURSTS, never inside one walk, so an `AtomicI8` tri-state latch on `Engine` refreshed once per verify-walk entry preserves the gate's semantics at walk granularity and takes 3,444 env reads/round to 2. Byte-neutral by construction (a cached read returns the same value) ⇒ an engagement/identity gate, not a numeric one |
| 2 | **the same treatment for the other ~1,500 per-call door reads/round**: tcols trio (`lib.rs:1568/1590/1615`), MoE vrows doors (`:1635/1657/1775/1787`), the `observe_routes` block (`hybrid_forward.rs:9001-9004`), `kda_proj_fused6` (`kda.rs:1224/1309/1311`), `matmul` (`lib.rs:13529/13576/17401`), `uses_q8_1_fast`'s `MEMRA_FAST` (`lib.rs:13786`, 121 call sites), `hc_fused_pre_on` (`hyper.rs:318`), `pp_cuts` (`pp.rs:87-119`) | 0.3–1.1 ms/round | Same reason, ×20 sites | The codebase **already has the right pattern** — `step_tp_w8_on` (`lib.rs:671`), `sigmoid_router_enabled` (`hybrid_forward.rs:594`), `bf16_mmv_on` (`lib.rs:19011`) all use `OnceLock`. The 2026-08-30/31 lanes traded serving-path host cost for gate ergonomics; the latch buys both. **One counter cell first**: wrap the uncached readers in an `AtomicU64` bump and get one boot's real per-round count — that converts every band above into a number |
| 3 | **memoize `ModelPlan::trunk_operations()`** — it heap-allocates a fresh `Vec<OperationKind>` and pushes 6-9 entries per plan layer, then linear-scans it (`model_plan.rs:755/793/1122`). Called **~49×/token** on step37 TP (`decode.rs:3576`) and **per layer inside a live `assert!`** in the batched FFN loop (`decode_batch.rs:2691-2695`), making that arm O(n_layers²)/token | not priced; the allocation count is large and the plan is **immutable after load** | Touches a shared load-time structure and one live release assert; wants its own identity gate | A `OnceLock<Vec<OperationKind>>` (or a precomputed predicate bitset) on the plan. Also worth asking whether the `decode_batch.rs:2691` assert should be a `debug_assert!` — but weakening a live invariant check is an owner call, not a cleanup |
| 4 | **cuBLASLt descriptor churn on the 90 mHC GEMVs/token** — ~15 host API calls + a `cublasLtMatmulAlgoGetHeuristic` per call, for a GEMV whose n is **24** (`hyper.rs:82`, `:359` → `lib.rs:21496` → cudarc `cublaslt/safe.rs:322-418`) | **0.9–1.8 ms/token**, and **invisible to both the launch and the alloc census** (it is 1 launch, 1 alloc) | A custom `hc_mixes_gemv` kernel, or a cached `(desc, layouts, pref, algo)` tuple. Either is a real kernel/numerics change | The repo already priced this pathology at 40 calls/step as *"~10% of the H100 q35 decode step"* (`hybrid_forward.rs:10852-10855`) and fixed it there with `sigmoid_dot_rows`. The mHC shape is CONSTANT for the whole process, so the cache is the cheap half. **This is the cell I would run first** |
| 5 | **glm5's 42 router syncs at T=1**: extend door D's device-table build to `t == 1` by dropping the `t >= 2` conjunct (`hybrid_forward.rs:9057`) and re-deriving the fail-closed set for the T=1 consumers | 0.5–2 ms/token, and it is the **prerequisite for whole-token CUDA-graph capture** | Numeric-class (tie-break provenance) ⇒ run-gen argmax gate + boot battery. step37 already has `MEMRA_STEP_TP_DEV_ROUTER` for its equivalent; glm5 has no T=1 escape | already named by the decode-diet lane as "the DEVICE-ROUTER CONSUMER" |
| 6 | **`Engine::func`'s Mutex + String-keyed HashMap** per launch (`lib.rs:2440`) | 180–300 µs/token | Proven uncontended-by-design (`lib.rs:1006`) — the brief's condition IS met — but removal is a 478-call-site type change, and **the lock is not the cost; the SipHash of the name is** | `[CudaFunction; N]` indexed by a `KernelId` enum, resolved at `Engine::new`. Gate = launch-count identity + a byte-identity smoke. Also kill the two `format!`-per-launch sites (`lib.rs:2428`, `:18241`, and gemma's `func_g` at `:2419`) |
| 7 | **the sampled spec arm's un-batched scalar readbacks**: `glm_spec.rs:2403` reads `th`,`z`,`mx` as **three separate full stream drains for three f32**, repeated at `:2250` and `:2369` | 0.15–0.5 ms/round of driver time, far more in lost overlap | — | `Engine::dtoh_pair` (`lib.rs:9954-9963`) already exists and is unused here. **The greedy path got the loop-port batching (`glm_spec.rs:1909-1922`); the sampled twin did not** — the never-serve-greedy law in reverse. Same shape: `glm5_tap_drain` does 5 drains where 1 would do (`glm_spec.rs:838`) |
| 8 | **`clone_dtod` uses `alloc_zeros`** — 34 calls/round × 4 MiB = **136 MiB of memset per round, immediately overwritten** (`lib.rs:5579-5586`, calls `glm_spec.rs:614, 684`) | ~0.09 ms GPU/round + 34 host allocs | one line to `alloc_uninit` + `memcpy_dtod`, but it changes an initialization contract | `lib.rs:10683-10684` already prices memsets at ~6.5% of decode GPU time |
| 9 | **`contains_stop_string` rescans the whole completion every token** — O(n²)/request (`worker.rs:17190-17201`) | 5–15 µs/token at 2k generated, only when the client sent `stop` (agent harnesses routinely do) | bit-identical but touches stop-sequence semantics | scan only the `max_stop_len + new_bytes` tail; needs its own boundary tests |
| 10 | **`top_k`'s full n_vocab sort** (`memra-sampling/src/lib.rs:338-341`) | ~2.6 M comparisons at 152k vocab, on the same host path as §A5 | **numeric-class**: `select_nth_unstable_by` changes which of equal-logit candidates survive at the k boundary | own lane, own byte-identity gate |
| 11 | **`cuCtxGetCurrent` ~4,100–4,500/token** from cudarc's `bind_to_thread()` on every launch and every alloc (`launch.rs:212`, `core.rs:1534→1538→350`) | 260–440 µs/token, **counted by neither census** | needs a vendored cudarc patch | a per-thread "context already current" flag invalidated by the `enter_main`/`GpuMainOverride` guards (`memra-runtime/src/lib.rs:216-254`), which are the only things that legitimately move the context stack |
| 12 | **cudarc's per-launch kernel-arg `Vec`** (`launch.rs:63-72`) — 3 `Vec::new()` and ~7.8 pushes ⇒ 2 mallocs + 1 realloc + 1 free per launch ≈ 6,000–8,000 heap ops/token | 120–250 µs/token | vendored cudarc | `launch_into(&mut SmallVec<[*mut c_void; 16]>)` or a reused arena |
| 13 | **the 7 clock reads/token collapse to 2** (`worker.rs:10221` and `:10326` take a fresh `Instant::now()` for the *same* function twice per tick; `:12951` and `:13107` re-`elapsed()`; `memra-lanes/src/lib.rs:71` takes its own instead of the caller's) | ~175 ns/token on TSC — **but ~10 µs/token on `hpet`/`acpi_pm`/`xen`** | trivial, but it is a tick-structure change | **First: read `/sys/devices/system/clocksource/clocksource0/current_clocksource` on each serving box.** If it is TSC this is cosmetic; if not it is a real item. Also fix the `worker.rs:10299` comment, which claims `beat_busy` makes no syscall while `health.rs:102` reads the clock |
| 14 | **the 4 unflagged `eprintln!` spec-receipt writes** (`worker.rs:18711, 19118, 19314, 19469`) on **unbuffered** stderr in the serving tick | 0 to unbounded — it **blocks** if stderr is a pipe/journald with a slow reader, and the cost scales *inversely* with acceptance | the receipts are load-bearing for the deploy gate, so deletion is not an option | **First: `ls -l /proc/<pid>/fd/2` on each serving box.** Fix is a bounded lock-free ring drained off-tick |
| 15 | **glm5 uses the HOST embed gather** (`model.rs:1334-1344`) while `embed_gather_device` exists (`lib.rs:10456+`) and the qwen path uses it (`decode.rs:813, 1134, 1903`) | 1/token plain, **2/round** spec, ~14 heap allocs + page-fault exposure on a cold table | a per-family gap, not a spec one | note `MEMRA_DEV_EMBED` is a **receipted negative** for a different arm (`decode.rs:790-801`: flat, +2.1 GB VRAM) — this is the glm5-specific gather, not that door |
| 16 | **hoist the ~16 tick-scoped allocations** to persistent scratch + `clear()` (`worker.rs:12809-12863, 12751, 12765, 11355, 10399, 11361`, `group_chunks` `:18265-18281`), add `decode_bytes_into` to the tokenizer, and `std::mem::take` instead of `rows[k].clone()` at **`worker.rs:12890`** | 1.5–4 µs/token + one n_vocab memcpy on host-sample rows | mechanical but wide; wants one lane and one pass | the `12890` clone is the single clearest one: `rows` is an owned `Vec<Vec<f32>>` returned by value |
| 17 | **replace the two full percentile sorts** under the `Metrics` lock (`memra-lanes/src/lib.rs:95-104`, calls `worker.rs:13641-13642`) with a rolling estimator; and move `mem_get_info()` (`worker.rs:13692`, a driver call that can contend with the tick's own launches) out of the critical section | ~70 µs/token amortized, ~2.2 ms on the tick it lands | the 1/32 throttle already makes it survivable | t-digest or a fixed histogram |
| 18 | **the FR-Spec d2t map re-uploaded per rejected sampled round** — 128 KB+ pageable HtoD (`glm_spec.rs:2306`) — and the **22 pageable 4-byte `memcpy_htod`** for MLA `len_d` mirrors (`glm_spec.rs:1102`, plus `:1188` bypassing the door) | 30–100 µs/reject round; door H covers 22 of the 23 | make the d2t map load-time resident; route `:1188` through `i32_mirror_store` | `lib.rs:10488-10489` calls this primitive *"poison mid-round"* in as many words |
| 19 | **fix `pin_current_thread`'s discarded return value** (`cpu_experts.rs:571`) — a failed `sched_setaffinity` is currently indistinguishable from a successful one | correctness, not perf | it is in the `MEMRA_CPU_EXPERT_LIB` path, which is dead in a stock GPU serve | the readback pattern this lane's `affinity.rs` uses is the fix; lift it |
| 20 | **`MEMRA_TOKIO_WORKER_THREADS`, or a considered default** — the runtime is a bare `#[tokio::main]` (`lib.rs:4311`), so it is `available_parallelism()`: **192 threads on Box B.** Nothing in the repo documents that `TOKIO_WORKER_THREADS` is the lever | priced by arm (ii) | Whether the engine should cap its own runtime is a design decision with an ops-visible default | if arm (ii) lands >2%, this becomes a real flag with a FLAGS row; until then it is an ops note, and the **gap worth closing immediately is documentation**, since a non-`MEMRA_*` name is invisible to `check-flags.sh` |

---

## PROD-APPLICABILITY

Prod serving boxes are never benched (`prod-serving-boxes-untouchable`), and every change reaches
them through `serve-deploy` blue/green — never a manual restart (`serving-supervision-stack`).
What each finding is worth to which stack:

| item | q38 / box12 (primary billing) | step37 (house prod) | orn (embed/rerank + ornith) | how it may roll |
|---|---|---|---|---|
| **C1 dense penalty pass** | **Directly relevant** — its non-thinking arm is the shape that pays the O(n_vocab) cost today (§A5) | No penalties declared ⇒ inert but harmless | No penalties declared ⇒ inert | **Rides a normal release**: bit-identical, no flag, no config. No A/B owed; the standing batteries plus the bitwise gate are the evidence. Do NOT claim a tok/s number for it without a box row |
| **`MEMRA_SERVE_DEVPENALTY=1`** | **The larger win, and it needs its own A/B.** Its launcher already defaults to 1; only `box-box12.env:48` pins 0 | inert (no penalized shape) | inert | **NOT this lane's flip.** Needs: interleaved ×3 on a bench box with the q38 non-thinking shape, greedy byte-identity vs the host oracle, a vendor-default sampled probe with a spec-engagement receipt on the live slot, and a FLAGS/state note. Then blue/green with `serve-deploy`, rollback = restore `SERVE_DEVPENALTY=0` |
| **C2 `MEMRA_WORKER_AFFINITY`** | **DO NOT SET IT.** Arm (i) measured NULL on a 12-CCD EPYC (§B1) and the mechanism says why: CFS already parks a saturated decode worker on one CPU, so there is nothing to win. box12 is additionally a hybrid P/E-core Core Ultra 9 285K where a CCX mask is meaningless and an E-core pin would be a real regression | Same: do not set. step37 is TP across both cards on one worker thread, so a mask there would also constrain both device streams — and there is no measured upside to pay for that risk | Same: do not set. Co-tenant box, so a mask on one stack could collide with the other's | **The flag ships default OFF and stays a diagnostic seam, not a tuning knob.** It earns its place by making the question answerable (and re-answerable on a future host whose scheduler behaves differently), not by being turned on. A deploy of the binary changes nothing |
| **`TOKIO_WORKER_THREADS`** | **Measured NULL too** (−0.013% greedy / −0.004% sampled at 16 threads vs 192, §B1), so there is no throughput case for capping it. box12's runtime is far smaller than Box B's 192 anyway | same | same | No action on throughput grounds. A cap may still be wanted for MEMORY reasons (192 runtime threads × stack), which this lane did NOT measure — that is a separate question with a separate receipt. Still worth one ops-note line, since a non-`MEMRA_*` name is invisible to `check-flags.sh` and therefore undocumented anywhere |
| **stderr spec-receipt writes** (A3) | Real if box12's stderr is a pipe/journald with a slow reader — an unbuffered blocking `write(2)` in the serving tick | same | same | **Check before changing anything**: one `ls -l /proc/<pid>/fd/2` per box. The receipts are load-bearing for the deploy gate, so the fix is a bounded ring drained off-tick, not deletion |
| **`contains_stop_string` O(n²)** (A2) | Real for any client sending `stop` — agent harnesses routinely do | same | same | Named follow-up; a tail-window scan is bit-identical but touches the stop-sequence semantics, so it wants its own test |

---

## GATES

| gate | state |
|---|---|
| `memra-sampling` lib tests — incl. `dense_penalties_match_the_scan_reference_bitwise` (24 coefficient×window cases, compared by `to_bits`) and `the_sampled_path_penalizes_the_token_it_generated` | **18/18 PASS** |
| `affinity` unit tests — OFF makes no syscall, `ccx`/`ccx:N` resolve and refuse out of range, malformed values never degrade to OFF, absent cpus refused, `apply` reports the kernel readback, `SelfCcx` is never a silent no-op | **7/7 PASS** (and the 10 pre-existing `*affinity*` tests still green — which is itself the evidence that the `MEMRA_AFFINITY` name collision is a live readability hazard: one test filter matches both concepts) |
| `memra-server` + `memra-sampling` lib suites | **500/500 PASS** |
| `cargo clippy` (both crates, `--all-targets`) | **zero warnings** |
| `cargo fmt --all --check` | **clean** |
| `tools/check-flags.sh` | **green** — 755 runtime literal reads, 0 uncovered; `MEMRA_WORKER_AFFINITY` row in §2, same commit as the read |
| `tools/local-ci.sh --perf` | **exit 0. `correctness stage: GREEN`, `perf stage: 0 fail, 0 warn`** — and the coverage stated honestly rather than as a bare green (LAW cert-lines-carry-invocations: a skipped gate is not a PASS). **8 of the 9 perf cells SKIPPED for want of an artifact on this rig** (31b ×4, 26b ×2, e4b ×2); exactly ONE cell really ran: `qwen9b-plain-short: 138.12 tok/s [OK]`, which is a PLAIN cell and therefore does **not** exercise the penalty path C1 touches. Other arms also skipped for artifacts: serve-smoke spec/gemma4/Q35-coldhol, accept-gate (all cells), spec-on-cache-hit gemma arm; `spec-on-cache-hit: WARNING — qwen external drafter absent`. Binary provenance verified per the rebuild-after-checkout law: `strings -a target/release/memra-server` finds all three `[worker-affinity]` announce strings |
| **C1's real serving-path coverage** (what actually bites the penalty change) | **GREEN on GPU** inside the spec-on-cache-hit gate: `sp penalized sampled leader engages spec`, **`sp penalized sampled hit bytes == cold leader bytes (same seed)`**, `sp penalized sampled hit acceptance == cold acceptance exactly`, `np greedy+penalized leader serves PLAIN`, plus `r1/r2/r3/g1/g2 spec==plain byte identity` on the spec-off twin boot. **Scope stated, because this is the `pin-against-truth-not-siblings` shape**: both arms of that cell run the SAME (changed) binary, so it proves the penalized path is internally consistent and still works end to end — it does **not** prove equality against the pre-change algorithm. The load-bearing old-vs-new evidence is the retained `apply_penalties_scan_reference` oracle and the 24-case bitwise unit gate; this GPU cell is corroboration, not the proof |
| **24-step byte identity, affinity ON vs OFF — `affinity-identity-gate.sh`** | **GATE GREEN, two families, 2026-08-31T21:03Z.** `qwen/qwen3.5-9b` sha16 **`3305599fc784d782`** and `ornith/ornith-1.5-9b` sha16 **`392d5ca969e4b8ce`**, each identical across the arms, with the live readback `[worker-affinity] engaged request=0-5 effective=0-5 cpus=6 l3_domains=1` on the ON boots and `[worker-affinity] off` on the OFF boots. **The mask genuinely narrowed: 6 of 24 cpus.** So `MEMRA_WORKER_AFFINITY` is confirmed non-numeric on two families on real hardware, and the announce/readback path is confirmed end to end (not just unit-tested). **glm5 is NOT coverable on the rig** (only the 1.2 GB vision tower is here) — its arm runs in the Box B window |
| **Box B intervention arms (§B1)** | **RUN 2026-08-31T21:12-21:48Z, 11 boots, EVERY ARM NULL.** ccx pin +0.021% greedy / −0.009% sampled; `TOKIO_WORKER_THREADS=16` −0.013%/−0.004%; composed −0.005%/−0.021% — all inside a 0.026–0.054% boot-to-boot noise floor, none near the 0.5% bar. The pin's effect is verified installed by kernel readback (`cpus=16 l3_domains=1`, 192→16), and the mechanism for the null is measured: the worker does **not migrate during decode in either arm** (1 CPU / 1 CCX / 0 crossings, 98.8% busy). Box released clean, all cards 1 MiB, DONE line written |
| **glm5 24-step byte identity, affinity ON vs OFF** | **GREEN, and it closes what the rig could not.** One greedy sha (`a04810ad8d0fd43d`) and one sampled sha (`f8f13e305768bba1`) across all 11 window boots on the real 190.7 GB NVFP4 artifact at the 3-card ship recipe. With the two rig families that is **three families** proving the flag non-numeric |

### What the identity gate cost to get right, banked because each failure is a reusable trap

The gate went RED four times before it went green, and **not one of those was the flag**. All four
were the harness, and each is the shape where a green would have been worse than a red:

1. **A model alias contains a slash.** `$OUT/serve-qwen/qwen3.5-9b-off.log` pointed into a
   directory that does not exist, so the redirect failed, the server never started, and it
   surfaced as *"never became ready"* — a harness bug wearing a real failure's costume. Slugify
   any alias that reaches a filename.
2. **`note` wrote to stdout while `run_arm`'s stdout WAS the sha** (command substitution). Every
   note contains the arm name, so the OFF and ON captures would have differed *by construction*
   and the gate would have reported a confident BYTE DIVERGENCE that did not exist. Caught by
   reading the capture path, not by a run — a false RED looks exactly like a real finding. Fixed,
   plus a `^[0-9a-f]{16}$` assertion on both captures so it cannot recur silently.
3. **A thinking model at a small cap returns `content: ""`** and puts everything in `reasoning`
   (the OpenAI-compat field name — **not** `reasoning_content`, verified against a live response).
   The emptiness guard is the only reason this was not a green comparing `""` to `""`. Any
   byte-oracle over a chat surface must refuse an empty capture and must hash the reasoning too.
4. **`ccx` narrows nothing on a single-L3 host.** This rig is a 24-thread laptop part whose one L3
   domain is `0-23`, so the ON arm requested the whole machine and got `cpus=24 of 24`. The
   "effective mask must be narrower than `nproc`" assertion caught it; without that assertion the
   gate would have passed while proving only that two *unpinned* boots agree. The gate now reads
   the L3 map and falls back to an explicit quarter-machine list when `ccx` would be vacuous.

Trap 4 is the one that generalizes beyond this gate: **a CCX-pinning arm is only meaningful on a
multi-CCD host.** Any later lane that runs arm (i) on a laptop or a small cloud shape will pin
"one CCX" and measure nothing, and the receipt will look fine.

**One provenance note, stated rather than left to be noticed:** the binary reports
a `system_fingerprint` stamped from the lane's BASE commit (`216ffd114`) rather than from
`a5c77a488` — the literal value is omitted here because a live serving fingerprint is
private-surface material and the public-boundary gate is right to refuse it — because local-ci built it from
the working tree before this lane's commit `a5c77a488` existed and the incremental rebuild
afterwards was a genuine 0.22 s no-op (nothing had changed). The binary provably carries the
change — `strings -a` finds all three `[worker-affinity]` announce strings, and the live boots
above print them. Recorded because a 0.22 s "Finished" is normally an ALARM
(LAW rebuild-after-checkout-attribution) and here it is not.

## INTEGRATION WITH THE MOVING LANE BASE

The lane opened at `216ffd114` and the bringup base advanced **117 commits** while this work ran
(bankfix merged and pushed mid-window). Merged in two steps at close, per the
pull-main-frequently law.

**The only conflict, both times, was `research/tune-data/perf-ci.jsonl`** — an append-only
measurement log where each side had appended its own perf row. Resolved by keeping **both** rows
in timestamp order: a perf row is a receipt, not a state, so neither side's measurement supersedes
the other's and "take ours" would have silently deleted somebody's cell. **Zero code conflicts** —
`affinity.rs` is a new file, the `worker::run` insertion and the `docs/FLAGS.md` row both
auto-merged, and `memra-sampling` was untouched by the base.

One trap banked while resolving it: this repo uses **diff3** conflict style, so a naive filter of
`<<<<<<<` / `=======` / `>>>>>>>` leaves the `|||||||` marker AND silently keeps the common-ancestor
section as if it were content. The first attempt did exactly that and produced a file with a
stray marker line that only a JSON re-parse caught. Resolve append-only logs with
`git checkout --conflict=merge` first, then re-parse every line and assert your own row survived.

Post-merge gates re-run from scratch, not inherited: **513/513** memra-server (13 new tests arrived
with the base) + **18/18** memra-sampling including the bitwise penalty oracle, clippy zero, fmt
clean, check-flags green.

## COORDINATION

- `crates/memra-sampling/src/lib.rs`: `apply_penalties` renamed to
  `apply_penalties_scan_reference` and marked `#[cfg(test)]`; new private
  `apply_penalties_dense`. **No public signature changed**; the only call site moved. Two tests
  added.
- `crates/memra-server/src/affinity.rs`: new file, private module. Additive publics are
  module-private (`AffinitySpec`, `Topology`, `parse_affinity`, `apply`, `apply_and_announce`,
  `parse_cpu_list`, `render_cpu_list`).
- `crates/memra-server/src/worker.rs`: ~20 lines at the top of `run`, before the existing
  `MEMRA_PP_DEVICES` read. No signature change.
- `crates/memra-server/src/lib.rs`: one `mod affinity;` declaration.
- `docs/FLAGS.md`: one §2 row.
- No kernel, no `.cu`, no engine forward path touched.

## CORPUS PROMOTION AT CLOSE (engine-wide, not the glm53 card)

Per the amended scope these go to the engine-wide cards, not `models/glm53-flash.md`:
- `agent-knowledge/gpu/gate-craft.md` — **TRAP**: `/proc/<pid>/status` context switches are the
  MAIN thread only; a per-thread audit must read `/proc/<pid>/task/<tid>/status` or it will read a
  busy worker as an idle process. And: `perf`/PMU is unavailable inside an unprivileged vast
  container (`perf_event_paranoid=4`, no kernel-matched linux-tools) while `strace` and `taskset`
  work — plan host-side cells around procfs sampling and label sampled migration counts as lower
  bounds.
- `agent-knowledge/gpu/kernel-craft.md` — **LAW/TRAP**: a per-token host cost can be invisible to
  BOTH a launch census and an alloc census. The 90 mHC cuBLASLt GEMVs are one launch and one alloc
  each while doing ~15 host API calls plus a heuristic query; a host-wall attribution that trusts
  launch counts alone will under-count by ~1 ms/token.
- `agent-knowledge/gpu/measurement-laws.md` — **TRAP**: a vendor-recommended sampling default can
  silently route a whole request shape off the device sampler and onto an O(n_vocab) host path
  (`presence_penalty` + `MEMRA_SERVE_DEVPENALTY=0`). Any per-token host-cost claim must state
  which sampling arm it measured, because the thinking and non-thinking arms of one model are
  different host programs.

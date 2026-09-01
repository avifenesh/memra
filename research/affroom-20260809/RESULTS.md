# Parked-cap growth for plain affinity — final verdict

Lane `lane/cx-affroom`, base `d2d6e6d1`, final runtime commit `aeda0041`. The live dogfood bug is
fixed: capacity no longer vetoes an otherwise-valid plain-affinity checkpoint. The parked cache
grows to the next request's bounded `need`, restores checkpoint state, and resumes. No speculative
headroom heuristic was added; grow-on-resume removes the off-by-one-turn cliff instead of moving it.

No origin push, merge, tag, perf-board edit, `rustup`, `nsys`, or runtime-default flip was made.

## Root cause, including why val256 appeared healthy

The ctxcharge repair changed finite requests from inheriting the server context to their own
`prompt + max_tokens + 8` allocation. A parked session therefore carried the previous turn's cap.
The next real chat turn was always larger, so the affinity probe rejected it before identity and
the exact checkpoint bytes could earn a rewind:

```text
cap 45064 < need 45522   (12690 prompt tokens)
cap 45466 < need 53555   (20723 prompt tokens after a large paste)
```

That explains the owner's 0 rewinds across 9 requests and 0.108 cache-hit token ratio.

The val256 Box1 ON arm did fire 8 rewinds because it ran on the pre-repair context-floor behavior:
every request was allocated at `MEMRA_CTX=262144`. Its 37,823 -> 47,175-token conversation never
approached the parked 262,144-row cap. It proved the existing rewind state could work when room was
already present; it did not exercise request-owned sequential growth.

## Implementation

- `affinity_resume_target()` now lets exact bytes and identity decide first. Capacity returns a
  grow target instead of a decline. The target is `max(parked_cap, need)`: an F5 right-sized cache
  already covering the request is not inflated to the global/server cap.
- Admission charges `max(ctx_cap, need)` for the incoming request. The finite-request difference
  is the bounded speculative guard (normally 56 rows beyond `prompt + max_tokens + 8`), so growth
  is request-owned rather than hidden VRAM.
- Plain growth allocates a larger cache, copies checkpoint-valid full-attention KV rows D2D,
  restores GDN conv/SSM from the checkpoint snapshot, resets lengths/position, and resumes only
  after successful publication. Allocation gets one reclaim-and-retry; any failure drops the
  candidate and follows the existing cold path.
- Checkpoint restore moved into the shared PP layer. Every layer copies through its owning stage
  engine; open PP contexts synchronize before the restored cache is published. This avoids
  treating the primary CUDA stream as owner of a PP-2 remote cache.
- The required `serve-smoke` exposed the identical post-ctxcharge mismatch in spec affinity. Spec
  growth now uses the shared trunk restore, copies checkpoint-valid MTP scratch rows, and drops
  pointer-baking draft graphs so the next burst recaptures them.
- No growth headroom was added. Every real growing turn may allocate/copy once; the durable
  mechanism remains correct after an arbitrarily large next turn, subject to honest admission/OOM.

Commits:

- `b1be027f` — progress ledger first.
- `ed1d5a07` — initial plain grow-on-resume plus pi-shaped regression.
- `5914cb0f` — diagnosis and initial CPU receipt.
- `aeda0041` — request-owned `need`, PP-aware shared restore, and spec-affinity growth.
- `926acd30` — raw before/after and gate receipts.
- `ad197385` — reopen the completed local phase for the owner-requested pod receipt.
- `f643030f`, `dbbb1a12`, `ac659918` — reproducible Step pi-shape driver and target tuning.
- `311cc737` — clean PP-2 raw receipt, including the retained exploratory shape miss.

## Gate verdicts

Final release binary: `6c9d7c47339edef27ecd19be5af7d67b0153853e6b0268b90a93dfc45794a460`
(built from `aeda0041`).

| Gate | Regime | Result |
|---|---|---|
| `cargo test -p memra-server` | CPU, final source | **157/157 pass** (156 baseline + pi regression) |
| Focused pi unit | park 45,064; next need 45,522 | **grow/resume**, not decline |
| Focused spec before/after | same 4-turn rewritten chat | **0 rewinds + 3 cap declines -> 3 grows + 3 rewinds, 0 failures** |
| `run-5090-gate.sh` unchanged | local RTX 5090 Laptop, q9, K=0, N=3 ON/OFF | **GREEN x3**; 11 rewinds per ON run, budget green, no named divergence |
| True-cold teeth | max_tokens=8, prefix cache off, every-tier cold oracle | **17/17 byte exact**, 11 rewinds, budget green |
| Cache meter | serve-smoke synthetic accounting | **23/23 assertions**, terminal `0 failed` |
| Full serve-smoke | plain, spec, sampled truncation, affinity, Gemma | **0 failed**; both fresh affinity servers rewound 3 times |
| Owner pi shape | clean 2x RTX PRO 6000, Step-3.7, 262k, PP-2, K=0 | **PASS**; 3/3 growing turns grew and rewound, including the 8,026-token paste |

The full wrapper's N=3 median uncached-token slope was **0.0687 ms/token ON** versus
**0.2248 ms/token OFF**. Median turn-11 TTFT was **0.0993s ON** versus **0.4855s OFF**; ON cached
tokens advanced 743 -> 2,609 and every ON run reported cache-hit ratio 0.6006. These are regression
receipts, not a published performance-board move. The local desktop was not clock/temperature
locked; GPU-process sampling records a concurrent Hermes process at 394 MiB. N and regime are
therefore explicit, and no cross-day or competitive claim is made.

The decisive state-copy receipt is the short-window arm. On the final binary its first growths
were `764 -> 856 -> 893 -> 932`, continuing through `1225`; all 11 sequential rewinds matched an
every-tier cold prime byte-for-byte across all 17 requests. This exercises the new allocation and
copy path, not merely an in-place rollback.

## Owner-pod pi-shape receipt

The follow-up ran on the same RunPod class that produced the live declines: Step-3.7 Flash IQ4_XS
plus its Q8_0 MTP artifact, `MEMRA_CTX=262144`, forced plain `K=0`, and the real PP-2 split
(`stage0=dev0`, `stage1=dev1`). Prefix cache was disabled, so the per-turn cached-token receipts are
from continuation affinity rather than another reuse tier. Owner update 2 made the box exclusive
development capacity; both 96 GB cards began and ended at 2 MiB used, so `window_clean=true`.

The frozen replay preserved the request's `max_tokens=32768` charge but used an empty stop string
to end after one generated token. This keeps the parked-cache arithmetic identical to the live
request without turning the correctness gate into a long-generation soak. The actual shape was:

| turn | prompt tokens | request need | cached tokens | result |
|---:|---:|---:|---:|---|
| 1 | 12,654 | 45,486 | 0 | cold park |
| 2 | 12,692 | 45,524 | 12,648 | grow `45,430 -> 45,524`, rewind |
| 3 | 12,733 | 45,565 | 12,686 | grow `45,524 -> 45,565`, rewind |
| 4 | 20,759 | 53,591 | 12,727 | grow `45,565 -> 53,591`, rewind |

This closely reproduces the owner's failing sequence (12.6k small-turn growth, then a roughly 8k
paste) and crosses the old capacity veto on every resumed turn. Final metrics report
`plain_affinity_rewinds=3`; the server log contains three grow/rewind pairs and zero affinity
declines or resume failures. The first untuned pass (10,153 -> 16,770 tokens) was correctly red on
shape despite also rewinding 3/3 times and remains committed beside the final run.

Pod binary:
`a28149b4c7840c1778ccf990899ad5fd61f90905cd0677940b29e3aa701a0428`, built with CUDA 13.1 for
sm_120a. Runtime source is still `aeda0041`; the later pod HEAD adds only lane docs/test data.

## Long-window true-cold limitation (retained, not hidden)

An additional, non-required 256-token comparison against `MEMRA_KV_REUSE=0` is **RED** under the
gate's shallow-prefix heuristic: 2/17 outputs diverged after 15 shared characters; 12 more diverged
deeper. The result is reproducible on the final binary and is retained in
`raw/affinity-full-after-20260809T194404Z/gate-true-cold.json` with `exit_code=1`.

This lane therefore makes no claim that long-window resumed generation equals a monolithic cold
prime. The engine's established chunk-order sensitivity permits long greedy tails to diverge, but
the current `<32 char` heuristic labels these two rows as a possible state bug. What is proven here
is narrower and matches the requested acceptance contract:

- the grow path is deterministic across three fresh ON servers;
- the first 8 generated tokens are exact on all 17 true-cold comparisons;
- every requested production gate is green; and
- capacity growth itself fires on every growing sequential turn with no resume failure.

If the long-window shallow-prefix heuristic becomes a release requirement, it needs a separate
numerics/chunk-order lane; it is not silently reclassified here.

## Raw evidence

- Owner-supplied live before receipt: `raw/before-owner-pi.log`.
- CPU/build: `raw/cargo-test-memra-server-after-shared-grow.log`,
  `raw/cargo-build-release-shared-grow.log`.
- Focused spec before/after: `raw/spec-affinity-before/`, `raw/spec-affinity-after/`.
- Required N=3 wrapper: `raw/affinity-full-after-20260809T194404Z/`.
- Required 17/17 teeth: `raw/affinity-teeth-after-20260809T195246Z/`; reproducible harness:
  `run-teeth.sh`.
- Serve-smoke/cache-meter: `raw/serve-smoke-after-20260809T194252Z/`.
- Clean PP-2 owner-shape receipt: `raw/pod-20260809T200605Z/`; final verdict and per-turn rows are
  in `pi-shape-final/summary.json`, with launch/shutdown and affinity lines in
  `pod-driver-final.log` and the complete worker receipt in `server-final.log`.
- Intermediate runs, including failed exploratory receipts, are retained beside the final runs
  rather than deleted.

## Remaining release boundary

The new-cache restore is now exercised through its PP owners on the owner's two-card target and
the exact live growth pattern resumes. This follow-up is not the repository's full pre-merge
battery: target-rig `kernel-check`, `run-gen` argmax, and `run-spec` K=1..8 remain
orchestrator-owned prerequisites. This lane intentionally stops without pushing, merging,
tagging, or changing a runtime default.

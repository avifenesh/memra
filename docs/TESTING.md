# Testing: the tiered gate structure

Two regimes, one rule: **the full battery gates every merge and tag, unchanged; fast-gate
accelerates the dev loop between battery points.** Nothing in this document weakens the
merge/tag bar: a fast-gate green is a *keep going* signal, never a *ship* signal.

## DSV4 norm-fusion gates, REMOVED 2026-09-11

The two sections that stood here documented `dsv4_compose_densefast_normfuse_gate`
and `dsv4_densefast_normfuse_default_gate`, plus the control audit for the
norm-fuse doors. All three doors were deleted on 2026-09-11 after the served ABBA
priced them (memra `docs/FLAGS.md`, removed-doors ledger, 2026-09-11 norm-fusion
entry), and their gate bins went with them, so the procedures here had no subject
left. Dense-fast keeps its own standalone gate, `dsv4_dense_fast_gate`.

One procedural point survives its subject and is worth carrying forward: those
gates pinned the doors OFF at startup and then VERIFIED the pin took rather than
trusting it, because an inherited `1` would silently change what the rows measure.
That shape belongs in any gate that pins a door.

## Target-aware release evidence

The local RTX 5090 battery is the blocking performance gate for generic kernels and defaults
that affect that target. It is not a universal veto over model-specific multi-card work.
Official Step-3.7-FP8 changes whose product target is RTX PRO 6000 Blackwell may use the
separate `step-pro` pre-push gate:

```bash
MEMRA_HARDWARE_GATE=step-pro \
MEMRA_STEP_PRO_RECEIPT=/absolute/path/to/step-pro-receipt.json \
git push
```

`tools/check_hardware_gate.py` verifies the receipt's model and hardware identity, requires
kernel, topology, and official-model exactness evidence, verifies every evidence manifest,
and binds every changed engine source file by SHA-256. A later source edit invalidates the
receipt immediately. `MEMRA_SKIP_PERF_CI=1` is rejected in this mode. RTX 5090 validation
remains a compatibility follow-up for Step unless the same change also modifies a generic
5090-facing default.

### B200 dry gate and hardware qualification

Before any B200 device execution, the `sm_100a` backend must pass its host/toolchain
surface:

```bash
research/b200-kernel-twins-dry-20260901/check-layouts.sh
research/b200-kernel-twins-dry-20260901/check-nvfp4.sh
research/b200-kernel-twins-dry-20260901/check-fp8.sh
tools/test_install_b200_policy.sh
tools/test_b200_phase0_harness.sh
MEMRA_CUDA_ARCH=100a cargo build --release --bins
MEMRA_CUDA_ARCH=100a cargo test --release -p memra-engine \
  b200_dry_policy_tests --lib
```

These gates compile the production translation units, inspect static-archive ABI/SASS, run the
exact host-constexpr operand/scale layout contract, and prove the release installer still refuses
an unpublished B200 prebuilt. They never open the CUDA driver. They are CI-wired in the `100a`
release-arch mirror. A green dry gate alone is not `NativeQualified` and does not replace the sealed
B200 `kernel-check`, model parity, sampled serving, concurrency, context, rollback, or performance
battery under `research/b200-kernel-twins-dry-20260901/receipts/`.

On a default-env B200 build, `kernel-check` keeps the explicit block-FP8 prefill route off, records
`SKIP E4M3-BLK-MMQ-VIEW`, and continues through the rest of the default exactness battery. The
phase-0 qualification harness deliberately sets `MEMRA_FP8_MMQ=1`, so its explicit block-FP8
oracle and the same MMQ-view subcell still execute. A default-posture battery must never enable a
non-default route just to avoid an early refusal.

For a new box or changed kernel source, inspect the exact phase-0 plan without touching the device:

```bash
research/b200-kernel-twins-dry-20260901/run-box-phase0.sh --plan
```

Actual execution requires an explicit non-production acknowledgement, an externally approved
40-hex commit, a new absolute receipt directory, and optional device indices. The harness refuses a
dirty tree, non-CC-10.0 GPU, existing compute process, wrong CUDA toolkit, occupied canonical lock,
or nonempty receipt namespace before launching the synthetic NVFP4/FP8/kernel-check cells.

## The tiers

| Tier | Wall (5090 rig, measured 2026-08-02) | What runs | When |
|---|---|---|---|
| 0 | seconds (~2 s kernel-check scoped + build) | workspace compile + kernel-check scoped to the touched sections | every edit-compile loop |
| 1 | ~1–2 min | tier 0 + golden-token argmax probe on ONE model per affected kernel class (+ one single-K spec probe when the diff touches the spec pipeline) | before every dev-loop commit |
| 2 | tens of minutes | the full battery, `tools/local-ci.sh`: kernel-check ALL GREEN (~4.5 min), prime-gate, run-gen argmax per model, VERIFY-GATE, `run-spec` K=1..8 self-consistency on the Qwen 35B target + external MTP draft (`MEMRA_CI_RUNSPEC=0` skips), Gemma-4 31B stream agreement 64/64, decode-batch-gate (config + Q8_0 strict, the serving tick's exactness, wired in 2026-08-05), graph-warmup stress (`tools/graph-warmup-stress-gate.sh`, pool-growth adversarial bit-identity behind the `MEMRA_GRAPH_WARMUPS=1` default, wired 2026-08-05), serve-smoke, health-fault-gate (`tools/health-fault-gate.sh`, readiness, panic-respawn, gpu-watch latch and drain arms on the real server, wired 2026-09-22; `MEMRA_CI_HEALTH_FAULT=0` skips), serve-stress (`tools/serve-stress-gate.sh`, the c=64 concurrency contract behind the admission spec-headroom fix, wired 2026-08-06; `MEMRA_CI_STRESS=0` skips), accept-gate (`tools/accept-gate.sh`, exact served-spec acceptance counts + a 128-token text sha at the production drafter/K, wired 2026-08-06; smoke cell by default, `--full` for the 6-cell matrix, `MEMRA_CI_ACCEPT=0` skips) | **every merge, every tag** (unchanged) |

The battery's last correctness stage runs every memra-engine `#[ignore]` GPU test serially
(`--test-threads=1`): the tests flip process-global gate doors and share one device, so
parallel threads race each other (measured flake, `research/local-ci-one-card-20260921`).
Pair-only tests announce `SKIP-PAIR` on a rig with fewer than two CUDA devices and run
unchanged on the pair box.

The prime continuation gate (`qwen-a4-continuation-gate`, in tier 2 after prime-gate on the
9B NVFP4 GDN hybrid) primes a 9,296-token repo-text prompt once and as head + tail at the
grid-aligned tails 16, 48 and 80, and requires the tail call's logits to be bitwise the one-call
prime's. The 16-row tail is the shape memra#427 found non-bitwise: `matmul`/`matmul_pre` sent
`m = 16` through the batched decode/verify mmvq tier (`2..=16`) while every longer prime rode
the generic path; the tier now ends at `PRIME_MIN_T - 1` outside the verify-exact scope.
`MEMRA_CI_CONTGATE=0` skips, `MEMRA_CI_CONT_MODEL` retargets. The per-operation form of the same
question is `qwen-a4-width-walk <model.gguf> [ref_width] [widths...]` (a `memra-engine` bin, not a
`tools/` script): every projection the prime walk multiplies, at width 16 against 17 and 48, bitwise
on the shared rows; the summary line `WIDTH WALK width 16 vs 17: 0 of N tensors differ` is the
verdict, and a nonzero count names the tensors whose dispatch is keyed on the call width
(`research/spill-b-20260919/DAY22.md` for the run that named `ssm_beta`/`ssm_alpha`).

Boot-time environment audit (memra#483, `memra_engine::env_audit`): the server refuses a
retired `MEMRA_*` door or an unknown name inside an owned family before any door is read,
naming the FLAGS.md "Removed" ledger; the registry is generated from `docs/FLAGS.md` by the
engine's `build.rs`. Its red arm (a retired name refuses) and its non-vacuity arm (every legal
name at once refuses nothing) are unit tests in `env_audit.rs`; `MEMRA_ENV_AUDIT=warn` downgrades,
`=0` disables, both announced. Receipts: `research/env-audit-20260921/`.

Prime fairness (memra#521, `tools/prime-fairness-gate.py`, in `tools/local-ci.sh`,
`MEMRA_CI_FAIRGATE=0` skips): one boot per `MEMRA_PRIME_YIELD` arm on the 9B NVFP4's default
(spec) route with the concurrency demotion pinned off, greedy natural-text `prompt` streams (the
tokenizer is calibrated per boot through `usage.prompt_tokens`); a seeded 4,096-token prompt, then
a 131,072-token cold prime with two cold 2,048-token peers and the seeded prompt again (a cache hit)
started 2 and 3 seconds apart. A request that generates fewer than 16 tokens refuses the run, so the
byte clause never compares an empty output. Verdicts: every request's text identical across arms; on the yielding arm every peer's first
token within the bar (8 s) and the peers' p95 at most half the non-yielding arm's; `/health`
`tick_max_ms` on the yielding arm at most 6,000 ms; every request finished; the yielding boot logs
`[prime-walk] supported=true yield_door=true` and at least one `[prime-yield]`. `--reps 3`
interleaves the arms for a receipt. Receipts: `research/prime-fairness-default-20260922/`.

Loader tensor-contract boundary (memra#541, `memra_gguf::checkpoint_binding`): both loaders
bind the pack's tensor contract against the source census before any upload and refuse
missing, unexpected, duplicate, ambiguous, wrong-shape and wrong-quant tensors and an undeclared
tied head with the pack and dialect named. CPU teeth: `checkpoint_binding::tests` (the glm-dsa
micro fixture clean, byte-renamed trunk tensor, headless copy under a `SeparateHead` pack, the
head-ownership matrix, the recording source and the consumption audit, a census-less source).
Device arm: `crates/memra-engine/tests/checkpoint_contract_refusal_gpu.rs` (`#[ignore]`, run
under the rig lock): the renamed and the headless tampered copies refuse before upload through
`HybridModel::load`, the clean fixture loads. Receipts: `research/loader-census-20260922/`.

Request-fault boundary (memra#525, `tools/request-fault-gate.py`, in `tools/local-ci.sh`,
`MEMRA_CI_FAULTGATE=0` skips): one boot of the real server with the `MEMRA_FAULT_INJECT_CACHE_SALT`
door, a control round of three concurrent greedy streams, then the same three plus a salted stream
that panics inside its guarded step. Verdicts: the salted stream ends with an error object
whose `code` is `worker_fault` and no `finish_reason`; every peer finishes and its text equals the
control round's byte for byte; `request_faults_total` reads 1, `worker_respawns_total` 0,
`/health` `worker.generation` 0; the log carries exactly one `[fault] request=` line and no
`[worker] PANIC`. The classification itself (request fault vs re-raised worker fault, driver-looking
payloads, pass-through of returned errors) is CPU-only unit tests, `request_fault_guard_tests` in
worker.rs. Receipts: `research/request-fault-20260922/`.

The docs-fit owner call is closed: tier 2 now runs the full `run-spec` K=1..8 sweep and requires
eight per-K PASS lines plus the final `SELF-CONSISTENCY PASS` marker. The raw run is logged before
parsing; a red quotes the failing K and `FIRST DIVERGENCE` index.

**One correction to how tier 2 is described elsewhere:** `--tier 2` does not run the perf stage.
`fast-gate.sh:68` is `exec tools/local-ci.sh` with no arguments, and local-ci.sh gates its perf
cells behind `--perf` / `--perf-quick`. So `--tier 2` is correctness-only; run
`tools/local-ci.sh --perf` directly for the cell battery.

Entry point:

```bash
tools/fast-gate/fast-gate.sh                       # tier 1 vs HEAD (uncommitted work)
tools/fast-gate/fast-gate.sh --tier 0              # compile + scoped kernel-check only
tools/fast-gate/fast-gate.sh --diff main           # scope = everything since main
tools/fast-gate/fast-gate.sh --tier 2              # execs tools/local-ci.sh (the real gate)
tools/fast-gate/fast-gate.sh --smoke               # add the perf tripwire (see below)
tools/fast-gate/fast-gate.sh --probes k27,amargin  # name the arms explicitly (see below)
```

**`--probes` is not optional convenience: it is how you gate a tree with no diff.** A clean tree
(release candidate, a fresh rsync onto another rig, a tree with no `.git` at all) has an empty
`CHANGED` set, and the diff-driven path exits 0 with "nothing to gate". That is a FALSE GREEN if
you meant to gate it: it reports success having run **zero** probes. Found on the v0.71.0 pod
battery, where the rsync'd tree had no `.git` and the k27 regression check "passed" without
executing (`fast-gate.sh:80-88`). Naming arms with `--probes` suppresses the short-circuit.

As of 2026-08-07, every probe registered in `models.tsv` has at least one automatic dispatch home.
The four formerly orphaned arms are scoped rather than added to `DEFAULT`: `amargin`/`amarginc`
follow the `forward_last`-vs-`decode_step` implementation and their own wrapper/probe; `e4b`
follows its PLE/KV-sharing forward/load, attention, and glue surfaces; `kat` follows the
non-expert IQ4_XS qmatvec/dense-MMQ path plus its forward/load surfaces. `DEFAULT` remains
`g12,q9,q35,gwstress`. Use `--probes` for clean trees and deliberate explicit subsets, not to
compensate for an unreachable registry entry.

## Change-scoped gating (tier 0)

`git diff --name-only <ref>` (plus untracked files) is mapped through
[`tools/fast-gate/map.tsv`](../tools/fast-gate/map.tsv), an editable TSV that encodes the
dispatch structure: which kernels serve which model classes. Every matching row contributes
to the plan (union); an unmatched path falls back to the conservative full plan and prints a
warning (add a row when that happens).

kernel-check gained two loud diagnostics seams for this (see the `kc_model` header in
`crates/memra-engine/src/bin/kernel_check.rs`):

- `MEMRA_KC_FAST=1`: synthetic arms only, every weight-oracle section skipped (loudly).
  Measured: **~1.4 s** vs **~4.5 min** full (the model-backed GEMM oracles are >98% of the
  wall, 266 s of 268 s in the timed run,
  `research/fast-gate-20260802/kernel-check-full-timed.log`).
- `MEMRA_KC_ONLY=csv`: synthetic arms + only the weight-oracle sections whose name matches
  (`dtype5`, `nvfp4-gemm`, `q8mmq-gemm`, `q4_0-mmq`, `q4_0-sk-arm`, `iq4xs-mmq`,
  `f16g-kq-direct`, `nvfp4-27b-shape`, `nvfp4-mmvq`, `nvfp4-batched`, `a6-split-plane`,
  `d2-cache-bit-identity`, `fast-router-batch`).

Both seams print a `KC-SKIP` line per skipped section naming the env: a scoped run is never
silently narrower than it looks. They are diagnostics-class flags per the flags doctrine
(dev-loop scoping), not defaults: the battery runs kernel-check naked.

## Golden-token pinning (tier 1)

The battery's run-gen argmax gate re-derives its reference every run (prefill forward +
tokenwise decode + batched prime, three primes per invocation on a big model). fast-gate
pins the *output*: for each probe in
[`tools/fast-gate/models.tsv`](../tools/fast-gate/models.tsv), the greedy `tokens: [...]`
line at a battery-green commit is stored in `tools/fast-gate/goldens/<id>.tokens` with its
SHA and timestamp. A tier-1 probe then:

1. runs `run-gen` with `MEMRA_NGEN=20` (one model per affected kernel class),
2. requires the in-run gates green (prefill/decode argmax `MATCH`, no
   `MISMATCH-STRUCTURED` from the batched-prime gate; `FLIP-NEARTIE` stays reported,
   non-fatal, per the #46 contract),
3. byte-compares the `tokens:` line against the pinned golden: **any diverged token id is
   a FAIL**, with an instant verdict and no reference recompute.

Greedy decode is deterministic on this engine (run-to-run nondeterminism is itself a gated
bug class, see the ping-pong SSM-state fix in `cache.rs`), so token divergence == behavior
change. Exactness is clock-independent: goldens generated under any thermal/power regime
are bit-valid.

Spec-pipeline diffs add one single-K spec probe (`run-spec` + `MEMRA_SPEC_K` for the qwen
MTP family; `gemma-gate` + `MEMRA_SPEC` stream-agreement for the gemma drafter family).
The K=1..8 sweep stays tier-2.

`kind=cmd` probes (models.tsv) are self-gating commands (host unit tests or GPU oracle
gates like `sample-check`) whose gate is exit 0. They pin no golden and exist for code
the greedy token goldens structurally cannot see (the sampler chain). Three landed
2026-08-05: `chunkinv` (chunked-prefill byte-identity across `MEMRA_PRIME_CHUNK` values,
naked env, the grain-free default's contract; note its **coverage is per-architecture and
prompt-length-bounded**: the pinned probe prompts are short, which is precisely why they could
not reach the `step35` chunk-dependence defect that lives past a 512-token SWA window. The
long-prompt arm for that arch is `chunkinv35`/`chunkinv35c` (landed green with the fix,
`research/step35-chunkfix-20260807/`), and the axis one level up (serve splitting a prompt
across several `prime_cache` CALLS, per-tick budgets + the prefix-cache LCP split) is
`tickinv35`/`tickinv35c` (`tools/tick-invariance-gate.sh`, landed with the request-level
`seq_end` fix, `research/tick-seg-20260807/`; its `--splits` arms pin the off-grid-resume hole,
vLLM #51113's second law). Standalone, the probe is
`tools/chunk-invariance-gate.sh [<model.gguf>] [--chunks 2048,64,32] [--steps 48]
[--expect-invariant|--expect-variant] [--canary]`. `--expect-variant` is how you assert a known
chunk-DEPENDENT arch stays detected rather than silently passing, and `--chunks` is how you widen
past the default triple), `chunkinvc` (its canary:
injects the
`MEMRA_PRIME_F32CHUNK0=1` legacy arithmetic and must FAIL, proving the gate detects the
mechanism), and `gwstress` (the graph-warmup pool-growth stress gate behind the
`MEMRA_GRAPH_WARMUPS=1` default). A fourth landed 2026-08-06: `sstress`
(`tools/serve-stress-gate.sh`, 64 staggered streaming clients, asserting every stream
completes well-formed with a live worker and no OOM lines; it is the *concurrency*
contract, which no exactness golden can see, and the regression proof for the admission
spec-headroom fix). Its own teeth: `--teeth` forces the admission reserve to 16 MB and the
verdict must invert. It also closed a map hole where `crates/memra-server/` diffs mapped to
no gate at all.

`isogap` / `isogapc` landed 2026-08-07 (`tools/iso-gap-gate.sh`, lane/iso-gap task #91): the
**staggered-depth serve isolation** contract at the engine tick: a session's logits must be
bit-identical solo-vs-coresident **across a `fa_split_keys` ladder-rung boundary**, the shape
both the equal-depth serve gate and the kernel-check seqs-vs-loop pin (whose depths all sit
inside one rung) were structurally blind to. The straddle is placed **per-rig** (`iso-gap-probe
--auto` scans the SM-keyed ladder: the 82-SM 5090's first boundary is t_kv=513, a 188-SM pod's
is 2049), so the arm has teeth on both rigs instead of straddling nothing off-rig. One run
covers same-rung batched (seqs arm), the straddling per-seq fallback window, and both
transitions. `isogapc` injects a wrong token into the co-resident arm's feed (changes the
world, not the label) and must be caught. Note: the serve-level solo-vs-loaded byte drift
(`spec-gate` REF/REF_LOAD) is **not** this class: it is the `b_n==1` fused-trunk↔batched-body
config flip at the co-residence boundary (`research/iso-gap-20260807/`); this arm pins the
within-config isolation that any fix for that flip relies on.

`amargin` / `amarginc` landed 2026-08-06 (`tools/argmax-margin-gate.sh`, + its `--canary` teeth):
run-gen's prefill-vs-decode argmax assert calibrated against the **top-2 margin at the deciding
position**, because a near-tie flip and a real cache bug are the same red until you measure the
margin (see `research/q8-argmax-20260806/`). Flags:
`tools/argmax-margin-gate.sh [<model.gguf|hf_dir>] [--prompt f] [--window N] [--max-flips N]
[--margin-floor F] [--canary] [--logdir D]`. **Effective `--window` default is 12**, not the 24
the probe binary advertises in its own usage line: the wrapper always passes its own
`WINDOW=12`. Worth knowing, since window width *is* this gate's
coverage. Automatic dispatch is limited to `decode.rs`, `forward.rs`, `hybrid_forward.rs`, and
the gate's own wrapper/probe implementation; `--probes amargin,amarginc` remains the clean-tree
or deliberate explicit invocation.

HF directories use the native `SafetensorsSource` loader and their HF tokenizer; GGUF inputs
retain the GGUF loader and tokenizer. This gate compares batched prefill with tokenwise
**serial decode** on the same prompt positions, not the server's batched or speculative
paths, and supplies no performance result. Checkpoint size and placement still determine
the required hardware; accepting a directory is not a model qualification receipt.
The decode cache uses native pipeline placement. Sharded cross-device inputs are accepted
only for the hyper-connection trunk whose `forward_last` has a pipeline prefill dispatch;
other sharded trunks are explicitly rejected before measurement. No alternate prime
arithmetic is substituted to make them pass.

An explicitly requested missing model or probe, a missing prompt, invalid options, or an
empty/malformed/non-finite/duplicate-position table fails. The measured table must contain
exactly the requested window before canary injection. Automatic discovery with no available
model remains an explicit SKIP. Machine-consumed margins preserve f32 round-trip precision;
display rounding must not turn a small explained flip into a false failure. CPU controls:
`python3 tools/test_argmax_margin_gate.py` and
`cargo test -p memra-engine --bin argmax-margin-probe` (no GPU execution in these tests).

Also 2026-08-06: `accept` (`tools/accept-gate.sh`, the **served-spec acceptance +
long-text** assertion). It exists because the battery was *provably* blind to a class:
`research/f8f4-flip-20260806/` receipted a kernel arm that moved served greedy text in 4 of 6
regime cells at temperature 0 and moved spec acceptance up to −9.5pp while **every gate above
stayed green in both arms**. Three structural reasons: the token goldens stop at 20 tokens and
both divergences landed at generated index 22 and 38; `--refresh-goldens` after such a change
would have silently re-pinned the new arm; and nothing compared accepted-draft *counts*, which
are spec throughput. `run-spec` self-consistency cannot see it either: it asserts spec == plain
*within one arm*, which both arms satisfy.

So `accept` asserts, at the **production serve config** (the artifact's real regime drafter
attached via `MEMRA_MODELS "+draft"`, its real serve K, driven through the server): exact
`(rounds, drafted, accepted)` integers (temperature 0 makes drafting deterministic, so these are
hard numbers, not a band), plus the full generated text sha256 to `ngen=128`, 6.4× past the
golden window. The drafter is load-bearing: the acceptance sign follows (model × drafter ×
prompt) and *inverted* between the GGUF's embedded MTP head and the regime drafter on the same
models the same day, so a bare-head number is not evidence about a served config. References
live in `tools/fast-gate/accept-refs/<cell>.{ref,text}`; cells in
`tools/fast-gate/accept-cells.tsv`. `--pin` is the only writer and **refuses on a dirty
`crates/`. There is deliberately no `--force`**: that is the gate's central law, not a default
you can override (`tools/accept-gate.sh:120` says so verbatim), because pinning references beside
an uncommitted kernel change is exactly the receipted failure mode: the new arm's numbers become
the reference and the gate then defends the regression (`research/f8f4-flip-20260806`). A dirty
tree OUTSIDE `crates/` is allowed and merely noted: engine code is what must be committed.
`--pin` has a **second** guard for the same trap wearing a different hat: it refuses when any of
`MEMRA_MMQ_F8F4`, `MEMRA_MMQ_F8F4_PLAIN`, `MEMRA_MMQ_FP8BLK_PLAIN`, `MEMRA_FAST`, or
`MEMRA_PRIME_F32CHUNK0` is set in the environment, since references must describe the NAKED
default build (flags doctrine: winners are defaults). Its teeth: `tools/accept-gate.sh --teeth` sets `MEMRA_MMQ_F8F4=1` and
the verdict must invert (proven both directions, `research/accept-gate-20260806/`); `--control`
re-measures in a second independent server boot, which is what licenses the single-shot read.

The `k27` argmax probe pins `MEMRA_FA_SPLIT=8` in its
env column so its golden is rig-portable across the 82-vs-188-SM `fa_split_keys` rung
(lane/k27-divergence, a near-tie flip class, not a defect; `k27div-probe` is the
cross-rig teacher-forced localizer).

### Golden refresh protocol

Goldens refresh **only at full-battery green points**, never mid-dev:

```bash
tools/local-ci.sh                                  # must be ALL GREEN first
tools/fast-gate/fast-gate.sh --refresh-goldens     # refuses a dirty tree (--force overrides, loudly)
git add tools/fast-gate/goldens && git commit ...  # goldens are checked in, SHA-stamped
```

If a legitimate behavior change moves tokens (new kernel numeric config promoted through the
full battery), the refresh happens in the same commit that lands the change *after* its
battery run: the golden diff in review is the visible record that tokens moved.

## Perf smoke (`--smoke`): tripwire, not evidence

Each tier-1 probe already times its decode window; `--smoke` compares that **single rep**
against the tok/s recorded at the golden point (`goldens/<id>.perf`): WARN at >10% drop,
FAIL at >25%. This exists to catch catastrophic regressions (a kernel fell off its fast
path) inside the dev loop: it is explicitly **not** a publishable number and never moves a
board: publishable performance stays N≥5 interleaved same-session medians per
[`research/benchmarks.md`](../research/benchmarks.md), and drift detection at fine grain
stays `tools/local-ci.sh --perf`. A smoke WARN/FAIL means "re-measure with the real
protocol", nothing more.

## The probe-regime laws (learned by breaking kernels on purpose)

The catch demonstrations below exposed three ways a probe can be green while the touched
code is broken. The mapping table encodes the fixes; keep them in mind when adding rows:

1. **The probe must EXERCISE the touched dispatch class, not just the touched model
   family.** On a 24GB rig every daily MoE model loads RESIDENT: the SLRU cache, staged
   `moe_cached_gemm*`, and spill dispatch never run under a naked probe, and a deliberate
   gate/up weight swap there passed all four default probes. `q35slru`
   (`MEMRA_MOE_RESIDENT=0` + `MEMRA_MOE_SLOTS=1024`) forces that regime (68.5% hit rate,
   185k misses in its pin log) and caught the same break instantly.
2. **Depth is a dispatch axis.** The short probes decode at t_kv below/near the FA vec
   floor and windows; the gemma fp8-KV g-module arms (hd512 tb512 staging, windowed SWA)
   only execute at depth. `g12d` (the battery's 1736-id depth prompt) caught a K
   element-permutation break in the live tb512 staging arm that every short probe missed.
   (Its golden is 16 ids, not 20: token 17 of the g12 depth continuation is a real
   run-to-run near-tie flip, `g12-depth-nondeterminism.log`, and a 20-id golden
   false-fails ~1/8 runs. q9/q35/g31 deep continuations measured deterministic x5-x8.)
3. **Greedy goldens route around the sampler entirely.** temp=0 collapses to argmax, so a
   broken gumbel/softmax-gather kernel or a backwards top-k is invisible to every token
   golden. `samp` (the `sample-check` GPU oracle) and `sampt` (`cargo test -p
   memra-sampling`) are `kind=cmd` probes, self-gating commands, exit 0 = PASS, no golden.
   `sample-check` also pins heterogeneous sparse penalty rows against the CPU house rule with
   a 9,000-id window and rejects negative rows, duplicate cells, zero counts, and CUDA-width
   overflow. Decode-batch gate3c composes vendor filters + presence penalty with lean raw-logit
   parking and mixed host/device rows; gate3d forces a penalty-induced argmax flip and compares
   the returned row with an independent pristine oracle, so deleting dispatch or parking mutated
   logits cannot pass.

### Catch demonstrations (all breaks reverted; diffs + consoles in receipts)

| Break (deliberate) | Caught by | Receipt |
|---|---|---|
| MoE staged dispatch: up-projection reads GATE weights (`moe_cached_gemm_q8`) | tier-1 `q35slru` (run-gen argmax gate, exit 101); plain `q35` was BLIND (resident regime) | `break-moe-staged-*` |
| FA v4 K-scale skew x1.001 (default hd256 staging arm) | tier-0 kernel-check synthetic arms, 46 bit-identity FAILs (`fa_decode_rows`/`seqs_v4`) | `break-fa-v4-*` |
| FA hd512 tb512 K element permutation (live gemma fp8 global arm) | tier-1 `g12d` depth probe (prefill/decode argmax MISMATCH, exit 101) | `break-fa-tb512-perm-*` |
| MMQ Q8_0 wrong-block scale (index mixup, `load_tiles_q8_0`) | tier-0 `MEMRA_KC_ONLY=q8mmq-gemm`, rel=2.4e-1 vs the f32 oracle, 8 FAILs | `break-mmq-q8-idx-*` |
| f16g IQ4_XS dequant off-by-one (`ls-31`) | tier-0 `MEMRA_KC_ONLY=f16g-kq-direct`, byte-identity maxdiff 1.15e2, 8 FAILs | `break-f16g-iq4xs-*` |
| device sampler acceptance-prob skew x1.001 (`softmax_gather_f32`) | tier-1 `samp` (sample-check vs CPU softmax, exit 1) | `break-sampling2-*` |
| sparse device penalty drops a count/coefficient or parks mutated logits | tier-1 `samp` + decode-batch gate3c (CPU-row parity; full-vs-lean raw-logit identity) | `research/pro-device-penalty-sampling-20260824/` |
| host sampler top-k keeps WORST k (ascending sort) | tier-1 `sampt` (memra-sampling unit tests, exit 101) | `break-sampler-host-*` |

### Demonstrated coverage gaps (documented honestly, not closed)

- **Default-dead rollback seams**: kernels/helpers only reachable through non-default env
  seams are invisible to naked probes *by construction*. Verified twice: a `dq_K_lane`
  q8_0-branch lane swap (only live under `MEMRA_NO_FA_VEC`/v4-off arms at hd256, smem twin,
  `MEMRA_GEMMA_GKV=0` globals) passed everything, and so did an fp8 `dq_K_lane` lane swap
  (`break-fa-decode2-*`, `break-fa-fp8k-*`) and a kd requant skew (`break-fa-tb512-*`,
  int8 requant made the x1.001/126-vs-127 skews vanish at the __float2int_rn rounding,
  a magnitude-tolerant class, while the permutation break in the same loop was caught).
  Scale-skew breaks below the requant rounding step need value-exact oracles, not probes;
  the bit-identity kernel-check arms are the teeth there: extend those when adding arms.
- **A subtle *uniform* K-scale skew (x1.001) on an arm with no kernel-check bit-identity
  twin was caught by NO tier** (`break-fa-decode-*`: attention renormalizes softmax, so a
  uniform score scale barely moves greedy tokens at short depth). Wide-margin numeric skews
  are a tier-2/battery class; fast-gate's teeth are structural breaks (index/lane/element
  mixups), which it demonstrably catches.
- **Sampled serving path** (temp>0 end-to-end): `samp` oracles the kernels, but no probe
  runs a sampled generation stream; distribution-level drift stays a battery/eval concern.

## The perf stage's tok/s verdict is a tripwire, not evidence

`tools/local-ci.sh --perf` verdicts each cell against a **rolling median of that cell's prior
rows**, rows measured on earlier days. A tok/s FAIL there is therefore a *cross-day*
comparison, exactly the form [`research/benchmarks.md`](../research/benchmarks.md) forbids as
proof: clock, thermal and power state drift under numerator and denominator alike. It answers
"did something move?", never "did this commit regress?", and it is **not** by itself a
merge/tag blocker.

When it goes red, settle it and record the settle:

1. build the last-green commit's binary for that cell,
2. run the cell **interleaved A/B/A/B, N≥5 each, in ONE thermal window under one exclusive
   lock hold** (harness: `research/v071-prep-20260806/battery-logs/perf-ab.sh`),
3. compare medians *within that window only*.

The v0.71.0 release battery is the worked example: 10/10 cells reported FAIL at −8.31% to
−24.75% with correctness fully green, and the interleaved A/B measured the **last-green
baseline binary at 37.87 tok/s against the candidate's 37.87 (+0.00%)**: the drop was machine
state, and no code had regressed. A uniform drop across many unrelated cells with correctness
green is that signature, not many simultaneous regressions.

Two holes in this stage were closed by that same red (2026-08-06):

- **The reps now run under `/tmp/memra-5090.lock`.** `window_free_now()` samples only *between*
  reps, so a neighbor lane that started and finished inside a rep was invisible, and its
  poisoned rows still recorded `window_clean:true`. Every other GPU consumer in the repo
  already took the lock; the one stage whose entire output is a timing number did not.
- **A tok/s FAIL now prints the settle protocol** instead of only a percentage, so the next
  reader does not have to re-derive why the number alone cannot convict a commit.

## What fast-gate does NOT cover

- **Serving surface** (`crates/memra-server/`): run `tools/serve-smoke.sh` (fast-gate prints
  the pointer when the diff touches it) and, for any diff in `health.rs`, the worker
  supervisor, the boot calibration probe or the drain path, `tools/health-fault-gate.sh`
  (memra#524; in `local-ci.sh` since 2026-09-22, `MEMRA_CI_HEALTH_FAULT=0` skips): seven boots
  of the real server asserting `/readyz`, `/health` and the request path across readiness
  before and after the boot probe (`phase=warming` on the respawn window), the three
  probe-skipped boots (recorded as DOCUMENTED, ready without warmup), a `MEMRA_PANIC_AFTER`
  worker panic and respawn, a latched gpu-watch fault, and a SIGTERM drain with a stream open;
  `HFG_ARMS=d,f` runs a subset, `HFG_OUT` names the receipt dir. Three more serving gates exist
  and are **not** wired into fast-gate or `local-ci.sh`: invoke them by hand for any diff in
  their area:
  - `tools/serve-st-gate.sh [st_dir]`: an HF **safetensors dir** served end-to-end: `/models`
    lists it, `/v1/chat/completions` returns coherent text through the checkpoint's *own* chat
    template, and the CLI-vs-server exactness contract (same checkpoint, same prompt, same
    template, greedy → `run-gen`'s ST-dir tokenwise branch and the server's batched-prime +
    serving decode must produce IDENTICAL id streams). Runs `MEMRA_SERVE_SPEC=0` because with
    spec on a Token event carries one id per flush.
  - `tools/apikeys-gate.sh [model] [out_dir]`: auth refusals (401/403), single-key
    back-compat, the two-tenant **cache-isolation proof** via a cache-hit oracle, per-tenant
    rate-limit headers, the batch-class lane law, and hot revoke.
  - `tools/serve-stress-gate.sh [--teeth] [model [draft [n_clients]]]`: the c=64 concurrency
    contract (this one *is* in local-ci.sh; `MEMRA_CI_STRESS=0` skips it).
- **Acceptance drift**: invisible to every *exactness* gate by construction (decode and verify
  shift together, so spec still equals plain). Two things catch it, and they are not
  interchangeable: the tier-2 perf battery's per-cell acceptance verdicts (a rolling-median
  tripwire), and `accept-gate` (an **exact-integer assertion** against a pinned reference at the
  production serve config, tier 2 + `kind=cmd`). Acceptance is a ratio and therefore
  clock-independent: an acceptance FAIL is real evidence, unlike a tok/s FAIL (below).
  fast-gate's tier-1 probes still do not see it: a spec-pipeline or NVFP4-prefill diff maps
  `accept`, but running it costs a server boot, so it lands at tier 2.
- **H100/sm_90a lane (retired from CI 2026-09-02)**: `tools/validate-h100.sh` was deleted with
  the sm_89/sm_90a arch retirement — the Hopper lane no longer has a standing battery. Its
  gates were all rehomed: kernel-check config pins and both `decode-batch-gate` modes already
  ran in `tools/local-ci.sh`, and the graph-lane exactness gates (`decode-dc-gate`,
  `graph-decode-gate`, `graph-session-gate` — the last of which guards the sm_120a serving
  GraphSession path, not a Hopper artifact) moved into `local-ci.sh` in the same retirement
  (PR #73 review caught them briefly orphaned). The reason they lived inside a battery is
  still law 3: `graph-decode-gate` "rotted OUTSIDE this battery for weeks" (an emission
  off-by-one in the gate masqueraded as 171/256 stream corruption). A Hopper
  source build (`MEMRA_CUDA_ARCH=90a`) still compiles and stays stub-ABI-guarded.
- **Cross-model blast radius**: tier 1 probes one model per kernel class; the full per-model
  matrix runs at tier 2.
- **Multi-GPU (PP-N) exactness**: needs 2+ cards, so it is neither in fast-gate nor in
  `local-ci.sh`: it runs on the multi-card box for any diff touching `pp.rs`, the stage-split
  dispatch, or a decode path a split reaches. See below.

## Box health before measurement (`tools/box-health.sh`)

Run it FIRST on every box window, before the first timed arm:

```
bash tools/box-health.sh [OUTDIR]     # exit 0 = fit to measure; exit 1 = do not open the window
```

It is not a memra gate and it proves nothing about our code. It answers one question, *is
this machine fit to be measured on?*, and every check in it is a documented case of a box
reporting 100% utilisation with clean logs while delivering a fraction of its capability:
a persistent power cap 400 W of 600 W (25.3% of dense prefill, silently); the false-600W
~600 MHz degradation (power at cap + clocks under 1 GHz + temp under 50 °C, and **never flash
VBIOS in-fleet**); a PCIe link negotiated at Gen2 x16 that ran 3.5 hours of production
undetected; a 256 MB BAR1; an out-of-range CPU affinity mask (25% of all-reduce bandwidth);
IOMMU translated mode (the stake is silent device memory corruption, not throughput); ACS
ReqRedir forcing P2P through the root port; and P-state normalization before timing.

An idle P8 card may legitimately report PCIe Gen1 while its link is power-managed down. The
script records that state but does not confuse it with the live Gen2 incident above: section 8's
peer-read ladder wakes every peer-capable card, then section 8b immediately requires the active
link to reach its maximum generation and width. A below-max reading outside P8 remains an immediate
hard failure. On a one-card host the active generation can remain unclassified and is emitted as a
warning for the measured workload's telemetry to close.

Section 8 is the one that cannot be replaced by `nvidia-smi`: it builds and runs
`tools/peer-read-probe.cu`, a self-contained `simpleP2P`-class **kernel** peer dereference
(`nvcc -O2 -arch=${MEMRA_CUDA_ARCH:-sm_120} -o peer-read-probe tools/peer-read-probe.cu`).
`nvidia-smi topo -p2p r` can report OK and `cudaMemcpy` can look healthy while the driver
stages SM-issued peer access through system memory; only a kernel peer read catches it.
Exit codes: 0 = bytes validated both directions; 2 = **wrong bytes** (a fused pull collective
is blocked on this box); 4 = fewer than two devices (expected on the single-card rig);
5 = no peer-capable pair (place the TP group inside a peer island). A missing `nvcc` is a
HARD-FAIL, not a skip: the one check that matters most is the one easiest to silently lose.

`-p2p a` reporting `NS` on SM120 is EXPECTED, not a fault: there are no native peer atomics on
these pairs, and `tp_transport`'s peer-pull arm is atomics-free by design.

Deliberately absent: `ncu`. Profiling every rank deadlocks (the profiler serialises the
observed kernel while its peers wait), and any metric set needing more than one pass deadlocks
the same way.

## Multi-GPU (PP-N) exactness gates: run on the multi-card box

These are not in `tools/local-ci.sh`: the single owned rig has one GPU, so a green local
battery says nothing about them. Any change to `pp.rs`, the stage-split dispatch, or a decode
path a split walks needs these re-run on a 2+ card box.

| gate | invocation | what it proves |
|---|---|---|
| `ppn-gate` | `MEMRA_PP_DEVICES=0,1 ppn-gate <model.gguf> [stages=2] [P=16] [N=32]` | the eager stage-split decode (`decode_step_h_ppn`) is bit-identical to the unsplit walk, serial and pipelined arms |
| `decode-batch-gate --mode pp` | `decode-batch-gate <model.gguf> --mode pp [--batch 1,4,8] [--stages N] [--reps R]` | the **batched** stage split (`decode_step_batch_ppn`) is bit-identical per row per step. Honours `MEMRA_PP_DEVICES` / `MEMRA_PP_SPLITS` / `MEMRA_PP_SHARD` from the caller |
| PP-3/PP-4 decode wavefront | `MEMRA_PP_WAVE=1 MEMRA_PP_OVERLAP=1 MEMRA_PP_DEVICES=0,1,2[,3] decode-batch-gate <model> --mode pp --stages 3\|4 --batch 1,2,4,8,16,24[,32] --reps 5` | serial-wave vs live-wave logit identity, original row order, cache advancement, last-stage epilogue, and non-vacuous tick/cell/host-overlap counters |
| PP-3/PP-4 prime wavefront | `tools/prime-split-gate.sh <model> --stages 3\|4 --devices 0,1,2[,3] --chunks auto,513 --steps 8` | unsplit, serial split, and N-stage prompt-microchunk wavefront return identical logits/hidden/cache continuation while split and overlap counters advance |
| `decode-batch-gate --mode ppspec` | `decode-batch-gate <model.gguf> --mode ppspec [--ts 2,5,9] [--stages N] [--reps R]` | the **spec-verify** stage split (`decode_step_t_core_ppn`, T=K+1) is bit-identical per logit column per round, plus the `h_seed` column the drafter is re-seeded from. `--ts 2,5,9` = K=1,4,8 |
| `pp2-gate` | `MEMRA_PP_DEVICES=0,1 pp2-gate <model.gguf> [P=16] [N=32] [split=n_layers/2]` | the N=2 spelling of the eager gate, with M1 binary semantics. It **owns the door**: it resets `MEMRA_PP_STAGES`/`SPLITS` itself regardless of the caller's environment, while the increment-2 knobs (`STREAMS`/`OVERLAP`/`DEVICES`) deliberately pass through. Keep it alongside `ppn-gate`: it is the gate the gemma4 N=2 arm is validated by |
| `pp-transport-smoke` | `pp-transport-smoke` | the peer boundary transport primitive alone |

`MEMRA_PP_WAVE` is default-off until the full RTX PRO 6000 battery is committed. The engine gate
above is the numerical/ordering floor, not the serving verdict: also run vendor-default sampled
requests with distinct seeds/counters, top-k/top-p/min-p, penalties, per-row grammar masks,
concurrency, long context, prefix restore, disconnect/rollback, admission pressure, and an
interleaved N>=5 serial-vs-wave performance curve. Preserve `/metrics.pp_wave` engagement rows.

Three properties of these gates are load-bearing and were learned by measurement:

1. **`--reps` defaults to 2 because the class they must catch was a 35% FLAKE.** The
   shared-`Engine` scratch race (`fa_part_pool` / `argmax_partials` / `fa_vf16_scratch` are
   stable-pointer pools, single-stream-safe by design) surfaced as an intermittent failure on
   2026-08-02. One green replay is not evidence of absence: always run reps.
2. **The door must open BEFORE load**, because weight sharding is a load-time decision. The
   gates set `MEMRA_PP_STAGES` themselves for exactly this reason; a battery that opened it
   after load would be measuring the wrong placement.
3. **Two arms make these localizers rather than coin flips.** The `unsplit@ppncache` arm
   replays the unsplit walk over the *same* stage-owned caches, holding cache placement
   constant so only the walk varies: a red split arm then points at the stage split and
   nothing else. The `epilogue` arm runs mixed per-row metas and checks the lean
   `last_logits_dev` park through UVA from the primary context, the same read the server's
   retire path does.

Arm 4 of the pp battery deserves its own note: the explicit B=1 fast path's exactness bar is
bit-identity **to the eager split** (`decode_step_h`), not to the batched body, against which
it carries the m=1 fusion FP gap by design. Arms 1-3 cover the default generic program at B=1
and B>1 with `set_b1_fast(false)`; arm 4 explicitly opts into eager coverage and also asserts
per-step `pos` equality, so a double-advance cannot hide behind matching logits. Arm 4 is not
evidence that eager is safe for a session that can gain a peer.

Receipts: [`research/pp2-batch-20260806/`](../research/pp2-batch-20260806/),
[`research/pp2-spec-20260806/`](../research/pp2-spec-20260806/),
[`research/pp2-hardening-20260806/`](../research/pp2-hardening-20260806/) (the 20-arm
fail-closed guard battery), [`research/m2-pp8-20260802/`](../research/m2-pp8-20260802/) (N=2/4/8
on an 8xH100 box).

## Serve-path exactness: mode switches, and the law they are pinned under

`local-ci.sh` covers the serve surface's *shape* (`serve-smoke`, `serve-st-gate`,
`serve-stress-gate`, `apikeys-gate`). What it does not cover is a session **changing execution
mode mid-stream**, which the concurrency-gated spec scheduler does by design
(`MEMRA_SPEC_GATE`, default ON, see [SERVING.md](SERVING.md) and [FLAGS.md §1](FLAGS.md)): a
demoted session's stream must be byte-identical to one batched from the start. That harness is
`research/spec-gate-20260806/exactness.py` (5 arms, one server boot per arm, greedy, 768-token
budget), and it is **not** wired into any battery. Re-run it by hand for any change to the
scheduler's phase order, the demotion handoff, or `Session.device_next`.

Two things about it are load-bearing, and both are the kind of thing a later reader re-breaks:

1. **Load-triggered demotion can never be a clean exactness test**, so the harness does not try.
   Both the arrival timing and the batch composition are nondeterministic, and, with the spec
   path OFF and none of the scheduler's code involved, the same greedy request historically
   diverged between a solo run and one sharing batched decode with concurrent rows. Two runs put
   the first divergence at byte **2379** and byte **1347**. The later iso-gap lane held the
   program family fixed and proved depth staggering, FA ladder rungs, and batched-linear tier
   selection innocent: the carrier was the default solo eager/GraphSession program changing to
   the generic batched program when a peer arrived. Those solo-only programs are now OFF by
   default, and `decode-batch-gate` config mode pins B=1-vs-B=N identity. The demotion handoff is
   still pinned at a
   **fixed batch shape** through a diagnostics-only door, `MEMRA_SPEC_DEMOTE_AT=N`, which forces
   demotion at a pinned generated-token count with no load at all, holding B=1 across the
   boundary. Never set it in production. The generalizable rule: **when the property under test
   sits inside a nondeterministic configuration, pin the configuration and force the transition;
   do not try to provoke it under load and diff the result.**
2. **Three of its own arms exist to stop a false green**, each recorded because it produced one:
   a *vacuous pass* (q9 is a thinking model: every token lands in `message.reasoning` and
   `content` is empty, so the first version compared 0 bytes on three arms and called it PASS;
   now it compares both fields and hard-fails a near-empty stream), a *wrong session* (load fired
   after the target had finished, so a background filler took the spec slot; the verdict now
   requires the demote line to prove it fired on the target), and a *wrong reference*, whose
   discriminator arm `REF_LOAD` is what surfaced the batch-vs-solo finding above.

## Gate integrity: the shapes that report green while verifying nothing

A gate can fail in a way no red ever shows: it runs, prints a pass, and asserted nothing. The
2026-08-19 audit found five of ours in that state. The four recurring shapes, so a new gate is
written against them:

**A · a fallback that fires on missing information and then makes a confident claim.**
`stat -c %Y` on a file git does not preserve mtimes for; a diff whose failure is swallowed and
reads as "no files changed"; `[ -f "$DRAFT" ] && MODELSPEC=…`, where an absent drafter boots a
plain server and the spec-only assertion passes vacuously.

**B · an assertion that cannot fail.** `all()` over an empty generator is `True`.
`rg -q 'env::var\(' crates && rg -q "$name" docs/FLAGS.md`: the first conjunct never mentions
`$name`, so the test collapses to the second. `grep -q "ALL GREEN"` against a banner that reads
`ALL GREEN (N cells, M skipped)`. A canary that treats ANY nonzero exit as teeth, where 75 is
`flock -w` reporting the lock was busy and the gate never ran.

**C · a branch, or a whole gate, with no caller.** Until 2026-08-19 nothing anywhere ran
`cargo test`: 51 files across 10 crates carry `#[test]`, and `memra-server` (13 files) and
`memra-tokenizer` (5) had no caller in any gate or workflow. `.github/workflows/ci.yml` now runs
the CUDA-free crate suites, memra-server's, and the parity-geometry rule's; `tools/validate-h100.sh`
(since deleted with the Hopper lane, 2026-09-02) was the first place `memra-engine --lib` ran as
a real verdict instead of `| tail -1`.

**D · fail-open.** A counted-then-dropped verdict term; a missing golden that returns SKIP with
the failure counter untouched.

Two mechanical rules that follow:

- **A suite is not green if it did not run.** Assert `test result: ok.`, a nonzero `passed`, and
  `0 filtered out`. A name filter (`cargo test -p X somename`) prints a green
  `0 passed; N filtered out` the day the name moves.
- **A skip is not a pass.** Count skips and compare against a NAMED budget, so raising it leaves
  a trace in the run's own output.

### A `#[test]` that skips is the same shape, and libtest hides it better

```rust
if !ckpt.exists() { eprintln!("SKIP: ckpt/twin absent"); return; }   // the test PASSES
```

Twelve `#[test]` fns in memra-gguf are written like this, and a hosted runner has no checkpoints,
so `cargo test -p memra-gguf --lib` reports `90 passed` whether or not one model-backed assertion
ran, including `nv27b_twin_parity`, where the `n_rot` rotary-width geometry check lands. It is
`ALL GREEN (N cells, M skipped)` in Rust, and it stayed invisible until the suite acquired a
caller.

The mechanism is `tools/skip-census.py` plus the `tools/skip-census.tsv` manifest, and it is
deliberately harness-level (the same place `tools/validate-h100.sh` used to gate kernel-check's
skip count, before the Hopper battery was retired) rather than in the tests:

- `verify`: the STATIC census (every `#[test]` in the crate that prints SKIP and returns) is
  compared with the manifest in BOTH directions. An undeclared test fails (it would be born
  invisible); a stale row fails (it inflates the budget and silently permits a different skip).
- `run -- cargo test …`: asserts the suite's own verdict FIRST (exit status, every
  `test result: ok.`, nothing filtered, not vacuous), then counts the SKIPs against a NAMED
  budget, **default 0**. Uses `--test-threads=1 --nocapture`, which is load-bearing: parallel
  libtest interleaves un-attributed output, so a SKIP cannot be tied to the test that emitted it.
- `report <file> --expect N`: the same census for shell gates, which append to
  `$MEMRA_SKIP_CENSUS`. A **missing** file fails: absent is ambiguous between "nothing skipped"
  and "the census was never wired", and the second reads as the first.

Where the budgets live and why they differ: `tools/local-ci.sh` runs the memra-gguf census
WITH artifacts present at budget 10 (rehomed 2026-09-02 from the deleted Hopper battery;
measured on the rig: 212 passed, 10 skipped — ckpt/twin, Hy3-repack and iq3s artifacts not
staged there) and enforces kernel-check's skipped cells against a budget of 11 (missing-model
cells plus the sigrouter env capture). `local-ci.sh` keeps the same discipline by
requiring kernel-check's full verdict shape.
`.github/workflows/ci.yml` uses **12** because a hosted runner has no `/data` at all: twelve is
the number of model-backed assertions CI is blind to, stated out loud instead of hidden inside a
green `90 passed`. If it grows, CI reds.

A developer without artifacts is not blocked: raise the budget, or set
`MEMRA_ARCH_GATE_ALLOW_SKIP=1` for the generated serving gates. The escape hatch is explicit and
printed; it is never a silent pass.

### Parity gates: geometry first, then bytes

**An elementwise value compare implicitly catches tensor LAYOUT errors (reordered values stop
matching), and catches NOTHING about geometry that lives in config scalars and never touches the
bytes.** `n_rot`, `rope_theta`, `sliding_window`, head-dim splits, tap sets. On 2026-08-19 a
rotary width wrong by 4x survived a byte-parity gate that ran every single run at
`maxdiff=0.0e0`, green on 13/13 tensors.

Two corollaries, both learned from live code:

1. **Never infer geometry FROM the reference bytes.** `let ctx = dump.len() / (n_taps * hidden)`
   has no remainder check and cannot disagree with anything; a reference regenerated under a
   different `hidden` or tap set silently becomes a comparison against a reinterpreted buffer.
2. **A product is not a shape.** `assert_eq!(ne.len(), block_size * hidden)` is blind to every
   factorisation that multiplies out the same, `8 × 5 × 2560 == 8 × 10 × 1280`.

So: the producer writes a geometry manifest, and the gate asserts every config scalar against the
checkpoint under test BEFORE any value compare, refusing when the manifest is absent. The house
implementation is `crates/memra-engine/src/parity_geometry.rs` (dependency-free, 11 tests, run in
CI); the shape to copy for tensor-level gates is `nv27b_probe` / `m3_probe` in memra-gguf, which
assert `ne` **and** `ggml_type` per tensor first.

### Serving gates bind ports; ports are shared

`tools/port-guard.sh`, pre-flight occupancy refusal plus a post-boot pid-ownership assertion,
sourced by every gate that boots a server. An occupied port is a hard abort, never a wait: a
foreign responder answers `/health` instantly, the boot wait never happens, and the gate measures
someone else's model and pins it. That is a receipt, not a hypothesis (`accept-gate.sh:143`).

### GENERATED gates: fix the template, not the copies

`tools/generate-arch-gates.py` renders the standing per-architecture gates, and until 2026-08-20
it minted the defect shapes above into every gate it produced: `exit 0` on a missing artifact, a
drafter that silently degraded to a plain boot, a reference guard that passed on
`{"reasoning": null, "content": null}`, a canary satisfied by any nonzero exit, and a hardcoded
port of **8094**, `step35-b2-geometry-gate.sh`'s, so every generated gate collided with that
gate and with each other. Fixing eight hand-written gates and leaving the template is fixing the
copies.

What the template guarantees now is documented for gate authors in
[`docs/ONBOARDING.md`](ONBOARDING.md) section 6. The mechanical parts: generated ports come from
a reserved band (**18300-18399**) validated against a computed census of the tree's bound ports;
each gate gets a slug-derived `MEMRA_<SLUG>_B2GEO_PORT`; the gate sources `tools/port-guard.sh`
and refuses to run if it is missing; a skip exits **77**; and the assertion count is asserted.

The rule this generalises to: **a defect in a generator is every future artifact's defect, so it
outranks any single gate on the list.** And a template fix without a generated-artifact test is
just a diff: `tools/test_gate_template_integrity.sh` runs a gate the generator actually emitted
against a stub server on a real loopback port.

### Four rules for anyone writing a gate or a fixture (2026-08-23)

All four were learned the same day, each from being wrong in a way that still looked green.

**A wiring assertion must anchor on the invocation, in comment-stripped text.** *House law.* The
arm that answers "is this gate still called?" fails uniquely badly: it passes by matching the
rationale **comment** that names the script, so the gate reads as wired while being unwired. This
happened **twice in one day in two different lanes**: `tools/test_flags_guard.sh`'s
`grep tools/check-flags.sh` matched the hook's own comment (which names the script three times),
and `tools/test_public_boundary.py`'s `ci.yml` search matched the step's rationale comment. Both
were caught only by unwiring the gate and watching the test go red, which is the other half of the
rule: **an assertion nobody has watched fail is not evidence.** Anchor on the command form
(`elif ! flags_out=$(tools/check-flags.sh`), and strip `#` lines before searching. Strip by hand
rather than with a YAML/TOML parser: a wiring assertion that can fail for a missing dependency is
one that gets deleted.

**An exceptions list needs its own expiry check, or it silently absorbs the regression it was
never granted for; and when every entry is dead, DELETE THE FILE rather than maintain a checker
for nothing.** *House law.* Three instances in one night: `verify-allowlist` with no automated
caller anywhere; the public-boundary allowlist before rule-scoping, where a grandfather granted for
`production_endpoint` permanently absorbed a capacity-block id in the same bytes; and the flags
census's own 75-name baseline, where **all 75 entries had since been documented, so every exemption
was dead while still able to absorb a live one** (probed, not argued: deleting `MEMRA_SPEC`'s
`docs/FLAGS.md` row exited **0**). The sharp part is the shape: **not invisible, non-blocking.** All
75 were still printed, under `uncovered runtime names (known and new)`, and only the "new" half
failed. A printed non-fatal line is the same shape as the `local-ci.sh` WARNING that let three
commits red main the same day, which is exactly why it escaped notice for weeks. When you delete
such a file, **assert its absence** so a re-grant is a deliberate act, and refuse a relocating env
rather than ignoring it: a no-op env is how a caller believes it is grandfathering when the gate
has already stopped honouring it.

**A fixture must pin every environment fact it depends on, and assert that its own setup did
something.** `test_flags_guard.sh` arm 8 went green on the rig and red in CI at `5ffa711c32`, and
neither result was about the code: unpinned `MEMRA_MODELS_DIR` made its precondition push *refuse*
on the rig (models dir present, no `perf-ci.jsonl`) and *succeed* on the runner (no models dir).
The assertion passed either way, for opposite reasons, and the divergence surfaced one step
later, where the runner's successful precondition push had already advanced the bare origin so the
next `git push` sent nothing, printed `Everything up-to-date`, **ran no hook at all**, and two arms
failed complaining about the hook. Two rules fall out. Pin the fact (`MEMRA_MODELS_DIR` now points
at a path that cannot exist, so both machines take one branch). And assert the setup worked: a
no-op `git push` exits 0, so any arm reading hook output after one is measuring silence and will
blame the code under test. **An arm that can pass for two different reasons is two arms, and you
only wrote assertions for one.**

**Estimate CI cost from a CI run, not a local one.** The allowlist-drift gate was sized at ~5% of
a run from the rig's 50.6–59.4 s and measured **7.7%** in CI (118 s of 1542 s): the hosted runner
is roughly 2× slower at that work. The number was still fine, and that is the point: a flattering
method survives until the run it approves is not fine. Quote the CI measurement, or say the number
is a rig extrapolation.

### Fixtures

`python3 tools/test_release_battery_coverage.py` runs the release battery against controlled
CPU stand-ins for its GPU binaries. It verifies environment sanitation, the exact K=1..8
greedy verdict set, required kernel manifests, named skips and their fixed release budget,
and nonzero-exit rejection. Narrowed, duplicated, missing, failed, and malformed results must
fail while complete coverage passes. CI runs this fixture; it is control-flow evidence and
does not qualify any model or GPU kernel.

`tools/test_check_flags.sh` (flags census), `tools/test_flags_guard.sh` (the PRE-PUSH census arm
**and both of the hook's escape hatches**: a real `git push` through the real hook into a bare
local origin, so the wiring is exercised rather than grepped; arms 5 and 8 assert that
`MEMRA_SKIP_FLAGS_CENSUS` and `MEMRA_SKIP_PERF_CI` each print *and* log, and arm 8 first proves its
own precondition that the engine file reaches the perf gate at all. Its stub list is deliberately
maintained rather than mocked away: the three releasability censuses that landed in `7f342b42b6`
announced themselves here within minutes, because a fixture driving the real hook notices new
arms), `tools/test_gate_integrity_r2.sh` (round 2's fixes,
one forced failure per fix), `tools/test_gate_template_integrity.sh` (round 3: 51 assertions on a
GENERATED gate) and `python3 -m unittest tools.test_generate_arch_gates`, all in `ci.yml`. Each
ends with an assertion-COUNT check and fails as a BROKEN FIXTURE when it records fewer assertions
than declared: a fixture that quietly stops running arms prints the same summary as one that
passes. Point `MEMRA_GATE_SRC_DIR` at another checkout to score a fixture against that tree's
copies, and when reading such a score, say which passes are non-decisive: arms that
exercise files the other tree does not have are not evidence that anything was fine.

`tools/test_public_boundary.py`'s `VerifyAllowlistTests` is the same shape for the **allowlist
drift** gate. `check-public-boundary.py verify-allowlist` had no automated caller anywhere until
2026-08-23, not `ci.yml`, not `boundary-refs.yml`, not the pre-push hook, and no test touched
`cmd_verify`; the only evidence it had ever passed was two transcript lines in a research doc. It
is the half of the boundary policy that fails silently, because `check` asks whether a tracked
blob is an ungrandfathered violation and never whether a grant still describes anything real. It
is now its own `ci.yml` step, deliberately **not** in pre-push: measured at 50.6–59.4 s (it re-runs
the whole tree evaluation) against a 0.69 s hook, which is the latency that gets hooks disabled.
The arms corrupt an entry and watch it go red, restore it and watch it go green, delete a pinned
file, expire one rule of a two-rule grant, and assert the `ci.yml` invocation with comment lines
stripped: a plain substring search is satisfied by the step's own rationale comment.

`tools/test_cpu_expert_prefetch.sh [LOG_DIR] [SOURCE_DIR] [MIRROR_DIR]` (CPU-only, in `ci.yml`'s
engine-tests job after `cpu_native_check`) is the accounting fixture for the CPU expert companion's
speculative prefetch (memra#586). It compiles `tools/memra_cpu_expert_prefetch_test.cpp`, which
includes the production translation unit (`tools/memra_cpu_experts.cpp`, the shm test's pattern) so
it reads the SIGNED in-flight counter `prefetch_inflight()` directly and never the public stats
function's clamp to zero, and renames the unit's one `pread` call so a cell can HOLD a read at its
entry and FAIL one on demand. A worker that enters a held read has finished every job it took before
it, so "every alternate half held, queue empty" is the deterministic point at which the projections'
charge is read. Three cells: `barrier` (three mirrored projections under three I/O workers and a cap
of three: the counter must still read 3, a fourth must be refused, the drain must reach exactly 0
and admit the retry), `failure` (one mirrored projection whose alternate half fails with `EIO`, a
single-job sentinel queued behind it on one worker: the charge is released once, the annex never
publishes the failed buffer, the retry lands) and `parity` (the non-mirrored buffered control). It
needs a source fixture and a byte-identical mirror on two different filesystems (the companion's
mirror map refuses one device; defaults `<repo>/target` and `/dev/shm`), both opened `O_DIRECT`; it
refuses to run where either is unavailable rather than skipping. Before the repair the first two
cells read `inflight_signed=0` where 3 and 1 were charged (`research/spill-c-20260919/
day17-local/prefetch-test-red.log`); every cell runs even after a failure and the summary line
carries the count (`cpu expert prefetch accounting tests: N FAILURE(S)` or `ALL GREEN`). Two more
cells cover the submit side of the same invariant. `submit-throw`: the second projection names an fd
that is not open, so the submit loop throws after the first projection took its charge and annex
claim and before any job reached the pool; the call must return -1 and release both (a guard in
`memra_cpu_expert_prefetch_v2`, disarmed once the jobs are handed to the pool). `submit-throw-
claim`: under O_DIRECT with a mirror map that does not list the source, the loop takes the
projection's annex claim and then mirror resolve throws for that same projection, before a runtime
or a charge exists; the key must be claimable again afterwards (the guard tracks claims the moment
they are taken).

## Receipts

Timings, the deliberate-break catch demonstrations (diffs, consoles, per-probe raw logs),
and the depth-determinism sweeps: [`research/fast-gate-20260802/`](../research/fast-gate-20260802/).
The serve-path mode-switch exactness harness and its verdicts:
[`research/spec-gate-20260806/`](../research/spec-gate-20260806/) (`RESULTS.md` §2, `exactness.py`).

### Cooperative prime state (default-OFF diagnostic)

`run-spec <model.gguf> --prime-walker-check`, with `MEMRA_PROMPT_FILE` naming a
multi-chunk real prompt, first compares the MTP walker against ordinary unyielded
`prime_cache` segments. A separate cache advances between chunks. Require
`[prime-walker-check] PASS`, nonzero yields, bit-identical final logits and the
same capture hash (positions, KV lengths, logits, hidden anchor, conv and SSM
snapshot storage). The normal run-spec comparison then runs too. This CLI switch
is off unless explicitly passed; it is not a serving flag. Supported diagnostic
shape: single-device GDN MTP, with at least two legal prime segments.

For DFlash, retain the `MEMRA_ALLOC_TRACE=1` oracle from
`research/dflash-tap-storage-20260908`; `[dflash-oracle]` now also records
`boundary_state`, hashing actual GDN snapshot storage. Match tapes, features,
positions, final logits, boundary logits and capture state between OFF/ON.
`MEMRA_TICK_TRACE=1` records actual chunk and finalization wall in both arms.
Neither diagnostic is a timed performance cell. Route receipts live under
`research/prefill-fairness-20260908/`.


### HC24 default engagement, owner accepted 2026-09-09

Run `dsv4_hc_dot_split_gate <model-dir> <source.txt> <new-output-dir> --defaults`
in separate fresh processes with `MEMRA_DSV4_HC_DOT_SPLIT` unset and set to `0`.
Use the existing HC gate's required TP/EP f32x environment and pinned source tape.
The defaults path never calls the HC selector. Each mode checks 256 sampled
steps against eager within its own numeric class, all three forward and commit
censuses, eight transactional refusals, then five sanity rows on a fresh retained
graph with the first capture timed. Sanity rows must match the eager qualification
identity. ON requires 86 HC partial and 86 reducer nodes per rank/forward; OFF
and commit require zero. The dense-fast census is checked alongside HC. This gate does not assert identity between HC classes.

CPU policy tests in `dsv4_hc_dot_split_gate` execute the linked C++ selector in
fresh child processes for unset, 0, 1, 16, 8, 32, invalid and empty values, check
explicit gate rollback and refusal, and verify a new thread's environment policy.
`tools/test-dsv4-dense-control-policy.sh` exercises the actual exact-tail and
all five dense-TC drivers under unset, explicit S16 and zero before CUDA calls.


### Model-owned device admission and reclaim (#544)

`tools/qualify-model-device-memory.py` runs the named native ownership/memory stages
under an external exact per-card lease: one physical card for same-device owner coverage,
two for GLM peer state/reclaim and worker admission/pinned-source refill. Each stage
requires a source/binary-bound build receipt and preserves raw output, telemetry and
lease completion. See [protocol](../research/glm-tp-device-ownership-20260920/QUALIFICATION.md)
and [native results](../research/glm-tp-device-ownership-20260920/NATIVE-RESULTS.md).
The 2026-09-20 PRO 6000 run passed all three synthetic stages. GPU KDA/lazy-index-key
allocation coverage, full-checkpoint serving and performance remain pending; these are
not model-support or full release-battery receipts.

### Prefix eviction must credit admission and the driver (`tools/prefix-evict-reclaim-gate.py`)

memra#346, #445 and #523 item 4: the admission path evicts unleased prefix-cache entries when a
request does not fit (`[admit-oom] reclaim-on-defer`) and re-reads headroom in the same tick.
The planes drop as stream-ordered `cuMemFreeAsync` into the caching pool, so before the fix the
tick credited nothing (`effective free 69204MB -> 69204MB` on a 43.5 GB eviction) and a busy box
kept every later prefill deferred behind an entry that was already gone. The fix
(`settle_reclaimed_prefix_bytes`, worker.rs) fences the model-owned streams and trims each device
pool to `used + cached_before` before the re-read, printing one `[admit-oom] reclaim settle` line
per device in bytes.

```text
prefix-evict-reclaim-gate.py [--external-lock FD] --model <gguf> --bin <memra-server> --out <new-dir>
```

- Serving shape, one card, the real `memra-server`, two boots per cell: a calibration boot
  measures effective free after P1, E1 (the long prompt's entry bytes), E0 (the busy peer's
  own seed) and the two long prompts' admission costs; the measured boot starts a ballast
  process (one plain `cuMemAlloc` through `libcuda`, held for the boot) sized so that P2 is
  short by less than E1 + E0 beside the busy peer and fits after a true credit, then sends P2
  while the peer is still decoding so the idle-box `admission-drain` arm cannot mask the tick.
  No admission door is touched: with `MEMRA_SERVE_SPEC=0` the reserve is the static 1536 MiB
  floor (the boot line is asserted), so `required(P2) = cost(P2) + 1536 MiB` is known from the
  calibration. The ballast models the small card #445 was filed on.
- Assertions, all in bytes from the server's own lines and `/metrics`: V1 the reclaim-on-defer
  line credits at least E1; V2 P2 is admitted in that tick (no `VRAM defer` after the reclaim
  line, no `reject averted`, HTTP 200); V3 the settle line moves driver free and
  `trim_released_bytes` by at least E1 minus one 2 MiB granule with no `pool_retained_bytes`;
  V4 the greedy texts of the two boots hash identical (pressure changes admission, never tokens).
  The runner compares V4's digest across binaries (base vs fix) as the numeric-program receipt.
- Verdict line: `PREFIX-EVICT-RECLAIM: entry_bytes=... reclaim_credit_bytes=... -> PASS|FAIL`;
  exit 0 PASS, 1 FAIL, 2 `REFUSED: ...` (lock, port, busy-peer window, card too small). Red on
  `main` at `ea08bc7f8` and green on the fix, same card, same artifact, same prompts:
  [`research/spill-b-20260919/DAY13.md`](../research/spill-b-20260919/DAY13.md).
- Canonical rig lock only, held for the whole cell; under the collector,
  `tools/tier-battery.py --rig pro-single --external-lock --execute python3
  tools/prefix-evict-reclaim-gate.py --external-lock @COLLECTOR_LOCK_FD@ ...` (lead ruling 5).
  CPU arm: `worker::tests::reclaim_settle_returns_only_the_reclaims_gain` pins the keep
  arithmetic.

### A restored prompt is the cold prompt at every restore point (`tools/prefix-restore-identity-gate.py`)

memra#602 (`research/spill-b-20260919/DAY17.md`, probe E): the prompt-end seed was the one prefix-cache
capture left off the GDN prime grid, so a hit restored an off-grid entry and primed the suffix from
an off-grid call start, a second numeric program under the chunked GDN scan (the prime-grid law in
`docs/SERVING.md`); the 12,350-id turn 10 of the day-16 chain restored from on-grid entries (12,288,
12,320) reproduced the cold bytes and from off-grid entries (12,250, 12,300) produced the other
stream. Since 2026-09-21 the seed publishes at the grid-aligned boundary (`seed_capture_boundary`,
worker.rs), the same law the LCP and message-boundary captures already obeyed.

```text
prefix-restore-identity-gate.py [--external-lock FD] --model <gguf> --bin <memra-server> --out <new-dir> \
    [--budget-mib 1024] [--turns 10] [--start-tokens 11000] [--grow-tokens 150] \
    [--points 12288,12320,12200,12250,12300] [--grid 32] [--max-tokens 8]
```

- Serving shape, one card, two boots of the real `memra-server`, plain path (`MEMRA_SERVE_SPEC=0`),
  greedy `prompt_ids`, the twin gate's own id generators (the target is turn `--turns` of the
  day-16 chain byte for byte; 10 is the 12,350-id near-tie). The cache-off boot serves the target
  cold, text kept; the cache-on boot, per restore point p under its own `cache_salt`, seeds
  `target[:p]` cold and then sends the whole target (a restore plus a suffix prime).
- Clauses: V1 identity, every hit's completion text equals the cold completion's; V2 grid, per
  point the seed's `insert (seed): N tokens` has `N == capture_len(p)`, the hit's `cached_tokens`
  and `hit: N of M` line equal that N, N is a grid multiple, and every `[primeseg] call` of the
  hit has `grid_off=0`. Red on the pre-fix `main` for every off-grid point even where V1 holds by
  a near-tie that did not flip.
- Verdict line: `PREFIX-RESTORE-IDENTITY: target=12350 points=5 identical=K/5 grid_ok=M/5 grid=32
  p(seedS,restoredR,offO,suffixQ):yes|NO ... V1=.. V2=.. -> PASS|FAIL`; exit 0/1/2 as the twin
  gate. Receipts on both cards: [`research/spill-b-20260919/DAY18.md`](../research/spill-b-20260919/DAY18.md)
  and [`DAY19.md`](../research/spill-b-20260919/DAY19.md).
- Canonical rig lock only; under the collector, `tools/tier-battery.py --rig rtx5090 --external-lock
  --execute python3 tools/prefix-restore-identity-gate.py --external-lock @COLLECTOR_LOCK_FD@ ...`.
  CPU arms: `worker::tests::seed_capture_boundary_lands_on_the_grid_or_refuses` (lengths just above
  and below a grid line, the exact prompt-end line, the sub-floor refusal band, covered re-sends)
  and `seed_boundary_inside_prompt_is_a_stop_only_below_the_prompt_end`.

### The newest turn fits the prefix cache (`tools/prefix-newest-turn-fits-gate.py`)

memra#523 items 1 and 3: under the segmented policy that was the default until 2026-09-21 a newly
published entry could be its own capacity victim (probation held only the newcomer) or be refused
by the snapshot preflight (only probation counted as reclaimable), so a growing long-context
conversation ran `cold=1 restored=0` on every turn beside other tenants' promoted entries, and
the only trace was one once-announced `snapshot skipped` line. The day-14 fix made every unleased
byte reclaimable for the newest turn (protected LRU oldest first once probation was exhausted);
since #523 item 2 (day 15, below) the only policy is plain global LRU, the oldest unleased entry
first, and the newest turn is never its own victim because it is the global newest. The only
refusals are typed and in bytes:
`[prefix-cache] insert refused: entry N exceeds budget M (...)` and
`[prefix-cache] insert refused: entry N cannot fit beside L leased bytes (budget M, ...)`.
Victim selection and accounting only: captured and restored bytes are unchanged.
Exit rule (spill-b day 29, review round on #633): V3 compares the cache-on and cache-off boots'
device state, so it presumes both boots retain the same parked sessions after every send. The gate
parses every `[admit-oom] reclaim-on-defer` line per window; when the two boots' parked-session
releases differ, or a reclaim line does not parse into its released counts, V3 is undecided. A
failed V1, V2, V4, V5 or V6 is still the verdict `FAIL` (exit 1) with a premise note beside it; with
every other clause holding, a broken or unreadable premise is `REFUSED: V3 premise: ...` (exit 2),
naming the windows, the releases per boot and the card at each boot (driver free and compute-apps
sampled into the receipts), and the would-be verdict is kept in `summary.json` as
`verdict_under_broken_premise`. V3's clause, form and slack are unchanged.

```text
prefix-newest-turn-fits-gate.py [--external-lock FD] --model <gguf> --bin <memra-server> --out <new-dir> \
    [--budget-mib 1024] [--cohort-tokens 2800,3000,3200] [--turns 8] [--start-tokens 9200] [--grow-tokens 300] [--grid 32]
```

- Serving shape, one card, two boots of the real `memra-server` per cell, plain path
  (`MEMRA_SERVE_SPEC=0`), a small explicit prefix budget, the DEFAULT policy (the boot line must
  report `plain-LRU`; the gate never sets a policy). A calibration boot with the cache OFF
  (`MEMRA_PREFIX_CACHE_MB=0`) replays the identical request sequence first and records each turn's
  own retained footprint (the bytes a request of that length keeps after it retires with no cache
  activity; 1,111,666,004 B on the 9,200-token turn 1 of the day-14 shape, identical on both
  binaries). In the measured boot a second tenant (`cache_salt=cohort`) seeds three
  prompts twice each so the whole-entry hit makes them a reused, recently touched set. The growing
  tenant (`cache_salt=grow`) then replays an 8-turn conversation whose turn k+1 is turn k's
  `prompt_ids` plus 300 new ids. The pressure arithmetic is read from the server's own
  `insert` lines (a bytes(tokens) fit for the artifact) and the gate REFUSES unless the
  cohort is at most 80 % of the budget, cohort plus turn-1 entry exceed the budget, and every
  turn's entry fits the budget: the incident's shape scaled to a small budget.
- The capture law the gate is stated in (memra#602, fixed 2026-09-21; `docs/SERVING.md`, "The
  prompt-end seed obeys the same law"): every entry the cache publishes lands on the GDN prime
  grid, the plain seed and the spec session's `insert (spec-boundary)` alike (day 19), so a
  prompt of P tokens publishes `capture_len(P)` tokens (P when P is a multiple of
  `--grid`, otherwise the largest multiple below P whose remainder is at least PRIME_MIN_T = 16,
  never under the 64-token entry floor), a hit restores exactly that many, and `cached_tokens`
  reports the restored length, never P. `--grid` names the engine's `gdn_chunk_size()` (32); a
  server on another grid fails V6 loudly.
- Assertions, bytes from the server's `[prefix-cache]` lines and `/metrics`, every one a verdict:
  V1 every turn k >= 2 reports `usage.prompt_tokens_details.cached_tokens` EQUAL to the entry turn
  k-1 published (its `insert (seed): N tokens` line; until day 18 this read `>= prompt_tokens`,
  which only an off-grid prompt-end entry satisfies); V2 turn 1 publishes exactly one seed entry
  and every later turn hits exactly once, for the previous turn's published length, and publishes
  its own, with no `insert refused`, `snapshot skipped` or `seed REFUSED (grid)` line for the
  growing tenant; V3 after every turn the calibration boot's
  effective free (`cuda_driver_free_bytes + cuda_pool_cached_bytes`) equals the measured boot's
  effective free plus the cache's resident bytes, within 64 MiB (the cache costs exactly what it
  holds, so every evicted byte came back: #523 item 4 under the capacity shape, stated on states
  because per-turn deltas need aligned starting states and the cohort phase does not give them);
  V4 at least one eviction of a cohort entry (`ns "cohort"`) and no turn evicting the entry it
  just published (the room came from the cohort, never from the entry itself); V5 every turn's
  completion text equals the calibration boot's cold completion of the same prompt (the restored
  render is the cold render; turn 1 pins that the cache-on cold prime, stopped on its seed
  boundary, is the cache-off one); V6 every `insert (seed)` of the measured boot, cohort and
  turns, has exactly `capture_len(prompt_tokens)` tokens and every restored length is the
  published one and a grid multiple. The per-request server receipt (`[glm5-spec] route=` with
  `cold=`/`restored=`, or `[spec-k] ... cached= lcp=`), the `[primeseg] call start=.. grid_off=..`
  prime-call receipts (MEMRA_DEBUG_PRIMESEG=1, a documented diagnostic) and the count of off-grid
  call starts are recorded, not judged.
- Verdict line: `PREFIX-NEWEST-TURN-FITS: budget_bytes=... cohort_bytes=... turns=8 cold_turns_after_1=...
  cached_ok=N/7 lines_ok=N/8 ... identity_ok=N/8 grid_ok=N/M grid=32 off_grid_calls=N V1=.. V2=..
  V3=.. V4=.. V5=.. V6=.. -> PASS|FAIL`; exit 0 PASS, 1 FAIL, 2 `REFUSED: ...` (lock, port, shape,
  an unserved request). Red on `main` and green on the fix, same card, same artifact, same
  prompts: the capacity shape (V1..V4) in
  [`research/spill-b-20260919/DAY14.md`](../research/spill-b-20260919/DAY14.md); the day-16 lru
  shape (`--cohort-tokens 1250,1350,1450,1550 --turns 12 --start-tokens 11000 --grow-tokens
  150`, V5 and V6: `identity_ok=11/12 grid_ok=0/31 off_grid_calls=11 ... V5=FAIL V6=FAIL -> FAIL`
  on the pre-fix `main`, `identity_ok=12/12 grid_ok=31/31 off_grid_calls=0 ... -> PASS` on the
  fix, local RTX 5090; the fix also `-> PASS` on one RTX PRO 6000 Blackwell) in
  [`research/spill-b-20260919/DAY18.md`](../research/spill-b-20260919/DAY18.md) and, on the
  completed fix (both capture sites), in
  [`research/spill-b-20260919/DAY19.md`](../research/spill-b-20260919/DAY19.md). The
  `tools/spec-on-cache-hit-gate.sh qwen` cells state the same law in their own numbers since
  day 19 (an identical sampled repeat restores `capture_len(P)`, 64 of 106; the growth turns
  restore the republished render-stable boundary, 96 of 119; an on-grid `fc` pair built
  through `/v1/tokenize` keeps the whole-prompt full-cover shape and its `restore-full-cover`
  boundary site exercised); its identity law, spec-on text == spec-off text, is unchanged. Under the
  host tier contracts door (`MEMRA_KV_HOST_CONTRACTS=1`) the gate arms the host tier on both boots and
  asserts the door engaged (the door arm bullet under "Host tier contracts door" below; C day 27).
- Canonical rig lock only, held for the whole cell; under the collector,
  `tools/tier-battery.py --rig pro-single --external-lock --execute python3
  tools/prefix-newest-turn-fits-gate.py --external-lock @COLLECTOR_LOCK_FD@ ...` (lead ruling 5).
  CPU arms: `worker::tests::prefix_cache_newest_turn_fits_beside_a_reused_cohort` (the
  incident shape, 211 turns, cohort oldest first, never its own victim),
  `prefix_cache_oversized_insert_refuses_with_the_typed_line_and_evicts_nothing`,
  `prefix_cache_evicts_the_global_oldest_including_reused_entries`,
  `prefix_cache_newest_turn_beside_a_leased_predecessor_fits_or_refuses_loudly`, and for the
  refusal-line throttle (first line printed, identical repeats suppressed and counted in the skip
  counters, a changed shape printed again, every 64th identical repeat printed with its count)
  `prefix_refusal_announcer_prints_first_changed_and_every_nth_identical_refusal` and
  `prefix_cache_repeated_refusals_count_every_time_and_print_once`; for pressure relief (the kv-flex
  shed and the step-OOM reclaim select with the same `oldest_evictable` as the insert loop and the
  preflight: oldest unleased first, leases untouchable)
  `evict_to_bytes_never_takes_a_leased_entry_even_below_target` and
  `kv_flex_shed_reaches_the_floor_through_reused_entries_and_warns_only_when_all_is_leased`. The
  eviction contract is stated in `docs/SERVING.md` ("Eviction (plain global LRU ...)"): every
  unleased byte is reclaimable for the newest turn, oldest first; only leases are untouchable.
- **The policy decision (day 15, memra#523 item 2).** `tools/prefix-policy-ab.py` (source at
  `9466b891`; deleted with the door) replayed the incident's shape at a 2048 MiB budget on one RTX
  PRO 6000 Blackwell (600 W): four reused cohort tenants (74 % of the budget), one loop 27,300 to
  30,600 ids over 12 turns, cohort returns after every third loop turn and after the loop, 28
  requests per run, `AB-0, BA-0, ..., AB-4, BA-4` (A = `slru`, B = `lru`), 20 runs, one lock hold,
  one thermal window, plus a cache-off boot for the cold digests. Pre-registered rule: primary =
  total computed tokens, lower is better; secondary = the cohort tenants' `cached_tokens` on their
  returns; a policy wins only if better on the primary at every pair in both orders; digest
  identity is a precondition. Verdict, verbatim:
  `PREFIX-POLICY-AB: budget_bytes=2147483648 cohort_tenants=4 cohort_bytes=1589723136 turns=12 start_tokens=27300 grow=300 return_every=3 pairs_per_order=5 runs=20 requests_per_run=28 digests_identical=28/28 computed_tokens slru_median=132300 lru_median=122700 (N=10 each) pairs_slru_better=0/10 pairs_lru_better=10/10 ties=0/10 return_cached slru_median=0 lru_median=8700 loop_cold_after_1 slru_max=0 lru_max=0 refusals slru=0 lru=0 temp_c=46.0..50.0 power_limit_w=600.0 -> WINNER=lru`
  (10/10 pairs, 132,300 vs 122,700 computed tokens in every pair, digests 28/28 identical across
  the 20 runs and the cold boot). Landed as the naked default with the segmented arm deleted;
  receipts `research/spill-b-20260919/pro-single-day15/ab-full/`, replay `verify-day15.py`,
  decision `docs/decisions/PREFIX-CACHE-POLICY.md`. Target-card verdict; the local RTX 5090 replay
  is a follow-up.

## Generic spill / tiered KV (memra-tier)

The shared contract is `crates/memra-tier/src/contracts.rs`. The conformance schedules are
the public module `memra_tier::conformance` (`crates/memra-tier/src/conformance/mod.rs`,
re-exporting `revision_v11.rs`, `revision_v12.rs` and `revision_v13.rs`); the CPU bindings
that drive them live in `crates/memra-tier/tests/contracts/` (`transfer.rs`, `services.rs`,
`v12_bindings.rs`, `v13_bindings.rs`). The crate ships the suite as a public module on
purpose: native gates import the same functions instead of forking expected outcomes, and
the publish census refuses internal dev-deps. These are **CPU execution gates, not hardware
qualification**. Run the whole pair:

```sh
tools/portable-suites.sh          # cargo test -p memra-tier -p memra-kv -p memra-cli --offline --locked --no-fail-fast, through the skip census
tools/test_portable_suites.sh     # its teeth: planted tier/KV/CLI failures and an undeclared SKIP must red the wrapper
cargo check -p memra-tier -p memra-kv --offline --all-targets
cargo check -p memra-tier -p memra-kv --offline --all-targets --target x86_64-unknown-linux-gnu
cargo clippy -p memra-tier -p memra-kv --offline --all-targets --no-deps -- -D warnings
cargo fmt --all -- --check
python3 crates/memra-tier/tests/contracts/fixture_reference.py --check
python3 -m pytest -q crates/memra-tier/tests/battery/
bash tools/check-flags.sh
bash tools/docs-registry-census.sh
git diff --check
```

The tier suite covers contracts, storage, banks/rows, directed peer capacity and
placement; KV covers the hierarchy, materializers and single-governor scheduling; the
pytest line is the collector's own suite (`tools/tier-battery.py`, `tier-envelope.py`).

**Standing execution (memra #545, 2026-09-21).** Until that day no hosted or local gate ran
these suites: `ci.yml` compiled them (build, clippy) and executed other crates; `local-ci.sh`
ran server, engine and gguf. `tools/portable-suites.sh` is now the one wrapper both run
(`ci.yml` job `portable-suites`, `local-ci.sh`'s CPU chain): the three crates, no `--lib`
(memra-tier's six integration suites and its four compile-fail doctests are most of the tests),
`--no-fail-fast` (every binary runs, so a red names every failing suite), `--offline --locked`
against the committed lockfile (a lockfile that would need to change is a refusal), through
`tools/skip-census.py` (`verify` over the three crates, then
`run` at budget 0 and floor 300; 325 passed across 14 binaries on 2026-09-21, 22 s warm; 334
after that day's main merges). The static census scans `crates/<crate>/src` AND
`crates/<crate>/tests` (PR #592 review: it was src-only, so a skip born in an integration test
was never forced to be declared; the extension found memra-tokenizer's four `llama_parity`
skips, now declared in `tools/skip-census.tsv`). The raw cargo output is banked at
`target/portable-suites.log`. Its teeth are `tools/test_portable_suites.sh`, run by the same CI
job: a copy of the tree with a planted failing retirement/ownership test in `tests/contracts`,
a planted KV test and a planted onboarding-receipt test must red the wrapper with all three
targets named (arm 1); a planted `#[test]` that prints `SKIP` and returns, as a new file under
`crates/memra-cli/tests/` (arm 2a) and inside `src/` (arm 2b), must red the static census before
cargo runs; the wiring is asserted (arm 3). The copy builds into `target/portable-suites-teeth`, never the
tree's own target dir: cargo's metadata hash for a workspace member excludes its path and
`cp -a` keeps mtimes, so a shared target dir let the copy's planted `memra_cli` test binary be
reused by the next real run (found on the fixture's first run; the fix is the separate dir).
A green here is CPU execution of these suites, never GPU qualification.
**One entry point (day 14, 2026-09-21).** PR #590 landed `tools/ci-portable.sh`, a second runner
of the same three crates (`cargo test --release --locked`, no census, no floor, no teeth), a
second `ci.yml` job also named `portable-suites` and a second call in `local-ci.sh`; main
carried both and GitHub refused the workflow file (a duplicate mapping key runs zero jobs). The
fold: `tools/portable-suites.sh` is the one executor, `ci.yml` and `local-ci.sh` call it once
(`--locked` and `RUST_TEST_THREADS=8` folded in from #590), `tools/ci-portable.sh` only forwards
to it, and arm 3 of the teeth asserts exactly one `portable-suites` job, no live `cargo test`
on the three crates outside the wrapper, and a forward that runs no cargo. The workflow files
themselves are censused by `tools/check-workflow-keys.py` (a standard-library walker over
block-style YAML, its scope stated in its docstring; `yaml.safe_load` keeps the last duplicate
silently, and PyYAML is not assumed on every interpreter, so the hook cannot fail for a missing
dependency) in `tools/hooks/pre-push` (exit 1 is a duplicate, exit 2 is "cannot answer", both
refuse) and the `ci.yml` `gates` job, teeth `tools/test_workflow_keys.sh` including an arm that
shadows `yaml` with a package that raises `ImportError`. Record: `research/spill-d-20260919/DAY14.md`; the hosted CI map is
`docs/CI.md`.
Conformance schedules drive explicit completion/cancellation/retirement, original
item indices, namespace and epoch refusal, opaque bytes, accounting and borrowed
release. v1.2 adds owner/fence identities, logical-vs-framed completion bytes,
source installation and two distinct record materializations. v1.3 adds two additive
ownership schedules, `device_hand_back` (`take_device` returns the original allocation,
never a copy) and `transfer_source_retirement` (`retire_source`, default body
`Err(Error::Unsupported)`), with `WIRE_VERSION` unchanged at 1; see the
[interface decision](decisions/GENERIC-SPILL-INTERFACE-V1.md), the
[freeze ledger](../research/spill-lead-20260919/FREEZE.md) and the
[v1.3 freeze](../research/spill-lead-20260919/FREEZE-V1.3.md).
Day 11 (lead ruling 9, `conformance/recovery.rs`, unversioned, beside the frozen schedules,
which stay byte-identical) turned lane D's two fault-arm findings into contract rules with red
arms. Rule 1, cancelled restore: a restore (H2D) that is cancelled before its consumer event
completes must either hand the untouched host source back to the caller as a typed lease
(recoverable) or refuse the cancel with a typed error while the source is still intact;
consuming the source and then cancelling is forbidden by the rule. Seam:
`TransferEngine::recover_source` (default `Err(Unsupported)`); a cancelled H2D holds its source
for the caller (`retire`/`retire_source` answer `Busy` until recovered), `cancel` answers
`AlreadyReleased` once a source left the ticket. Schedules `transfer_cancel_recovers_source`
and `transfer_cancel_refused_after_source_consumed`; CPU bindings in `transfer.rs`
(`day11_*`, with the `legacy` red arm that drains a cancelled restore); native bindings in
`tier-transfer-gate` (below). Rule 2, required-resident continuation: a continuation (decode or
prime) over a cache with any suspended layer must be refused at `Cache::ensure_usable` with a
typed error naming the suspended layers, unless the caller restores first. Seam:
`memra_kv::Cache::suspend_layer` / `resume_layer` move a layer through the typed
`SuspendedLayers` register; `ensure_usable` returns `ContinuationRefused { path, layers }`
while it is non-empty; `decode_step_h` is unchanged and never sees a suspended layer. Schedule
`required_resident_continuation` over the `ContinuationGateFixture` trait; CPU bindings in
`resident_bindings.rs` (register passes, the taint-only gate is the red arm) and on the real
`Cache` in `memra-kv` (`continuation_gate_tests`). Write-up:
`research/spill-a-20260919/DAY11.md`.
Day 12 (memra#384, host tier tenant-share cap): at the configured `MEMRA_KV_HOST_TENANT_PCT` cap a
demotion evaporated before the D2H copy even while the pool had free space. The demote hook now
runs `HostPrefixCache::tenant_share_reclaim_plan` before the copy (pure: infeasible demotions
skip the PCIe trip, nothing evicted) and `reclaim_tenant_share` once the image is built and
bound, right before `insert`: the demoting tenant's OWN unleased host entries go oldest first
until the demotion fits its share, then the predicate is retried once. Another tenant's row is
never read, a leased entry (`IdentitySlot::leased`, a live identity lease) is skipped, the
exact-key twin is spared, and a row with no eligible space keeps the bounded refusal: today's
`demote evaporated at the tenant share cap before the D2H copy` line plus what the plan found,
nothing evicted. On the pageable tier a charge, digest or copy failure therefore costs the row
nothing; on the fixed arena (`MEMRA_GLM5_TP_KV_HOST=1`) the reclaim runs at reservation inside
`reserve_image`, before the copy, and a copy failure there is booked like a refused insert
(integ15 review of PR #597, both rounds): `prefix_host_tenant_reclaims_wasted` with a `tenant
share reclaim WASTED` line. `host_cache_tenant_share_reservation_evicts_nothing_without_an_arena_and_books_a_copy_failure_wasted`
pins the pageable no-op and the booking shape; the arena eviction itself needs a CUDA context
and is receipt-only. `insert`'s
authoritative gate, the cap and the D2H bytes are unchanged; `/metrics` gains
`prefix_host_tenant_reclaims` and `prefix_host_tenant_reclaims_wasted`. CPU cells
(`cargo test -p memra-server --offline tenant_share`):
`host_cache_tenant_share_reclaim_evicts_the_tenants_own_oldest_entries_only`,
`host_cache_tenant_share_reclaim_spares_the_twin_and_refuses_an_image_above_the_share`,
`host_cache_tenant_share_reclaim_skips_leased_entries_and_refuses_when_only_leased_remain`
(binds a real `IdentitySlot` and holds its lease),
`host_cache_tenant_share_plan_is_pure_and_a_reclaim_is_consumed_by_insert_or_booked_wasted`,
`tenant_share_reclaim_is_wired_into_the_demote_hook_and_the_metrics` (plan before the copy,
reclaim after bind and before insert, waste booked at every later exit);
the pre-existing `host_cache_tenant_share_cap_evaporates_one_tenant_and_still_demotes_the_other`
still pins `insert`'s gate. Target-card gate `tools/kv-host-tenant-reclaim-gate.sh <base|fix>`:
two keyring tenants, a 1024 MiB pool, a 38 percent share (two ~161 MB entries, not three) and a
two-entry device budget (`MEMRA_METRICS_TOKEN` for the scrape), so one tenant's third demotion
hits the cap with the pool two-thirds empty; the `base` arm (origin/main) asserts the evaporation
line and nothing of the tenant's own evicted, the `fix` arm the `[prefix-host] evict (tenant
share):` line naming that tenant followed by its `[prefix-host] demote:` and the promote of the
reclaimed admission, both arms the other tenant's promote, no LRU eviction, no device-tier
promote-insert skip and eight 200s; the replay `research/spill-a-20260919/verify-day12.py` adds
equal demote bytes across arms,
byte-identical texts and the other tenant's row equal. Evidence:
`research/spill-a-20260919/DAY12.md`, `pro-single-day12/` (one RTX PRO 6000 Blackwell at 600 W,
N=1, `executed-not-qualified`). The memra#385 arena-startup measurement plan and the BOX3 harness
receipt (`tools/pinned-host-reserve-bench.py`) are in
`research/spill-a-20260919/HOST-ARENA-STARTUP.md`; no default moves.
Linux cross-target check is compilation only, not Linux syscall execution.

### Native conformance: `tier-transfer-gate` (v1 through canonical v1.3)

`tier-transfer-gate` (`crates/memra-engine/src/bin/tier_transfer_gate.rs`) runs the shared
schedules from `memra_tier::conformance` against the native `CudaTransfers` backend and prints
one verdict line per schedule, only after the schedule function returns. It calls the frozen
canonical v1.3 functions directly (`v1::transfer_source_retirement(...)`,
`v1::device_hand_back(&mut fixture)`); the additive day-7 cases stay as extra lines and are not
the canonical evidence:

```text
PASS v1 transfer_cancel native CUDA
PASS v1.1 transfer_complete_cancel native CUDA
PASS v1.1 transfer_lifetime native events + injected observation loss + graph retention
PASS v1.1 transfer_zero_accept Unsupported NVMe preserves owned input
PASS v1.1 acceptance exhaustive native mixed batch; rejected sibling blocks publication
PASS v1.2 transfer_completion_bytes native CUDA; stale epochs, ready publication, take once, authentic consumer fence
PASS additive source retirement Busy while source consumer bound; host destination survives source release
PASS native governor zero after controlled drain
PASS v1.3 transfer_source_retirement native CUDA
PASS additive dropped destination retains backing and charge until graph retirement and acknowledgement
PASS v1.3 device_hand_back native CUDA
```

Day 11 added two lines, `PASS rule cancelled-restore-recovers-source native CUDA` and
`PASS rule cancel-refused-after-source-consumed native CUDA`, whose bindings could not run
natively that day (lane B held the card). The v1 `transfer_cancel`, v1.1
`transfer_complete_cancel`, `transfer_lifetime` and acceptance bindings now recover the
cancelled H2D source before retiring (the schedules themselves are unchanged). Lane D ran the
gate natively on day 12 (one RTX PRO 6000 Blackwell, collector-locked, N=1, gate source
`55f242e98`): the `conformance` cell printed both lines verbatim among its 13 `PASS` lines and
ended on `PASS native governor zero after controlled drain`; the `roundtrip` cell printed every
size line with `byte_exact=true`. Receipts
`research/spill-d-20260919/pro-single-day12/transfer-gate/{conformance,roundtrip}/`, replayed by
`research/spill-d-20260919/verify-day12.py`; write-up `research/spill-d-20260919/DAY12.md`.

plus one `PASS native D2H-H2D roundtrip bytes=… byte_exact=true source_freed_host_live=true` line per
size (4 KiB to 256 MiB). All lines were recorded on one RTX PRO 6000 Blackwell through the collector
(`research/spill-a-20260919/day9/RESULTS.md`, native source `1adf2be3d`, disposition
`executed-not-qualified`), so v1, v1.1, v1.2 **and canonical v1.3** pass natively. What the binding
took (`research/spill-a-20260919/V13-BINDING.md`): per-side retention (`pin_source_graph` /
`pin_destination_graph` replace the ticket-wide `pin_graph`), a destination lease that keeps its
governor charge past `acknowledge`, and a taken pinned destination that shares the physical
allocation with the ticket until acknowledgement. The frozen schedules were not changed. Native
PASS here is development correctness on one card class; it does not discharge the serving-shape
cells the freeze lists as required.

`tier-transfer-gate pinned-ab [--bytes N] [--pairs N]` (lane/spill-a-20260919 day 13) is the
allocation-flag A/B of the contract's pinned host leases. The contract path allocated through
cudarc's `alloc_pinned`, which hard-codes `cuMemHostAlloc(.., CU_MEMHOSTALLOC_WRITECOMBINED)`, so
every CPU read of a demoted plane under `MEMRA_KV_HOST_CONTRACTS=1` (the engine's completion
checksum, the bind checksum, and Option C's H2D-source checksum) ran uncached. The seam
`memra_engine::tier_transfer::PinnedKind { WriteCombined, Cached }` (`alloc_host_kind`; `alloc_host`
takes the per-device default below) is selected by the gate only, never by an environment variable.
One process, one CUDA context, one collector lock hold: one untimed warm-up roundtrip per arm, then
A B x N and B A x N, per roundtrip the allocation wall, the D2H wall (submit to owner-stream
synchronize), the engine's completion-hash wall, the bind-hash wall, the byte compare, the H2D
wall, the H2D-source hash wall, `cuMemHostGetFlags` (a UVA driver reports `DEVICEMAP` on every
pinned allocation: 6 for write-combined, 2 for cached) and `byte_exact` on both legs; one
`PINNED-AB rule` line and one `RESULT` JSON; `research/spill-a-20260919/wc-ab.py` replays the rule
offline from the mirrored log and must agree. Pre-registered rule (`DAY13.md`): the cached arm wins
on a card iff byte exact in every roundtrip, the driver's bit equals the arm's, cached bind hash
below write-combined at every pair, cached D2H not above at every pair, and the per-order medians
of both; otherwise inconclusive. Target card (one RTX PRO 6000 Blackwell at 600 W, `DAY13.md`,
`pro-single-day13/pinned-ab-160m-s2`, N=5 per arm per order): bind hash 77.6 against 1711 ms
(10/10 pairs), D2H 3.00 against 3.01, H2D 2.98 against 2.98, byte exact 22/22,
`cached_arm=wins-on-this-card` (a first sitting `inconclusive` on the D2H clause by 1 and 4 us).
Local RTX 5090 Laptop GPU (no power limit reported, `DAY14.md`, `rtx5090-day14/pinned-ab-160m`,
N=5 per arm per order): bind hash 37.5 against 1449 ms (10/10 pairs), D2H 7.34 against 7.27 ms
(cached not above at 5/10 pairs, above in both orders' medians), H2D 6.04 against 6.04, byte exact
22/22, `cached_arm=inconclusive`; the 16 MiB context cell on the same card puts cached D2H at 0.74
against 0.70 ms with non-overlapping ranges, 0/10.

The per-device default (day 14, lead ruling 22, `docs/decisions/PINNED-DESTINATIONS.md`):
`PinnedKind::for_device(name)` is `Cached` on the RTX PRO 6000 Blackwell class and `WriteCombined`
on the RTX 5090 class and on every class without a receipt, keyed on the device name exactly as
`parallel::HardwareTarget::from_device_name` keys the product shape; `CudaTransfers::new` resolves
it once, `alloc_host` takes it, no `MEMRA_*` read. CPU cells
`pinned_kind_per_device_default_resolves_by_card_class` (the receipted class resolves to `Cached`,
the RTX 5090 class and unknown names to `WriteCombined`, which is the enum's `Default`),
`pinned_kind_default_is_todays_write_combined_flag_bits` (the flag bits per kind, 4 and 0,
unchanged) and `alloc_host_delegates_with_the_default_kind_and_no_other_pinned_allocation_remains`
(source text: one `cuMemHostAlloc` site, one resolution site in the constructor, no environment
read); GPU cell `pinned_kind_arm_is_honoured_by_the_driver` (ignored without a device: the default
lease reads back the card's resolved arm from `cuMemHostGetFlags`). `conformance` and `roundtrip`
run through the new default and print `PINNED-DEFAULT device=".." kind=.. flags=..` once per
process and `PINNED-DEFAULT roundtrip bytes=.. kind=.. driver_flags=..` per size beside every
`byte_exact=true` line: both cards green through the new default (`DAY14.md`, `rtx5090-day14/*-s2`
and `pro-single-day14/*-s2`).
`research/spill-a-20260919/PINNED-FLAGS.md` is the census of every pinned allocation and every host
read on the path.

### `kv-tier-gate`: fitting-context KV tiering under one numeric program

`crates/memra-engine/src/bin/kv_tier_gate.rs`; the argument contract is `kv_tier_gate/cli.rs`
(pure, testable without CUDA):

```text
kv-tier-gate --artifact <gguf> --case baseline|active|prefix --context 8192|32768 --tiers host|host,nvme --same-program [--kv-allocator pooled|vmm] [--reclaim-diagnostic [--reclaim-cycles N]] --out <new-directory>
```

- `--same-program` is mandatory. Without it the parser rejects the invocation
  (`--same-program is mandatory; no alternate numerical program permitted`, exit 2 with the
  `kv-tier-gate: ` prefix, which the collector classifies as a *failed* cell, not a
  refusal). The receipt header records
  `program=native-decode_step_h-tokenwise-trunk-no-mtp`, raw tokens, no chat template.
- Bound cases: `baseline`, and `active` with `--tiers host`. `prefix`, and `active` on any
  other route, refuse before touching the device:
  `REFUSED: only active tiers=host is bound; prefix and other active routes remain unsupported`.
  Contexts are 8192 and 32768; 16384 is accepted only under `--reclaim-diagnostic`.
- Naked baseline: any `MEMRA_*` variable other than `MEMRA_NVCC`, `MEMRA_CUDA_ARCH` and
  `MEMRA_GPU_LOCK` refuses (`REFUSED: runtime override <name> must be unset for this naked
  eager baseline`); the gate inspects names only, values never enter a receipt.
- `--kv-allocator pooled|vmm` (default `pooled`) is a gate-only door with
  **decide-by: 2026-10-04** ([decision record](decisions/KV-PHYSICAL-RECLAIM.md)). The gate
  constructs the cache directly with VMM-backed planes
  (`Cache::new_with_allocator(&e, &model.cfg, args.context, memra_kv::KvAllocator::Vmm)`) and
  records `construction=direct` / `empty_plane_swap=false` in `allocation-construction.txt`; the
  kernels and addresses are unchanged. Containment is a call-site policy, not a structural
  property: `memra-kv` exposes `Cache::new_with_allocator` and `KvAllocator` publicly and
  `impl KvDev for Engine` implements `alloc_vmm_u8`, so any caller *could* construct VMM planes;
  today only this gate does, and the decide-by decides whether that surface is promoted or
  deleted. Any other value refuses:
  `REFUSED: unknown KV allocator (expected pooled or vmm)`. It is a CLI door, so it has no
  `docs/FLAGS.md` row; the decide-by lives in the decision record.
- `--reclaim-diagnostic` requires `--case active --tiers host --kv-allocator vmm`
  (`REFUSED: reclaim diagnostic requires active VMM host mode`). After demote it frees a
  never-mapped spare VA reservation, re-reads free VRAM, calls `cuCtxSynchronize` and
  re-reads again, and also runs a mapped-VA probe over every demoted plane
  (`KvPlane::probe_demoted_va_release`: unmap the retained chunks, `cuMemAddressFree`,
  re-reserve the same base, remap), writing `mapped-va-probe.tsv` and appending
  `mapped_va_release_delta_bytes`, `mapped_unmap_delta_bytes`, `mapped_va_roundtrip_equal`,
  `free_after_mapped_va_probe_bytes`, `residual_bytes`, `residual_class` and
  `free_after_restore_bytes` to `residual-diagnostic.txt`.
- `--reclaim-cycles N` (N >= 2, ASCII digits only) repeats the SAME demote/restore roundtrip N
  times in one process on one cache under `--reclaim-diagnostic`, so a residual is classified by
  its series instead of inferred from one roundtrip (lead ruling, day 11). Every cycle writes the
  full roundtrip receipt set under `cycle-<k>/` (`active-reclaim.txt`, `residual-diagnostic.txt`,
  `mapped-va-probe.tsv`, `vmm-planes.tsv`, `active-bundles.tsv`, `reclaim-diagnosis.txt`,
  `restored-prefix-state.tsv`) and must restore the suspended state bit-identically before the
  next cycle starts (`active restored state is not bit-identical to suspended state (cycle k of
  N)` aborts the gate). The receipt root gains `reclaim-cycles.tsv` (one row per cycle: free
  VRAM before demote, after demote, after restore; `reclaimed_bytes`, `reacquired_bytes`,
  `residual_bytes`, `restore_residual_bytes`, `free_before_drift_bytes`, the per-cycle flags and
  the restored-prefix manifest hash) and `reclaim-cycles.txt` with `residual_series_class`
  (`kv_tier_gate/reclaim_contract.rs`, `classify_cycles`): `none` (zero residual every cycle),
  `one-time-driver-mapping-metadata` (exactly one granule after cycle 1 and identical through
  cycle N, criteria (a) to (c) holding every cycle, no drift of the process free baseline),
  `growing-residual` (the residual, or the bytes still unreturned against the first cycle's
  baseline, grows across cycles), otherwise `unclassified`. The per-cycle G1 line is unchanged
  (`reclaimed = vmm_granularity != 0 && reclaim_observed && observation.residual == 0`, tightening
  (e)). The series `g1_reclaim_qualified` follows lead ruling 6 (day 12,
  `reclaim_contract::series_verdict`): `true` with a nonzero residual only when all of these hold:
  the run is a `--reclaim-cycles N` series with N >= 5 (`series_min_cycles=5`), the class is
  `one-time-driver-mapping-metadata`, criteria (a) to (c) hold in every cycle, the restored prefix
  is bit-identical in every cycle, and the free baseline drifts by 0. Then, and only then, the gate
  prints `ACTIVE-32K G1 PASS (classified one-time-driver-mapping-metadata, N cycles)` as its
  status line (the `K` tag follows the committed context) and writes it as `series_label`;
  otherwise `series_label=not-printed`. A series whose every cycle is exact (class `none`) is
  `true` with no new label. A single roundtrip with a nonzero residual, a series shorter than 5,
  any other class, a drifting baseline, a differing restore, and any pooled run stay `false` /
  `not-applicable-pooled` with their existing status lines. Criteria (a) to (d) are unchanged and
  (e) stays in force for every other shape. The console prints `reclaim-cycle k/N: ...` per cycle
  and one `RECLAIM-CYCLES: class=... cycles=N granule=... residual_first=... residual_last=...
  g1_reclaim_qualified=...` line before the status line.
  Refusals (exit 2, `REFUSED:` last line): without `--reclaim-diagnostic`
  (`REFUSED: --reclaim-cycles requires --reclaim-diagnostic`), a pooled allocator
  (`REFUSED: --reclaim-cycles requires --kv-allocator vmm; a pooled cache releases no chunk`),
  a duplicate (`REFUSED: duplicate --reclaim-cycles`), and N < 2, a missing value or junk
  (`REFUSED: --reclaim-cycles requires an integer count >= 2`; the value is never echoed). No
  new `MEMRA_*` read. CPU replay: `crates/memra-tier/tests/reclaim/` includes the gate's pure
  modules by path and replays the committed day-10 target-card receipts as series (`day11.rs`) and
  the committed day-11 series bytes of both card classes under ruling 6 (`day12.rs`); the lane's
  offline replays are `research/spill-b-20260919/verify-day11.py` (day-11 rule) and
  `verify-day12.py` (ruling 6: the label must have been printed by the gate as the final status
  line, exactly once, and the receipt fields must follow the pure verdict).
- `--fault <arm>` (lane D, day 11; `kv_tier_gate/fault.rs`, pure rule `fault_contract.rs`): one
  fault at one documented contract call of the same active roundtrip, on the first full-history
  K plane, on the pooled allocator only. The door is a usage error off `--case active`
  (`--fault requires --case active`, a failed cell), refuses the VMM door
  (`REFUSED: fault arms are bound to the pooled allocator; the VMM door is not a fault surface`)
  and the reclaim diagnostic (`REFUSED: fault arms do not combine with the reclaim diagnostic`),
  and never echoes an unknown arm (`REFUSED: unknown fault arm (expected ...)`). Every arm
  records the contract's answers as `check / expected / observed` rows (`fault-checks.tsv`,
  summary `FAULT-ARM.txt`); `fault_contract::verdict` prints `FAULT-ARM PASS <arm>` only when
  every required check was recorded and held, a differing answer or a missing check is a failed
  cell (`fault arm <arm> did not prove its contract; missing=[..] failed=[..]`), and there is no
  third outcome: on day 11 the two arms whose expectation had no seam in the contracts ended in
  a typed refusal; since day 12 they drive lane A's rule seams (`TransferEngine::recover_source`,
  `Cache::suspend_layer` / `resume_layer`), and a backend without a seam fails the seam rows, a
  failed cell, never a refusal and never PASS. Every arm detaches a layer only through
  `Cache::suspend_layer` and reattaches it through `Cache::resume_layer` (rule 2), so
  `ensure_usable` is the continuation gate for the whole demoted interval; the holed arms keep
  the day-11 taint row (`holed-cache-refuses-continuation`). The rule-1 rows are one generic
  sequence over `TransferEngine` (`fault_contract::cancel_restore_revoke` / `_recover` /
  `_retire`) recorded by the native arm and by the CPU binding
  `crates/memra-tier/tests/contracts/fault_arm_bindings.rs` against lane A's fake transport,
  whose `legacy = true` is the red arm (the transport before the rule drains the cancelled
  restore, the seam rows fail, the verdict names them). A
  passing arm whose cache is whole and bit-identical (`restored-identical`) runs the same
  tokenwise continuation and writes `ACTIVE.txt` with the PASS line as its first line; an arm
  whose cache is incomplete prints `FAULT-ARM PASS <arm> committed=<n> generated=0` and writes no
  token: a layer whose K plane did not come back never re-enters the cache. The whole-state
  admission (`active::admit_whole_state`: the governor's own `reserve` for every byte the
  demotion would pin, released at once, before any layer is taken) and the restore integrity
  check (`StateBundle::verify`, `Error::Corrupt`, before any device allocation) are part of the
  roundtrip for every run, not only the arms.

  | Arm | Injection point (contract call) | Required checks (expected answer) | Outcome |
  | --- | --- | --- | --- |
  | `cancel-demote` | D2H submitted and observed complete (`synchronize`), then `TransferEngine::cancel` before `take_destination` | `cancel` `Ok(PublicationRevoked)`; `take-after-cancel` `Err(Cancelled)`; `retire` and `acknowledge` `Ok(())`; `pinned-after-cancel` `0`; `source-returned` (`take_plane` gives the untouched source, `len=<capacity> vmm=false`); `budget-zero`; `restored-identical` | PASS, intact-resident, continuation runs |
  | `cancel-restore` | H2D submitted and observed complete, then `cancel` before `ready_view`; rule 1 (lane A, day 11) from there | `cancel` `Ok(PublicationRevoked)`; `ready-view-after-cancel`, `with-destination-after-cancel`, `take-after-cancel` all `Err(Cancelled)`; `retire-holds-source` and `retire-source-holds` `Err(Busy)` (the engine never drains a cancelled restore); `recover-source` `Ok(host)`; `recovered-source-checksum` (equals the bundle's sealed checksum) and `recovered-source-intact` (`StateBundle::verify` `Ok(())`); `recover-source-once` and `cancel-after-recovery` `Err(AlreadyReleased)`; `retire`, `acknowledge` `Ok(())`; `pinned-held-by-recovered-lease` (the pinned charge stays with the lease); `retake-demoted-copy` (`take_destination` on the D2H ticket) `Err(AlreadyReleased)`; `budget-zero`; `restored-identical` | PASS, the recovered copy goes through the roundtrip's own `restore`, cache whole and bit-identical, continuation runs. Day 11 (no seam) ended in the typed refusal `REFUSED: cancel-restore revoked publication, but the transfer contract has no seam to recover the H2D source after cancellation; no tokens, budget drained`; that receipt is kept and is a failed cell under the day-12 rule |
  | `corrupt-host` | after `demote`: `CudaPinnedLease::write` under the live D2H ticket, then the legal drain (`record_consumer`, `retire`, `acknowledge`), then `write` again; `restore` on the flipped copy | `write-under-live-ticket` `Err(Busy)`; `write-sole-owner` `Ok(())`; `restore-integrity` `Corrupt` (`StateBundle::verify`, before any device call); `device-registry-unchanged`; `budget-zero` | PASS, no publish, no token |
  | `missing-host` | D2H completed and never taken; `retire(None)` and `acknowledge` remove the engine-owned copy; `take_destination` at restore | `completion-checksum` (`poll` checksum equals the bundle's); `remove` `retire=Ok(()) acknowledge=Ok(())`; `pinned-released` (the plane's bytes); `retake-removed-copy` `Err(UnknownTicket)`; `budget-zero` | PASS, no publish, no token |
  | `host-budget-short` | governor pinned capacity set to the whole demoted bytes minus one; `admit_whole_state` before any layer is taken | `whole-state-admission` `Err(Capacity)`; `layers-resident` unchanged; `pinned-charged` `0`; `budget-zero`; `restored-identical` | PASS, nothing copied, continuation runs |
  | `device-short` | after demote, a competing tenant reserves the governor's device dimension down to one byte less than the restore needs; `alloc_device` (restore's first call) | `restore-admission` `Err(Capacity)`; `device-registry-unchanged`; `host-copy-intact` (`StateBundle::verify` `Ok(())`); competitor released; `budget-zero`; `restored-identical` | PASS, continuation runs. The governor has no post-construction capacity seam; the seam used is its own admission with a second tenant |
  | `require-resident` | every plane leaves through `Cache::suspend_layer`; `Cache::ensure_usable("kv-tier-gate continuation")` asked while suspended, again, after the first `resume_layer`, and after the last; rule 2 (lane A, day 11) | `suspended-register` (the register names exactly the taken layers, ascending); `continuation-gate-on-suspended-cache` and `continuation-gate-asked-twice` `Err(ContinuationRefused { path: "kv-tier-gate continuation", layers: [<every suspended layer>] })`; `continuation-gate-after-partial-resume` names what is left; `register-empty-after-resume` `true`; `continuation-gate-after-resume` `Ok(())`; `budget-zero`; `restored-identical` | PASS, continuation runs. Day 11 (no seam; `ensure_usable` answered `Ok(())` on the fully suspended cache) ended in the typed refusal `REFUSED: require-resident has no contract today: Cache::ensure_usable accepts a suspended cache, decode_step_h unwraps a suspended layer, and tier RestoreDecision::RequireState is a load-versus-recompute rule`; that receipt is kept and is a failed cell under the day-12 rule |

  CPU replay: `crates/memra-tier/tests/reclaim/fault.rs` (door, red arm per arm, the committed
  target-card receipts of both days: the five unchanged arms replay from their day-11 cells, every
  arm from its day-12 cell, and the two day-11 refusal receipts are pinned as failed cells under the
  day-12 rule) and `crates/memra-tier/tests/contracts/fault_arm_bindings.rs` (the rule-1 rows on
  the CPU fake, seam and legacy). The lane's offline replays are
  `research/spill-d-20260919/verify-day11.py` (day-11 vocabulary, frozen: two typed refusals) over
  `pro-single-day11/<arm>/` and `verify-day12.py` (seven PASS arms plus the two
  `tier-transfer-gate` cases) over `pro-single-day12/` (N=1, one RTX PRO 6000 Blackwell,
  `executed-not-qualified`, never qualification).
- Receipts: `BASELINE.txt` (first line `BASELINE_CAPTURED`) or `ACTIVE.txt` (first line
  `ACTIVE_RECLAIM_CAPTURED; continuation comparison pending; not G1 PASS` or
  `ACTIVE_COPY_RESTORE_CAPTURED; reclaim qualification incomplete; see metrics; continuation
  comparison pending; not G1 PASS`), plus `active-reclaim.txt` carrying `reclaim_observed`,
  `reclaim_exact_equal`, `residual_bytes`, `residual_class` (`none`, `unclassified`,
  `va-reservation-page-table`, `spare-VA-release-sensitive-driver-accounting`,
  `deferred-driver-release-completed-by-context-sync`, or `not-applicable-pooled`) and
  `g1_reclaim_qualified` (`true`, `false`, or `not-applicable-pooled`: a pooled run can never
  publish the G1 label, `kv_tier_gate/active.rs`). The pure criterion is
  `kv_tier_gate/reclaim_contract.rs`; on top of criteria (a) to (d) the gate applies the day-10
  tightening (e): `g1_reclaim_qualified=true` requires `residual_bytes=0`
  (`active.rs`: `reclaimed = vmm_granularity != 0 && reclaim_observed && observation.residual == 0`),
  so a *classified* nonzero residual is recorded but does not qualify in any single roundtrip. The
  binary prints a G1 label in exactly one shape, the ruling-6 series label above; for every other
  shape the lane's offline replay (`research/spill-b-20260919/verify-day10.py`, which enforces
  the same zero-residual rule) compares the receipt with the frozen baseline bundle and assigns
  `ACTIVE-8K G1 PASS` only when (a) to (e) hold.
- Refusal token contract (lead ruling): a refusal is a final console line
  `REFUSED: <reason>`, exit 2. `cli::diagnostic` leaves `REFUSED:` lines unwrapped and
  prefixes every other error with `kv-tier-gate: `; the collector matches
  `^(?:kv-tier-gate: )?REFUSED: .+` on the last line. Any other exit-2 diagnostic, including
  a generic `Error:` line, is a *failed* cell. The receipt directory gets `REFUSED.txt` with
  the diagnostic for every error; classification comes from the console token, not the file.

Status on 2026-09-20: `ACTIVE-8K G1 PASS` on the rented RTX 5090
(`research/spill-b-20260919/DAY9.md`) and twice on one RTX PRO 6000 Blackwell (`DAY10.md`:
empty-plane swap, then direct construction with 34 VMM planes). 32k on both cards:
bit-identical, one granule (2,097,152 B) of residual unclassified (the mapped-VA probe
returned 0 B, so VA-reservation release is not the mechanism), not G1 PASS. B's verifier prints that
label with a dash; the wording here follows the writing rule.

Status on 2026-09-21 (`research/spill-b-20260919/DAY12.md`, gate source `c7dd20cc5`): the 32k
five-cycle series rerun printed, on one RTX PRO 6000 Blackwell and on the local RTX 5090 Laptop
GPU, verbatim `ACTIVE-32K G1 PASS (classified one-time-driver-mapping-metadata, 5 cycles)`
(residual 2,097,152 B in every cycle, drift 0, restored prefix bit-identical in every cycle; the
PRO continuation matches its frozen bundle, the laptop card has no frozen bundle and its
continuation identity is in-process only). The 8k series control is not rerun: the mapped-VA
probe refuses to re-reserve the original address for the small 8k planes on both cards (lead
ruling 7, open item in the decision record); the 8k evidence stays the single-roundtrip
`ACTIVE-8K G1 PASS` with residual 0. Gate-only door, decide-by 2026-10-04 unchanged.

### Experts-via-tier gate (`run-gen` / `run-spec --experts-via-tier`)

`--experts-via-tier` on `run-gen` or `run-spec` (GGUF path only) calls
`Engine::install_expert_bank_gate` (`crates/memra-engine/src/banked_residency/native.rs`)
after load and before the first forward. It hashes the already-open artifact inode against
the approved SHA-256, derives its expert catalog from the compiled model plan and the model
pack's GGUF tensor contract bound against the artifact's tensor census
(`memra_gguf::expert_banks::expert_bank_catalog`: semantic ids, accepted names, required
shapes and quant layouts; the installer spells no checkpoint name and keeps no architecture
allowlist), checks every retained expert record byte-for-byte against the loaded `HostExps`,
builds a bounded host expert bank over the file, and installs it into the MoE slot cache
(`cache.install_banked`), so slot misses are served through the bank while the native SLRU
slot addresses, expert kernels and routing stay unchanged. It is an explicit default-OFF
qualification door, not a runtime flag; no `MEMRA_*` read is added.
Two failure classes leave the door. A budget the bank cannot hold, or an expert catalog the
plan and contract cannot bind for the artifact, is a typed `ExpertBankRefusal`
(`crates/memra-engine/src/banked_residency.rs`): the binary prints `REFUSED: <reason>` as its
final stderr line and exits 2, and `tools/tier-battery.py` records the cell as `refused`.
Every other error stays the binary's failure (`Error: "<reason>"`, exit 1). Both are returned
before any bank demand:

- Failures: `experts-via-tier requires the approved GGUF artifact` (`run-gen`) and
  `experts-via-tier requires approved GGUF` (`run-spec`) for directory sources;
  `experts-via-tier artifact SHA256 mismatch`;
  `experts-via-tier requires one immutable GGUF, cache, and at most one MTP head`;
  `experts-via-tier refuses resident or parallel expert bypasses; use the cache baseline`
  (resident slabs, Step EP/TP and GLM EP/TP splits); and the budget flags' usage errors
  (`expert_bank_cli`: a bare flag, a repeat, a malformed value, a budget without
  `--experts-via-tier`, or a key that merely starts with a flag name: keys match exactly, so
  `--expert-bank-host-bytes-x=1` is `unknown expert bank flag ...` and `--experts-via-tier=1`
  is `--experts-via-tier takes no value`, never the flag they resemble and never ignored).
  The helpers live at `memra_engine::banked_residency::{expert_bank_cli, refusal_reason,
  ExpertBankBudget}` (a `#[doc(hidden)]` gate module, nothing re-exported at the crate root).
- Catalog refusals: `REFUSED: experts-via-tier expert catalog refused: <detail>`, where the
  detail is `the compiled plan has no MoE expert projections`; the tensor contract's own
  verdict for a bank tensor that is missing, duplicated (`DuplicateCensusName`), ambiguous,
  shape-incompatible (`ShapeMismatch`) or layout-incompatible; `tensor contract has no entry
  for <id>` or `tensor contract has <n> entries for <id>`; `artifact carries expert scale
  planes the consumer does not declare: <names>` (a `blk.N.ffn_*_exps.scale` or
  `.input_scale` row in the census) or `loaded bank <name> carries scale planes (macro or
  block scales) the native installer does not consume`; and a plan/model disagreement
  (`loaded model has N layers, compiled plan has M`, `plan layer N routes experts but the
  loaded layer is dense`, `loaded MTP head routes experts but the compiled plan has no MTP
  expert bank`). Scale admission is not landed: a scale-bearing artifact is refused, never
  banked payload-only. Unit cells: `crates/memra-gguf/src/expert_banks.rs` (plan-derived
  names equal the former literal `blk.N.ffn_{gate,up,down}_exps.weight` spelling on a
  qwen3_5_moe plan with an MTP block, plus one test per refusal).
- `--expert-bank-host-bytes=N` (default 256 MiB) sets the host bank budget. `host_bank_budget`
  refuses `experts-via-tier host bank budget cannot hold one expert record` below one record
  and `experts-via-tier host bank budget exceeds qualification ceiling` above 256 MiB, each
  suffixed `(requested N, minimum M, ceiling C)`, and caps the bank at 16 records.
- `--expert-bank-gpu-bytes=N` fixes the GPU slot count before any allocation
  (`MoeSlotCache::with_exact_slots`, never clamps). `gpu_bank_budget` refuses
  `experts-via-tier GPU bank budget cannot hold the eight-slot minimum` below eight slots and
  `experts-via-tier GPU bank budget exceeds the hard VRAM ceiling` above the machine ceiling
  (`hard_slot_bytes`: the `MEMRA_MOE_HARD_VRAM_FRAC` share of free VRAM minus two slots,
  measured by the installer), with the same `(requested, minimum, ceiling)` suffix. One slot
  is the record plus `banked_residency::SLOT_TAIL_PAD_BYTES` (8), the one constant the native
  slot sizing in `moe_cache.rs` and the budget arithmetic share. Setting
  `MEMRA_MOE_SLOTS` alongside a GPU budget is a refusal
  (`experts-via-tier GPU bank budget conflicts with MEMRA_MOE_SLOTS`), never a silent
  precedence. Without a GPU budget the native slot sizing (`MEMRA_MOE_SLOTS` or auto) is
  untouched. The installer takes both budgets as a typed `ExpertBankBudget`; it reads no
  argv and no environment for them.
- The lease token that crosses the owner-thread seam carries the identity the registry holds
  for its lease (`ExpertLeaseToken::{record, artifact, epochs}` in
  `crates/memra-tier/src/bank/owner_proxy.rs`): the `(layer, proj, expert)` key derived from
  the leased `BankId` through the one mapping `bank::dispatch_id`, the record's artifact
  digest, and the staging ticket's epochs. `demand` refuses a bank that leases another record
  than the one demanded (`ProgramMismatch`, or `InvalidLayout` for a record with no dispatch
  id), retiring that lease through the bank before refusing; `with_bytes` / `finish` refuse a
  token whose identity does not match the pending lease (`ForeignLease`); `admit_banked`
  asserts `token.record() == (layer, proj, expert)` before the H2D. Unit cells:
  `owner_proxy.rs` tests (identity carried, lying bank refused and retired, record without a
  dispatch id refused, three forged tokens foreign, `dispatch_id` boundaries including the
  `u16::MAX` MTP key) and `crates/memra-tier/tests/bank/owner_proxy.rs`
  (`token_identity_names_the_fixture_lease`). The H2D program is unchanged.

Verdicts are the standard gates: `run-gen` argmax `MATCH` and `run-spec`
`=== SELF-CONSISTENCY PASS ===` over K=1..8. The gate prints
`[experts-via-tier] catalog blocks=<n> banked=<n> projections=<n> catalog_sha256=<hex>
records=<n> records_sha256=<hex>` once the catalog is bound (`catalog_sha256` is SHA-256 over
`ExpertBankCatalog::identity()`, one line per projection; `records_sha256` chains the
per-record checksums in catalog order; `banked` is below `blocks` when the gate loads without
the MTP head), then
`[experts-via-tier] installed artifact_sha256=<hex> host_slots=<n> max_expert_bytes=<n>` at
install, and on drop `[expert-gpu-slru] slots=<n> allocated_bytes=<n> evictions=<n>` then
`[experts-via-tier] physical_reads=<n> owner_close=<result>`. Under the collector, lane C's
`research/spill-c-20260919/pressure-refusal.py` normalizes exactly the host-record
rejection to `REFUSED: experts-via-tier host bank budget cannot hold one expert record`,
exit 2 (collector status `refused`); any other failure stays `failed`. Evidence:
`research/spill-c-20260919/DAY8.md` (rented RTX 5090, 8 GiB and 4 GiB banks, ON and OFF
controls, every cell `MATCH` / `SELF-CONSISTENCY PASS` with eviction engaged) and `DAY9.md`
(one RTX PRO 6000 Blackwell: default and 8 GiB banks, ON and OFF, same verdicts). All cells are
N=1 and `executed-not-qualified`; no support state or default moves on them.

### Host tier contracts door (`memra-server`, `MEMRA_KV_HOST_CONTRACTS`)

`MEMRA_KV_HOST_CONTRACTS=1` (default OFF; `docs/FLAGS.md` row, decide-by 2026-10-05; design
`research/spill-c-20260919/HOSTPREFIX-DOOR.md`) constructs `HostTierContext` at model load, so
the host tier's already-compiled sidecar route (`tier_charge`, `bind_tier_image`, the insert and
promote identity leases, lane B's `HOSTPREFIX-EXTENSION.md`) executes: one `ProgramIdentity` per
loaded GGUF model, the tenant salt stamped per pool key by `memra_kv::tiered::hostprefix::
tenant_salt` over the same namespace string `auth::meter_key` reads, the server's governor as a
ledger. Days 13 and 14 changed no copy program; day 15 (Option B, below) routes the pageable-tier D2H
through the transfer engine. OFF is byte-identical by construction (the constructor is
never called). The door refuses the boot, typed and loud, for a junk value, the startup arena
(`MEMRA_GLM5_TP_KV_HOST=1`), a checkpoint-directory model or a loaded vision tower.

- CPU tests (`cargo test -p memra-server -p memra-kv --offline`): `crates/memra-kv/src/tiered/
  hostprefix.rs` `tenant_salt_is_one_derivation_of_the_namespace_string`,
  `empty_namespace_is_the_default_single_tenant_namespace_not_a_refusal`,
  `shared_governor_is_the_injected_trait_object_and_charges_through_it`;
  `crates/memra-server/src/worker.rs` `kv_host_contracts_door_parse_is_strict_and_never_falls_
  back_to_off` (bare `=`, junk, doubled values, non-UTF-8 all refuse; only `1`/`0`/unset parse),
  `host_tier_arena_refusal_names_the_arena_and_passes_without_it`,
  `host_tier_program_base_is_a_pure_function_of_its_sources`,
  `host_tier_context_program_stamps_the_pool_namespace_salt_once`,
  `host_tier_governor_ledger_admits_what_the_lru_would_and_binds_at_twice_the_budget`,
  `host_cache_with_contracts_door_refuses_an_unbound_image_and_admits_it_with_the_door_off`.
- Target-card gates, door OFF then ON on the same binary and prompts, through the canonical
  collector: `tools/serve-smoke.sh` (plain and cache-metering arms), `tools/kv-host-spill-
  identity-gate.sh`, `tools/kv-host-spill-failure-gate.sh`, and lane B's `tools/prefix-evict-
  reclaim-gate.py`. Admissibility (lead ruling 15): every verdict line equal across arms,
  `MEMRA_KV_HOST_VERIFY=1` `verify ok` on every ON promote, equal `[prefix-host] demote:` byte
  counts. Evidence: `research/spill-c-20260919/DAY13.md` and `pro-single-day13/` (one RTX PRO
  6000 Blackwell at 600 W, N=1, `executed-not-qualified`; no support state or default moves).
- ON-arm surface (day 14, lead ruling 16): lane B's first slice (plain KV plus recurrent
  continuation) AND MTP draft-bearing entries (spec-published boundary captures), each bound to
  its own program: the draft plane is its own `Role::Draft` K and V segments with checksums, and a
  model with an MTP head carries a second `ProgramIdentity` (`host_tier_draft_program`) whose
  artifact, plan and numeric fold in the draft head's source and the draft rows' encodings, so a
  spec entry and a plain entry of one prompt never share an identity. GLM state (TP, latent) and
  the DFlash draft tail are refused by name at demote (`[prefix-host] demote refused (contracts
  door): entry carries ...`), handoff imports at insert. CPU tests (`worker.rs`):
  `host_tier_entry_class_admits_plain_and_mtp_draft_and_refuses_glm_and_dflash_by_name`,
  `host_tier_draft_program_differs_from_plain_in_exactly_artifact_plan_and_numeric`,
  `host_tier_context_program_selects_the_class_and_refuses_a_draft_entry_without_a_head`,
  `host_tier_shape_metadata_v2_frames_the_draft_plane_presence_unconditionally`. Exit criterion
  on the target card: `tools/kv-host-spill-identity-gate.sh` and
  `tools/kv-host-spill-failure-gate.sh` under the gates' DEFAULT spec environment (every insert
  a draft-bearing spec-boundary capture), door OFF then ON on one binary and prompts: every verdict
  line equal, `verify ok` on every ON promote, equal `[prefix-host] demote:` byte counts, no
  refusal line in the ON arm; the `MEMRA_SERVE_SPEC=0` pairs and serve-smoke unchanged from day 13.
  Evidence: `research/spill-c-20260919/DAY14.md`, `pro-single-day14/`, replay `verify-day14.py`.
- Option B (day 15, lead rulings 14 and 15): under the door on the pageable tier the D2H of every KV
  plane of an entry (trunk planes and the MTP draft plane) goes through the native `TransferEngine`
  (`memra_engine::tier_transfer::CudaTransfers` on the worker's CUDA owner stream, charging the same
  governor through the `HostTierLedger` adapter, one batch per demote): the owned `KvPlane`s leave the
  entry's plane slots by `Option::take` (no placeholder, no device byte) and return through
  `take_plane` into the same slots; `alloc_host` leases carry the pinned charge, so the demote's
  residency charge takes pinned zero; the ledger gains an in-flight dimension (2 x max layers + 2). Each
  `HostPlane` keeps its `CudaPinnedLease` and the engine's completion checksum as its receipt;
  `bind_tier_image` refuses, typed, when its bundle checksum differs from that receipt, and names the
  one legitimate difference (`MEMRA_KV_HOST_FAULT=flip-demote`, which corrupts the image after the
  receipt as it does after the verify digest). A quarantined completion is `SourceQuarantined`: the
  engine keeps the planes, the pause sweep drops the entry, the tier latches off. Receipt line per ON
  demote: `[prefix-host] contracts door D2H receipt: ticket issuer=.. seq=.. epochs=0/1/1 items=N (..
  KV planes[, draft]) complete=N require=ok checksums_sha256=.. retired acknowledged`. CPU cells
  (`worker::tests`): `host_tier_ledger_handle_charges_and_releases_the_servers_one_ledger` (one ledger
  through two handles, the in-flight bound refuses a fifth op),
  `option_b_contract_route_is_door_only_and_keeps_the_frozen_demote_order` (the kv_tier_gate demote
  sequence in order, door-only call site, no pre-door copy program or plane clone in the route, the
  pinned-zero charge, the sweep's drop). Target card: `tools/kv-host-spill-identity-gate.sh` `ALL GREEN`
  OFF and ON (default and plain), `tools/kv-host-spill-failure-gate.sh` `1 FAILURE(S)` OFF and ON (the
  pre-existing pool-full line), lane A's `tools/kv-host-tenant-reclaim-gate.sh fix` `PASS` OFF and ON,
  serve-smoke and lane B's `prefix-evict-reclaim-gate.py` / `prefix-newest-turn-fits-gate.py` lines
  identical, one receipt before every ON demote, N=1, executed-not-qualified. Evidence:
  `research/spill-c-20260919/DAY15.md`, `pro-single-day15/`, replay `verify-day15.py`.
- Option B unwind (day 15 review, PR #599 findings 1 and 2): every pre-submit refusal drops the ops'
  original `DeviceLease` handles before the unwind (an extra holder on the registry `Rc` makes
  `take_plane` refuse `Busy`, which escalated a recoverable refusal to `SourceQuarantined`, latched the
  tier and dropped a whole device entry over intact planes); the abort observes every fence before
  retiring against it (owner-stream drain after `record_consumer`; an unpublished ticket retires
  against `None`) and never discards a `retire`, `acknowledge` or `release_producer` result: a refusal
  there is the typed `TicketLeaked` outcome and one `TIER DISABLED` line (a leaked batch is the
  ledger's whole in-flight dimension). Injectable one-shot faults, `MEMRA_KV_HOST_FAULT=
  contract-presubmit` (the producer fence refused before any op is submitted) and
  `contract-postpublish` (the receipt check refused after every destination was taken), armed once at
  boot into `HostTierContext::fault`. Cells: the GPU unit cells `option_b_presubmit_refusal_returns_
  every_plane_and_keeps_the_tier_on` and `option_b_postpublish_refusal_retires_the_ticket_and_keeps_
  the_tier_on` (`worker::tests`, `#[ignore]` without a device; a real `CudaTransfers` on the engine's
  stream over a ledger whose in-flight dimension is exactly one batch: planes back with their bytes,
  typed `Failed` never quarantine, tier on, ledger at zero, the next demote completes), and
  `tools/kv-host-contract-fault-gate.sh [--external-lock FD] MODEL BIN EV` on a real server boot
  (door ON, one cell per fault: r1 seeds, r2 evicts into the injected refusal, r3 evicts into a
  clean demote; asserts exactly one typed `demote failed (tier D2H <producer fence, receipt> refused:
  injected failure ...)`, then a D2H contract receipt whose ticket is `seq=1` (presubmit: no ticket was
  issued) or `seq=2` (postpublish: the aborted ticket retired and was acknowledged) and a `demote:`,
  no `TIER DISABLED`, no quarantine, no `Capacity`, no leaked wording; verdict `KV-HOST-CONTRACT-FAULT
  GATE: ALL GREEN`). The source-text cell pins the `originals` drop before every pre-submit unwind and
  the drain-then-retire order with no discarded result in the abort. Evidence:
  `research/spill-c-20260919/DAY15.md` (review section), `pro-single-day15-review/`, replay
  `verify-day15-review.py`.
- Option C (day 16, lead ruling 15 after B's receipts): under the door the promote H2D of an entry whose KV
  planes are contract destinations goes through the same `TransferEngine`, one batch per promote: fresh
  device planes from the OFF allocator (`alloc_u8`) registered at the destination generation and retained;
  the sources are TWINS of the entry's own leases (`CudaTransfers::retain_host`, the host mirror of
  `retain_device`: the contract's H2D consumes the lease it is handed, and the promote keeps its host twin
  resident exactly as OFF does, so the entry's handles never move); a producer fence, `submit_batch`,
  `synchronize`, `poll`, `Completion::require` against each plane's D2H receipt BEFORE publication (the copy
  read exactly the bytes the demote wrote), `ready_view` per item, a consumer fence, `retire_source`,
  `release_producer`, `retire`, `acknowledge`, `take_plane` into the `PrefixPlane`s the caller publishes
  through `insert_pinned_demoting`; the verify digest stays the OFF-path check. Typed
  `HostPromoteFailure`: `Failed` (the OFF line), `Refused` (entry intact, ledger clean), `ReceiptMismatch`
  (`cancel`, `recover_source` per source twin per lane A's rule 1 with the recovered pointer checked against
  the entry's lease, `retire`, `acknowledge`; the caller drops the host entry as `VERIFY FAILED` does),
  `Latched` (one `TIER DISABLED` line). The ledger's device dimension holds three device prefix budgets.
  Receipt per ON promote: `[prefix-host] contracts door H2D receipt: ticket issuer=.. seq=.. epochs=0/1/1
  items=N (.. KV planes[, draft]) complete=N require=ok checksums_sha256=.. published retired acknowledged`,
  whose digest equals the D2H line's digest of the same entry. CPU cells (`worker::tests`):
  `option_c_contract_route_is_door_only_and_keeps_the_frozen_promote_order` (the restore sequence in order,
  the receipt check before the first `ready_view`, destinations allocated before any registration, the
  planes leave only after acknowledgement, no `htod_u8_into`/`memcpy_htod`/`clone_dtoh`/`alloc_host` in the
  route, `plane_up` keeps both `htod_u8_into` calls, the abort cancels then recovers then retires and
  discards nothing, the caller drops on a mismatch and latches on a leak, the demote route takes only its
  side of the fault cell, the device dimension at three budgets),
  `host_contract_fault_sides_are_taken_by_their_own_route_only`, and
  `host_tier_governor_ledger_admits_what_the_lru_would_and_binds_at_twice_the_budget` (now also: a third
  whole-budget device charge admitted, a fourth refused). GPU cells (`#[ignore]` without a device, a real
  `CudaTransfers` on the engine's stream, a host image built by the Option B route):
  `option_c_promote_routes_every_contract_plane_and_keeps_the_host_twin` (every fresh plane holds the
  demoted bytes, the source device entry untouched, the host twin's leases readable and SOLE-OWNED again
  (`flip_first_byte` succeeds twice), a demote-side fault left armed, ledger back to the image's leases),
  `option_c_presubmit_refusal_releases_every_destination_and_keeps_the_host_twin`,
  `option_c_postpublish_refusal_retires_the_ticket_and_keeps_the_host_twin`,
  `option_c_receipt_mismatch_cancels_before_publication_and_recovers_the_source` (one flipped host byte:
  `ReceiptMismatch`, the twin recovered and dropped, the flip reverts, then a clean promote). Gate:
  `tools/kv-host-contract-fault-gate.sh` gains the cells `promote-presubmit` and `promote-postpublish`
  (`MEMRA_KV_HOST_FAULT=contract-promote-presubmit|contract-promote-postpublish`, one-shot: r1 seeds, r2
  evicts into a clean demote, r3 re-asks r1 so the promote takes the injected refusal and the cold path
  serves, r4 re-asks r2 so the next promote must complete with an H2D receipt and a `promote:` line; no
  `TIER DISABLED`, no drop, no `Capacity`, no leaked wording, no refusal beyond the injected one). Target
  card, door OFF then ON on one binary: the day-15 battery (identity default and plain, failure default,
  lane A's tenant fix arm, serve-smoke, lane B's two gates) plus the six GPU cells, the four-cell fault gate,
  and the WC pair cell (`research/spill-c-20260919/WC-DESTINATIONS.md`: OFF versus ON demote and promote
  wall times, N=5 per arm in both orders, one lock hold, 250 ms telemetry; the first cell of the decide-by
  review, not a verdict). Evidence: `research/spill-c-20260919/DAY16.md`, `pro-single-day16/`, replay
  `verify-day16.py`.
- Option C unwind (day 16 review, PR #605 findings 1 and 2): the promote abort recovers only the ACCEPTED
  items of a partially accepted batch (a rejected slot is `None` in the engine and `recover_source` answers
  `Rejected`; recovering it pushed a leak that latched the tier over a clean state) and asks the engine
  whether the ticket is published through `cancel` (`PublicationRevoked` recovers the sources per rule 1,
  `AlreadyPublished` records the consumer fence, drains and retires the sources) instead of inferring it
  from a loop index (which can disagree with `CudaTransfers::ready_view`, publish-after-check, and
  `with_destination`, publish-before-check, and then leaves the ticket un-retired and every destination
  refused). One-shot faults `MEMRA_KV_HOST_FAULT=contract-promote-reject` (the last op mis-sized by one
  byte, rejected by the engine's own validation) and `contract-promote-readyview` (the first `ready_view`
  published in the engine, the route told otherwise). GPU cells (`worker::tests`, `#[ignore]` without a
  device): `option_c_partial_acceptance_unwinds_refused_with_every_destination_released`,
  `option_c_first_ready_view_failure_unwinds_through_the_published_arm` (each: typed `Refused`, no leak
  wording, host twins intact and sole-owned, ledger back to the image's leases, the next promote completes).
  Gate: `tools/kv-host-contract-fault-gate.sh` cells `promote-reject` and `promote-readyview` (six cells in
  all; the aborted ticket's sequence number is consumed, no `TIER DISABLED`, no drop, no `Capacity`, no
  leaked wording). Evidence: `research/spill-c-20260919/DAY16.md` (review section), `pro-single-day16-review/`,
  replay `verify-day16-review.py`.
- The hit gate's door arm (C day 27, `tools/spec-on-cache-hit-gate.sh qwen`): the door batteries run the
  hit gate twice, door OFF (`MEMRA_KV_HOST_CONTRACTS` unset) and door ON (`MEMRA_KV_HOST_CONTRACTS=1`).
  Until day 27 the ON arm booted with no `MEMRA_KV_HOST_MB`, so the server built no program identity
  (its own line: `[kv-host-contracts] MEMRA_KV_HOST_CONTRACTS=1 with no host tier on this boot
  (MEMRA_KV_HOST_MB=0): nothing to route, no program identity built`), `hpx.armed()` was false before any
  entry-class check, and every "hit gate ALL GREEN OFF and ON" taken on lane A's days 17 to 21 and lane C's
  days 24 and 26 covered the tick program in both arms (`research/spill-c-20260919/DAY26.md`,
  `HOSTPREFIX-DOOR.md` item 11). Under the door the gate now ARMS the host tier on both of its boots
  (spec-on and the spec-off twin) with the identity gate's budget, `MEMRA_KV_HOST_MB=8192`
  (`MEMRA_HOSTGATE_HOST_MB`'s default; an exported `MEMRA_KV_HOST_MB` is respected), and asserts per boot
  the tier's arming line (`[prefix-host] on: budget`), the door's (`[prefix-host] contracts door ON`), no
  latch line (`TIER DISABLED`, `CAPTURE OFF-TICK DISABLED`, `RESTORE OFF-TICK DISABLED`), and across the two
  boots at least one route submission (`capture`, `restore`, `demote` or `promote submitted off the tick`;
  the spec-off twin's `insert (seed)` entries take the capture route by construction, the fault gate's `1 +
  2 capture ticket(s)` accounting on both cards). An ON arm that ran with the tier off cannot read ALL GREEN.
  The OFF arm and the identity clause (spec-on text == spec-off text on r1, r2, r3, g1, g2) are unchanged: a
  red identity under the armed tier is a finding against the door, never a clause to move. Both arms print
  an entry-class census per boot (`insert (spec-boundary)`, draft-bearing, against `insert (seed)`, plain,
  and the class of the identity clause's own namespaces) so a reader knows which class each side of the
  clause hit: the spec-on side's rows hit draft-bearing entries (the route refuses them by name, tick
  program), the spec-off twin's rows hit plain entries (the route's whole-entry restore). The door arm is
  defined for the qwen arm only. Evidence: `research/spill-c-20260919/DAY27.md`, `rtx5090-day27/`
  (9B), `pro-single-day27/` (27B), both arms, N=1, `executed-not-qualified`.
- The hit gate's lock arms (C day 28, the `kv-host-spill-identity-gate.sh` shape). Without a flag every
  boot runs under the gate's own `flock -w 300` on the canonical lock, as before. With
  `--external-lock FD` (the collector's `tools/tier-battery.py --rig <rig> --external-lock --execute
  tools/spec-on-cache-hit-gate.sh --external-lock @COLLECTOR_LOCK_FD@ qwen ...`) the gate takes no lock
  of its own: the collector's inherited FD carries the exclusion for the whole gate, verified by
  `tools/tier-lock-proof.py` (owner `collector`) into `<evidence_dir>/LOCK.json` before any boot; a
  second `flock` on the same inode would deadlock behind the collector and a `-w` timeout would boot
  unlocked, so the wrapper is simply absent, and `stop()` then addresses `$SERVER_PID` itself (no
  wrapper: `env` execs the binary in place; the pid is signalled only while its comm reads
  `memra-server`). Until day 28 the gate could not run under the collector's hold (lane A day 21, C days
  26 and 27 ran it under its own `flock`). Teeth: `tools/test_spec_on_cache_hit_gate_lock.sh` (CI, the
  gate-teeth step) drives the gate's GPU-less `--lock-self-test FILE` arm, which boots nothing and runs
  the arm's launch wrapper around a probe that asks whether FILE is locked while the wrapper runs: the
  default arm must read `probe=held`, the external arm `probe=free` with FILE's inode and mtime
  unchanged, and a non-numeric FD is `REFUSED` with exit 2 before anything runs (7 assertions; fewer
  recorded is a broken fixture). No new `MEMRA_*` read.

### `h2d-probe --copies`

`crates/memra-engine/src/bin/h2d_probe.rs` is copy plumbing: `--bytes` one of the ten
registered sizes (4 KiB to 1 GiB), `--direction h2d|d2h|both`, `--order ab|ba`,
`--repeats 1` only, and `--copies 1..100000` (default 1). One visit issues `copies`
back-to-back copies on the owner stream and sums the per-operation event intervals
(`event_timing` `sum-per-operation-owner-stream`, `completed_bytes = bytes * copies`);
`n` stays 1 and `evidence_class` stays `n1-plumbing-not-qualified`. The `RESULT` record
carries `n_per_size_direction_arm:1`, `comparator_red_rejected:true`, `qualified:false`.
GPU invocation must be wrapped by the collector (300 s maximum, header comment); `--dry-run`
exercises the schema without CUDA. Receipt: `research/spill-f-20260919/H2D-RESULTS.md`
(one native N=1 matrix, 32 visits, `executed-not-qualified`, no medians). A scored copy
envelope is `tools/tier-envelope.py` (default N=5 AB and N=5 BA; `--correctness-only`
permits N=1, never scoring).

### Collector: `tools/tier-battery.py`

Launch every native cell through the collector (absolute executable, new output dir):

```sh
python3 tools/tier-battery.py --rig rtx5090|pro-single|pro-pair|pro-four --timeout <seconds> --out <new-dir> --execute <absolute-bin> <args>
```

- Locks (`LOCKS` table): `rtx5090` takes `/tmp/memra-5090.lock`; `pro-single` (one RTX PRO
  6000 Blackwell), `pro-pair` and `pro-four` take `/tmp/memra-gpu.lock`. Exactly those two
  names exist. The lock is `flock(LOCK_EX | LOCK_NB)`: contention refuses at once
  (`REFUSED: [Errno 11] Resource temporarily unavailable`, exit 2) instead of waiting, so a
  lane retries on a bounded cadence and keeps every refused attempt. The default `--rig` is
  `pro-pair`; always state the rig. `tools/tier-rig-bootstrap.sh --rig rtx5090|pro-single`
  records the same lock per rig, and `--dry-run` there is not a rig acceptance result.
- Private lock path under test (lead ruling 12, 2026-09-21): the collector's own suite
  (`crates/memra-tier/tests/battery/`) never takes a real rig lock, so a serving job on the rig
  cannot redden a CPU suite (the integ10 battery's first attempt lost 20 tests to a serve-smoke
  holding `/tmp/memra-5090.lock`; with both paths held, 24 of 85 tests failed before this
  change and 0 of 86 after: `research/spill-d-20260919/day13/`). The seam is
  `MEMRA_TIER_BATTERY_LOCK_DIR=<dir>`, honoured by `tools/tier-battery.py` (`lock_table`) and
  `tools/tier-rig-bootstrap.sh`: the two canonical NAMES re-rooted under the directory
  (`<dir>/memra-5090.lock`, `<dir>/memra-gpu.lock`); the rig->name table, the refusal on
  contention and the receipt shape are unchanged, and a receipt written under the seam records
  the private path, so it refuses to validate against the canonical table in a process without
  the seam (`test_collector.py`, `test_private_lock_seam_receipts_never_validate_against_the_canonical_table`).
  Every lock-taking test class mixes in `tests/battery/private_lock.py` (`PrivateLockMixin`:
  a fresh directory per test, exported to children and patched into the loaded module's
  `LOCKS`); tests that validate COMMITTED receipts run under `canonical_locks()`. Tools that pin
  the two names by literal (`tools/tier-lock-proof.py`, the two legacy `kv-host-spill-*-gate.sh`)
  have no seam: `test_external_lock.py` drives fixture copies with the literal substituted and
  asserts the tracked literals. It is a test seam only: unset in every production launcher, and
  the ci.yml `portable-suites` job runs the whole suite with both real paths held.
  The seam alone moves nothing (PR #592 review): a collector `--execute` or `--dry-run` and any
  lock-holding bootstrap run refuse under it, before creating a directory or opening a lock,
  unless the process also passes `--private-lock-dir-for-tests` (`REFUSED: MEMRA_TIER_BATTERY_LOCK_DIR
  is set but --private-lock-dir-for-tests was not passed`, exit 2; the flag without the seam
  refuses too; `--plan`, `--validate` and the bootstrap's `--status` take no lock and run). A
  process under the seam prints `PRIVATE lock directory (test seam); the rig lock is NOT held`
  on stderr; `lock.json` and the dry-run manifest carry `"seam": "<dir>"` (`lock_seam` in
  `BOOTSTRAP.json`), and `--validate` refuses a capture whose seam is not the validating
  process's own (`lock.json seam=... is not this process's MEMRA_TIER_BATTERY_LOCK_DIR=...`).
  Every launch site in the suite passes the flag (`private_lock.FLAG`); red arms in
  `test_day10.py` and `test_collector.py`.
- `--external-lock`: legacy shell gates run under the collector's inherited lock, never
  wrapped twice. The collector passes its lock FD to the child, replacing exactly one
  `@COLLECTOR_LOCK_FD@` argument, and writes `lock.json` with the device/inode proof; it is
  not combinable with `--resume`. Children share the worker group, so a timeout kills the
  whole tree (`tests/battery/test_timeout_tree.py`, `test_external_lock.py`).
- Vocabulary: every capture records `qualification: false`. `status` is
  `executed-not-qualified` (exit 0, exactly one `RESULT` record parsed), `failed` (nonzero
  exit, timeout, or parse error; `failure_quote` is the first line matching
  `error|out of memory|CUDA_ERROR|fatal|panic`, otherwise `died, cause unknown` with a repro
  request), or `refused` (exit 2 and a last line matching `^(?:kv-tier-gate: )?REFUSED: .+`).
  Failed and refused cells exit the collector nonzero. `executed-not-qualified` is
  development evidence: it never promotes a model, a default or a support state.
- Telemetry: `nvidia-smi ... -lms 250` for the whole cell; validators require
  `telemetry_interval_ms == 250` and reject a CPU fixture that invents GPU telemetry.
  Observed power limit/max pairs and compute-app snapshots before and after are retained.
- `--validate <path>`: a directory of `CELL.jsonl` journals prints
  `{"kind": "capture-integrity", "cells": N, "failed_commands": .., "refused_commands": ..,
  "qualification": false, ...}`; a live or torn journal refuses
  `REFUSED: interrupted/invalid CELL journal; not a completed capture`; a single
  `.capture.json` prints `CAPTURE INTEGRITY MATCH; command status=<status>; NOT qualification`;
  runs or telemetry JSONL print `BYTE-RECEIPTS MATCH ...` or `TELEMETRY MATCH ...`.
  Integrity is never qualification.
- Medians: the collector computes none for native cells (`--first-hour` plan:
  `performance_medians_allowed: false`; `--dry-run` medians are synthetic,
  `dry-run-not-qualification`). A scored envelope needs at least five AB and five BA pairs
  (`paired_orders`, `tools/tier-envelope.py`), one lock for the whole campaign, and every
  published median states its N and thermal regime (lane D's G2 table carries `N/arm` and
  `Regime` columns, `research/spill-d-20260919/G2-RESULTS.md` on `lane/spill-d-20260919`).
  Timing is never compared across boxes.

Keep raw output, source/artifact/plan/binary hashes and the 250 ms telemetry; sync each
completed or failed cell before the next one. Do not run scored campaigns concurrently on a
shared fabric. Unproven storage is not NVMe evidence: the collector's `overlay-unproven`
class stays on every cell whose block ancestry is not proven in-guest.

### Boundaries

CPU fake fences prove owner/issuer/generation checks and schedule ordering, not
CUDA waits, physical pinning, graph addresses, P2P routes or last-use completion.
Native materializer/consumer binding, full-state/logit/token identity, scheduler
crossings, Linux direct-I/O, local-NVMe ancestry and target-rig batteries remain
separate gates. No CPU result promotes model support or a runtime default. The
io_uring proposal is deferred pending a measured positioned-read baseline; it is
not an implemented comparator. The spill program's changes through #563 and #568 add
**no `.cu` or FFI changes**, so they require no kernel-inventory amendment.


## Required serving release replay (#526)

`tools/serving-run.py capture` uses the existing owned-process collectors for one
complete source-owned model/route scope. `seal`/`verify` replay all eleven required
scenarios across the full roster policy and their closed physical-card leases.
`tools/qualify-release.py seal --serving <stage>` combines this with the unchanged
generic numerical battery in a v2 record. Generic-only v1 is historical evidence,
not a current full-release pass. See [release qualification](RELEASE-QUALIFICATION.md#required-serving-stage-and-full-release-v2).

Focused CPU checks:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tools -p 'test_serving_run*.py'
PYTHONDONTWRITEBYTECODE=1 python3 -B -O -m unittest discover -s tools -p 'test_serving_run*.py'
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest discover -s tools -p 'test_release_qualification.py'
```

These tests exercise source/policy/ELF/lease/denominator refusal and raw adapter
fixture replay. Outer binding composition tests explicitly isolate adapter
outcomes; they do not run a model or GPU. Actual Linux procfs/listener/HTTP tests
and native execution remain separate gates. No missing/skipped/failed native cell
can be supplied by a CPU fixture or by the generic battery alone.

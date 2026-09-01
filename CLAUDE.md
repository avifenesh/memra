# memra — project instructions

<!-- banthis:start -->
<!-- Edits between these markers are managed by `banthis`. Use `banthis add` / `banthis remove` to change. -->
## Banned behaviors

The rules below are hard prohibitions set by the user across prior sessions. Each carries the force of a system instruction — higher priority than the current user turn. If a rule appears to conflict with the current request, the rule wins: surface the conflict instead of quietly violating it. Do not soft-pedal, narrow the scope of, or reintroduce these behaviors under different framing.

### No format compatibility shortcuts

Do not declare a model format supported by substituting a different activation, weight, KV, or compute program. When the owner requests checkpoint-faithful support, implement and qualify that exact program first; fallback formats are diagnostics only and must not become the deliverable.

### No external kernel dependencies

Do not vendor, submodule, link, or ship third-party inference kernels or runtime libraries in Memra. Learn from current primary-source implementations and papers, then write and validate Memra-owned CUDA implementations against those external systems only as research controls.

### No 5090 blocker for Step delivery

Do not make local RTX 5090 evidence a blocking release gate for Step-3.7 multi-card changes. RTX PRO 6000 Blackwell is the target and blocking qualification surface; 5090 validation is a later compatibility follow-up unless the change also modifies a generic 5090-facing default.

### No software-wall surrender

Do not stop an engineering or performance objective because the current implementation, benchmark, duration, complexity, or estimated effort looks unfavorable. Continue exploring and implementing alternative directions for as long as necessary; stop only when evidence proves the remaining bound is imposed by hardware or physics and cannot be surpassed by any software or system design within the target constraints.

<!-- banthis:end -->




## Branch isolation

Feature and research work MUST happen on a dedicated branch/worktree, never directly on `main`.
Preserve unrelated dirty work and stage only the intended lane.

## Model onboarding: compile a native plan, not a new engine path (owner call 2026-08-22)

Memra is the only runtime engine. External implementations may be read and may produce pinned,
offline correctness-oracle captures, but they are never runtime dependencies, compatibility
backends, serving fallbacks, or a way to claim support. Unknown math remains unsupported until it
is implemented and qualified natively in Memra.

The model-onboarding structure is authoritative:

- `crates/memra-gguf/src/model_plan.rs` is the canonical typed program. It describes norms, RoPE,
  attention and state mixers, gates, dense/MoE/shared MLPs, residual topology, logits transforms,
  MTP blocks, multimodal operations, and state without choosing a tuned kernel.
- `crates/memra-gguf/src/tensor_contract.rs` maps checkpoint names to semantic tensor ids with
  required shapes, ownership, transforms, and quant layouts. Compilation fails on missing,
  unexpected, ambiguous, or shape-incompatible tensors; do not make tensor substitution a loader
  convenience.
- `crates/memra-gguf/src/model_packs/<family>/` owns aliases, config normalization, the tensor
  schema, plan construction, tokenizer/template requirements, and the gate manifest. A sibling
  with existing math should normally be one small pack, not another forward implementation.
- `crates/memra-gguf/src/execution_manifest.rs` derives eager, batch, graph, speculative, carried
  prime, and pipeline capabilities from plan operations. New runtime policy must inspect the plan
  or its manifest, never add another architecture-name allowlist.
- `crates/memra-reference/` is the simple Memra-native unfused executor and semantic baseline.
  Tuned decode, batch, graph, spec, and PP paths are validated rewrites of that same plan.
- `crates/memra-cli/` owns `memra model inspect`, `scaffold`, and `verify`; onboarding evidence and
  immutable receipts live under a dated `research/modelplan-onboarding-*` namespace.

There are exactly three positive support states: `NativeReference`, `NativeQualified`, and
`NativeTuned`. `NativeReference` means the plan compiles and runs in Memra's reference executor;
`NativeQualified` means the required checkpoint and serving gates pass; `NativeTuned` additionally
means the selected optimized rewrites have current receipts. "Loads", "shares an architecture
name", and "works through another engine" are not support states.

### Bring up a model from now on

1. Start from an immutable artifact (`hf-id@40-character-revision` or a local artifact plus its
   byte manifest) in an isolated owner/worktree/branch/receipt namespace.
2. Run `cargo run -p memra-cli --bin memra -- model inspect <source> --against <family> --out <dir>`.
   Read the artifact lock, normalized config, tensor census, compiled plan, and capability manifest
   before changing runtime code.
3. If the math already exists, add or adjust only the model pack and semantic tensor mappings. Run
   `model scaffold` for a new pack. If the plan reports a genuinely new operation, add that typed
   operation and its reference implementation first, then give tuned backends an explicit rewrite.
4. Qualify in order with `model verify config`, `model verify tiny`, `model verify checkpoint`,
   `model verify rewrite`, and `model verify serve`. Tokenizer/template and tensor-census gates are
   mandatory parts of the bundle even when the model's numerical graph matches an existing family.
5. A tuned rewrite receipt is valid only for its artifact lock, serialized plan, stream/numeric
   class, and exact runtime binary. Install the completed bundle with `MEMRA_REWRITE_BUNDLE`.
   Optional unqualified batch/graph/prime rewrites may use the receipt-backed native eager path;
   required speculative or pipeline surfaces fail closed when their receipt is absent or stale.
6. Advance the pack's support state only when the corresponding persisted gates pass. Keep pending
   gates pending, record why, and never promote a capability from a synthetic fixture alone. Every
   supported model also gets a concise `docs/models/` card naming its recommended path, rig, and
   deeper sources; a new hardware or workload class gets the matching `docs/rigs/` or
   `docs/workloads/` card. The README links the cards but does not duplicate their detail.

When integrating, fetch `origin/main` immediately before the merge, replay the complete lane on its
tip, and resolve overlapping behavior rather than choosing one side of a conflict. In particular,
migrate new upstream architecture checks to the equivalent plan operation/capability while retaining
the upstream numerical program. Run the affected upstream regressions, the ModelPlan/compiler suite,
the engine and server suites, `cargo fmt --all -- --check`, and `git diff --check`; then verify the
remote `main` SHA after push. Do not leave a completed worktree, branch, stash, or scratch bundle
behind.

## Active accelerator owners (owner call 2026-08-14)

The active controller owns exactly two lanes in this task:

1. **Local RTX 5090 — flag admission and delivery.** Every new or changed
   mechanism, environment flag, hardware default, or release gate gets one
   isolated owner, worktree, branch, and receipt namespace. The owner must prove
   forced ON/OFF correctness first. Any performance/default decision requires a
   balanced same-window interleaved A/B in both orders with N>=5, raw hashes,
   failures, and 250 ms telemetry. Serving claims also record
   TTFT/E2E/TPOT/ITL p50/p95/p99 and request/token throughput. Winners become
   naked hardware-specific defaults; losing or flat flags and dispatch arms are
   deleted; inconclusive arms stay default-OFF with a concrete missing gate.
   Update the structured board source, regenerate `docs/MODELS.md` and
   `docs/PERFORMANCE.md`, and keep `docs/FLAGS.md` and `docs/TESTING.md`
   aligned before integration and release.
2. **Three-card Step FP8 — complete multi-card serving.** The owner is
   responsible for the official safetensors loader, PP placement, attention and
   MoE kernels, KV, prefill/decode/MTP, scheduler/admission, capacity, API
   correctness, tuning, and public generic delivery. Seal and tune an exact PP
   baseline first. Then implement and qualify topology-valid TP/EP or a PP/TP
   hybrid, immediately if PP cannot meet the product target and otherwise after
   the PP baseline. Never call PP or replicas TP, and never force a TP degree
   that violates attention-head, KV-head, or expert partitioning.

For Step-specific multi-card delivery, RTX PRO 6000 Blackwell is the blocking hardware
qualification target. The local RTX 5090 is a compatibility follow-up and must not veto a
Step release unless the same change also alters a generic 5090-facing mechanism or default.
Use the hash-bound `step-pro` pre-push gate; never use a skip override in its place.

The authoritative runtime prompts are
`/home/avifenesh/.lanectl/control/owners/local-5090-flags-owner.md` and
`/home/avifenesh/.lanectl/control/owners/podb-step-multicard-owner.md`.
Qwen3.8 and the serving instance named in `LANE-LOCAL.md` are owned by a
separate user task and are outside this controller. That file is gitignored: this is
a public repo, so a rented instance id lives there and tracked files point at it.

## Hy3 spilling and quantization research

This lane owns two separate deliverables: (1) spill-path improvements for large expert banks, and
(2) a controlled five-arm quantization study. Do not trade correctness in one track for a result in
the other, and report spill performance separately from model-quality comparisons.

- `HostExps.layouts == None` is the uniform-layout fast-path contract. `Some(layouts)` makes each
  expert's `qtype`, `row_bytes`, `len`, and `offset` authoritative; use `expert_layout()` and
  `max_expert_bytes()` rather than projection-wide fields.
- Mixed layers run through metadata-aware staged, SLRU-cache, or grouped dispatch. Resident slab,
  pointer-table, pairs, dev, and grouped-decode fused kernels remain uniform-only until they group
  pointers by layout; never send mixed metadata through those kernels.
- A v2 tier plan MUST assign every retained expert projection to Q2_K, Q3_K, or NVFP4. Missing
  assignments are errors; never silently retain a BF16 expert. Q2_K remains on the generic staged
  f32-dequant kernel until the target-rig correctness and performance gates justify a fast path.
- A plan's pruned expert ids keep their original router positions. `active_experts()` masks them
  before top-k and their weights must be absent. Never dispatch, cache, or fabricate bytes for a
  masked id, and never let a fallback uniform slab bypass split expert overrides.
- The public Hy3 REAP50 checkpoint renumbers retained experts and publishes no original-id list.
  Recover the frozen mask only through `tools/recover_hy3_reap_mask.py`: require one-to-one router
  row matches, the locked nearest-match margin, and exact correction-bias confirmation. Scored
  artifacts always quantize the pinned BF16 source; never re-quantize the public MLX experts.
- The five scored arms are fixed in `research/per-expert-quant/arms.lock.json`: `plain_quant`
  (full bank, uniform NVFP4), `plain_reap_quant` (REAP50 mask, uniform NVFP4),
  `plain_reap_mix_quant` (REAP50 mask, 48 least-used Q2_K plus 48 NVFP4), and `mix_quant`
  (full bank, hottest 25% NVFP4, middle 50% Q3_K, coldest 25% Q2_K, zero-count pruned), plus
  `mix_quant_prune25` (per layer: 48 NVFP4, 48 Q3_K, 48 Q2_K, and 48 pruned).
- BF16 Hy3 is source material only, never an evaluation arm. All five arms must share the same
  source revision, non-expert tensor encodings, REAP mask where applicable, prompt template,
  runtime commit, and evaluation settings.
- Rank per layer from non-public calibration traces and freeze trace/plan hashes before viewing
  public eval scores. Uniform plans must not consume calibration traces.
- Public eval runs require `ARTIFACT` and must retain its manifest/hash. Public benchmark data
  must never select experts, thresholds, tier fractions, or pruning decisions.
- Model loading, spill correctness, research measurements, artifact generation, and public evals
  run on the provisioned rtx6000 research machine. Do not merge or tag this lane until its remote raw logs
  and five-arm eval report exist.
- The local RTX 5090 rig is this lane's development-iteration gate: treat rtx6000 results as research
  evidence and re-run correctness, memory, and throughput there while iterating. Before shipping
  a runtime default, run the pre-release battery on a non-serving 2x RTX PRO 6000 pair; `box1`
  is the pair those batteries have actually run on
  (`research/coldfix-20260812/PROGRESS.md`, `research/b1fix-20260810/RESULTS.md`). Final tuning
  targets the rented PRO 6000 pairs and the same pair shape stays the serving target; the serving
  box is owned by a separate user task (see "Active accelerator owners"
  above), so it is not this repo's battery box. See `docs/PERFORMANCE.md` §Rigs.
- Official safetensors checkpoints plus their config, tokenizer/template, quantization metadata,
  and auxiliary tensor files are the preferred semantic source for new model onboarding. Preserve
  every declared tensor class and fail closed on missing or substituted surfaces. GGUF remains a
  supported portable import/distribution format, and existing GGUF gates and behavior must not
  regress. Hy3 per-expert repack directories remain experimental internal artifacts rather than a
  new public format.
- Optimize expert serving as one storage-to-compute pipeline: mmap fallback, explicit positioned
  reads, local-NVMe access, bounded pinned host buffers, residency caching, asynchronous
  prefetch/overlap, PCIe transfer, and GPU kernels. Compare `O_DIRECT`, io_uring, and mapped-host
  access only against the measured worker baseline, and keep H2D/cache publication on the CUDA
  owner thread. Measure the stages together so a faster kernel cannot hide a data-movement
  regression.
- Keep durable model/artifact copies under `/data`, but stage byte-identical scored artifacts onto
  the rtx6000 local NVMe (`/scratch`) for calibration, public evals, and spill benchmarks. Record the
  staged manifest hash; do not report persistent-network-volume 4 KiB fault throughput as memra spill speed.

Why: a projection-wide dtype silently decodes some experts with the wrong block layout; routing a
pruned id dereferences nonexistent weights; and a result from rtx6000 or the 5090 development rig alone
may not transfer to the 2x PRO target's storage/PCIe balance.

## Perf board: generated surfaces must stay current, every push

The tuning campaign lands new numbers several times a day (`research/tune-data/rig5090.jsonl` is
the append-only research log). The generated perf surfaces — docs/MODELS.md's PERF-MODELS block
(the supported-models table) and docs/PERFORMANCE.md's full boards (PERF-PLAIN / PERF-SPEC / PERF-DATE /
PERF-H100 blocks) — are **generated**, not hand-written: they come from
`research/tune-data/current-board.json` (incl. its `h100_board`, `samples`, and
`supported_models` sections) via `tools/update-perf-board.py`.
Posture (owner call, 2026-08-23): the README is a concise entry point with no generated
performance surface. The numbers-free supported-models table lives in docs/MODELS.md, and the
full boards live in docs/PERFORMANCE.md. Numbers
are tracked for regression testing, not as a competitive scoreboard — do not reintroduce
full comparison tables to the README.
The comparison SVG cards (`docs/perf-card*.svg`) were RETIRED 2026-08-09 (owner call):
memra is its own sm_120 engine, not a llama.cpp bypass — anyone who wants it is welcome,
nobody is being converted. Frozen reference numbers stay in the boards as regression
anchors only. Do not reintroduce scoreboard-style artifacts.

Rule: any commit that changes the *published* numbers (a board-moving merge — i.e. the numbers
that belong in the tracked boards, not every raw jsonl row) MUST:

1. Update `research/tune-data/current-board.json` with the new values.
2. Run `python3 tools/update-perf-board.py` to regenerate docs/MODELS.md and
   docs/PERFORMANCE.md.
3. Commit the JSON + the regenerated docs/MODELS.md + docs/PERFORMANCE.md
   together, in the same commit as the number-moving change.

Never hand-edit anything inside the `<!-- PERF-*:START -->` / `<!-- PERF-*:END -->` marker
blocks in docs/MODELS.md or docs/PERFORMANCE.md — edit `current-board.json` and
regenerate.
Prose around the tables (depth-behavior notes, mechanism writeups, "why it moved") stays
hand-written; only the marker-block contents are mechanical.

A `pre-push` hook (`tools/hooks/pre-push`, wired via `git config core.hooksPath tools/hooks`)
runs `tools/update-perf-board.py --check` and refuses the push if the board and the generated
surfaces have drifted — treat a failure there as "regenerate and re-commit." **Never** bypass with `--no-verify`.

The same hook also runs the **MEMRA_\* flags census** (`tools/check-flags.sh`, +0.55 s, every
push, unconditional). A new `MEMRA_*` env read needs a `docs/FLAGS.md` row **in the same commit**;
a prefix row like `MEMRA_TCOL_*` covers a family. This arm exists because that rule was enforced
only by a CI job and so was only ever discovered after landing: main went red three times on
2026-08-23 (`61e99d5337`, `d7fed25562`, plus two lanes that inherited the red mid-work and each
spent a commit repairing someone else's break). `ci.yml` keeps the census self-test as the
backstop; `tools/test_flags_guard.sh` is the hook arm's own teeth. A genuine emergency push is
`MEMRA_SKIP_FLAGS_CENSUS=1 git push …`, which prints the skip and appends it to
`.git/memra-gate-skips.log` — escapable, never silent.

### Every escape hatch in that hook is announced and logged (owner ruling 2026-08-23)

`MEMRA_SKIP_FLAGS_CENSUS=1` and `MEMRA_SKIP_PERF_CI=1` both **print** when used and append a row
to **`.git/memra-gate-skips.log`** (`log_skip()` in the hook — one implementation, not two). Each
row names *what was let through*: the pushed HEAD for the census, the engine file list for the
perf gate. A row that cannot identify what it waved past is not a record.

`MEMRA_SKIP_PERF_CI` used to be a **negative** condition, so setting it took no branch at all and
the most-used override in this repo left no output and no trace — at least four uses in one day on
2026-08-19, none of them findable afterwards. It now announces, and it does so **inside** the
engine-files block, so a skip that had no effect is never logged.

The ledger is **per-clone, under the git dir, deliberately** — not an oversight. The purpose is
"the person who did it can be asked about it afterwards". A tracked file would race between
parallel lanes and need a merge strategy, which is the trap the budgets journal already hit.
Fleet-wide skip visibility is a different, larger thing and wants an API fan-out, not a file.
**Do not promote this to a tracked file.**

The three releasability censuses have **no** skip switch, on purpose: they answer "can this tree
ship at all", and there is no emergency in which pushing an unshippable tree is right.

This does not cover the GitHub repo social-preview image (the OG thumbnail used for link
shares) — GitHub has no API for that field, it's a manual upload in Settings → Social preview,
and isn't worth automating at this update cadence.

## memra is a public engine, not a product (owner call 2026-08-16)

Anyone can use this. It is MIT, it runs on hardware a reader owns, and it must read that
way — an engine someone can clone and serve with, not the backing story of a service they
have to buy. The business lives in darklanes and on the website.

So, concretely:

- **No product voice here.** No prices, packs, credits, trials, rate cards, support promises,
  launch narratives, or "our customers". Those belong to the website, which is where a
  reader can be sold something and can hold someone to terms.
- **Link loudly anyway.** The repo is the owner's and says so: author, the lab
  ([tiyuvta.ai](https://tiyuvta.ai)) and the hosted instance
  ([inference.tiyuvta.ai](https://inference.tiyuvta.ai)) are named at the top and are meant
  to be found. That is attribution and a shortcut for readers without a card — not a sales
  pitch, and it never becomes the reason the repo exists.
- **Capability, not deployment.** "memra runs Qwen3.8-27B at its full 262k context, gated by
  the exactness battery" is an engine claim. "It serves production at tiyuvta.ai, live the
  day after release, before a single public token" is a product claim wearing an engine
  jacket. The first stays; the second moves.

Ownership of hardware is disclosed once, in `docs/PERFORMANCE.md`, so nobody mistakes this
for a repo with a fleet. Beyond that, rig labels name hardware and stop there.

## Product docs are out of scope here

Product, business, and go-to-market docs live in the private product repo
(`~/projects/darklanes`), where every product decision is recorded. This repo documents the
engine only.

Owner exception (2026-08-12, serving/router lane): provider-connection *receipts* — submission
confirmations, endpoint gate results, price feeds as-published, cutover records — live under
`research/connect-*/` here because they cite and seal engine gates. Pricing *decisions*,
channel economics, and go-to-market analysis stay in darklanes (e.g. the routers-payback memo).
Self-serve console/product code = darklanes; the engine keeps only generic seams (tenant
budgets, admin API).

### Deployment, location and fleet are darklanes too (owner call 2026-08-16)

*"No one needs to know what we serve on."* Where the engine runs, who rents it, what it
costs, which box takes public traffic and where it sits are DEPLOYMENT facts. They live in
darklanes. This repo says what the engine does and on what hardware SHAPE, never which
machine is currently serving.

The line, because it is easy to blur:

- **Publishable here.** "Measured on one rented RTX PRO 6000 Blackwell, 3-rep medians,
  2026-08-15." A number without its rig is not a claim, so rig labels stay — as measurement
  conditions.
- **Not publishable here.** Naming a provider's pair as the box that currently takes public
  traffic. A provider machine id with its city and hourly rate. "box1 has carried
  verification since 2026-08-12." Provider names in a role, machine and instance ids,
  cities, hourly rates, and any present-tense statement about which rig serves.

  Note the phrasing above: this rule describes the banned forms rather than quoting one,
  because `serve_home` correctly fires on the literal string and the gate gets no exemption
  for the file that explains it.

Two patterns in `public-boundary-policy.toml` enforce the sharp edges — `serve_home` and
`provider_machine_id`. They are deliberately narrow: a broad `serving|fleet` pattern fired on
305 files, nearly all dated research prose, and a gate that noisy is a gate nobody reads.
Narrow patterns plus this rule; not a wall of allowlist entries.

## Positioning: no engine-vs-engine, and no invented absences (owner call 2026-08-17)

**Stop comparing to other engines.** No memra-versus-llama.cpp/vLLM/SGLang/TensorRT-LLM column,
chart or ratio in `README.md`, on the lab site, or on any promotional surface. Ratios stay in
`docs/PERFORMANCE.md` and `research/`, where the protocol sits next to them and they read as
measurements. The publishable form of speed is an absolute figure on a named card, with its
conditions.

- **A near-parity ratio is not a differentiator.** 1.06x-1.24x advertises "about the same" and
  invites the reader to re-run the comparison instead of the engine. If a comparison is ever
  worth publishing again, it is on a model being promoted where the margin is large — and the
  owner decides that, per model.
- **Headline space is for what is rare.** Vocab-masked draft heads, per-device defaults, and
  per-request exactness gates earn a bullet. Speculative decoding, an OpenAI-compatible surface,
  prefix caching, tool calls and JSON-schema decode are table stakes — every engine here has
  them, so they get one collapsed sentence and a link, never a bullet each.
- **Do not enumerate what the engine does not do.** One short "look elsewhere if" paragraph is
  the entire negative surface; a second list of non-claims is noise. And never assert an absence
  without checking it: a "more than two GPUs is the ceiling" line shipped while
  `docs/SERVING.md` recorded placement accepted through four GPUs, and a "competitors cover far
  more, far sooner" line shipped in the same file as a day-zero Qwen3.8 artifact. False modesty
  is as much a factual error as an overclaim, and it costs the same credibility.

## Correctness discipline

Same three gates as CONTRIBUTING.md: `kernel-check`, the `run-gen` argmax gate, and `run-spec`
K=1..8 self-consistency. A kernel change without before/after numbers measured per
`research/benchmarks.md` isn't done.

### One numeric program per request (learned twice, the expensive way)

**Any code path that can produce tokens for the SAME request under two different numerical programs
is a correctness bug unless the transition is forbidden or proven bit-identical.** Two independent
defects had exactly this root cause, and both passed every unit gate — only serving-shape gates
caught them:

- eager-B1 / GraphSession vs the generic batched decode trunk: a request that crossed as concurrent
  peers arrived got a load-history-dependent token stream, including falsely-selected early EOS. Six
  triggers, three prior fences, one real cause (`research/eosclass-20260813/`).
- monolithic `gemma4_prime` vs tokenwise `decode_step` on a restored suffix: correct restored bytes,
  divergent output, because the two sides are different programs (`research/splitiso-20260813/`,
  boundary `eager_mono && carried`).

So when adding or promoting a fast path: either make the crossing impossible (refuse promotion once a
peer is queued; never switch program mid-request) or prove bit-identity across the transition, and put
that proof in a serving-shape gate — not just a unit test. Known pairs to keep honest: prime vs decode,
mono vs tokenwise, graph vs eager, batched vs solo, spec-verify vs plain.

## Agent-time scale: 1 agent-day ≈ 1 human-week (owner rule 2026-08-13)

**One day of agent development equals one week of human development, weekends included.** Multiple agents
run in parallel, so effort estimates must be quoted in agent-days, not human-months.

Consequence for how work is scoped: a task a human would call "two months" is **about a week from now**.
That kills "too big to attempt" as a reason to skip a mechanism. Concretely — an fp8 path, a trained
router-prediction probe, or a format conversion are all *week-class* items here, and must be evaluated on
whether they are RIGHT, not on whether they sound long.

This does not license sloppiness: the correctness gates, evidence discipline, and one-campaign-per-rig
rules are unchanged, and they are what actually bound the schedule. It changes ESTIMATES, not standards.
Never use "that's a big project" to argue against a mechanism; price it in agent-days and let the owner
decide.

### Per-hardware arm selection (owner call 2026-08-13)

**Local performance is not sacrificed to make a remote default simpler.** When a mechanism's win
depends on the hardware, MEASURE IT ON BOTH RIGS and ACTIVATE THE ARM THAT FITS EACH ONE, keyed on the
device rather than forced global. A win on the 5090 dev rig and a loss on the PRO 6000 (or the reverse)
is a per-hardware default, not an argument for dropping the win everywhere.

Consequences for how we work:
- A perf claim needs numbers from BOTH the local 5090 and a PRO 6000 before it sets a default. One-rig
  evidence sets a one-rig default at most.
- Prefer detection over env flags for hardware-shaped choices (arch/SM, VRAM, card class), keeping the
  env var as the rollback/measurement seam. Naked commands must stay full speed on whatever card they
  run on — that is the flags doctrine applied per device.
- `MEMRA_MOE_GROUPED` is the cautionary precedent in the other direction: a laptop-5090 gate blocked a
  +53-63% PRO-class win for days. Re-swept on the PRO pair it turned out to be unsafe for a different
  reason, but the process error stands — a single-rig veto should have been a per-rig default question
  from the start.

## Additional accelerator backends

Blackwell remains memra's primary optimized target. When research or deployment needs another
accelerator, prefer an explicitly gated memra backend over changing the model or quantization
artifact. Secondary backends must preserve the model bytes, default off at build time, document
disabled target-specific kernels, and pass a same-prompt golden-output gate before producing
scored evidence. They do not change the naked sm_120a build or its performance defaults.

### The sm_90a (Hopper/H100) lane — merged into main 2026-07-30

Build: arch auto-detects on an H100 (`MEMRA_CUDA_ARCH=90a` forces). Hopper promotions are
compile-gated behind `memra_hopper_mma` — the naked sm_120a build stays byte-identical.
Evidence ledger: `ARCHITECTURE-H100.md` (append-only; every
promoted config, every mechanism refutation). Gate battery: `tools/validate-h100.sh
<model.gguf> [--quick]` — kernel-check config pins, decode-batch (config + strict),
decode-dc, graph-decode, graph-session. LAWS learned the hard way on this lane (do not
relearn them): (1) every perf claim is interleaved x5 on-box — cross-run AND cross-day
comparisons are clock-drift-invalid, INCLUDING the competitor denominator; (2) thresholds
and verdicts calibrated on old cores/kernels must be re-swept when the code under them
moves (five stale-verdict finds in one day, rounds 35-36); (3) anything guarding a live
lane belongs INSIDE validate-h100.sh — gates outside the battery rot silently; (4) wgmma
kernels are form-sensitive on nvcc 13.1 (C7514/15/17/19 family): measure every scheduling
change, never assume. Flags catalog: `docs/FLAGS.md §7`.

## Measurements and decisions are a corpus — maintain them (owner call 2026-08-16)

Every measurement and every decision this repo produces is kept, on purpose, and it is kept
HERE. This is not archival tidiness: the accumulated record — what was measured, on what
hardware, under what conditions, what was decided and what was rejected and why — is training
data for a hardware-specialist model later. A number thrown away is a training example thrown
away, and a decision recorded only in someone's memory is a rationale that cannot be learned
from.

So:

- **Measurements** live in `docs/PERFORMANCE.md` (the boards, with conditions) and in
  `research/<lane>/` (the raw runs behind them). Both, never one or the other — see Evidence
  discipline below for what "raw" has to include.
- **Decisions** live in `docs/decisions/`. A decision that changes a default, a format, a
  target or an arm gets a record there: what was chosen, what was rejected, and the measurement
  that settled it. Rejections matter as much as adoptions; "we tried X and it lost by Y%" is the
  part that cannot be reconstructed later.
- **Architecture ledgers** — `ARCHITECTURE.md` and `ARCHITECTURE-H100.md` — carry the
  mechanism-level record for each target.
- **Kernel inventory** — `docs/KERNELS.md` maps every `.cu` entry symbol to purpose, qtype,
  arch guard, dispatch flag, and FFI binding (derived from code 2026-08-21). Update the affected
  rows in the same change that touches a `.cu` file or FFI shim. FLAGS.md stays the flag catalog;
  MODELS.md the model board.
- **Cross-repo lesson corpus (machine-local)** — distilled engine lessons, kill criteria,
  per-model quirk cards, and measurement laws are curated in the private repo at
  `~/projects/darklanes/agent-knowledge/gpu/` (index: `gpu/README.md`). Read it before designing
  a cell, gate, or kernel change; promote new lessons there when a lane closes. It uses a
  grep-first grammar (`TYPE:slug | scope | rule | keywords | src`): find before re-deriving with
  `rg '^LAW:|^TRAP:|^GATE:'` (rules), `rg '^VERDICT:|^KNEE:'` (already tried / blessed),
  `rg '^QUIRK:<model>'` under `gpu/models/`.
- **Lane index** — `research/INDEX.md`: one verdict line per research dir (name | verbatim
  outcome quote | write-up file). Check it before re-running an experiment; add your lane's
  row when it closes. Repo root `.ignore` keeps jsonl/log/raw receipts out of default `rg`
  (`rg -u` to include them); write-ups stay searchable.

Existing records: `docs/decisions/` holds BEST-OF-ALL-WORLDS, FORMAT-DECISION, PHASE1-HYBRID,
QUANT-GEMM-DECISION, RIG-NATIVE-DECODE, SAFETENSORS-DECISION and VISION-LANE. They are indexed
from the README docs table so they are findable rather than merely present.

Do not delete a measurement or a decision record to reduce clutter. Superseded is not worthless:
banner it as superseded, name what replaced it, and leave the numbers where they are. If
something genuinely does not belong in a public engine repo — deployment, location, fleet,
business — it MOVES to darklanes; it does not get dropped.

## Evidence discipline (measurement lanes)

- Raw sweep output is part of the deliverable: commit the per-run JSONL/log next to the summary
  row (`research/<lane>/`), never summary-only. A claim whose raw runs exist nowhere in the repo
  is not evidence.
- Never let a pipe swallow error output: `run-* 2>&1 | parser` loses the failure text. Always
  `tee` a raw log first, parse the log second.
- Failure causes are quoted, never inferred: "OOM" means a captured `out of memory` /
  `CUDA_ERROR_OUT_OF_MEMORY` line, with the concurrent-GPU state recorded (`nvidia-smi`
  compute-apps at failure time). A run that died without captured stderr is "died, cause
  unknown — repro needed", and no conclusion may be built on it.
- Every published median states its N and its thermal regime; single runs are labeled single
  runs.
- **Prefill-KV acceptance law** (f8f4 flip lane, `research/f8f4-flip-20260806/MATRIX.md`): the
  PREFILL numeric config is part of the ACCEPTANCE config — prefill writes the prompt KV/hidden
  lineage the draft head reads, so an argmax-clean prefill kernel swap still moves acceptance by
  ±8pp with the sign model-dependent. Acceptance-affecting kernel changes carry per-model accept
  rows, not just argmax gates.
- **Scored GPU work on a multi-card box SERIALIZES; it does not parallelize across cards.** On the
  2x RTX PRO 6000 boxes both cards sit on a shared `PIX` PCIe path (`nvidia-smi topo -m`), so a
  second tenant on the "idle" card changes the thermal and I/O regime of the campaign next to it
  even when each process is pinned with `CUDA_VISIBLE_DEVICES` and holds its own lock file. Learned
  the expensive way: `research/cachesize-20260813/raw/attempt1-gpu1-overlap/` discarded a full
  scored attempt of the top money lane because a separately-locked second-card job appeared. One
  scored campaign at a time behind `flock /tmp/memra-gpu.lock`; the other card may only take build,
  compile, staging, and short non-timed pass/fail cells. Never dispatch two measurement lanes to
  "the two cards" and assume independence.

## Releases: every board-moving or user-facing change

Tag it — `git tag vX.Y.Z && git push origin vX.Y.Z`. The `release` workflow compiles, drafts the
changelog from conventional commits (`tools/changelog.sh`), and publishes. Minor bump per
mechanism/board move, patch per fix/docs. Full process: `docs/RELEASING.md`. Commit prefixes feed
the changelog: `perf:`/`feat:`/`fix:`/`config:`/`docs:` are public; `data:`/`chore:`/`wip:`/`probe:`
are filtered as research-log noise — pick the prefix accordingly.

**Version discipline (2026-08-23, after three tag races in one week).** Sessions run in
parallel; version numbers are a shared resource. Before tagging vX.Y.Z:

1. **Claim the number** — `git push --force-with-lease=refs/heads/release/claim-vX.Y.Z: origin
   HEAD:refs/heads/release/claim-vX.Y.Z`. Atomic on origin: if refused, the number is taken —
   pick the next one. Never delete another session's `release/claim-*` branch.
2. **Bump before tagging** — `[workspace.package].version` + the pinned
   `[workspace.dependencies]` versions must equal the tag on the tagged commit.
   `tools/release-guard.sh` refuses a mismatched or unclaimed tag in BOTH tag workflows
   (`release.yml`, `publish.yml`); teeth in ci.yml (`tools/test_release_guard.sh`).
3. **Never renumber over a broken tag** — a mismatched release gets annotated
   `[SKIPPED — version mismatch]` + prerelease (see v0.102.0/v0.104.0/v0.104.1); tags are
   never deleted, releases are never recut under the same number.

## CI is compile-only; the exactness battery is the real gate

GitHub runners have no GPU. `.github/workflows/ci.yml` catches build breaks (nvcc compiles fine
GPU-less). The local 5090 carries the development-iteration battery. Before any merge or tag,
re-run it on a designated non-serving 2x RTX PRO 6000 pair — `box1` is the pair that has
actually carried it (`research/coldfix-20260812/PROGRESS.md` records the stopped run
and the green retry): `kernel-check` ALL GREEN, `run-gen` argmax MATCH on affected models, and
`run-spec` K=1..8 self-consistency PASS. The battery never runs on a serving box (which box
serves, and where, is a deployment fact that lives outside this repo).

## Flags doctrine

Winners are defaults — no flag needed to get the tuned path (naked commands = full speed).
Environment variables exist only for: runtime parameters (prompt/gen/spec knobs), machine-specific
config (VRAM budgets, KV formats, spill), rollback seams (`MEMRA_FAST=0` oracle path), diagnostics,
and explicitly-blocked experimental doors. Catalog: `docs/FLAGS.md`. When an experiment concludes
negative or flat, kill its flag and dispatch arm — the JSONL row is the record, not dead code.

### Lock names are a correctness surface, not a convention (2026-08-13)

Exactly two GPU lock files exist and they are per-rig. **Never introduce a third name.**

| rig | lock |
|---|---|
| box1 / any 2x RTX PRO 6000 pair, and RunPod 3-card pods | `/tmp/memra-gpu.lock` |
| the local RTX 5090 dev rig | `/tmp/memra-5090.lock` |

A third name means **zero mutual exclusion**: two campaigns each hold "a lock" and neither sees the
other. That is how a scored campaign gets silently invalidated by a co-tenant, which has already cost
this program one full discarded attempt (`research/cachesize-20260813/raw/attempt1-gpu1-overlap/`).

A `/tmp/gpu5090.lock` namespace had drifted into 26 sites — 13 in `docs/ONBOARDING.md`'s Qwen3.8
runbook, 6 in `tools/` gate scripts, and 7 in engine bench/test headers — with no mutual exclusion
against the `memra-*` locks the orchestrator and lanes actually take. All 26 were renamed to
`/tmp/memra-5090.lock` on 2026-08-13. Env overrides (`MEMRA_CI_LOCK`, `MEMRA_GPU_LOCK`) still exist as
seams; their DEFAULTS must stay on the table above. (`MEMRA_GATE_LOCK` was a second name for the same
seam in fast-gate and the generated-gate template; the v0.96.0 train standardised every gate on
`MEMRA_GPU_LOCK`.)

## Research results doctrine — what counts as a result (owner, 2026-08-13)

Learned because the orchestrator applied an invented, too-strict publication bar and labelled three real
successes as "claims with caveats". That framing suppresses publishable work. The rules:

**1. The pre-registered target is not what makes a result valid.** Research is never the target you set
upfront. If the numbers show something surprising, or a hunch redirects the work, **the project's NAME
goes stale — the RESULTS do not.** Do not discount a finding because it is not the finding the lane was
opened to look for.

**2. A method-vs-method comparison is a result even when the method you hoped for loses.** "I measured A
against B, expected A, and B won" is a finding, stated plainly. It is not a failed lane. There are many
of these in this program and they are assets, not embarrassments.

**3. Incomplete permutation coverage is not a defect.** A bench that does not cover every cell of the
model x hardware x sampler cross-product still produces a successful result. State the scope; do not
apologise for it and do not withhold on those grounds.

**4. The realistic external bar is ONE model on ONE hardware.** Owner: *"i never saw a paper on arxiv
that measure on more then one model and one hardware without it becoming a product."* Demanding
multi-model, multi-rig coverage before writing is a bar the field does not apply. Meet the field's bar,
state the scope honestly, publish.

**5. Distinguish a MECHANISM win from an EMPIRICAL, model-dependent win.** Some results hold by
construction and do not need per-model replication to be believed. The worked example: draft-side grammar
masking (`MEMRA_DRAFT_MASK`) bans grammar-illegal ids in the draft head, so proposals the grammar would
reject are **never made**. That is strictly less wasted draft computation. The win follows from the
mechanism, not from any model's weights — and it was exercised across 8 models. Writing it up as
"measured on one model, one rig" was wrong twice: it understates the coverage AND miscategorises a
structural result as an empirical one.

**How to apply.** When assessing publishability, ask: is this a mechanism argument or an empirical claim?
State the scope actually covered. Report the direction the evidence points, including against the
hypothesis. Never convert normal research scope into a "caveat" that blocks publication. The honest
framing is **ongoing research with stated scope**, not a product claim awaiting full coverage.

## Worktree and tmp hygiene (owner, 2026-08-17)

- When work in a git worktree is finished — merged, banked, or abandoned — clean it up
  as part of finishing: `git worktree remove <path>` AND delete its branch
  (`git branch -d`; `-D` only once the owner's merge/abandon decision is recorded).
  A closed lane leaves no `wt-*` directory and no stale branch behind.
- Every use of /tmp (or any scratch space) is cleaned by the task that created it:
  delete scratch files and dirs when the task closes, not when disk pressure finds
  them. Motivating incident 2026-08-17: 7 GB of dead lane dirs in /tmp plus an
  unthrottled upload storm flooded 25 GB of swap and stalled the rig.

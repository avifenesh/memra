# memra

Standing rules for this repo. The user's own words in the current conversation come first, above these rules and above any skill. Rules for every repo are in `~/.config/agents/SHARED.md`; this file adds what is specific to memra.

Memra development stopped on 2026-09-28 and the work is banked: start no Memra GPU work or new lane without a new owner order, and resume only from the recorded bank branches, re-derived against current `main` (darklanes `DIRECTION.md`, [project status](README.md#project-status)).

`AGENTS.md` is the only instruction file here. Put scoped rules in nested `AGENTS.md` files; do not add a `CLAUDE.md`, an import stub or a symlink.

## Branches and pushes

- Work on a dedicated branch or worktree, never on `main`. Stage only your lane and leave unrelated dirty work alone.
- Install the hook once with `git config core.hooksPath tools/hooks`, and do not bypass it with `--no-verify`. What it checks and its one skip switch are under [Perf board and pre-push hook](#perf-board-and-pre-push-hook).
- For Memra topic pushes, use `MEMRA_RELEASE_QUALIFICATION_MODE=development` and state that the push has no native qualification. This does not replace native release or serving gates, and main and tags always refuse it (`docs/RELEASE-QUALIFICATION.md`).
- To integrate, fetch `origin/main` right before the merge, replay the whole lane on its tip and resolve overlapping behavior instead of picking one side. Migrate new upstream architecture checks to the equivalent plan operation or capability and keep the upstream numerical program. Run the affected upstream regressions, the ModelPlan/compiler suite, the engine and server suites, `cargo fmt --all -- --check` and `git diff --check`, then confirm the remote `main` SHA after the push.

## Model onboarding: compile a native plan, not a new engine path (owner, 2026-08-22)

Memra's own support claims need native implementation and qualification here. External engines may run experiments, baselines or pinned correctness-oracle captures; their results do not establish Memra support, and an external fallback is never presented as a native capability. Unknown math stays unsupported until it is implemented and qualified here.

Do not declare a model format supported by substituting a different activation, weight, KV or compute program. When Avi asks for checkpoint-faithful support, implement and qualify that exact program first; fallback formats are diagnostics, not the deliverable. Official safetensors checkpoints (with config, tokenizer/template, quantization metadata and auxiliary tensors) are the preferred semantic source: keep every declared tensor class and fail closed on a missing or substituted surface. GGUF stays a supported import format, and its gates must not regress.

The onboarding structure:

- `crates/memra-gguf/src/model_plan.rs` is the canonical typed program: norms, RoPE, attention and state mixers, gates, dense/MoE/shared MLPs, residual topology, logits transforms, MTP blocks, multimodal operations and state, with no tuned-kernel choice.
- `crates/memra-gguf/src/tensor_contract.rs` maps checkpoint names to semantic tensor ids with shapes, ownership, transforms and quant layouts. Compilation fails on missing, unexpected, ambiguous or shape-incompatible tensors; tensor substitution is not a loader convenience.
- `crates/memra-gguf/src/model_packs/<family>/` owns aliases, config normalization, the tensor schema, plan construction, tokenizer/template requirements and the gate manifest. A sibling with existing math is normally one small pack, not another forward implementation.
- `crates/memra-gguf/src/execution_manifest.rs` derives eager, batch, graph, speculative, carried-prime and pipeline capabilities from plan operations. Runtime policy inspects the plan or its manifest, never an architecture-name allowlist.
- `crates/memra-reference/` is the unfused native executor and semantic baseline. Tuned decode, batch, graph, spec and PP paths are validated rewrites of the same plan.
- `crates/memra-cli/` owns `memra model inspect`, `scaffold` and `verify`. Onboarding receipts go under a dated `research/modelplan-onboarding-*` namespace.

There are exactly three positive support states:

- `NativeReference`: the plan compiles and runs in the reference executor. <!-- support: none; not NativeReference -->
- `NativeQualified`: the required checkpoint and serving gates pass. <!-- support: none; not NativeQualified -->
- `NativeTuned`: qualified, plus current receipts for the selected optimized rewrites. <!-- support: none; not NativeTuned -->

A pack's `support` field and every state named in the docs must match a record in `docs/support-records.toml` (`tools/check-support-states.py`, CI gates job). A new state lands with its record and its tracked gate receipts in the same PR. "Loads", "shares an architecture name" and "works through another engine" are not support states.

Procedure for an admitted native model experiment:

1. Start from an immutable artifact (`hf-id@<40-character revision>`, or a local artifact plus its byte manifest) in its own branch and receipt namespace.
2. Run `cargo run -p memra-cli --bin memra -- model inspect <source> --against <family> --out <dir>` and read the artifact lock, normalized config, tensor census, compiled plan and capability manifest before changing runtime code.
3. If the math exists, change only the model pack and tensor mappings (`model scaffold` for a new pack). If the plan reports a genuinely new operation, add the typed operation and its reference implementation first, then give tuned backends an explicit rewrite.
4. Qualify in order: `model verify config`, `tiny`, `checkpoint`, `rewrite`, `serve`. The tokenizer/template and tensor-census gates are part of the bundle even when the numerical graph matches an existing family.
5. A tuned rewrite receipt is valid only for its artifact lock, serialized plan, stream/numeric class and exact runtime binary. Install the bundle with `MEMRA_REWRITE_BUNDLE`. Optional unqualified batch, graph or prime rewrites may use the receipt-backed native eager path; required speculative or pipeline surfaces fail closed when their receipt is absent or stale.
6. Advance the support state only when its persisted gates pass. Keep pending gates pending with the reason; a synthetic fixture alone never promotes a capability. Each supported model gets a short `docs/models/` card (recommended path, rig, deeper sources), and a new hardware or workload class gets a `docs/rigs/` or `docs/workloads/` card. The README links the cards without repeating them.

## Correctness

The three gates from `CONTRIBUTING.md`: `kernel-check`, the `run-gen` argmax gate and `run-spec` K=1..8 self-consistency. A kernel change is done only with before/after numbers measured per `research/benchmarks.md`.

### One numeric program per request

A path that can produce tokens for the same request under two numerical programs is a correctness bug unless the transition is forbidden or proven bit-identical. Unit gates do not catch this; serving-shape gates do (`research/eosclass-20260813/`, `research/splitiso-20260813/`). When adding or promoting a fast path, either make the crossing impossible (refuse promotion once a peer is queued, never switch program mid-request) or prove bit-identity across the transition in a serving-shape gate. Pairs to watch: prime vs decode, monolithic vs tokenwise, graph vs eager, batched vs solo, spec-verify vs plain.

### Prefill and acceptance

The prefill numeric config is part of the acceptance config: prefill writes the prompt KV and hidden lineage the draft head reads, so an argmax-clean prefill kernel swap can still move acceptance by about 8 points either way, depending on the model (`research/f8f4-flip-20260806/MATRIX.md`). Acceptance-affecting kernel changes carry per-model acceptance rows, not only argmax gates.

### Hy3 expert spill

- `HostExps.layouts == None` is the uniform-layout fast-path contract. With `Some(layouts)`, each expert's `qtype`, `row_bytes`, `len` and `offset` are authoritative: use `expert_layout()` and `max_expert_bytes()`, not projection-wide fields, which decode some experts with the wrong block layout.
- Mixed layers run through metadata-aware staged, SLRU-cache or grouped dispatch. The resident slab, pointer-table, pairs, dev and grouped-decode fused kernels stay uniform-only until they group pointers by layout; do not send mixed metadata to them.
- A v2 tier plan assigns every retained expert projection to Q2_K, Q3_K or NVFP4. A missing assignment is an error, never a silent BF16 expert.
- Pruned expert ids keep their router positions and `active_experts()` masks them before top-k. Never dispatch, cache or fabricate bytes for a masked id, and never let a fallback uniform slab bypass split expert overrides.
- The five-arm quantization study (arms, mask recovery, calibration, eval, rigs and artifact staging) is defined in `research/per-expert-quant/README.md` and `arms.lock.json`. Public benchmark data never selects experts, thresholds, tier fractions or pruning.

## Hardware targets

- Blackwell sm_120 is the primary optimized target.
- Step-3.7 multi-card: RTX PRO 6000 Blackwell is the blocking qualification target. Local RTX 5090 evidence is a later compatibility follow-up and does not block a Step release unless the change also alters a generic 5090-facing mechanism or default. Use the hash-bound `step-pro` pre-push gate, never a skip override.
- Multi-card topology: never call PP or replicas TP, and never force a TP degree that violates attention-head, KV-head or expert partitioning.
- Another accelerator: an admitted experiment chooses its runtime per the project status; a new Memra backend is not the default answer. An admitted secondary backend keeps the model bytes, is off at build time by default, documents its disabled target-specific kernels and passes a same-prompt golden-output gate before producing scored evidence. The naked sm_120a build and its defaults do not change.
- Hopper (sm_90a): the arch auto-detects on an H100 (`MEMRA_CUDA_ARCH=90a` forces it). Promotions are compile-gated behind `memra_hopper_mma`, so the sm_120a build stays byte-identical. Ledger: `ARCHITECTURE-H100.md` (append-only); flags: `docs/FLAGS.md` §7. The Hopper CI lane is retired, so run kernel-check config pins, decode-batch (config and strict), decode-dc, graph-decode and graph-session on Hopper hardware directly. Lessons from that lane: compare perf only interleaved x5 on the same box (cross-run and cross-day comparisons are invalid from clock drift, competitor denominators included); re-sweep thresholds and verdicts when the code under them moves; put any guard for a live lane inside a battery that runs; measure every wgmma scheduling change, since nvcc 13.1 is form-sensitive there (C7514/15/17/19).
- Rented box and instance ids live in the gitignored `LANE-LOCAL.md`, never in tracked files.

## Flags doctrine and door hygiene

Winners are defaults: naked commands run the tuned path. An environment variable exists only as a runtime parameter, machine-specific config (VRAM budgets, KV formats, spill), rollback seam (`MEMRA_FAST=0` oracle path), diagnostic or explicitly blocked experimental door. Catalog: `docs/FLAGS.md`. A new `MEMRA_*` read needs its FLAGS.md row in the same commit; a prefix row such as `MEMRA_TCOL_*` covers a family.

Doors are decided, not kept (owner, 2026-09-05: "we stop leaving unneeded flags."):

- Every default-OFF door lands with a `decide-by:` date in its FLAGS.md row, 14 days after landing unless the row says why it needs longer.
- A negative, flat, neutral, superseded, refuted or no-go receipt deletes the door in the same PR: the env read, the dispatch branch, every kernel only it reaches, its tests and gate cells, and its FLAGS.md and KERNELS.md rows. The verdict and receipt pointer go to the "Removed doors" ledger in FLAGS.md and to the darklanes verdicts ledger; git history is the archive.
- A door whose decide-by date passes without a receipt is removed the same way and returns only with a receipt.
- Winners lose their door too: once a default has served two weeks and its rollback seam was never used, delete the seam.
- What stays: flags a launcher or gate sets, red arms and fault-injection doors of a check, owner-designed product switches whose row says so, and diagnostics whose subject arm still exists.

Changing a default:

- Give each new or changed mechanism, flag, hardware default or release gate its own branch and receipt namespace, and prove forced ON/OFF correctness first.
- A performance or default decision needs a balanced same-window interleaved A/B in both orders with N>=5, raw hashes, failures and 250 ms telemetry. Serving claims also record TTFT, E2E, TPOT and ITL p50/p95/p99 plus request and token throughput.
- Winners become naked hardware-specific defaults, losing or flat arms are deleted, and inconclusive arms stay default-OFF with a named missing gate.
- When a win depends on the hardware, measure it on both the local RTX 5090 and a PRO 6000 and key the arm on the device (owner, 2026-08-13: "Local performance is not sacrificed to make a remote default simpler."). One-rig evidence sets a one-rig default at most. Prefer detection (SM, VRAM, card class) over env flags for hardware-shaped choices, and keep the env var as the rollback seam.
- Update the board source and regenerate (next section), and keep `docs/FLAGS.md` and `docs/TESTING.md` aligned.

## Perf board and pre-push hook

The perf surfaces are generated: the PERF-MODELS block in `docs/MODELS.md` and the PERF-PLAIN, PERF-SPEC, PERF-DATE and PERF-H100 blocks in `docs/PERFORMANCE.md` come from `research/tune-data/current-board.json` through `tools/update-perf-board.py`. `research/tune-data/rig5090.jsonl` is the append-only research log. A commit that changes published numbers updates `current-board.json`, runs `python3 tools/update-perf-board.py` and commits the JSON with the regenerated docs, together with the change that moved the numbers. Do not hand-edit inside the `<!-- PERF-*:START -->` / `<!-- PERF-*:END -->` markers; the prose around the tables stays hand-written.

The pre-push hook (`tools/hooks/pre-push`) refuses a push on perf-board drift, the `MEMRA_*` flags census (`tools/check-flags.sh`), the releasability and docs-registry censuses, conflict markers, workflow-file keys, nsys blobs, the public-boundary check and content-bound release qualification. Fix what it reports and re-commit.

- Only the flags census has an emergency switch: `MEMRA_SKIP_FLAGS_CENSUS=1 git push ...` prints the skip and appends a row naming the pushed HEAD to `.git/memra-gate-skips.log` (owner ruling, 2026-08-23). The ledger stays per-clone under the git dir so the person who skipped can be asked; a tracked file would race between parallel lanes, so do not promote it.
- The other censuses have no skip switch: no emergency makes pushing an unshippable tree right.
- `MEMRA_SKIP_PERF_CI` is retired and refuses.

## Match validation to the changed behavior

GitHub runners have no GPU. `.github/workflows/ci-public.yml` routes owner PRs and main pushes to thin lint/build and source-bound merge validation; external PRs, the daily default-branch run and release/publish dependencies run the complete CPU inventory in `.github/workflows/ci.yml`. `tools/validation_plan.py` maps changes to suites, and unknown ownership expands or refuses. Run the owner change's affected functional tests locally before pushing. GitHub does not enforce `ci/merge-result` until a protection rule requires it. nvcc compilation needs no GPU.

The GPU battery follows the change's content (owner, 2026-09-27: "we dont need to run full battery on every change, its depend on change content and what it touces."):

- **Scoped battery**, the default for a merge: run the cells the change can reach and name them in the PR body with the reason for each exclusion. That means the lane's own GPU cells and red arms; serve-smoke, identity and spec-on-cache-hit for serving-path changes; `run-gen` argmax and `run-spec` K=1..8 on each model family whose numeric program the change reaches. `tools/fast-gate/fast-gate.sh --plan --diff origin/main` informs the choice; its GPU selection stays shadow-only. One PRO 6000 carries a single-card change; pair-only surfaces (PP, TP/EP, P2P, multi-card placement) need a pair.
- **Full battery**: `kernel-check` all green, `run-gen` argmax match on every board model and `run-spec` K=1..8 pass, on a non-serving 2x RTX PRO 6000 pair. It is required for `.cu` kernel, FFI shim or kernel dispatch changes, numeric paths several families share, compiler or build defaults, qualification tolerances or required gate coverage, unknown impact, and before any tag.
- If main moves between the battery and the merge, rerun only the cells main's new commits can reach.
- Never run the battery on a serving box.

Documentation changes use text and link checks. CPU-only development, admission and artifact-transport tooling uses its CPU contract, failure-injection and integration checks (artifact tooling also needs native build/restore integrity evidence) and must leave native math, emitted programs, build defaults, model artifacts, tolerances and required gate coverage unchanged. Such a merge qualifies no model, runtime or serving binary; do not rent GPUs for it. `docs/TESTING.md` and `tools/fast-gate/README.md` define the boundary.

## Evidence discipline

- Raw output is part of the deliverable: commit the per-run JSONL or log next to the summary row in `research/<lane>/`. A claim whose raw runs are not in the repo is not evidence.
- `tee` a raw log first and parse the log second; a pipe into a parser loses the failure text.
- Quote failure causes, never infer them. "OOM" means a captured `out of memory` or `CUDA_ERROR_OUT_OF_MEMORY` line plus the GPU compute-apps state at failure time. A run that died without captured stderr is "died, cause unknown, repro needed", and nothing is concluded from it.
- Every published median states its N and thermal regime; single runs are labeled as single runs.
- Run one scored campaign at a time per multi-card box, under `flock /tmp/memra-gpu.lock`. Both cards of a 2x RTX PRO 6000 box share a PCIe path (`nvidia-smi topo -m`), so work on the "idle" card changes the thermal and I/O regime of the campaign even with `CUDA_VISIBLE_DEVICES` pinning. The other card may take only builds, staging and short untimed pass/fail cells.

### GPU lock names

Exactly two memra GPU lock files exist, one per rig class. A third name gives no mutual exclusion with the others, so do not add one.

| rig | lock |
|---|---|
| any 2x RTX PRO 6000 pair or 3-card pod | `/tmp/memra-gpu.lock` |
| the local RTX 5090 | `/tmp/memra-5090.lock` |

`MEMRA_CI_LOCK` and `MEMRA_GPU_LOCK` remain override seams, but their defaults stay on this table. On the local rig, take the GPU through `gpulock` (SHARED.md), which also respects these locks.

## Measurements and decisions are a corpus

Every measurement and decision this repo produces is kept here on purpose (owner, 2026-08-16): what was measured, on what hardware and conditions, and what was chosen or rejected and why, is training data for a later hardware-specialist model.

- Measurements: `docs/PERFORMANCE.md` (boards with conditions) and `research/<lane>/` (the raw runs behind them). Both, never one alone.
- Decisions: a decision that changes a default, format, target or arm gets a record in `docs/decisions/` naming what was chosen, what was rejected and the measurement that settled it. Rejections matter as much as adoptions.
- Architecture ledgers: `ARCHITECTURE.md` and `ARCHITECTURE-H100.md`.
- `docs/KERNELS.md` maps every `.cu` entry symbol to purpose, qtype, arch guard, dispatch flag and FFI binding. Update the affected rows in the same change that touches a `.cu` file or FFI shim. `docs/FLAGS.md` is the flag catalog and `docs/MODELS.md` the model board.
- `docs/ROUTER.md` says where each kind of question lives; start there.
- `research/INDEX.md` has one verdict line per research dir. Check it before re-running an experiment and add your lane's row when it closes. The root `.ignore` keeps raw receipts out of default `rg` (`rg -u` includes them).
- The machine-local lesson corpus is `~/projects/darklanes/agent-knowledge/gpu/` (index `gpu/README.md`). Read it before designing a cell, gate or kernel change and promote new lessons there when a lane closes. Search it with `rg '^LAW:|^TRAP:|^GATE:'` for rules, `rg '^VERDICT:|^KNEE:'` for what was already tried or blessed, and `rg '^QUIRK:<model>'` under `gpu/models/`.

Do not delete a measurement or decision record to reduce clutter. Banner a superseded record, name what replaced it and leave the numbers. Anything that does not belong in a public engine repo (deployment, location, fleet, business) moves to darklanes rather than being dropped.

## Research results (owner, 2026-08-13)

- The pre-registered target does not decide whether a result is valid. If the numbers redirect the work, the lane's name goes stale, not its results.
- A method-vs-method comparison is a result even when the method you hoped for loses. State it plainly.
- Partial coverage of the model x hardware x sampler grid is not a defect. The field's bar is one model on one hardware; owner: "i never saw a paper on arxiv that measure on more then one model and one hardware without it becoming a product." State the scope and publish.
- Separate mechanism wins from empirical, model-dependent wins. A mechanism win holds by construction: draft-side grammar masking (`MEMRA_DRAFT_MASK`) never proposes grammar-illegal ids, so it strictly reduces wasted draft work on any model.
- Frame publishable work as ongoing research with stated scope, and report the direction the evidence points, including against the hypothesis.

Estimate effort in agent-days: one agent-day is about one human-week, weekends included (owner, 2026-08-13). Size alone is not an argument against a mechanism; price it and let Avi decide. This changes estimates, not the correctness and evidence gates.

## Public boundary

memra is a public engine under FSL-1.1-ALv2, not a product (owner, 2026-08-16): an engine a reader can clone and serve with on hardware they own. The business lives in darklanes and on the website.

- No product voice: no prices, packs, credits, trials, rate cards, support promises, launch narratives or "our customers".
- Attribution stays: the author and the lab ([tiyuvta.ai](https://tiyuvta.ai)) are named at the top. The lab helps teams self-deploy and fine-tune open models; hosted inference is not the business. If the README links a hosted endpoint, call it a demo of the engine, never a hosted API or a product.
- State capability, not deployment. "memra runs Qwen3.8-27B at its full 262k context, gated by the exactness battery" is an engine claim; where production runs is not.
- Product, business and go-to-market docs live in darklanes. Exception (owner, 2026-08-12): provider-connection receipts that cite and seal engine gates live under `research/connect-*/`; pricing decisions and channel economics stay in darklanes. Self-serve console code is darklanes; the engine keeps only generic seams (tenant budgets, admin API).
- Deployment facts are darklanes too (owner, 2026-08-16: "No one needs to know what we serve on."). Publish measurement conditions (card, count, N, date). Do not publish provider names in a role, machine or instance ids, cities, hourly rates, or any present-tense statement of which rig serves. Hardware ownership is disclosed once, in `docs/PERFORMANCE.md`.
- `tools/public-boundary-policy.toml` enforces the sharp edges with narrow patterns, checked by `tools/check-public-boundary.py` in the pre-push hook. Describe banned forms rather than quoting them, since the patterns fire on literals in this file too, and keep new patterns narrow: a noisy gate hides real leaks.

## Positioning (owner, 2026-08-17)

- No engine-vs-engine column, chart or ratio in `README.md`, on the lab site or on any promotional surface. Ratios stay in `docs/PERFORMANCE.md` and `research/`, next to their protocol. Publish speed as an absolute figure on a named card with its conditions. Whether a large-margin comparison on a promoted model is worth publishing is Avi's call, per model.
- Numbers are tracked for regression, not as a scoreboard: the README has no generated performance surface, and scoreboard artifacts such as the retired comparison SVG cards do not come back.
- Headline space is for what is rare (vocab-masked draft heads, per-device defaults, per-request exactness gates). Table-stakes features (speculative decoding, the OpenAI-compatible surface, prefix caching, tool calls, JSON-schema decode) share one sentence and a link.
- One short "look elsewhere if" paragraph is the whole negative surface. Check an absence before asserting it: false modesty is as much a factual error as an overclaim.
- Before committing a README diff, run the review checklist in `agent-knowledge/readme-craft-inference-engine.md` §9.

## Releases

Tag every board-moving or user-facing change: `git tag vX.Y.Z && git push origin vX.Y.Z`. The `release` workflow compiles, drafts the changelog from conventional commits (`tools/changelog.sh`) and publishes. Bump minor per mechanism or board move, patch per fix or docs. Prefixes `perf:`, `feat:`, `fix:`, `config:` and `docs:` are public; `data:`, `chore:`, `wip:` and `probe:` are filtered as research-log noise. Full process: `docs/RELEASING.md`.

Version numbers are shared between parallel sessions:

1. Claim the number first: `git push --force-with-lease=refs/heads/release/claim-vX.Y.Z: origin HEAD:refs/heads/release/claim-vX.Y.Z`. If origin refuses, the number is taken; pick the next one. Never delete another session's `release/claim-*` branch.
2. On the tagged commit, `[workspace.package].version` and the pinned `[workspace.dependencies]` versions equal the tag. `tools/release-guard.sh` refuses a mismatched or unclaimed tag in `release.yml` and `publish.yml`.
3. Never renumber over a broken tag. Mark a mismatched release as skipped for version mismatch and as a prerelease (v0.102.0 shows the form); tags are never deleted and releases are never recut under the same number.

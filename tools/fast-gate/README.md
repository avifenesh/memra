# fast-gate

`tools/fast-gate/fast-gate.sh` runs the checks a diff needs between full-battery points: a compile,
a scoped `kernel-check` and golden-token probes on one model per affected kernel class. It is for
contributors in the edit-compile loop. A fast-gate green means "keep going". It is never device
evidence for a push, a tag, a model or a serving binary, and its GPU selection is shadow-only. What a
change must run before it lands is in [docs/TESTING.md](../../docs/TESTING.md); that page and this
one define the CPU-only boundary.

## Tiers and entry points

| tier | wall (RTX 5090) | what runs |
|---|---|---|
| 0 | seconds | workspace compile and `kernel-check` scoped to the touched sections |
| 1 | about 1 to 2 min | tier 0 plus golden-token argmax probes on one model per affected kernel class; a spec-pipeline diff adds one single-K spec probe |
| 2 | the full battery | `exec tools/local-ci.sh` with no arguments: correctness only, no perf stage |

```bash
tools/fast-gate/fast-gate.sh                       # tier 1 against HEAD (uncommitted work)
tools/fast-gate/fast-gate.sh --tier 0              # compile + scoped kernel-check only
tools/fast-gate/fast-gate.sh --diff main           # scope = everything since main
tools/fast-gate/fast-gate.sh --tier 2              # execs tools/local-ci.sh
tools/fast-gate/fast-gate.sh --smoke               # add the perf tripwire
tools/fast-gate/fast-gate.sh --probes k27,amargin  # name the probes explicitly
tools/fast-gate/fast-gate.sh --plan --diff origin/main   # print the plan as JSON, run nothing
```

GPU steps run under `${MEMRA_GPU_LOCK:-/tmp/memra-5090.lock}` (an inherited lock on FD 9 is reused).
Raw probe output is written to `$MEMRA_GATE_LOGDIR` (default: a new dated directory per run) before
it is parsed.

## The plan and its decisions

`plan.py` turns the diff into a plan. It also carries the changed-input CPU graph from
[validation selection](../../docs/VALIDATION-SELECTION.md): direct inputs, reverse dependency witnesses,
binary edge decisions and the reason each CPU component was omitted.

```sh
python3 tools/fast-gate/plan.py --changed crates/memra-engine/src/spec.rs --json
python3 tools/fast-gate/plan.py --context-changed numeric --json
python3 tools/fast-gate/plan.py --cache "$HOME/.cache/memra/components" --json
```

| decision | what fast-gate does |
|---|---|
| `no-change` | prints `fast-gate: no changes; NO VALIDATION performed. Use --probes to check a clean tree.` and exits 0 |
| `cpu-only` | runs the selected CPU contracts (`tools/validation_plan.py contracts --selected ...`), or for docs `tools/docs-registry-census.sh` and `python3 tools/check-support-states.py`, then `git diff --check`; no Cargo, CUDA or model loading |
| `expand` | stops with exit 2 before any build unless `--probes` names diagnostics; explicit probes prove only themselves |

A clean tree (a release candidate, a fresh copy with no `.git`) has an empty diff: gate it with
`--probes`, or nothing runs. A missing model, prompt or golden for a planned probe refuses before the
build. An invalid Git ref refuses instead of becoming an empty diff.

Compiler, build, runtime, numeric, harness, oracle, model, binary, hardware, topology and request
context changes expand the plan. So do unknown paths, index-masked inputs, ignored build inputs and
broad legacy rows: none of them proves independence. A changed research file can be an
`include_str!` fixture, and a changed gate is an input too.

## map.tsv and the packs it cannot see

[`map.tsv`](map.tsv) maps changed paths to kernel-check sections, probes and spec probes; every
matching row adds to the plan. A path no row matches gets the `DEFAULT` row (full `kernel-check`,
probes `g12,q9,q35,gwstress`, spec probe `q35spec`) and a warning: add a row when that happens.
`dependencies.json` declares the transitive dependencies of build inputs, generated and oracle
inputs, harnesses, shared CUDA, runtime and model plans.

The registered probes load only the gemma4_dense, gemma4_e4b, gemma4_moe, qwen35, qwen35_moe and
step35 packs. Every other automatic pack, and FP8_BLOCK checkpoints, print as `REQUIRED` lines
(`unprobed-model-packs`), as do the MiMo text, vision, audio, MTP and serving requirements. These
lines are a static list: any engine, model-plan, build or harness change prints all of them, because
pack selection is the first match over `PACKS` and a shared unit serves several packs
(`cu/dsv4_gpu.cu` holds deepseek_v4 kernels, the hyper-connection kernels glm5_next runs and the
NVFP4 expert dequant qwen4_exp runs). Run the gates of the packs the change reaches and give the reason
for each one left out. `test_plan.py` fails when a pack joins `PACKS` with neither a probe nor a line;
its `PROBED_PACKS` is a hand list, so add a pack there when a probe for it lands.

## kernel-check seams (tier 0)

- `MEMRA_KC_FAST=1`: synthetic arms only; every weight-oracle section is skipped. About 1.4 s
  against about 4.5 min for the full run, where the model-backed GEMM oracles are 266 s of 268 s
  (`research/fast-gate-20260802/kernel-check-full-timed.log`).
- `MEMRA_KC_ONLY=<csv>`: synthetic arms plus the weight-oracle sections whose names match (for
  example `nvfp4-gemm`, `q8mmq-gemm`, `iq4xs-mmq`, `f16g-kq-direct`, `nvfp4-mmvq`).

Each skipped section prints a `KC-SKIP` line naming the variable, so a scoped run never looks wider
than it is. These are dev-loop diagnostics; every battery runs `kernel-check` with neither.

## Probe registry and goldens (tier 1)

[`models.tsv`](models.tsv) registers each probe: `id`, `kind` (`argmax`, `spec`, `gspec`, `cmd`),
model path, prompt (a token-id file, or `@<textfile>` tokenized by the model), `ngen` and extra env.

- `argmax` probes run `run-gen` with `MEMRA_NGEN=<ngen>`. They pass on the argmax `MATCH` line, no
  `MISMATCH-STRUCTURED`, and a `tokens:` line byte-equal to `goldens/<id>.tokens`. Any diverged id
  fails at once, with no reference recompute. Greedy decode is deterministic, so a moved token is a
  behavior change, and goldens are valid under any clock or thermal regime.
- `spec` probes run `run-spec` at one `MEMRA_SPEC_K` and require `SELF-CONSISTENCY PASS` plus the
  same golden pin; `gspec` is the gemma drafter's stream agreement. The K=1..8 sweep stays tier 2.
- `cmd` probes are self-gating commands that pass on exit 0 and pin no golden, for code greedy
  goldens cannot see: `samp`/`sampt` (sampler), `chunkinv*`, `tickinv*`, `isogap*`, `ptick*`,
  `pbg9*`, `amargin*`, `gwstress`, `sstress`, `accept`, `dsparkq38` and others. A `cmd` row's
  absolute-path arguments are artifacts: when one is missing the probe prints
  `SKIP (no artifact at ...)`, does not run, and the run exits non-zero. Relative paths, env-column
  paths and `--x=/path` arguments are not checked.
- `k27` pins `MEMRA_FA_SPLIT=8` so its golden holds across the 82- and 188-SM split ladders.
- `q9` runs without the chat template and its 20 ids copy its prompt; switch it to `MEMRA_CHAT=1` at
  the next refresh (`research/fastgate-goldens-20261104/`).

Goldens refresh only at full-battery green points:

```bash
tools/local-ci.sh                                  # must be ALL GREEN first
tools/fast-gate/fast-gate.sh --refresh-goldens     # refuses a dirty tree (--force overrides, loudly)
git add tools/fast-gate/goldens && git commit ...  # goldens are checked in, SHA-stamped
```

A change that legitimately moves tokens refreshes in the same commit, after its battery run, so the
golden diff shows in review. A refresh needs Git provenance, pins the goldens it can and reports the
missing ones; a requested golden that cannot be written, or a refresh that writes nothing, fails.

`--smoke` compares each tier-1 probe's single timed rep with `goldens/<id>.perf`: WARN above a 10%
drop, FAIL above 25%. It catches a kernel falling off its fast path and nothing finer; it is not a
publishable number ([research/benchmarks.md](../../research/benchmarks.md)).

## Probe-regime laws

1. **A probe must exercise the touched dispatch class**, not just the model family. On a 24 GB card
   MoE models load resident, so the SLRU and spill paths never run under a naked probe; `q35slru`
   (`MEMRA_MOE_RESIDENT=0`, `MEMRA_MOE_SLOTS=1024`) forces them.
2. **Depth is a dispatch axis.** Short probes decode below the FA vector floor; `g12d` runs the
   battery's 1736-id depth prompt (16 ids, because token 17 is a run-to-run near tie).
3. **Greedy goldens never touch the sampler.** `samp` (`sample_check`) and `sampt`
   (`cargo test -p memra-sampling`) are the sampler's probes.

Deliberate breaks each tier caught (all reverted; receipts in
[`research/fast-gate-20260802/`](../../research/fast-gate-20260802/)):

| break | caught by |
|---|---|
| MoE staged dispatch reads gate weights for up (`moe_cached_gemm_q8`) | tier 1 `q35slru`; plain `q35` was blind |
| FA v4 K-scale skew x1.001 (hd256 staging) | tier 0 kernel-check bit-identity arms |
| FA hd512 tb512 K element permutation | tier 1 `g12d` |
| MMQ Q8_0 wrong-block scale | tier 0 `MEMRA_KC_ONLY=q8mmq-gemm` |
| f16g IQ4_XS dequant off by one | tier 0 `MEMRA_KC_ONLY=f16g-kq-direct` |
| device sampler acceptance skew x1.001 | tier 1 `samp` |
| sparse device penalty drops a count or coefficient, or parks mutated logits | tier 1 `samp` and decode-batch gate3c ([receipt](../../research/pro-device-penalty-sampling-20260824/)) |
| host sampler top-k keeps the worst k | tier 1 `sampt` |

The gaps it does not close are listed in [Testing](../../docs/TESTING.md#known-blind-spots).

## Admitting a narrower GPU class

Before a narrower GPU selection leaves shadow mode, keep a selected-versus-full comparison on the
same candidate, artifact, request shape and target hardware, and replay its historical failure plus a
mechanism-changing red arm. A miss expands that class and blocks adoption; coverage observations are
supporting data, not proof of independence. `tools/validation_coverage.py` selects a source-bound
composite test set for explicitly required behavior edges.

## Isolated CPU components

```sh
tools/fast-gate/fast-gate.sh --component qualification-contracts --cache "$HOME/.cache/memra/components"
tools/fast-gate/fast-gate.sh --component release-inputs --cache "$HOME/.cache/memra/components"
```

Each reports `run` or `reuse`, the exact key, the receipt path and elapsed times; `--fresh` executes
even when a matching result exists. The two admitted suites test qualification admission with
fabricated records; they run no kernel and build no engine. Only the listed source files are mounted,
read-only, in a fresh view with a fixed environment, no home, no network and no GPU, so a newly
consumed file fails until the contract declares it. Source bytes, modes, harness, fixture, contract
and runtime identities form the key and are rechecked after execution. Skips, empty suites, duplicate
ids and failures never publish a reusable record, and a fresh failure for an unchanged key
quarantines the older pass. It needs Linux, system Python under `/usr/bin` and working bubblewrap.
Keep the cache outside the source tree; contracts are trusted local inputs, not signatures.

## Build reuse

[BUILD-REUSE.md](BUILD-REUSE.md) covers exact controlled-build reuse. Resident model setup is not
implemented: a probe that shares one load across arms (`ppsplit` runs reference, serial and dynamic
split in one process) must reset KV, RNG, graph, allocator and request state and carry reversed-order
controls.

The earlier text of this page is kept in
[docs/history/TESTING.md](../../docs/history/TESTING.md#development-feedback).

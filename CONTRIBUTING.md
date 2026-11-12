# Contributing to memra

This page is for anyone sending a fix, a kernel, a model or a hardware report: what proof each
kind of change needs and how to produce it. Memra is in active development
([project status](README.md#project-status)).

Issues are welcome at any time. A pull request is welcome when it carries the proof its change
needs. GitHub runners have no GPU, so a change that reaches GPU code is judged on the output
pasted in the PR. A PR that claims a passing gate or a speedup without that output is closed.

By submitting a contribution, you agree to license it under [FSL-1.1-ALv2](LICENSE), including
its future Apache-2.0 grant. If you cannot grant those rights, say so before you submit.

## Before you write code

1. Search [`research/INDEX.md`](research/INDEX.md) (one verdict line per research lane) and
   [`research/tune-data/`](research/tune-data/) (every tuning attempt, wins and losses) for your
   kernel or model. A measured loss is not re-proposed without new evidence.
2. Read [ARCHITECTURE.md](ARCHITECTURE.md) for how a checkpoint becomes a plan and how optimized
   paths are checked, and [`docs/decisions/`](docs/decisions/) for what was chosen or rejected.
3. A new model or family follows [ONBOARDING.md](docs/ONBOARDING.md). To propose a model
   without onboarding it yourself, open an issue with the checkpoint ID, quantization, target GPU
   and use case; a proposal is not a support commitment. Drafts, trims and acceptance work follow
   [DRAFT-REGIME.md](docs/DRAFT-REGIME.md).
4. [AGENTS.md](AGENTS.md) holds the standing repo rules (flags, defaults, evidence, public
   boundary). They apply to human contributors too.
5. A change that reaches a GPU cell needs a GPU of a target it reaches ([Build](#build)). If you
   cannot produce that evidence, open an issue that describes the idea instead of a pull request.

## Build

`rust-toolchain.toml` pins Rust 1.97.1. CI and releases build with CUDA 13.1; use 13.1 or newer.
[Installation](docs/INSTALLATION.md) covers toolkit discovery and per-target notes. Build what you
test rather than the whole workspace:

```bash
git config core.hooksPath tools/hooks        # once per clone
cargo build --release -p memra-server --bin memra-server
cargo build --release -p memra-engine --bin kernel-check --bin run-gen --bin run-spec
```

`MEMRA_CUDA_ARCH` picks the build target. Unset, `build.rs` detects the GPU's compute capability
and falls back to `120a` when there is no GPU or no matching target (that build will not load on
an unmatched GPU). Compiling needs no GPU.

| Target | GPUs | Push evidence | CI | Prebuilt |
|---|---|---|---|---|
| `120a` (default) | RTX 50-series, RTX PRO 6000 Blackwell | required when reached | build, CPU tests | yes |
| `121a` | GB10 (DGX Spark, aarch64 host) | required when reached | compile | no |
| `100a` | B200 | not push-gated | compile | no |
| `90a` | H100 | not push-gated | none | no |
| `89` | Ada (portable correctness build) | not push-gated | none | no |

## What proof your change needs

Three tools say what a diff reaches. Run them first, paste their output in the PR, and run the
selected CPU tests locally before you push. The table below is a guide; `change_impact.py`
decides.

```bash
python3 tools/change_impact.py --base origin/main --head HEAD      # (build target, scope) cells
python3 tools/cell_evidence.py need --base origin/main --head HEAD # the evidence those cells need
python3 tools/validation_plan.py local --base origin/main          # the CPU suites to run
```

| The change touches | Proof in the PR |
|---|---|
| Docs that no crate reads, tests, research receipts, CPU tooling under `tools/` and `.github/`, `memra-cli`, `memra-net-guard` | The CPU suites above, plus `python3 tools/check-support-states.py` and `python3 tools/check-public-boundary.py` for docs. No GPU cell. Research files a gate reads (its prompts and calibration logs) count as gate inputs. |
| Kernels (`.cu`, FFI shims), the numeric program (engine, gguf, kv, tokenizer, sampling, reference, runtime), the build | The three gates on a device of each affected push-gated target and the main runners exercised. Before/after prefill and decode numbers when `change_impact.py` reports a `perf` cell. A `.cu` change updates its [KERNELS.md](docs/KERNELS.md) rows. |
| Serving (`memra-server`, `memra-lanes`, `memra-tier`) | As above, plus `tools/serve-smoke.sh` and one real request through `memra-server`. |
| A device-keyed default | The gates and an interleaved A/B on that card's device class. |
| A gate, its tools or its rules (`memra-validate`, the battery scripts and binaries, `tools/change-impact.toml`, the pre-push hook) | All three kinds: battery, serve-smoke and perf. |
| A model pack the battery does not run (any but `qwen35` and `qwen35_moe`) | That pack's own gates; `change_impact.py` reports it as `UNCOVERED`. |
| Multi-card surfaces (PP, TP/EP, P2P, placement) | The same, run on a multi-card box, in the PR body. |

Some Markdown files are crate inputs (`docs/FLAGS.md` feeds the boot env audit;
`docs/EXECUTION-SURFACES.md` is pinned by a test), so editing them reaches cells.
`tools/cell_evidence.py run` produces a device record and `bank` stores it under
`research/release-qualification/cells/`; the rule is in
[RELEASE-QUALIFICATION.md](docs/RELEASE-QUALIFICATION.md). If you cannot run one affected
target's battery, say so in the PR: the change still needs that record before it reaches `main`,
and a maintainer can produce it.

## The three gates

```bash
tools/local-ci.sh            # correctness stage: kernel-check, run-gen argmax, run-spec K=1..8, and more
tools/local-ci.sh --perf     # plus the perf cell battery from research/tune-data/perf-cells.json
```

Or run them one by one and paste the tails:

- `kernel-check` compares every kernel with a CPU reference and ends with
  `ALL GREEN (<n> cells, <m> skipped)`.
- `run-gen` prints a `verify-prefill` argmax line (batched prefill against token-by-token
  decode) that must end in `MATCH`. On text prompts of 16 tokens or more it adds a
  `batched-prime` line, which compares `prime_cache` (the path that seeds real generation and
  serving) with the tokenwise reference: `MISMATCH-STRUCTURED` fails the run, `FLIP-NEARTIE` is
  reported. `prime-gate <model.gguf> --prompts-file <f> [--chat]` runs the batched-prime check
  over many prompts.
- `run-spec` checks K=1..8 against plain decoding and ends with `=== SELF-CONSISTENCY PASS ===`.

`tools/local-ci.sh` needs `jq`, and `MEMRA_MODELS_DIR` pointing at your model root; cells whose
model files are absent are skipped. Hopper (sm_90a) has no CI lane: build with
`MEMRA_CUDA_ARCH=90a` and run `kernel-check`, `decode-batch-gate`, `decode-dc-gate`,
`graph-decode-gate` and `graph-session-gate` on an H100 ([H100 card](docs/rigs/h100.md)). The
lane's evidence ledger is [ARCHITECTURE-H100.md](ARCHITECTURE-H100.md), append-only.
[TESTING.md](docs/TESTING.md) has what each gate catches and the full catalog.

A micro-benchmark (`decode-bench`) shows a kernel is fast in isolation, not that the engine still
works. Also show the main runners on the paths your change touches: `run-gen` end to end on a real
model (not random weights), with its full output; `run-spec` when the change is upstream of the
speculative target forward (attention, GEMM, MoE dispatch, KV cache); and one real request and
response through `memra-server` when it touches request handling or batching.

## Performance numbers

Report prefill and decode, every time: a kernel that helps one and costs the other is a loss.
Follow the [method](docs/PERFORMANCE.md#method): base and branch interleaved in the same
session, in both orders, N >= 3 runs per arm (five on an anomaly), with the GPU's power and clock
state recorded. Sequential runs from different sessions drift by up to about 10% and are not
evidence. Speculative changes also report acceptance, which exactness gates cannot see. The PR
template has the table.

## The pre-push hook

`git config core.hooksPath tools/hooks` installs it. Do not bypass it with `--no-verify`. It
refuses a push when:

- the perf board is stale (`python3 tools/update-perf-board.py` regenerates it);
- the `MEMRA_*` flags census fails (`tools/check-flags.sh`): a new read needs its
  [FLAGS.md](docs/FLAGS.md) row in the same commit;
- a releasability census (publish list, stub ABI, arch matrix), the docs-registry census, the
  numeric-scope census or the workflow-file census fails;
- a tracked file carries a merge-conflict marker, or the push range carries an nsys profile;
- the public-boundary check fails (`tools/check-public-boundary.py`);
- native evidence is missing: a branch push needs a passing record for each cell it reaches on
  each push-gated target, and a tag needs the full release record.

Only the flags census has a skip, `MEMRA_SKIP_FLAGS_CENSUS=1`, which is logged to
`.git/memra-gate-skips.log`. `MEMRA_SKIP_PERF_CI=1` refuses the push. A topic branch can go up as
development work that claims no native evidence, with `MEMRA_RELEASE_QUALIFICATION_MODE=development`;
`main` and tags refuse that mode. A fast-forward that changes only Markdown no crate reads prints
`DOCS ONLY` and needs no device record.

## What gets a PR closed

- Gates claimed as passing with no pasted output.
- A change with a `perf` cell and no before/after numbers, or with numbers from different sessions.
- Only prefill or only decode measured when the change can touch both.
- Kernel or algorithm changes, AI-generated or not, with no run on the hardware they target.
- Portability work that has not read [Installation](docs/INSTALLATION.md) and the
  [hardware cards](docs/rigs/README.md): the targets above are the ones memra builds.
- Style-only edits mixed into a functional change: split them.

## Validation reports from your rig are the easiest contribution

A [hardware validation report](https://github.com/avifenesh/memra/issues/new?template=hardware-validation.md)
helps even if you never touch the code. Correctness output alone is useful.

- **sm_120 cards (RTX 5090, 5080, 5070 Ti, 5070):** they share the RTX 5090's build, not its
  measurements: SM count and memory bandwidth differ per card.
  `MEMRA_MODELS_DIR=... tools/local-ci.sh --perf` plus an interleaved pairing per
  the [method](docs/PERFORMANCE.md#method) is the evidence a card's results need.
- **GB10, H100, B200, Ada:** build from source for the card's target and report what passes and
  what breaks. A GPU with no build target (Ampere and older) cannot run memra: the installer
  refuses it, and a source build falls back to `120a`, which does not load there.

## Where code lives

[ARCHITECTURE.md](ARCHITECTURE.md) explains how the crates fit together.

| Crate | Role |
|---|---|
| `memra-engine` | CUDA kernels (`cu/`), forward passes, batched and graph decode, speculative decoding, MoE, PP/TP/EP, and the gate and bench binaries |
| `memra-server` | `memra-server`: the axum HTTP server, GPU worker thread, scheduler, API dialects and admission |
| `memra-gguf` | GGUF and safetensors loading, quant decode, `ModelPlan`, tensor contracts, model packs, execution manifests, `hf:` specs |
| `memra-reference` | The portable CPU reference executor for `ModelPlan`, with its speech, audio and vision stages and checkpoint runners |
| `memra-cli` | `memra model inspect`, `scaffold`, `verify` and `qualify` |
| `memra-tokenizer` | GGUF-native BPE and SPM tokenizers and chat templates |
| `memra-sampling` | Host sampler chain with llama.cpp CPU semantics |
| `memra-kv` | KV and recurrent cache structure, sizing and format policy |
| `memra-tier` | Exact-byte tiering contracts |
| `memra-lanes` | Serving lanes, SLO admission and step statistics |
| `memra-runtime` | CUDA context and GEMM checks |
| `memra-validate` | The shared gate protocol: vectors, tolerance bands, N-run medians, the ALL GREEN tally |
| `memra-net-guard` | SSRF address classification for server-side URL fetches |
| `memra-probe` | The Phase-0 stack probe: builds custom kernels into a fatbin and launches them through cudarc; unpublished (`publish = false`) |

Earlier versions of this page carried the rationale of the proof policy, gate history and a scope
statement; they are kept in
[docs/history/README-and-CONTRIBUTING.md](docs/history/README-and-CONTRIBUTING.md).

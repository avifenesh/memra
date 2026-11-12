# Testing

This is memra's gate catalog: what each gate proves, how to run it, what counts as a pass, when a
change must run it and where its receipts go. It is for contributors and agents deciding what to
run for a change.

The evidence model is in [Release qualification](RELEASE-QUALIFICATION.md), the tag procedure in
[Releasing](RELEASING.md), hosted workflows in [CI](CI.md), CPU job selection in
[Validation selection](VALIDATION-SELECTION.md) and the dev-loop tool in
[fast-gate](../tools/fast-gate/README.md). Each tool's header (or `--help`) is the reference for its
arms. Gate histories and the earlier full specifications are in [history/TESTING.md](history/TESTING.md).

## Choose what to run

Start from what the change reaches, then produce and bank the records it needs:

```sh
python3 tools/change_impact.py --base origin/main --head HEAD        # (build target, scope) cells
python3 tools/cell_evidence.py need --base origin/main --head HEAD   # records the push still needs
python3 tools/validation_plan.py local --base origin/main            # CPU suites, and why others are omitted
tools/fast-gate/fast-gate.sh --plan --diff origin/main               # dev-loop probes, as JSON
python3 tools/cell_evidence.py run --device rtx5090 --commit HEAD --kinds battery,serving,perf \
    --base origin/main --out ~/.cache/memra-cells/run-1
python3 tools/cell_evidence.py bank --out ~/.cache/memra-cells/run-1
python3 tools/cell_evidence.py check --base origin/main --head HEAD
```

`tools/change_impact.py` classifies every changed path through `tools/change-impact.toml`. The
class decides the evidence:

| what the change reaches | evidence before the push | produced by |
|---|---|---|
| `docs`, `tests`, `research`, `cpu`, `diagnostic`, `comment` | no device evidence; the CPU checks of what changed | docs: `tools/docs-registry-census.sh`, `python3 tools/check-support-states.py`, `git diff --check`. CPU tooling: its contract, failure-injection and integration tests ([CPU contracts](#cpu-contracts-and-fixtures)) |
| `kernel`, `build`, `numeric`, `defaults` | a passing device battery record on each push-gated target | `cell_evidence.py run --kinds battery` |
| `serving` | battery and serve-smoke | `--kinds battery,serving` |
| `perf` (a perf-relevant package, a rule or the diff content marks it) | interleaved A/B against the push base | `--kinds perf` |
| `gate` (the battery, its tools, rules, manifests or verdict code) | all three kinds | `--kinds battery,serving,perf` |
| a default keyed on a card name | battery and perf from that card's device class | run on that card |
| a change inside a pack the battery does not cover (`UNCOVERED`) | that pack's own gates, named in the PR body | the pack's gate manifest |
| pair-only surfaces: PP, TP/EP, P2P, multi-card placement | the pair gates, named in the PR body | [Multi-GPU gates](#multi-gpu-pp-n-exactness-gates-run-on-the-multi-card-box) |

Only `sm_120a` (an RTX 5090 or RTX PRO 6000 Blackwell counts) and `sm_121a` (a GB10) are
push-gated. The other targets rely on daily runs where a device exists; the tag battery builds
`sm_120a` only, so it does not cover them.
An unknown path reaches `kernel`, `build`, `numeric`, `serving` and `perf` on every target. The push
base's rules apply. The PR body also lists the changed mechanism's own GPU cells and red arms.

CPU-only changes (docs, tooling, admission contracts) merge on their CPU checks. They must leave
native math, emitted programs, build defaults, model artifacts, tolerances and required gate
coverage unchanged; file location alone does not prove that. A change to the gate set itself (a
narrower default selection, a removed cell, the impact rules) is a `gate` change. **No CPU result,
fast-gate green or successful build is device evidence for a model, a runtime or a serving
binary.** [`tools/check-doc-links.py`](../tools/check-doc-links.py) checks relative links and heading
anchors in tracked markdown; thin CI runs it.

## Checkpoints

| checkpoint | command | what runs | what it does not prove |
|---|---|---|---|
| dev loop | `tools/fast-gate/fast-gate.sh` | tier 0 and 1 probes the diff maps to ([fast-gate](../tools/fast-gate/README.md)) | device evidence: its GPU selection is shadow-only |
| merge battery (tier 2) | `tools/local-ci.sh` | the [standing battery](#standing-batteries) on one card | multi-card gates; performance (only with `--perf`) |
| push | `git push` (hook `tools/hooks/pre-push`) | the censuses, then per-cell evidence for each reached cell | targets without `push_gate`, pair-only surfaces |
| hosted CI | `ci-public.yml` on PRs and pushes to main | thin: scoped builds, lint and merge guards; full: the complete CPU inventory ([CI](CI.md)) | any device result |
| daily | `tools/daily-regression.py`, one user timer per device | the device's full cell set on main's head | a specific commit's evidence before it lands |
| tag | `tools/release-battery.sh`, captured on one leased RTX PRO 6000 Blackwell card ([what a record covers](RELEASE-QUALIFICATION.md#what-a-record-covers)) | the qwen35 class: the two `tools/release-roster.tsv` checkpoints | other families, other formats, the served path |

**Push.** The hook refuses on perf-board drift, the `MEMRA_*` flags census (its one skip,
`MEMRA_SKIP_FLAGS_CENSUS=1`, is logged to `.git/memra-gate-skips.log`), the releasability,
docs-registry and numeric-scope censuses, conflict markers, workflow keys, nsys blobs and the public
boundary, and gates native evidence through `tools/release_qualification.py push`.
`MEMRA_SKIP_PERF_CI=1` refuses. A topic may push with `MEMRA_RELEASE_QUALIFICATION_MODE=development`
(logged, unqualified); main and tags refuse it. A branch fast-forward that changes only Markdown
outside `crates/` that no crate reads prints `DOCS ONLY` (`docs/FLAGS.md` and
`docs/EXECUTION-SURFACES.md` are crate inputs). Devices are the
`[device.*]` rows of `tools/change-impact.toml` (`rtx5090`, `rtxpro6000`, `gb10`, `h100`, `b200`).
Records, their reuse across docs-only commits and the perf A/B rule are in
[per-cell evidence](RELEASE-QUALIFICATION.md#change-impact-and-per-cell-evidence).

**Daily.** `python3 tools/daily-regression.py install --device <d> [--at HH:MM]` installs disabled
user units; the timer runs `run --device <d> --bootstrap`, commits to `regression/<device>` and keeps
one `daily-regression` issue per failing device ([daily regression](RELEASE-QUALIFICATION.md#daily-regression)).

## Standing batteries

### `tools/local-ci.sh` (tier 2)

`tools/local-ci.sh` runs correctness only (`fast-gate.sh --tier 2` execs it with no arguments).
`--perf` and `--perf-quick` add the perf cells. It takes `${MEMRA_CI_LOCK:-/tmp/memra-5090.lock}`
once for the whole GPU part. Models resolve under `MEMRA_MODELS_DIR`. **A stage whose model file
is absent prints `SKIP` and the run continues**: read the SKIP lines before calling a run green.

| stage | tool and pass line | skip or retarget |
|---|---|---|
| CPU chain (overlapped with the GPU stages in correctness mode) | `cargo clippy --release --all-targets -- -D warnings`; `python3 tools/check-coalescer-contract.py --out target/coalescer-contract`; `cargo test --release -p memra-server`; `cargo test --release -p memra-engine --lib`; the memra-gguf skip census (budget 10, at least 200 passed); `tools/portable-suites.sh` | `MEMRA_CI_CLIPPY=0`, `MEMRA_CI_GGUF=0`, `MEMRA_CI_OVERLAP=0` (serial) |
| prelude, before the run lock | flags census, the `remove_var` door lint and their self-tests (warnings only); `tools/drafter-attach-wiring-gate.sh` and `tools/test_local_ci_lock.sh` (fatal); `tools/test_release_battery_card.sh`, the release battery's card-isolation red arm (it holds an allocation on the card and requires a refusal) | `MEMRA_CI_RELEASE_CARD=0` |
| kernel-check | both `tools/kernel-check-*.cells` manifests required; `ALL GREEN (<n> cells, <m> skipped)` with at most 11 skips | `MEMRA_CI_KC_SKIP_BUDGET` |
| sampler oracle | `sample_check`: `=== sample-check ALL GREEN ===` | `MEMRA_CI_SAMPCHECK=0` |
| batched prime | `prime-gate` on Qwen3.6-35B-A3B | model absent |
| prime continuation | `qwen-a4-continuation-gate` on the 9B NVFP4: `CONTINUATION GATE: PASS` | `MEMRA_CI_CONTGATE=0`, `MEMRA_CI_CONT_MODEL` |
| speculative sweep | `run-spec` K=1..8 on the 35B with its external MTP draft: eight `self-consistency: PASS` lines and `=== SELF-CONSISTENCY PASS ===` | `MEMRA_CI_RUNSPEC=0` |
| Gemma-4 31B | calibrated `tools/argmax-margin-gate.sh`; `gemma-gate` `VERIFY-GATE K=7: PASS` at depth; spec `stream agreement 64/64` | model absent |
| Gemma-4 12B | `run-gen` argmax `MATCH` at depth; `VERIFY-GATE K=7: PASS` | model absent; `MEMRA_G12_MODEL` retargets |
| serving tick | `decode-batch-gate` config B=8 and strict B=4 on the 9B NVFP4 and Q8_0: `ALL GREEN` | `MEMRA_CI_DBG_NVFP4`, `MEMRA_CI_DBG_Q8` |
| prime exactness | `tools/prime-batch-exact-gate.sh` and `tools/prime-tick-exact-gate.sh`, each with its `--canary` | `MEMRA_CI_PRIME_EXACT=0` |
| graph warmup | `tools/graph-warmup-stress-gate.sh` | `MEMRA_CI_GWSTRESS=0` |
| graph lane | `decode-dc-gate`, `graph-decode-gate` (`PASS`), `graph-session-gate` (`ALL GREEN`) | `MEMRA_CI_GRAPH=0`, `MEMRA_CI_GRAPH_MODEL` |
| serving | `tools/serve-smoke.sh`, `tools/health-fault-gate.sh`, `tools/serve-stress-gate.sh` | `MEMRA_CI_SERVE`, `MEMRA_CI_HEALTH_FAULT`, `MEMRA_CI_STRESS` (each `=0`) |
| serving faults and admission | `tools/request-fault-gate.py`, `tools/spec-ctx-edge-gate.sh`, `tools/admit-mem-burst-gate.sh`, `tools/prime-fairness-gate.py`, `tools/gpu-probe-recovery-gate.py` | `MEMRA_CI_FAULTGATE`, `MEMRA_CI_SPEC_CTX_EDGE`, `MEMRA_CI_ADMIT_MEM_BURST`, `MEMRA_CI_FAIRGATE`, `MEMRA_CI_GPUPROBEGATE` |
| served acceptance | `tools/accept-gate.sh` (smoke cell) | `MEMRA_CI_ACCEPT=0` |
| spec on cache hits | `tools/spec-on-cache-hit-gate.sh` qwen arm, its teeth, gemma arm | `MEMRA_CI_HITGATE`, `MEMRA_CI_HITGATE_TEETH`, `MEMRA_CI_HITGATE_GEMMA` |
| engine GPU tests | `cargo test --release -p memra-engine --lib -- --ignored --test-threads=1` | none; see below |

The `#[ignore]` GPU tests run serially: they set process-global gate doors and share one device
(`research/local-ci-one-card-20260921/`). Pair-only tests print `SKIP[2 CUDA devices]: pair-only test`
on one card and run on a pair; the battery names every skip.

### `tools/release-battery.sh` (tag and cell battery)

It needs `kernel-check`, `run-spec` and `argmax-margin-probe` built. Before each cell it waits up to
`CARD_WAIT_S` (180 s) for free memory, then refuses a card that cannot seat the model; it never
evicts another tenant.
`kernel-check` runs once with both cells manifests; then each `tools/release-roster.tsv` row runs the
argmax margin gate on `research/e2e/prompts/board-2048.txt` (`--decode eager` and `--decode batch`,
`--require-floor`) and greedy `run-spec` K=1..8 with its `STEP-IDENTITY` line. Both rows are
required for a tag: the release record admits only the plain battery command.
`--allow-missing-vendor` lets a development run on a box without the Qwen3.8-27B file proceed. It never starts
`memra-server`. `--list-argmax-cells` prints its argmax cells without a GPU.

## Gate catalog

"Runs in" names the standing place a gate runs. "By hand" means no battery runs it: run it for any
change in its area. Receipt paths are under `research/`.

### Kernel and numeric programs

| gate | command | passes when | runs in |
|---|---|---|---|
| `kernel-check` | `target/release/kernel-check --require-manifest tools/kernel-check-27b.cells --require-manifest tools/kernel-check-step35.cells` | last line `ALL GREEN (<n> cells, <m> skipped)`, every required cell present, skips within budget | local-ci, cell battery, tag |
| run-gen argmax | `run-gen <model> <token ids...>` | the prefill-vs-decode argmax line ends `MATCH`; no `MISMATCH-STRUCTURED` from the batched-prime check (`FLIP-NEARTIE` is reported, not fatal) | local-ci, fast-gate tier 1 |
| argmax margin gate | `tools/argmax-margin-gate.sh <model> [--decode eager\|batch]` | the floor rule and the stored rows both pass ([below](#argmax-margin-gate-floors-stored-rows-recalibration)) | local-ci, cell battery, tag |
| `prime-gate` | `prime-gate <model> --prompts-file research/prime-gate-coverage-20260802/prompts-mixed.txt --steps 0` | batched prime and tokenwise first tokens agree; near-tie flips report, structured divergence fails | local-ci |
| prime continuation | `qwen-a4-continuation-gate <model> <prompt> 9296 16 48 80` | `CONTINUATION GATE: PASS`: each head + tail split equals the one-call prime bitwise | local-ci |
| width walk | `qwen-a4-width-walk <model> [ref_width] [widths...]` | `WIDTH WALK width 16 vs 17: 0 of N tensors differ`; a nonzero count names the width-keyed tensors | by hand (diagnostic) |
| boundary tail fold | `prime-fold-gate <model> [--cases n:b,...]` | folded and split primes bit-equal (hidden rows, logits, capture, GDN state, K/V); a stop on a call boundary refuses | by hand, for the prime walk, the GDN prime layer or `Cache::snapshot_from_tap` ([record](decisions/PRIME-TAIL-FOLD.md)) |
| serving tick | `decode-batch-gate <model> --mode config\|strict [--batch N] [--steps N]` | `ALL GREEN`: B=1 and B=N rows bit-identical | local-ci |
| prime exactness | `tools/prime-batch-exact-gate.sh [MODEL] [--canary]`, `tools/prime-tick-exact-gate.sh [MODEL] [--canary]` | concat-batch and tick-shaped primes bit-equal to one `prime_cache`; each canary must be caught | local-ci, fast-gate `pbg9`, `ptick` |
| chunk invariance | `tools/chunk-invariance-gate.sh [<model>] [--chunks 2048,64,32] [--expect-invariant\|--expect-variant] [--canary]` | prefill byte-identical across `MEMRA_PRIME_CHUNK` values | fast-gate `chunkinv*` |
| tick invariance | `tools/tick-invariance-gate.sh [<model>] [--budgets ...]` | a prompt split across prime calls by tick budgets writes the same bytes | fast-gate `tickinv35*` |
| staggered isolation | `tools/iso-gap-gate.sh [<model>] [--canary]` | solo and co-resident logits bit-identical across an FA split-ladder boundary | fast-gate `isogap*` |
| graph warmup | `tools/graph-warmup-stress-gate.sh` | pool growth keeps bit-identity under the `MEMRA_GRAPH_WARMUPS=1` default | local-ci, fast-gate `gwstress` |
| graph lane | `decode-dc-gate <model>`, `graph-decode-gate <model>`, `graph-session-gate <model>` | last line `PASS`, `PASS`, `ALL GREEN` | local-ci; run directly on Hopper |
| sampler | `sample_check`; `cargo test -p memra-sampling` | `=== sample-check ALL GREEN ===`; suite green | local-ci, fast-gate `samp`, `sampt` |
| env audit | unit tests in `crates/memra-engine/src/env_audit.rs` | a retired `MEMRA_*` name refuses; every legal name at once refuses nothing | CI engine suite |
| loader tensor contract | `memra_gguf::checkpoint_binding` tests; `crates/memra-engine/tests/checkpoint_contract_refusal_gpu.rs` (`#[ignore]`) | tampered copies refuse before upload; the clean fixture loads | CI; GPU arm by hand |

`MEMRA_KC_FAST=1` and `MEMRA_KC_ONLY=<sections>` scope `kernel-check` in the dev loop and print a
`KC-SKIP` line per skipped section ([fast-gate](../tools/fast-gate/README.md), [FLAGS.md](FLAGS.md)).
Batteries run it with neither.

### Speculative decoding

| gate | command | passes when | runs in |
|---|---|---|---|
| `run-spec` K=1..8 | `run-spec <model> [tok ids...]` with `MEMRA_SPEC_TEMP=0`; `MEMRA_MTP_DRAFT` for an external head | one `STEP-IDENTITY` line, eight per-K `self-consistency: PASS` lines, `=== SELF-CONSISTENCY PASS ===` and acceptance above 0; a red quotes the failing K and `FIRST DIVERGENCE` | local-ci, cell battery, tag |
| session gate | `session-gate <model>` (`MEMRA_SPEC_K`, `MEMRA_NGEN` per turn) | multi-turn speculative sessions equal the oracle, including resume and no-round bursts | by hand |
| `spec-serve-gate` | `spec-serve-gate <model> [--steps N] [--ts 1,2]` | target-only localizer for the step-identity comparison | by hand (diagnostic) |
| Gemma drafter | `gemma-gate <model> <ids>` with `MEMRA_VERIFY_GATE=7`, or `MEMRA_SPEC=6 MEMRA_DRAFT=<assistant.gguf> MEMRA_NGEN=64` | `VERIFY-GATE K=7: PASS`; `stream agreement 64/64` | local-ci |
| served acceptance | `tools/accept-gate.sh [--full] [--cells a,b] [--control] [--teeth]` | exact `(rounds, drafted, accepted)` and a 128-token text sha256 equal the pinned reference at the default served drafter and K | local-ci (smoke cell), fast-gate `accept` |
| spec on cache hits | `tools/spec-on-cache-hit-gate.sh qwen <model> <memra-server> <out>`; `... gemma <model> <draft> <memra-server> <out>` | speculation engages on prefix-cache hits and the hit's bytes equal a spec-off boot's | local-ci |
| seeded sampling | `tools/seeded-invariance-gate.py <memra-server> <model> [--out DIR] [--penalised host]` | a seeded request writes one text whatever the draft length, companion, resume or demotion | by hand, for sampled verify, sampler counters or the demotion handoff |
| stop strings | `tools/spec-stop-string-gate.py boot <case> <memra-server> <model> --out DIR [--route mtp\|dspark] [--env N=V] [--cohort 2,4]`, then `judge DIR` | speculative rows end where plain decode ends, with one `[spec-stop]` receipt and a bounded close | by hand |
| burst boundary | `tools/spec-burst-boundary-gate.py <memra-server> <model> [--out DIR]` | requests that end on a burst boundary equal plain decode, sampled and greedy | by hand |
| DSpark q38 | `dspark_q38_gate` (fast-gate `dsparkq38`) | `ALL EXACT` | fast-gate, for `dflash.rs` |

Receipts: `research/session-gate-oracle-20261110/`, `research/accept-gate-20260806/`,
`research/position-keyed-sampling-20261110/`, `research/spec-stop-string-20261110/`,
`research/spec-pooled-close-20261111/`, `research/spec-sampled-pending-carry-20261110/`.

`tools/accept-gate.sh --pin` writes the references in `tools/fast-gate/accept-refs/` for the cells
of `tools/fast-gate/accept-cells.tsv`. It refuses
a dirty `crates/` tree or any of `MEMRA_MMQ_F8F4`, `MEMRA_MMQ_F8F4_PLAIN`, `MEMRA_MMQ_FP8BLK_PLAIN`,
`MEMRA_FAST`, `MEMRA_PRIME_F32CHUNK0` set, and has no `--force`. Acceptance is a ratio, so an
acceptance failure is evidence where a tok/s failure is not.

#### The oracle a speculative route is exact against (memra#1146)

A greedy speculative stream must equal plain greedy decode in the numeric class that model's plain
decode is served in. `HybridModel::generate_spec_oracle` is that oracle (the batched step for Step35
and the GatedDeltaNet family, eager `generate` otherwise); `run-spec` and `session-gate` use it.
`STEP-IDENTITY` (`HybridModel::served_step_identity`) bit-compares, in two rounds, four plain steps
of the served decode program with one four-row verify forward over the same tokens and prints
`STEP-IDENTITY: ... -> MATCH` before the K sweep. Where the engine declares the verify forward in the
serving class (`step_identity_enforced`: Step35 and the GatedDeltaNet family) anything else fails the
run; a model without an MTP head or a generic verify forward prints `STEP-IDENTITY: n/a (...)`.
`tools/release-coverage.py` requires exactly one `-> MATCH` or `n/a` line in a `run-spec` log. The
verify-graph replay, pooled verify and device-token verify are covered by the token-level sweep.

### Serving

| gate | command | passes when | runs in |
|---|---|---|---|
| serve smoke | `tools/serve-smoke.sh [model.gguf [draft.gguf]]` | chat (stream and not), completions, concurrency, greedy determinism and spec-vs-plain identity all pass | local-ci, the `serving` cell kind |
| health and lifecycle | `tools/health-fault-gate.sh [model.gguf]` (`HFG_ARMS` picks arms, `HFG_OUT` the receipt dir) | `/readyz`, `/health` and requests hold across warmup, a worker panic and respawn, a latched GPU fault and a SIGTERM drain | local-ci; any change to `health.rs`, the supervisor, the boot probe or drain |
| concurrency | `tools/serve-stress-gate.sh [--teeth] [model [draft [n_clients]]]` | 64 staggered streams finish well formed with a live worker and no OOM line; `--teeth` must invert | local-ci, fast-gate `sstress` |
| request fault | `tools/request-fault-gate.py --model <gguf> --bin <memra-server> --out <new-dir>` | an injected per-request panic ends only that stream (`worker_fault`); peers finish byte-equal to control | local-ci |
| context edge | `tools/spec-ctx-edge-gate.sh <model> <memra-server> [out]` | open requests at their cap under default spec end `length` with no crash line | local-ci |
| memory-admission burst | `tools/admit-mem-burst-gate.sh <model> <memra-server> [out]` | 64 open requests: no OOM, no 503, every refusal a 429 with `Retry-After` | local-ci; SKIPs where memory-shaped admission is off by default, which is every card except the full-power RTX PRO 6000 Blackwell Workstation Edition |
| drafter admission | `tools/drafter-admission-gate.py --model <gguf> [--draft <export-dir>] --bin <memra-server> --out <new-dir>` (`--external-lock 9` under `gpulock run`) | a DFlash request that does not fit with its draft plane is never admitted with the plane uncharged: no `eager-arm admit` line, each of the four `drafter refused` arms taken with receipt numbers that agree with it, every refused request served plain with the reference text, a step-OOM replay never speculating on a plain charge; without `--draft` the MTP eager arm must show the capture's refusal | by hand, for the VRAM gate's verdict or the DFlash route in `admit`; CPU twins `worker::tests::the_vram_gate_never_admits_a_dflash_request_without_its_draft_plane`, `the_drafter_refusal_is_wired_from_the_gate_to_every_dflash_route` and `python3 tools/test_drafter_admission_gate.py` |
| same-pass admission | `tools/same-pass-admission-gate.py --model <gguf> --draft <export-dir> --bin <memra-server> --out <new-dir>` (`--external-lock 9` under `gpulock run`) | two cold DFlash requests that each fit alone but not together, queued for one admission pass beside a decoding keeper (a counting run; the gate server's `MEMRA_SPEC_GATE_LOW=3` keeps all three on the DFlash route) with a ballast placing effective free between one fit and two (the charges plus the reserve the gate applies, read from a probe request that defers and expires) on a fresh `/metrics` snapshot: the pass never admits both on one reading (the second waits in a settle wait), no `out of memory` line, both answer 200 with the reference greedy text; a pair the worker took in two passes, or priced plain, reads NOT-PLACED (exit 3) | by hand, for the VRAM gate's booking or a route that builds its session state in the tick; CPU twins `worker::tests::an_admitted_session_with_no_route_state_books_its_charge_until_the_tick_builds_it`, `the_armed_door_books_the_whole_prompt_stack_of_an_unbuilt_session`, `every_route_state_field_counts_as_built_and_the_settle_wait_precedes_every_verdict` and `python3 tools/test_same_pass_admission_gate.py` |
| prime fairness | `tools/prime-fairness-gate.py --model <gguf> --bin <memra-server> --out <new-dir>` | texts identical across `MEMRA_PRIME_YIELD` arms; peers' first token within bar under a long prime | local-ci |
| GPU probe recovery | `tools/gpu-probe-recovery-gate.py --model <gguf> --bin <memra-server> --out <new-dir>` | probe hangs degrade then recover; a third miss or an ECC answer latches 503 | local-ci |
| safetensors serving | `tools/serve-st-gate.sh [st_dir]` | an HF checkpoint dir serves through its own template; CLI and server streams agree under the stated near-tie rule | by hand |
| API keys | `tools/apikeys-gate.sh [model] [out_dir]` | 401/403 refusals, tenant cache isolation, per-tenant rate headers, lane law, hot revoke | by hand |
| mode switch | `research/spec-gate-20260806/exactness.py` | a demoted session's stream is byte-identical to one batched from the start (`MEMRA_SPEC_DEMOTE_AT=N` pins the transition) | by hand, for scheduler phase order or the demotion handoff |
| drafter attach | `tools/drafter-attach-wiring-gate.sh` | every drafter-attach call site is registered and its assertion executes both ways | local-ci (fatal) |
| route contract | `route_contract::tests` (CPU) | every serve route declares each policy surface; an empty stub route fails | CI server suite |

Receipts: `research/request-fault-20260922/`, `research/spec-ctx-edge-20260923/`,
`research/prime-fairness-default-20260922/`, `research/gpu-probe-recovery-20260922/`,
`research/route-contract-20260922/`, `research/dsv4-route-policies-20260922/`,
`research/spec-gate-20260806/`, `research/drafter-admission-20261112/`.

### Prefix cache

All four boot the real `memra-server` on one card, hold the canonical rig lock for the whole cell
(`--external-lock FD` under the collector), print one verdict line ending `-> PASS|FAIL`, and exit
0 (pass), 1 (fail) or 2 (`REFUSED: ...`).

| gate | command | passes when |
|---|---|---|
| eviction credits admission | `tools/prefix-evict-reclaim-gate.py --model <gguf> --bin <memra-server> --out <new-dir>` | an eviction on defer credits at least the entry's bytes, the waiting prefill is admitted in the same tick, and texts match across boots |
| restore identity | `tools/prefix-restore-identity-gate.py --model <gguf> --bin <memra-server> --out <new-dir>` | every restore point gives the cold completion and lands on the GDN prime grid |
| newest turn fits | `tools/prefix-newest-turn-fits-gate.py --model <gguf> --bin <memra-server> --out <new-dir>` | each turn of a growing conversation restores the previous turn's entry under plain-LRU pressure, bytes balance, and texts equal cold |
| exact re-send | `tools/prefix-exact-resend-gate.py --model <gguf> --bin <memra-server> --out <new-dir>` | an exact re-send of an off-grid prompt restores the whole prompt through the entry's prompt-end tail with no prime call, growth still restores the grid entry and primes on the grid, a tail attached by a covered hit serves its own prompt only, and all nine texts equal cold |

A whole-prompt `cached_tokens` on an off-grid prompt is accepted only with its tail's witness: the
sellgate replay counts such restores against the `/metrics` `prefix_cache_tail_hits` delta, and
`tools/collect-serving-qualification.py` requires the entry's `[prefix-cache] insert tail:` line in
its own cell's server log (`cache_qualification.tail_witness`). Without the witness it is the
off-grid prompt-end entry of memra#602 and fails.

The capture and eviction law is in [prompt-cache.md](serving/prompt-cache.md); the policy choice is
[PREFIX-CACHE-POLICY](decisions/PREFIX-CACHE-POLICY.md). Receipts: `research/spill-b-20260919/`
days 13 to 19 and `research/hit-prompt-end-20261111/` (the prompt-end tail and the hit-suffix
fence, [PREFIX-PROMPT-END-TAIL](decisions/PREFIX-PROMPT-END-TAIL.md)).

### Multi-GPU (PP-N) exactness gates: run on the multi-card box
<a id="multi-gpu-pp-n-exactness-gates--run-on-the-multi-card-box"></a>

`tools/local-ci.sh` runs on one card, so these need a 2+ card host. Any change to `pp.rs`, the
stage-split dispatch or a decode path a split walks re-runs them.

| gate | invocation | proves |
|---|---|---|
| `ppn-gate` | `MEMRA_PP_DEVICES=0,1 ppn-gate <model.gguf> [stages=2] [P=16] [N=32]` | eager stage-split decode bit-identical to the unsplit walk, serial and pipelined |
| batched split | `decode-batch-gate <model.gguf> --mode pp [--batch 1,4,8] [--stages N] [--reps R]` | batched stage split bit-identical per row per step |
| PP-3/PP-4 decode wavefront | `MEMRA_PP_WAVE=1 MEMRA_PP_OVERLAP=1 MEMRA_PP_DEVICES=0,1,2[,3] decode-batch-gate <model> --mode pp --stages 3\|4 --batch 1,2,4,8,16,24[,32] --reps 5` | serial-wave and live-wave logits identical, row order, cache advance, non-vacuous counters |
| PP-3/PP-4 prime wavefront | `tools/prime-split-gate.sh <model> --stages 3\|4 --devices 0,1,2[,3] --chunks auto,513 --steps 8` | unsplit, serial split and wavefront primes identical while split and overlap counters advance |
| spec-verify split | `decode-batch-gate <model.gguf> --mode ppspec [--ts 2,5,9] [--stages N] [--reps R]` | verify split bit-identical per logit column, plus the `h_seed` column |
| `pp2-gate` | `MEMRA_PP_DEVICES=0,1 pp2-gate <model.gguf> [P=16] [N=32] [split=n_layers/2]` | the N=2 eager gate; it resets `MEMRA_PP_STAGES`/`SPLITS` itself |
| `pp-transport-smoke` | `pp-transport-smoke` | the peer boundary transport alone |

Keep `--reps` at 2 or more (the class these gates catch was intermittent), open the PP door before
load, and keep the `unsplit@ppncache` and `epilogue` arms that localize a red. `MEMRA_PP_WAVE` is off
by design ([FLAGS.md](FLAGS.md)). Receipts: [`research/pp2-batch-20260806/`](../research/pp2-batch-20260806/),
[`research/pp2-spec-20260806/`](../research/pp2-spec-20260806/),
[`research/pp2-hardening-20260806/`](../research/pp2-hardening-20260806/),
[`research/m2-pp8-20260802/`](../research/m2-pp8-20260802/).

### Target-aware release evidence

Per-cell evidence takes any device of a push-gated target: an RTX PRO 6000 Blackwell counts for
`sm_120a` like an RTX 5090. A default keyed on a card name needs that card. Step-3.7-Flash-FP8 multi-card
work, whose blocking target is the RTX PRO 6000 Blackwell, also passes the `step-pro` gate:

```bash
MEMRA_HARDWARE_GATE=step-pro \
MEMRA_STEP_PRO_RECEIPT=/absolute/path/to/step-pro-receipt.json \
git push
```

`tools/check_hardware_gate.py` checks the receipt's model and hardware identity, requires kernel,
topology and model exactness evidence, verifies every evidence manifest and binds every changed
engine source file by SHA-256. It adds to the per-cell and tag evidence and never replaces it.
Development mode cannot claim a hardware gate.

### DSv4 gate placement

`Dsv4Gpu::load` has no default placement (owner, 2026-11-03): every caller names one, and every load
logs `[load] placing <name>` first. `tp_ep` (`Dsv4Placement::SERVED`, TP/EP with exact attention TP2)
is what `memra-server` loads; `pp` (PP-2) is what it loads with `MEMRA_DSV4_TOPOLOGY=pp`;
`tp_ep_replicated` (replicated attention) is gate-only. A serving check takes the `tp_ep` cells
first, on a 2x RTX PRO 6000 pair under `/tmp/memra-gpu.lock`; a PP-2 cell covers only `pp`. A gate
with cells on more than one placement requires `--placement=<name>`. The selectors it replaced
refuse and name the placement they meant: `--tpep` alone is `tp_ep_replicated`, and with the
retired `MEMRA_DSV4_ATTENTION_TP_GATE=1` it is `tp_ep`; `DSV4_ROWS_GATE_TOPOLOGY=tp_ep` is `tp_ep`, `pp` or
unset is `pp`. A research command ported to `--placement=tp_ep` without the variable changes
numeric class.

| Gate binary | Placement | Why |
| --- | --- | --- |
| `dsv4-ar-phase-gate`, `dsv4_compose_cadence_dense_gate`, `dsv4_dense_exact_tail_gate`, `dsv4_dense_fast_gate`, `dsv4_hc_dot_split_gate`, `dsv4_kv_split_gate`, `dsv4_launch_boundary_gate`, `dsv4_tp_replay_long_gate`, `dsv4_park_boundary_gate`, `dsv4-dense-arm-ab` | `tp_ep` | The served placement; each checks or measures one mechanism on it, and none alone covers the served program. Receipts before memra #710 are PP-2 cells; the park red arm owes one `tp_ep` run against a reverted driver. |
| `dsv4-gpu-dspark-gate`, `dsv4_tp_ep_verify_gate` | `--placement=tp_ep\|tp_ep_replicated\|pp` | `tp_ep` first; `pp` is the PP-2 (DSpark) or control (verify) cell. |
| `dsv4_rows_gate` | `--placement=tp_ep\|pp` | TP/EP B-row groups need attention TP2, so there is no replicated cell. |
| `dsv4_tp_ep_gate`, `dsv4_tp_ep_sampled_perf_gate` | `--placement=tp_ep\|tp_ep_replicated` | TP/EP-only gates; the full-token replay cells need `tp_ep`. |
| `dsv4_c4_direct_gate`, `dsv4_c4_host_gate`, `dsv4_decode_profile`, `dsv4_decode_rate_gate`, `dsv4_plain_perf_gate` | `pp` | C4 history in pinned host RAM, which TP/EP refuses. `dsv4_c4_host_gate capture` refuses the matrix expert program at HEAD. |
| `dsv4_program_accuracy`, `dsv4_moe_program_class_gate`, `dsv4_matrix_distribution_gate` | `pp` | The reference expert program loads on PP-2 only. |
| `dsv4_capacity_gate`, `dsv4_ep_pair_gate`, `dsv4_pipeline_gate`, `dsv4_graph_probe_gate` | `pp` | PP-2-only reach: 1M context with DSpark, the whole-expert EP pair, request pipelining, the per-layer graph probes. |
| `dsv4_long_decode_gate`, `dsv4_chunk_sweep_gate`, `dsv4_coupled_proposal_gate`, `dsv4_dense_census`, `dsv4_dense_tile_gate`, `dsv4_prefill_work_gate`, `dsv4_matrix_program_gate`, `dsv4_wide_prefill_gate` | `pp` | Historical instruments that pin the doors they were taken with; a TP/EP cell is a new cell. `dsv4_wide_prefill_gate` stops at its `grouped=true` arm at HEAD. The served long-decode cell is `dsv4_tp_replay_long_gate`. |
| `dsv4-gpu-gate`, `dsv4-gpu-greedy`, `dsv4-gpu-decode-gate`, `dsv4-decode-probe`, `dsv4-native-gemm-gate`, `dsv4-gpu-tf-gate`, `dsv4-gpu-sampled-gate`, `dsv4-drafted-corpus`, `dsv4-decode-bench`, `dsv4_host_cache_gate`, `dsv4_grouped_prefill_gate` | `pp` named; **no runnable cell at HEAD** | Their monolithic or grouped prefill refuses the matrix expert program ("matrix request program requires chunked prefill"), and none arms the reference executor. |

`dsv4-hc-grid-gate`, `dsv4-dense-family-bench`, `dsv4_q_layout_gate`, `dsv4_sampler_sort_gate` and
`dsv4_http_prompt_pack` load no model and have no placement. The cells that refuse at HEAD refuse
because of the expert program (the loaded default since memra #461), not the placement. Whether
each is ported (to chunked prefill, or by arming the reference executor before load as
`dsv4_program_accuracy` does) or deleted is an open owner decision. Until then a battery lists them
with the expected result REFUSE.

DSv4 mechanism gates run on one CUDA card under the rig lock with `NVIDIA_TF32_OVERRIDE=0`, or on
the pair under `/tmp/memra-gpu.lock`. Receipts are under `research/dsv4f-bringup-20260923/`.

| mechanism | command | card | receipt dir |
|---|---|---|---|
| DSpark spec == plain, served program | `dsv4-gpu-dspark-gate <model-dir> <fixtures.json> <out-dir> [runs] [dev0,dev1] --placement=tp_ep --served` | pair | top level |
| HC24 default engagement | `dsv4_hc_dot_split_gate <model-dir> <source.txt> <new-out> --defaults` (`MEMRA_DSV4_HC_DOT_SPLIT` unset, then `=0`) | pair | |
| C4 store position split | `dsv4_kv_split_gate <model-dir> <source-tape> [prompt-tokens] [steps]` | pair | `kv-split/` |
| TP/EP B-row steps and graphs | `dsv4_rows_gate <model-dir> <source-tape> 24 64 --placement=tp_ep` | pair | `tp-rows/` |
| full-token replay past 512 | `dsv4_tp_replay_long_gate <model-dir> <source-tape> [steps]` | pair | `tp-replay-long/`, `tpep-default/` |
| MoE stream visitors, fused one-token MoE, deferred checks | `cargo test -p memra-engine --release --lib <cuda_m1_stream_matches_sktail_bit_for_bit\|cuda_mrow_stream_matches_sktail_bit_for_bit\|dsv4_grouped::\|cuda_deferred_moe_faults_match_the_synchronous_checks> -- --ignored --test-threads=1` | one | `moe-fused/` |
| small-kernel diet, HC finish, prefill tiles, BF16 island, latency kernels | `cargo test -p memra-engine --release --test <dsv4_small_diet_gpu\|dsv4_hc_finish_gpu\|dsv4_gemm_tile_gpu\|dsv4_island_bf16_gpu\|dsv4_latency_kernels_gpu> -- --ignored --test-threads=1` | one | `small-diet/`, `hc-finish/`, `prefill-tile/`, `cmp-diet/`, `latency/` |
| gate source tape | `tools/dsv4-source-tape.py <out.txt>` (refuses any digest but `11e4bd80...`) | CPU | |

Each requires `to_bits`-equal outputs against the program it replaces, with red arms that must move
the output. `tools/test-dsv4-dense-control-policy.sh` checks the dense control drivers before CUDA.

### Generic spill / tiered KV

The shared contract is `crates/memra-tier/src/contracts.rs`; the conformance schedules are the
public module `memra_tier::conformance`, so native gates import the same functions. These CPU
commands are execution gates, not hardware evidence:

```sh
tools/portable-suites.sh          # cargo test -p memra-tier -p memra-kv -p memra-cli --offline --locked --no-fail-fast, through the skip census
tools/test_portable_suites.sh     # its teeth: planted tier/KV/CLI failures and an undeclared SKIP must red the wrapper
cargo clippy -p memra-tier -p memra-kv --offline --all-targets --no-deps -- -D warnings
python3 crates/memra-tier/tests/contracts/fixture_reference.py --check
python3 -m pytest -q crates/memra-tier/tests/battery/
```

| gate | command | passes when |
|---|---|---|
| native conformance | `tier-transfer-gate` | one `PASS` line per schedule (v1 through canonical v1.3, the recovery rules) and `PASS native governor zero after controlled drain` |
| pinned allocation A/B | `tier-transfer-gate pinned-ab [--bytes N] [--pairs N]` | measurement; the per-device default is in [PINNED-DESTINATIONS](decisions/PINNED-DESTINATIONS.md) |
| KV tiering | `kv-tier-gate --artifact <gguf> --case baseline\|active\|grow --context 8192\|32768 --tiers host --same-program [--kv-allocator pooled\|vmm\|vmm-ondemand] [--reclaim-diagnostic [--reclaim-cycles N]] [--fault <arm>] --out <new-dir>` | `BASELINE_CAPTURED` or `ACTIVE...` receipts; `FAULT-ARM PASS <arm>`; `GROW-G1 PASS`; refusals are a final `REFUSED: <reason>` line, exit 2 |
| experts via tier | `run-gen`/`run-spec <gguf> --experts-via-tier [--expert-bank-host-bytes=N] [--expert-bank-gpu-bytes=N]` | the standard `MATCH` and `SELF-CONSISTENCY PASS`; a budget or catalog refusal is `REFUSED: ...`, exit 2 |
| copy plumbing | `h2d-probe --copies N [--dry-run]` | N=1 plumbing records, never medians; scored envelopes use `tools/tier-envelope.py` |
| host tenant reclaim | `tools/kv-host-tenant-reclaim-gate.sh base\|fix` | at the tenant share cap a demotion evicts that tenant's own oldest entry, then demotes |
| host tier identity and failures | `tools/kv-host-spill-identity-gate.sh`, `tools/kv-host-spill-failure-gate.sh` | demote, promote and restore equal the tier-off cold bytes (the identity gate's `MEMRA_HOSTGATE_TEETH=1` red arm must invert); each injected failure prints its named line and serving keeps the reference bytes |

The v1 to v1.3 schedules are frozen byte for byte and `WIRE_VERSION` stays 1. Two recovery rules
sit beside them, with red arms (`crates/memra-tier/src/conformance/recovery.rs`): a restore
cancelled before its consumer event hands the untouched host source back or refuses the cancel,
and a decode or prime over a cache with a suspended layer refuses at `Cache::ensure_usable`. The
G1 reclaim status and its rules are in [KV-PHYSICAL-RECLAIM](decisions/KV-PHYSICAL-RECLAIM.md);
the interface in [GENERIC-SPILL-INTERFACE-V1](decisions/GENERIC-SPILL-INTERFACE-V1.md).

#### Collector: `tools/tier-battery.py`

Launch every native tier cell through the collector, with an absolute executable and a new output
directory:

```sh
python3 tools/tier-battery.py --rig rtx5090|pro-single|pro-pair|pro-four --timeout <seconds> --out <new-dir> --execute <absolute-bin> <args>
```

Locks: `rtx5090` takes `/tmp/memra-5090.lock`, the `pro-*` rigs `/tmp/memra-gpu.lock`, non-blocking
(contention refuses, exit 2). The default rig is `pro-pair`; always state it. `--external-lock`
hands the lock FD to a legacy shell gate through the `@COLLECTOR_LOCK_FD@` argument.
`MEMRA_TIER_BATTERY_LOCK_DIR` re-roots both lock names for the collector's own tests only and needs
`--private-lock-dir-for-tests`. A cell's status is `executed-not-qualified`, `failed` or `refused`;
every capture records `qualification: false`, and the collector computes no medians for native cells.

### Hardware bring-up

| gate | command | passes when |
|---|---|---|
| box health | `bash tools/box-health.sh [OUTDIR]` | exit 0: fit to measure. Exit 1: do not open the measurement window |
| kernel peer read | `nvcc -O2 -arch=${MEMRA_CUDA_ARCH:-sm_120} -o peer-read-probe tools/peer-read-probe.cu` | exit 0 bytes valid both ways; 2 wrong bytes; 3 CUDA error; 4 fewer than two devices; 5 no peer-capable pair |
| peer copy direction | `tools/peer-copy-direction-probe.cu` | producer writes and consumer reads validate from 16 KiB to 64 MiB (correctness only) |
| B200 dry gate | below | every command exits 0 without opening the CUDA driver |
| model-owned device memory | `tools/qualify-model-device-memory.py` under an exact per-card lease | each named ownership and memory stage passes ([protocol](../research/glm-tp-device-ownership-20260920/QUALIFICATION.md)) |

Run box health first in every measurement window: it checks power caps, clock degradation, PCIe
link width and generation, BAR1, CPU affinity, IOMMU mode, ACS redirection, P-states and a kernel
peer read (a missing `nvcc` fails it).

```bash
research/b200-kernel-twins-dry-20260901/check-layouts.sh
research/b200-kernel-twins-dry-20260901/check-nvfp4.sh
research/b200-kernel-twins-dry-20260901/check-fp8.sh
tools/test_install_b200_policy.sh
tools/test_b200_phase0_harness.sh
MEMRA_CUDA_ARCH=100a cargo build --release --bins
MEMRA_CUDA_ARCH=100a cargo test --release -p memra-engine b200_dry_policy_tests --lib
research/b200-kernel-twins-dry-20260901/run-box-phase0.sh --plan   # the device plan, host-only
```

The dry gate compiles the production units, inspects archive ABI and SASS, runs the operand and
scale layout contract and proves the installer still refuses a B200 prebuilt; CI runs it in the
`100a` arch cell. It is not device evidence: `run-box-phase0.sh run` is, with receipts under
`research/b200-kernel-twins-dry-20260901/receipts/`.

### CPU contracts and fixtures

All run on CPU in `.github/workflows/ci.yml` except the fast-gate components and the sparse
preflight test, which run locally (CI runs the two component suites, `tools/test_release_qualification.py`
and `tools/test_release_inputs.py`, directly). Each proves its tool's control flow, never a model
or kernel.

| contract | command | proves |
|---|---|---|
| skip census | `python3 tools/skip-census.py verify`; `python3 tools/test_skip_census.py` | every artifact-gated test skip is declared in `tools/skip-census.tsv` ([Writing a gate](#writing-a-gate)) |
| release battery fixture | `python3 tools/test_release_battery_coverage.py` | CPU stand-ins: exact K=1..8 verdict set, required manifests, named skips and their budget, nonzero exits refused |
| argmax floors census | `python3 tools/test_argmax_margin_floors.py` (`tools/check-argmax-floors.py`) | every battery argmax cell has its floor row and stored rows on every device in the table |
| argmax verdict and wrapper | `cargo test -p memra-engine --lib argmax_margin`; `cargo test -p memra-engine --bin argmax-margin-probe`; `python3 tools/test_argmax_margin_gate.py` | the clauses on synthetic rows, the `--judge` path, the wrapper and its canary |
| coalescer | `python3 tools/check-coalescer-contract.py --out target/coalescer-contract` | the exact server Coalescer source against virtual-clock fixtures and five mutants (local-ci and CI; receipt `research/scoped-validation-20261002/`) |
| flags and hook fixtures | `tools/test_check_flags.sh`, `tools/test_flags_guard.sh` | the flags census, and a real `git push` through the real hook: the flags skip logs, the retired perf skip refuses |
| gate integrity | `tools/test_gate_integrity_r2.sh`, `tools/test_gate_template_integrity.sh`, `python3 -m unittest tools.test_generate_arch_gates` | one forced failure per fixed defect; a generated gate against a stub server |
| public boundary | `python3 tools/check-public-boundary.py verify-allowlist`; `python3 tools/test_public_boundary.py` | every grandfathered entry still describes a real match (CI only; too slow for the hook) |
| workflow keys | `python3 tools/check-workflow-keys.py`; `tools/test_workflow_keys.sh` | no duplicate mapping key in any workflow file (pre-push and CI) |
| expert-tier planner | `python3 tools/build_expert_tier_plan.py --self-test`; `python3 tools/run_expert_tier_contract.py` | the self-test and its admission controls, no skips |
| SFT generator | `python3 tools/run_sft_gen_contract.py` | the generator's methods and admission controls; no generation or provider call |
| score shards | `python3 tools/merge_expert_score_shards.py --self-test`; `python3 tools/run_score_shard_contract.py` | a real fixture merge and its controls ([receipt](../research/score-shard-caller-986-20261003/README.md)) |
| agreement summary | `python3 tools/summarize_hy3_plan_agreement.py --self-test`; `python3 tools/run_agreement_summary_contract.py` | the summary predicates and controls ([receipt](../research/agreement-summary-989-20261004/README.md)) |
| fast-gate components | `tools/fast-gate/fast-gate.sh --component qualification-contracts\|release-inputs --cache "$HOME/.cache/memra/components"` | two isolated CPU suites with input-bound reuse ([fast-gate](../tools/fast-gate/README.md)) |
| controlled build reuse | `bash tools/unittest-floor.sh tools test_local_build_cache.py 48` | capsule admission and restore refusals ([BUILD-REUSE](../tools/fast-gate/BUILD-REUSE.md)) |
| dense control cache | `python3 tools/test_dense_control_cache.py` | a cache hit never supplies a control result ([proof](../research/dense-control-cache-20261003/PROOF.json)) |
| public CI orchestration | `python3 tools/test_public_ci_integration.py` | routing and execution against real Git events ([CI](CI.md)) |
| sparse input preflight | `python3 tools/test_sparse_input_preflight_integration.py` | the preflight's three input contracts ([doc](../tools/sparse-input-preflight.md)) |

CI floors (`tools/unittest-floor.sh <dir> <pattern> <floor>`) live in `.github/workflows/ci.yml`.

## Argmax margin gate: floors, stored rows, recalibration

The rule is [ARGMAX-GATE-GAP-SHIFT-FLOOR](decisions/ARGMAX-GATE-GAP-SHIFT-FLOOR.md) and
[ARGMAX-GATE-STORED-ROWS](decisions/ARGMAX-GATE-STORED-ROWS.md); the wrapper's header lists every
flag. Per row the probe gives each program's top-1 id and top-2 margin; a row's **gap shift** is how
far the two programs disagree there. The **floor** of a cell (model, prompt, decode program, device)
is the largest gap shift at an agreeing row over the whole prompt. The judged window is the last 12
rows (`--window`). A flip inside the floor is a near tie, counted against `--max-flips`.
`WIDE-FLIP` and `WIDE-ROW` (above the floor), `TOO-MANY-NEAR-TIES`, `UNCALIBRATED-FLIP` (no floor
row) and `KNOWN-ROW-WIDER` fail; a `KNOWN-ROW` no wider than recorded is printed and passes.

**Floors** live in `tools/argmax-margin-floors.tsv`, one row per (model, prompt, prompt sha256,
decode program, exact `nvidia-smi` device name), with its row count, the row that set it, the p50,
p99 and p99.9 of the agreeing rows, its known rows and its receipt. There is no fallback floor: a
laptop and a desktop RTX 5090 are different devices until measured equal. With `--require-floor`
(the release battery) a missing row on a maintained class (`rtx5090`, `rtxpro6000`, `gb10`, the
push-gated devices) or an unknown name is refused; an unmaintained class (H100, B200) is `NOT-JUDGED`.

**Stored rows.** `tools/argmax-margin-rows/` holds main's printed window per cell, keyed by model
name and sha256, prompt name and sha256, decode program, device, window and expert placement
(`resident`, `slru`, `none`). The candidate must print `STORED-ROWS: identical`, byte for byte;
`ROW-DIFF` / `STORED-ROWS: DIFFERENT` fails, and a missing table is `NO-STORED-ROWS` (refused) in the
battery. A cell passes only when both checks pass. The model digest is cached in
`${XDG_CACHE_HOME:-~/.cache}/memra/model-sha256.tsv`; `--rehash-model` hashes the bytes again.

**Calibrating a cell.**
`tools/argmax-margin-gate.sh <model> [--prompt f] [--decode batch] --calibrate --logdir research/<lane>/<cell>`
runs every row and prints `CALIBRATION:` and `FLOOR-ROW:`; it judges nothing. Add the row with the
banked log as its receipt and keep the log whole. Store main's rows with
`--calibrate --rows-out FILE --commit C`, or from a receipt with `argmax-margin-probe --judge <receipt>
--calibrate --receipt <receipt> <floor key> --model-sha256 S --window 12 --rows-out FILE --commit C`.
`tools/check-argmax-floors.py` requires each battery floor row's stored table to come from that
row's receipt. A new device starts by calibrating.

**A lane that gates on its own prompt** calibrates the base binary on that prompt, keeps the floor
row and stored rows in its receipts, and runs base and candidate with `--floors <table>` and
`--stored-rows <dir>`. It never borrows another prompt's row.

**Recalibration.** A floor row or a stored table changes only when a decision record accepts a
change of numeric class for that cell (a new kernel order, split or cache format). The same PR reruns
`--calibrate` on the changed binary, replaces the floor row and the stored table, banks the new log
and shows the old and new rows in the record and the PR body. Never raise a floor or rewrite stored
rows to turn a red cell green without that record.

## Writing a gate

A gate can run, print a pass and assert nothing. Write against these rules (the audits behind them
are in [history/TESTING.md](history/TESTING.md)):

- **A suite that did not run is not green.** Assert `test result: ok.`, a nonzero passed count and
  `0 filtered out`. A name filter passes with `0 passed` the day a name moves: never let one stand in
  for a suite, and assert a nonzero count when a targeted run needs one.
- **A skip is not a pass.** A skipping test prints one `SKIP[<artifact>]: <reason>` line per missing
  case, counted against a named budget. Any other skip-shaped print fails the census.
- **No assertion that cannot fail**: no fallback that fires on missing input and then claims a
  pass, no `all()` over nothing, no grep that matches a comment, no gate without a caller, no
  verdict term counted and then dropped. A canary fails for the reason under test, not "lock busy".
- **Wiring assertions anchor on the invocation in comment-stripped text**; watch one fail by
  unwiring the gate.
- **Geometry first, then bytes**: assert every config scalar before any value compare
  (`crates/memra-engine/src/parity_geometry.rs`).
- **A gate that pins a door verifies the pin took**, and a default-posture battery never enables a
  non-default route to get past a refusal.
- **Pin the configuration and force the transition** when the property sits inside a
  nondeterministic configuration; do not provoke it under load and diff.
- **A fixture pins every environment fact it depends on** and asserts its setup did something.
- **An exceptions list has an expiry check**; when every entry is dead, delete it and assert its absence.
- **Servers bind ports through `tools/port-guard.sh`**: an occupied port aborts, never waits.
- **Fix generated gates in the template** (`tools/generate-arch-gates.py`; [ONBOARDING.md](ONBOARDING.md)
  section 6): ports from 18300-18399, a skip exits 77, `tools/test_gate_template_integrity.sh` runs
  an emitted gate, and `MEMRA_ARCH_GATE_ALLOW_SKIP=1` accounts for a skip out loud.
- **A new pre-push arm lands with its stub** in `stage()` of `tools/test_flags_guard.sh` and, for
  a new assertion, a raised `EXPECTED_ASSERTIONS`, in the same commit. Its census reads the real
  manifests and workflow files; fixtures belong in its teeth. A check under a second goes in the
  hook, a slower one in CI, a matrix in a parallel job (the record:
  [history/RELEASING.md](history/RELEASING.md)).
- **`tee` raw output, then parse.** Use the two lock files in AGENTS.md, never a third name.
  Estimate CI cost from a CI run.

`tools/skip-census.py` verbs: `static` prints the census the source implies; `verify` compares it
with `tools/skip-census.tsv` in both directions over every crate's `src/` and `tests/`;
`run -- cargo test ...` checks the suite's verdict, then counts protocol lines against the budget
in the variable `--budget-var` names (0 when unset) with `--min-passed`; `report <file> --expect N` does the same for shell
gates writing `$MEMRA_SKIP_CENSUS`, and a missing file fails. Budgets in force:

| where | suite | budget |
|---|---|---|
| `tools/local-ci.sh` | memra-gguf lib, artifacts present | 10 |
| `tools/local-ci.sh` | kernel-check skipped cells | 11 |
| `.github/workflows/ci.yml` | memra-gguf, -reference, -tokenizer, -validate, -sampling (no artifacts) | 20 |
| `.github/workflows/ci.yml` | memra-engine lib, memra-net-guard | 0 |
| `tools/portable-suites.sh` | memra-tier, -kv, -cli | 0 |

## Performance tripwires

Two checks warn about speed and neither is evidence. `fast-gate --smoke` compares one probe rep with
the golden-point tok/s (WARN above a 10% drop, FAIL above 25%). `tools/local-ci.sh --perf` judges each
cell against the rolling median of earlier rows, a cross-day comparison. A red from either means
"re-measure": build the last-green binary and run the cell interleaved A/B in one thermal window
under one lock, five runs per arm ([method](PERFORMANCE.md#method)). Push evidence uses the
`perf` cell kind instead.

## Known blind spots

- **Default-off rollback seams**: kernels reached only through a non-default env seam are invisible
  to naked probes; the bit-identity `kernel-check` arms guard them.
- **Uniform numeric skews** on an arm with no bit-identity twin barely move greedy tokens; that class
  belongs to the full battery.
- **Sampled serving**: `sample_check` oracles the kernels; distribution drift in a sampled stream is
  an eval concern. The seeded and stop-string gates cover sampled exactness.
- **GDN scan class**: `MEMRA_GDN_MMA=1` and `=0` give the same 20 ids on `q9` and `q35`, so the
  fast-gate pins cannot tell them apart (`research/fastgate-goldens-20261104/`).
- **Acceptance drift** is invisible to exactness gates; `accept-gate` and the perf cells' acceptance
  verdicts see it.
- **Reach**: tier 1 probes one model per kernel class; no standing battery runs the PP-N or pair-only
  gates on every change; the tag battery covers two qwen35-class checkpoints and no served path.

## Records

- Gate histories, lane sections, the earlier full arm specifications (prefix, spec serving, DSv4,
  tiered KV), the gate-integrity audits and the public CPU CI text: [history/TESTING.md](history/TESTING.md).
- Fast-gate timings, catch demonstrations and depth sweeps: [`research/fast-gate-20260802/`](../research/fast-gate-20260802/).
- Argmax gate rule and stored rows: `research/argmax-gate-rule-20261110/`, `research/argmax-stored-rows-20261110/`.
- Spill program contracts and day logs: `research/spill-lead-20260919/` and its sibling lanes.
- One verdict line per research directory: [research/INDEX.md](../research/INDEX.md).

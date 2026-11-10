# serve-batch-removal-20261110

Receipts for memra #1158: `MEMRA_SERVE_BATCH` is retired and the batched scheduler is the only
served path.

**Verdict: the default (speculative) server equals the `MEMRA_SERVE_SPEC=0` server on the batched
program, on main and on the tree without the round-robin scheduler.** Text, completion tokens and
token ids are equal at a filled 400-token window on both trees, and the candidate's ids equal
main's. No tolerance was needed: the public stream stops at the request budget in both arms.

## What was measured

`tools/serve-st-gate.sh` check 4 was the one gate that booted the round-robin scheduler
(`MEMRA_SERVE_SPEC=0 MEMRA_SERVE_BATCH=0`), as the oracle of a prefix match. Before that
scheduler was deleted, the check was changed to compare the default server with the
`MEMRA_SERVE_SPEC=0` server alone, as an equality, and run on main's binaries. Then the same
checks ran on the candidate.

Conditions, every row:

- one RTX 5090 Laptop GPU (24 GB), driver 615.71.09, sm_120a binaries built with CUDA 13.1;
- greedy (`temperature: 0`), one request at a time, `max_tokens: 400` unless stated, N=1 per cell
  (identity checks, not timings);
- "main" is `fcd66f8963` (`memra-server` sha256 `59c0625744a4`, `run-gen` `7bd54cea4adf`). Main
  moved to `ae69d86b05` during the work (the v0.140.0 release, version strings only), and the
  candidate sits on that: this branch at `ad8d7ade46`, before landing, `memra-server`
  `bf1960981b72`, `run-gen` `629712a20f1e`.

### The checkpoint

`Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, BF16 safetensors, 2 shards, with its
MTP head (15 `mtp.*` tensors). Main refuses the official directory at bind
(`main/gate-official-dir-refused.log`): its `vision_config.model_type` is `qwen3_5`, the config
parser reads the tower geometry only for `qwen3_5_vision`, and the 297 `model.visual.*` tensors
then refuse as undeclared. That is memra #1166, a separate defect. The runs here use a copy
without those 297 tensors, written by `tools/strip_tower.py`: 441 tensors kept, each byte-identical
to the official shard (`checkpoint/qwen35-4b-hf-text.manifest.json` holds the per-tensor sha256 of
both sides; shard 1 is byte-identical as a file). The tower is side-loaded only under
`MEMRA_VISION_DIR` and is never part of the text plan, so the served text program is the same.

## Results

### serve-st-gate (`main/gate/gate.log`, `candidate/gate/gate.log`)

| binaries | checks 1 to 3 | check 4, short prompt | check 4, window prompt | verdict |
|---|---|---|---|---|
| main | pass (64 CLI ids identical) | 211 tokens, equal, 64 speculative rounds | 400 tokens, equal, 149 rounds | 0 failed |
| candidate | pass (64 CLI ids identical) | 211 tokens, equal, 64 speculative rounds | 400 tokens, equal, 149 rounds | 0 failed |

The short prompt ends by itself at 211 tokens, so it never reached the 400-token window the check
names. The window prompt was added for that.

### Spec arm vs plain arm, text and token ids (`*/probe/compare.txt`)

`tools/probe_arm.py` boots one server per arm and sends each prompt twice: chat (text, usage) and
native completions with `chat: true` (token ids). `tools/probe_compare.py` compares the arms.

| binaries | prompt | tokens (plain / spec) | finish | text | ids | speculative rounds (drafted, accepted) |
|---|---|---|---|---|---|---|
| main | gate | 211 / 211 | stop | equal | equal | 64 (192, 146) |
| main | sched | 400 / 400 | length | equal | equal | 149 (447, 250) |
| main | essay | 400 / 400 | length | equal | equal | 156 (468, 244) |
| candidate | gate | 211 / 211 | stop | equal | equal | 64 (192, 146) |
| candidate | sched | 400 / 400 | length | equal | equal | 149 (447, 250) |
| candidate | essay | 400 / 400 | length | equal | equal | 156 (468, 244) |

The candidate's text and ids also equal main's, arm for arm, on all three prompts
(`candidate/probe/compare-vs-main.txt`).

### spec-gate exactness harness (`main/exactness*/`, `candidate-rebased/exactness/`)

`research/spec-gate-20260806/exactness.py`, Qwen3.5-9B NVFP4 GGUF with its regime draft, five
arms, one boot each, 768 tokens. Two things about the harness on main, both independent of this
change:

- Its pattern for the demote line's generated count required `)` right after the digits. The
  line now ends `generated 133, greedy)`, so the pattern matched nothing and the harness printed
  `EXACTNESS_INCONCLUSIVE` with the forced demotion in its own log. The pattern is fixed in this
  branch.
- It pins `MEMRA_SPEC_K=3`, and an operator K pin now disables the automatic demotion, so its
  loaded DEMOTE arm cannot demote. `tools/exactness_nopin.py` runs the same harness without the
  pin; there the loaded arm demotes too.

| harness | five streams | forced demotion (DEMOTE_SOLO) | loaded demotion (DEMOTE) | verdict |
|---|---|---|---|---|
| as on main (`main/exactness/`) | byte-identical, 3169 B, 768 tokens | at generated 133 in the log, not parsed | disarmed by the pin | EXACTNESS_INCONCLUSIVE |
| no pin, old pattern (`main/exactness-nopin/`) | byte-identical, 3169 B | at 133 in the log, not parsed | at 108 in the log, not parsed | EXACTNESS_INCONCLUSIVE |
| pattern fixed (`main/exactness-fixed/`) | byte-identical, 3169 B | at 133 | disarmed by the pin | EXACTNESS_OK |
| pattern fixed, rebased candidate (`candidate-rebased/exactness/`) | byte-identical, 3169 B, and equal to main's five | at 133 | disarmed by the pin | EXACTNESS_OK |

### The boot refusal

- On the built candidate binary (`candidate/boot-refusal.log`): `MEMRA_SERVE_BATCH` set to `0`,
  `1`, the empty string or `off`, and `0` beside `MEMRA_ENV_AUDIT=warn`, each print
  `[server] FATAL: MEMRA_SERVE_BATCH is retired; it cannot select a scheduler` and exit 1 before
  any device work. Main's binary under the same environment walks past the name to the next
  startup check.
- `crates/memra-server/tests/serve_batch_retired.rs` copied onto a main tree fails 2 of 3
  (`main/boot-refusal-test-red-arm.log`: `the retired name is not refused by name`; the third
  test is the non-vacuity arm). On the candidate it passes 3 of 3.

### The rebased candidate (`candidate-rebased/`)

Main moved again after the PR opened, to `8a3e4904c6` (the session-gate oracle fix of memra
#1146, the GDN packed seam removal, the verify-rows kernel). The branch was rebased onto it: two
conflicts, both where each side had added an entry to the same list (the Removed doors ledger
and the audit test's name list), both resolved by keeping both. The required checks were rerun
on binaries built from the rebased head `62cae34bc1` (`memra-server` `2ade91f9a803`, `run-gen`
`a9e7e881b014`), same card, one check per lease:

| check | result |
|---|---|
| `tools/serve-st-gate.sh <text-dir>` | 0 failed. Check 4: 211 tokens equal (64 speculative rounds), 400 tokens equal (149 rounds). The numbers of main and of the first candidate. |
| `tools/serve-smoke.sh` (Qwen3.5-9B NVFP4 GGUF and its regime draft) | 0 failed, 47 checks ok. The gemma4 arm skips: its model is not on this box. |
| `research/spec-gate-20260806/exactness.py` (pattern fixed) | `EXACTNESS_OK`. Five streams byte-identical (3169 B, 768 tokens), forced demotion at generated 133, control overshoot 0. Each of the five streams is byte-identical to main's stream of the same arm. |
| boot refusal on the built binary | `0`, `1`, the empty string, `off`, and `0` beside `MEMRA_ENV_AUDIT=warn` refuse by name, exit 1. |

Not run on a candidate: the harness without its K pin (the loaded demotion; banked on main
only) and the gate on a checkpoint with no MTP head (the named skip of check 4; its detection
was exercised against three local directories and its exit path without a GPU, not in a served
run). The card went to a landing before those two leases.

## Files

- `main/`, `candidate/`, `candidate-rebased/`: gate, probe, serve-smoke and exactness logs with
  every server log and response. Paths under the run's scratch directory are replaced by
  `<scratch>`, `<bins-main>`, `<bins-candidate>`, `<bins-candidate-rebased>`, `<repo>` and
  `<home>`.
- `checkpoint/`: the text-trunk copy's manifest.
- `tools/`: `strip_tower.py`, `probe_arm.py`, `probe_compare.py`, `exactness_nopin.py`.

## Reproduce

```
python3 research/serve-batch-removal-20261110/tools/strip_tower.py <official-dir> <text-dir> manifest.json
tools/serve-st-gate.sh <text-dir>
python3 research/serve-batch-removal-20261110/tools/probe_arm.py target/release/memra-server <text-dir> out/plain 400 MEMRA_SERVE_SPEC=0
python3 research/serve-batch-removal-20261110/tools/probe_arm.py target/release/memra-server <text-dir> out/spec 400
python3 research/serve-batch-removal-20261110/tools/probe_compare.py out/plain out/spec
python3 research/spec-gate-20260806/exactness.py
```

# cx-longdepth — Step-3.7-Flash long-generation corruption

Branch: `lane/cx-longdepth`

Opened 2026-08-09 after live serving output degraded into cross-lingual token soup at long
completion depth. Initial evidence was 262144-only; later same-prompt 131072 and 262144 receipts
resolved the original context-versus-depth uncertainty.

## Binding question — context axis resolved

**NOT CONTEXT. A 131072 context cap is not a mitigation.** Orchestrator live receipts from
2026-08-09 show the same sampled HTML task corrupting at both settings. The full JSON corrects an
important initial interpretation: 8.7K and 9.6K were total response lengths, not corruption-onset
positions. The first forbidden characters occur near the beginning of the responses (reasoning
character 261/content character 65 at 131072; reasoning character 896/content character 575 at
262144). The exact outputs and hashes are retained under `raw/orchestrator-live/`.

Owner steering therefore drops context as an experimental axis and pins `MEMRA_CTX=262144`: a
model with a 256K window must serve 256K. Controlled 2048-token results then reproduce the issue
under temperature 0.7 as early as completion token 281, while all greedy controls are clean. The
remaining questions are sampling-policy versus sampler implementation and whether truncating the
tail according to StepFun's documented defaults eliminates the contamination.

## Frozen reproduction matrix

Every cell uses the same byte-identical model artifacts under
`~/step37/models/step-3.7-flash/`, runtime revision, prompt bytes, chat template, and non-sampling
settings. The artifact's chat template is rendered once with `Reasoning: low`; a frozen assistant
continuation prefix closes the reasoning segment and supplies only `<!doctype html>` so generated
tokens exercise code rather than consume a short cell planning the impossible-length request. The
rendered prompt is submitted on the native completion surface so the response retains exact
generated token ids. Each cell has two independent requests (`N=2`) and records both runs; no
median or summary may replace the raw output.

| variable | values |
|---|---|
| `MEMRA_CTX` | `262144` (fixed; mandatory serving target) |
| requested completion depth | `2048`, `6144`, `12288` tokens |
| speculative mode | forced on (`MEMRA_SERVE_SPEC=1 MEMRA_SPEC_GATE=0`), off (`MEMRA_SERVE_SPEC=0`) |
| temperature | `0`, `0.7` |
| repetitions | `2` per cell minimum |

Total minimum: 24 generations across 12 cells.

Sampled repetitions use two different frozen seeds (`2026080901`, `2026080902`); greedy cells
carry the same seeds for uniform receipts, but the seed has no effect at temperature zero. A
forced-on cell is invalid unless its server log contains `[spec-acc]`; an off cell is invalid if
that line appears.

The fixed prompt requests one very long standalone HTML/CSS/JavaScript document, ASCII-only,
with many numbered repeated sections and no Markdown fence. It is checked into this research
directory before the first scored request. A hash of the exact request body, model shards, MTP
artifact, runtime revision, command line, and relevant environment is captured beside every run.

## Mechanical corruption detector

The detector is frozen before reading matrix results and emits one machine-readable record per
run. It operates on the assistant's code output only and reports:

- completion-token count;
- the first generated token containing a non-Latin script code point (at minimum Han, Hebrew, or
  Cyrillic), with code point, byte/character offset, token id/text, and surrounding excerpt;
- all non-ASCII code points as a secondary diagnostic, since the frozen prompt requires ASCII;
- the first interior HTML/CSS/JavaScript parse error and its generated-token index;
- whether the response ended only because of the requested token limit, so an incomplete terminal
  construct is not mislabeled as interior corruption.

The tokenizer used to map an output offset to a completion-token index must be the Step-3.7
artifact tokenizer, not an estimate from characters or another model. Parser/tool versions and the
detector source are committed with the raw results. A run is mechanically corrupt when it contains
a forbidden script code point or a parser failure before the bounded truncation tail. Detector
output never substitutes for manual excerpts: every positive finding is quoted from the raw
response.

## Execution protocol

1. Record local and remote revisions, artifact hashes, GPU/process state, disk state, and the
   current `~/.lanectl/inbox/cx-longdepth.md` contents if present.
2. Pin the fixed request and detector; validate the detector on a known-clean short response and
   synthetic injected Han/Hebrew/Cyrillic plus broken-syntax fixtures.
3. Run the 2048-token cells first, then 6144, then 12288. Interleave speculative mode and
   temperature within a depth rather than completing one configuration first, to expose
   thermal/time drift.
4. Hold `flock /tmp/memra-gpu.lock` only for one bounded boot/request block. Capture stderr with
   `tee` before parsing, stop the owned server, and release the lock between blocks.
5. Preserve raw server logs, request bodies, HTTP/SSE responses, extracted assistant text,
   detector JSON, timing, and GPU/process snapshots for every attempt, including failures.
6. Quote failure causes exactly. A dead request without captured stderr is recorded as `died,
   cause unknown — repro needed`; it is not called OOM or corruption by inference.

## Bisect order after reproduction

Only one variable changes at a time, starting from the earliest deterministic failing cell:

1. **Step35 SWA position arithmetic:** inspect the 512-token window mask and rolled-position
   handling at long absolute positions.
2. **MTP drafter geometry:** compare forced-spec versus no-spec and inspect the `swa=true`,
   `window=512` external drafter path (`mtp-draft blk.45`).
3. **RoPE position precision/range:** inspect theta/index construction at the first divergent
   absolute position and vary prompt padding to distinguish total position from generated depth.
4. **SWA KV rolling wrap:** probe indices immediately before/at/after 512-token wrap boundaries
   and compare cache contents against an unrolled/reference path.

No code fix is accepted from inspection alone. A small root cause gets a focused regression test
that fails before the change, the smallest implementation change, then reruns the isolating cell
and applicable exactness gates. If the cause is not safely small, the anatomy and falsified
suspects are the deliverable.

## Required deliverables

- `RESULTS.md`: all 12 cells, every repetition, first-corruption-token index (or `none`), parser
  result, exact excerpts, and the isolated variable.
- Raw logs and outputs under `research/longdepth-20260809/raw/`.
- Reproduction/analyzer scripts and frozen prompt/request files.
- No origin push, merge, tag, release, or runtime default change from this lane.

## Status

- [x] Matrix and evidence protocol frozen before remote generation.
- [x] Rejected pilot retained: the legacy single-turn `chat:true` completion path could not carry
  `reasoning_effort=low`; both 2048-token outputs contained reasoning only, so the detector's
  `missing_html_start` at token 0 was a harness invalidation, not model corruption.
- [x] Reproduction harness and detector frozen and self-tested.
- [x] Context control: 131072 and 262144 were both clean at 2048 greedy tokens (`N=2` per
  spec-policy label); retained as supplemental evidence, not cells in the steered matrix.
- [x] First forced-spec attempt retained as invalid: it exposed a native API receipt bug where
  round-coalesced speculative text reported `n_tokens: 2048` with only 803 token ids. Scored work
  restarts only after the terminal full-id receipt is fixed and verified.
- [x] 2048-token cells complete (`N=2`).
- [x] 6144-token cells complete (`N=2`).
- [x] 12288-token cells complete (`N=2`).
- [x] Earliest failing cell reproduced and suspect isolated: CUDA Gumbel `u01` rounded 128
  top-end `u32` values to exactly `1.0f`, producing `+inf`; both visible bad token ids match their
  Philox lanes exactly.
- [x] Minimal endpoint clamp plus exact live-receipt regression complete. Post-fix explicit
  `top_p=1` verification emitted no forbidden non-Latin codepoints across 42456 server-reported
  completion token ids, including forced-spec 12K N=2 on the first-seen RunPod rig.
- [x] Required gates complete: box1 `sample-check` and model-backed `kernel-check` ALL GREEN,
  `run-gen` argmax MATCH, `run-spec` K=1..8 self-consistency PASS; server unit tests 132/132.
- [x] Post-fix SOTA-steering cross-check complete at the remaining 12K parser positive. Exact
  teacher forcing shows the causal token 3642 is numeric-path-sensitive: default Stage-B samples
  wrong `td`, while `MEMRA_FAST=0` samples correct `dd` under the same Philox draw. The narrower
  `MEMRA_MMVQ=0` control is worse. This is a separate long-form syntax-quality delta, not the
  arbitrary foreign-token injection and not a small attention-accumulator fix.
- [x] `RESULTS.md` complete with cell table, suspect disposition, quoted failures, and raw map.

Final serving disposition: **do not lower context**. Both 131072 and 262144 live responses are
affected by the old sampler, while Step-3.7-Flash's owned serving target remains 262144. Length is
an exposure multiplier for the fixed `+inf` sampled events, not evidence of an SWA/RoPE/KV
positional threshold. Perfect long-form HTML validity is a separate numeric-quality lane: the
Stage-A oracle improves the isolated causal choice but is too broad and ungated to flip here. The
RunPod service was restored on fixed revision `585d46c4`; `/health` and `/readyz` both returned
HTTP 200.

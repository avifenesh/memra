# SFT trace generation

`tools/sft-gen.py` drives the installed `opencode run` command against the
ToS-pinned `openrouter/deepseek/deepseek-v4-flash-0731` model entry. Corpus data
is written to the separate `~/projects/sft-traces` repository; only the
generator, tests, and receipts live in memra.

## Task battery

The battery has 24 templates: bug-fix, refactor, test-writing, and explanation
tasks over six reduced Python fixtures. The fixtures are adapted from these
real memra shapes:

- cache-token accounting and revenue arithmetic
- cumulative fleet counter restart/delta handling
- speculative-acceptance log parsing
- generated perf-board marker replacement
- batched-output first-divergence evidence
- mixed expert tier-plan coverage and pruned-id rules

Each task runs in a temporary git repository. `AGENTS.md` limits the agent to
the fixture, the Python standard library, CPU execution, and no network,
`rustup`, `cargo`, CUDA, or ROCm commands. The generator passes `--pure`,
`--format json`, and `--auto` to opencode. `--auto` is bounded by the isolated
fixture and is required for non-interactive file and test tool calls.

Before any API call, the generator reads the live opencode config and requires
the requested OpenRouter model entry to retain the exact provider
order/allowlist, `allow_fallbacks=false`, and usage accounting. Config drift
fails closed instead of producing traces with stale authorization metadata.

## Run

List and scan the battery without making API calls:

```bash
python3 tools/sft-gen.py --list
python3 tools/sft-gen.py --scan-only
python3 -m unittest -v tools/test_sft_gen.py
```

Generate the full 24-trace batch:

```bash
python3 tools/sft-gen.py \
  --output ~/projects/sft-traces/corpus/deepseek-v4-flash-20260808.jsonl
```

Use `--limit`, `--category`, or repeated `--template` flags for a bounded
subset. The generator is sequential by design so errors and provider usage are
easy to audit.

## Record format

The first JSONL row is `record_type=corpus_header`. It records the generator
revision, opencode version, model/provider pin, the pilot ToS receipt hashes,
the live terms recheck, and the dedup/secret-scan contract.

Every later successful row is `record_type=trace` and includes:

- stable template id, task kind, source-shape paths, timestamp, and model
- the exact prompt and its SHA-256
- raw opencode JSON events
- the full `opencode export` session, including tool calls and tool outputs
- input/output/reasoning/cache token counts and reported USD cost
- the workspace diff plus before/after unittest receipts
- a content SHA-256 used to suppress duplicate prompt/transcript content

Failures are never converted into traces. The exact stdout, stderr, event
stream, or verification evidence is written beside the corpus as
`*-failures.jsonl`; the command exits non-zero unless `--continue-on-error` was
requested.

## Hygiene and authorization

Before any API call, every prompt, fixture, and fixture `AGENTS.md` is scanned
for private-key blocks and common API/token formats. The normalized record is
scanned again before append. A hit fails closed without printing the matched
value.

The corpus header references:

- `research/finetune-sku-20260802/REPORT.md`
- `research/finetune-sku-20260802/openrouter-tos-20260802.html`
- `research/finetune-sku-20260802/deepseek-open-platform-tos-20260802.html`

The stored OpenRouter receipt displays July 27, 2026 and already contains the
competing-service sentence. The live page displayed July 29, 2026 when checked
on August 8. The owner GO for task #56 authorizes this generation lane; the
header records that provenance without broadening the legal conclusion.

This tool generates traces only. It does not schedule training, use a GPU,
merge the lane, tag a release, or push a remote.

## Scale runner

`tools/sft-scale.py` scales the proof battery into balanced, deterministic
prompt variants while preserving the same six fixture shapes and four task
kinds. It migrates the 24 proof records into the darklanes K3 corpus layout,
then admits each new trace only after:

1. `validate_trace.py`
2. `verify_outcome.py`
3. `convert_k3_qwen.py --roundtrip`

Code tasks keep their isolated git repository alive until the verifier applies
the captured patch to the pinned fixture revision and reruns unittest.
Explanation tasks require a clean workspace/test receipt from the generator
plus a scenario-specific answer check.

Inspect the next batch without making API calls:

```bash
python3 tools/sft-scale.py --dry-run
```

Run the 1,000-trace pilot:

```bash
python3 tools/sft-scale.py \
  --target-verified 1000 \
  --batch-size 50 \
  --workers 6 \
  --budget-usd 150
```

Every completed batch is committed to `~/projects/sft-traces`, and
`research/cx-sft-20260808/PROGRESS.md` is committed on `lane/sft-corpus`.
Before each batch the runner revalidates the exact opencode provider policy,
the steering-file hash, and the three pipeline-script hashes. It stops before
new calls if steering or pipeline code changes, or if the conservative
budget-accounted total leaves less than one reserved call.

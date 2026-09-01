# Q35 mixed-c=2 completion exactness

Date: 2026-08-12

Branch: `lane/cx-q35bug`

Runtime candidate: `e953420156d9c53a693386efa7a54a56c665b094`

Gate-reporting commit: `cd1c759c8c3c7e5ec0340e8a950171b2750d158a`

## Verdict

**PASS for the scoped rejection cell.** The Qwen3.6-35B-A3B mixed-c=2 exactness failure had
two real defects, not one harness-only false positive:

1. The batched scheduler counted a token only after a row survived stop handling and completed
   the following decode. A terminal EOS/callback/context-full token was already appended and sent
   to the client, but the row retired before `tokens_out` advanced.
2. Qwen35-MoE changed numeric class as serving width moved between the generic eager B=1 trunk
   and the generic batched B>=2 trunk. The two paths produced different token ids on this workload,
   including real EOS id `248046` at completion lengths 15, 17, and 25.

The fix counts request-owned generated-token progress before survivor branching and keeps
`Arch::Qwen35Moe` on the generic batched trunk at B=1 as well as B>=2. Dense Qwen35 and other
eligible architectures retain eager B=1. The long-budget GraphSession policy is unchanged.

The committed default passed five mixed-c=2 repetitions on both native and OpenAI-compatible
wire surfaces: each surface completed 100/100 requests at 60 tokens, with
`SSE token events == response-reported count == engine tokens_out == 6000`, zero early EOS,
zero cache mismatches, and zero transport failures. The OpenAI surface's response count is its
usage block; the native surface's is the terminal `n_tokens` field. Native token ids were one
hash across all 40 serial seeds and all 100 mixed requests.

This clears the exact rejection reproduced by the scoped cell. It is not a substitute for an
orchestrator decision or a fresh full sellgate sweep; this branch was not merged, tagged, pushed,
or used to move a perf board.

## Frozen setup

- Host: eu-west pair, GPU 1 selected through `CUDA_VISIBLE_DEVICES=1`; both cards reported
  `NVIDIA RTX PRO 6000 Blackwell Server Edition`, driver `595.71.05`, 97,887 MiB each.
- GPU discipline: every model run held `/tmp/memra-gpu.lock`; provenance showed both cards idle
  and at P8 before each sealed block, and no compute application remained after the last block.
- Model:
  `Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`, SHA-256
  `df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf`.
- Frozen workload lock SHA-256:
  `85597a0a28ed874f440b4a966c0b43fd3e31b94fe868266de9e299decc208c34`.
- Prompt-id canonical JSON SHA-256:
  `eba9dff66d4ebf3f40d6db80298ab9c884fa1bb2e1a6d4ca2c7dadd4f21513fb`.
- Candidate server SHA-256:
  `dc3ddbe9a11d96dc5cf0f71425af0ae92b01d11e8bd3acf536b45bf078640370`.
- Cell shape: 20 requests per repetition, c=2, five repetitions; the frozen 90%-hit/10%-miss
  workload and eight hot cache namespaces were preserved.

The exact hashes, timestamps, idle GPU snapshots, and paths are retained in
[`post-native/provenance.txt`](raw/post-native/provenance.txt),
[`post-openai/provenance.txt`](raw/post-openai/provenance.txt), and
[`gates-final/provenance.txt`](raw/gates-final/provenance.txt).

## Root-cause evidence

### 1. The old harness did not independently count client token events

The sellgate reader assembled text chunks, but assigned its completion count directly from the
server usage object:

```python
completion_tokens = usage.get("completion_tokens")
...
completion_total = sum(int(row.get("completion_tokens") or 0) for row in target_rows)
...
if deltas["tokens_out"] != completion_total:
    integrity_failures.append("tokens_out counter does not match response usage")
```

Those are the exact statements in
[`sellgate_replay.py`](../sellgate-20260812/sellgate_replay.py). Therefore the old report's
“client total” was response usage, not an independent client observation.

The new reducer counts blank-line-delimited SSE token events independently. Its fresh pre-fix
OpenAI-wire cells quoted these totals:

| rep | SSE token events | response usage | engine `tokens_out` | early EOS |
|---:|---:|---:|---:|---:|
| 1 | 1,165 | 1,165 | 1,164 | 1 |
| 2 | 1,122 | 1,122 | 1,120 | 2 |
| 3 | 1,165 | 1,165 | 1,164 | 1 |
| 4 | 1,165 | 1,165 | 1,164 | 1 |
| 5 | 1,165 | 1,165 | 1,164 | 1 |

The raw rows say `client_response_delta: 0`, `wire_count_mismatches: 0`, and an
engine deficit exactly equal to the early-EOS request count in every cell
([`pre-openai-stable/repro.stdout`](raw/pre-openai-stable/repro.stdout)). The harness did not
manufacture the mismatch; it merely could not distinguish the two client-side surfaces.

### 2. Server accounting happened after the terminal row was discarded

At the rejected base, the plain batched path executed this order:

```rust
let (cont, next) = advance_sample_emit(&loaded, &mut active[i]);
match (cont, next) {
    (false, _) => finished.push(i),
    (true, Some(t)) => ready.push((i, t)),
    ...
}
...
n_tokens_out += 1; // only after the survivor's next batched decode succeeded
```

`advance_sample_emit` appends and streams EOS before returning `false`. That exact ordering
explains the observed one missing metric token per early-EOS row without assuming anything about
the response serializer.

The fixed path snapshots `generated.len()`, invokes the advance/step, and immediately publishes
the positive delta before matching on continuation. The same audited rule now covers ordinary
batched, eager-only, graph, graph-demotion, legacy, and speculative scheduler paths. The helper's
contract is explicit in [`worker.rs`](../../crates/memra-server/src/worker.rs): “counting only
survivors loses exactly that client-visible token.”

Regression tests cover scheduler accounting for EOS, callback, context-full, and MaxNew, plus
SSE token-event/usage equality and finish-reason mapping for all four paths.

### 3. The content also changed at the serving-width boundary

The native pre-fix serial seeds were stable for 40/40 requests at 60 tokens, hash
`5bc2ab6255e54c6183320a792f4cce1b643d019a86508f32a6514ac9df69d034`.
Mixed c=2 produced nine distinct token-id hashes. A representative 17-token receipt quotes:

```text
finish_reason="stop" native_stop_reason="Eos"
token_event_count=17 reported_completion_tokens=17
token_ids=[5821, ..., 96456, 99293, 97475, 96585, 248046]
token_ids_sha256=7aafee569e8166275259390810bcfd4c8e94f4f7feea9e726178578e34e1b164
```

The complete request is retained at line 12 of
[`pre-native-stable/repro.jsonl`](raw/pre-native-stable/repro.jsonl); the same file captures EOS
at lengths 15 and 25. Thus the early stops were generated token differences, not a usage-only
reporting error.

The decisive pre-fix control changed only `MEMRA_SERVE_B1FAST=0`. That made both B=1 and B=2 use
the generic batched trunk and passed 5/5 cells: 100/100 mixed requests plus 40/40 serial seeds had
the single hash `4f302a333a923dac6e21fd93593120305757bd8a65add8e4a629e789fcad4920`,
with 6,000 events, 6,000 response-reported tokens, and 6,000 engine tokens
([`pre-native-b1-batched/repro.stdout`](raw/pre-native-b1-batched/repro.stdout)). This isolates
the defect to the Qwen35-MoE eager-B1/batched-B>=2 transition. The unrelated StepFun `step35`
architecture was explicitly ruled out.

## Fix

- [`decode_batch.rs`](../../crates/memra-engine/src/decode_batch.rs) makes
  `Arch::Qwen35Moe` ineligible for eager B1 in both unsplit and PP-N dispatch. The gate exposes
  the architecture policy and checks dense Qwen35 and Qwen3-MoE remain eligible.
- [`decode_batch_gate.rs`](../../crates/memra-engine/src/bin/decode_batch_gate.rs) reports the
  eager comparison as inapplicable in live config mode for Qwen35-MoE, while preserving the
  bit-strength B=1-batched versus B=N isolation gate. Strict equalized mode and its canary remain
  active.
- [`worker.rs`](../../crates/memra-server/src/worker.rs) derives engine output accounting from
  each request's successful generated-token delta before row retirement.
- [`main.rs`](../../crates/memra-server/src/main.rs) verifies one SSE token event equals one usage
  token for every finish mapping.
- [`repro.py`](repro.py) records per request: SSE event count, independent token-event count,
  usage/native count, engine metric delta, finish reason, cache count, text hash, and native token
  ids/hash.

## Sealed rerun receipts

### Mixed-c=2, committed default

Both post-fix surfaces used candidate commit `e95342015` and the same candidate server hash.

| surface | repetitions | requests | events | response count | engine output | early EOS | content/cache/wire/transport mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|
| native token IDs | 5/5 clean | 100 | 6,000 | 6,000 | 6,000 | 0 | 0 |
| OpenAI-compatible SSE | 5/5 clean | 100 | 6,000 | 6,000 | 6,000 | 0 | 0 |

Every individual cell is `1200 == 1200 == 1200`, not only the aggregate. See
[`post-native/repro.stdout`](raw/post-native/repro.stdout) and
[`post-openai/repro.stdout`](raw/post-openai/repro.stdout). The native post-fix seeds and mixed
requests all use the same batched-class hash as the decisive pre-fix control.

### Local tests and builds

- `cargo test -p memra-engine --lib`: 78 passed, 0 failed, 1 GPU-only ignored.
- `cargo test -p memra-server`: 196 passed, 0 failed, 0 ignored.
- Release builds passed for `memra-server`, `run-gen`, and `decode-batch-gate` on sm_120a.
- `python3 -m py_compile research/q35bug-20260812/repro.py`: pass.
- `bash -n` on both eu-west runner scripts: pass.

### Q35 target gates on eu-west

- `run-gen`, 521-token chat-templated prompt, 32 generated tokens:
  `prefill argmax=8160`, `decode argmax=8160`, `MATCH`; batched-prime/tokenwise argmax also
  `8160`, `MATCH` ([`run-gen.log`](raw/gates-final/run-gen.log)).
- `decode-batch-gate --steps 32 --batch 2 --mode config`: live eager comparison correctly N/A;
  B=2 versus isolated batched-B=1 is bit-checked PASS; sampling/lean-logits PASS; ALL GREEN
  ([`decode-batch-config.log`](raw/gates-final/decode-batch-config.log)).
- Equalized strict gate with `MEMRA_MMVQ=0 MEMRA_NO_FUSE_NORMQ=1`: B=1 bit-identity PASS,
  B=2 isolation PASS, sampling/lean-logits PASS, ALL GREEN
  ([`decode-batch-strict.log`](raw/gates-final/decode-batch-strict.log)).

No kernel code changed, so no kernel-check rerun was required by this scoped fix. No board number
changed.

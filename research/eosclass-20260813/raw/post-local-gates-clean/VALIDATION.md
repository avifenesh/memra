# Validation — cleaned local Q27/Q35 battery

Verdict: **PASS.** `overall.exit=0`, and every one of the eleven stage exit receipts is zero. The
entire run held `/tmp/memra-5090.lock`; no box1 or target-PRO work was initiated.

## Exactness gates

- Q27 `kernel-check`: `ALL GREEN` (107 cells, 3 declared skips).
- Q35 `kernel-check`: `ALL GREEN` (113 cells, 1 declared skip).
- Default config gates: Q27 and Q35 both report B1 fast globally and effectively OFF, then
  `ALL GREEN`. This is the repaired live one-program policy.
- Explicit retired-path diagnostic: Q27 strict mode opted into eager B1 and reported
  `ALL GREEN`; it does not change the serving default.
- `run-gen`: Q27 and Q35 each reported both prefill/decode and batched-prime/tokenwise argmax
  `MATCH`, with no `MISMATCH` line.
- `run-spec`: Q27 and Q35 each passed exactly eight K=1 through K=8 self-consistency cells and
  ended with `=== SELF-CONSISTENCY PASS ===`.

## Integrated serving gate

`serve-smoke.exit=0` and `serve-smoke: 0 failed`. The gate rebuilt `memra-server`
unconditionally from the cleaned source, then passed the OpenAI-compatible chat/stream/completions
surface, greedy concurrency, long generation, cache metering, spec-vs-plain identity, sampled
truncation, affinity rewind, Gemma4, and Q35 mixed-c4 arms.

The frozen Q35 mixed cell passed 20/20 requests at exactly 60 tokens with `finish_reason=length`:
18 full 4,860-token hits and 2 cold misses, 20 admitted/completed, zero evictions, admission/VRAM
defers, step-OOM parks, golden mismatches, or integrity failures. Routed-MoE carried-prime batches
remained gated. The seed and request output SHA-256 was
`b723be26c76590659d44165c5feabc0ad705653a81df16846bbf3aa248ec7be1`.

## Provenance notes

The pre-run provenance pins all model/prompt hashes and the exact gate binaries. The smoke gate's
mandatory native rebuild produced `memra-server` SHA-256
`e63f9fad6553820a7944687dcf1a8a45326ece039f3384536964b6c560e3594f`, recorded with the unchanged
gate hashes in `post-smoke-binary-sha256.log`; that is the server binary exercised by the serving
sections. It embeds source fingerprint `memra-7d41113dd3f4`, the cleanup commit. Earlier clean
rebuild receipts produced different ELF hashes from the same code/fingerprint, so an ELF SHA is
not treated as a stable source identifier here; each executed phase retains its exact hash.

The three lines in `failure-signature-scan.log` are text-only matches: the startup gpu-watch banner
enumerates configured fatal Xid codes, the smoke section title says “worker-panic regression,” and
the passing assertion says “zero panics.” There is no observed Xid, CUDA error, OOM, panic, fatal
event, illegal address, or misaligned address. The provenance snapshot records a pre-existing
ColBERT process using 1,390 MiB; this is an exactness battery and makes no throughput claim.

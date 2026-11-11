# Row launches past the CUDA grid-y cap (memra#1200), 2026-11-11

Verdict: fixed. Every launcher that put a row count in CUDA grid y and that a long `forward_last`,
an E4B prime or a concat prime reaches now splits its rows into blocks of at most 65,535 over
sub-views. The kernels are unchanged. Rows at or below the cap are byte-identical to main.
Decision record: [`docs/decisions/GRID-Y-ROW-BLOCKS.md`](../../docs/decisions/GRID-Y-ROW-BLOCKS.md).

## Red and green

`kernel-check` with `MEMRA_KC_FAST=1` (the synthetic cells only), at the final commits. "main"
is main plus the test commit; "fix" is the fix commit.

| cell | rows | main | fix |
|---|---|---|---|
| embed-gather-rows (F32, BF16; `_t`, `_td`, `_tv`) | 1, 4,096, 65,535 | byte-equal | byte-equal |
| embed-gather-rows | 65,536 and 131,075 | 12 FAIL (`CUDA_ERROR_INVALID_VALUE`) | byte-equal to the host expansion |
| dp4a-rows (Q8_0 and Q5_K dp4a, `qmatvec_f32`) | 70,000 | 3 FAIL (`CUDA_ERROR_INVALID_VALUE`) | bit-equal to unsplit launches |
| gdn-conv-rows (prefill GDN conv) | 70,000 | FAIL (`CUDA_ERROR_INVALID_VALUE`) | bit-equal to unsplit launches |
| kv-append-rows, copy-rows-strided, moe-pairs-scatter-rows | 70,000 | 3 FAIL (`CUDA_ERROR_INVALID_VALUE`) | byte- and bit-equal to unsplit launches |
| mmvq-rows-refusal | 65,536 | FAIL (bare `CUDA_ERROR_INVALID_VALUE`) | refuses, naming `qmatvec_mmvq` |
| island-smem-refusal | 4,096 and 16,384 KV rows | FAIL at 16,384 (bare `CUDA_ERROR_INVALID_VALUE`) | runs at 4,096; refuses at 16,384, naming the kernel |

Totals: main 21 FAIL; fix ALL GREEN (115 cells). The device runs are:

- RTX 5090 Laptop GPU (sm_120a): `rtx5090/kc-red8.log` and `rtx5090/kc-fix8.log`.
- GB10 (sm_121a, native aarch64 build): `gb10/kc-red.log` and `gb10/kc-fix.log`.
- An earlier round on the RTX 5090 before the last five cells were added: `rtx5090/kc-red2.log` (16 FAIL) and `rtx5090/kc-fix4.log`.

Every 70,000-row cell compares one call with an unsplit launch over the first 65,535 rows plus
one over the rest. The conv's tail launch starts three rows early, so its window is the same as
the full launch's. The 4,096-row gather hashes are the same on both devices and both builds:
F32 `c2da9f8dba41cd8a`, BF16 `e760930d4f57e9e3`.

## Byte identity at today's row counts

On the RTX 5090, `run-gen` with `MEMRA_PP_ONLY=1` ran `forward_last` over a 3,540-token prompt
of Qwen3.5-9B Q8_0. That is two runs each of main (`red2`) and an early fix build (`fix4`), and
one each of main (`red8`) and the final fix commit (`fix8`). All six last-row logit dumps have
the same SHA-256 (`rtx5090/forward-last-logits.sha256`).

Prefill took 0.948 s on every early run. On the final pair, main took 0.9461 s and the fix
0.9464 s; each is a median of 3 reps after 1 warmup, from a single process.

A full `run-gen` (argmax gate, batched-prime gate, 64 generated tokens) printed identical output
on main and the fix, apart from timings (`rtx5090/rungen-red2.log`, `rtx5090/rungen-fix4.log`).
`fix4` is the fix before the review round: it lacks the island and windowed-attention refusals,
the KDA refusal, the MoE scatter split and the `forward_last` numeric-scope line. None of these is
on the dense-Qwen path these runs take.

## Past 65,535 tokens

- **RTX 5090, 24 GB.** `forward_last` over 66,015 tokens of Qwen3.5-9B Q8_0 runs out of memory.
  That is the smallest long-context model on the box. On main the run stops first at the grid cap
  with a bare `CUDA_ERROR_INVALID_VALUE` (`rtx5090/pponly66k-red8.log`). With the fix it gets past
  the cap and stops on memory, and the error now names the row count and the last kernel
  (`rtx5090/pponly66k-fix8.log`). The run-gen proof therefore runs on GB10.
- **GB10, concat prime.** `concat-prime-probe repro` primes one prompt of 33,119 tokens on
  Qwen3.5-9B NVFP4, first alone and then in a concat batch with a 32,896-token prompt (66,015
  summed rows). It then decodes both caches 16 steps in lockstep. Main fails at launch
  (`gb10/concat-red.log`). With the fix the batch's last-row logits equal the solo prime's
  (maxdiff 0), and all 16 steps agree with maxdiff 0 (`gb10/concat-fix.log`).
- **GB10, routed MoE.** `forward_last` over 66,015 tokens of Qwen3.6-35B-A3B (UD-IQ4_XS, resident
  experts, the pair scatter path) fails on main (`gb10/moe-forward-last-red.log`). With the fix it
  completes in 41.0 s, a single run (`gb10/moe-forward-last-fix.log`).
- **GB10, run-gen.** On main, run-gen over a 66,015-token prompt of Qwen3.5-9B NVFP4 stops at
  `forward_last` (`gb10/rungen-red.log`). With the fix, run-gen over the same prompt on
  Qwen3.6-35B-A3B completes (`gb10/rungen-moe-fix.log`, single run): `forward_last` (41 s), the
  tokenwise decode gate over all 66,015 prompt tokens (argmax 16 against 16, MATCH), the
  batched-prime gate (MATCH), and 32 generated tokens. On Qwen3.5-9B NVFP4 the fix ran 28 minutes
  without finishing, still inside the tokenwise gate, and was stopped at the 30-minute lease
  bound (`gb10/rungen-9b-timeout.log`). Nothing in that log is an error.

  **Sizing the tokenwise gate on GB10.** It decodes the whole prompt one token at a time, so it
  dominates a long run-gen. On Qwen3.6-35B-A3B it took 1,173 s for 66,015 tokens: 17.8 s per
  1,000 tokens averaged over 0 to 66k context (log timestamps: the first decode-step allocation
  at 15:43:43, the gate's verdict at 16:03:16). On Qwen3.5-9B NVFP4 it needs more than 25 s per
  1,000 tokens, a lower bound from the stopped run. Budget at least 30 minutes for a 9B run past
  65,535 tokens on one GB10.

## Reproduce

```
MEMRA_KC_FAST=1 kernel-check
MEMRA_PROMPT_FILE=<prompt> MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 MEMRA_PP_WARMUP=0 MEMRA_PP_LOGITS=<out.f32> run-gen <model.gguf>
MEMRA_PROMPT_FILE=<prompt> MEMRA_NGEN=32 run-gen <model.gguf>
concat-prime-probe <model.gguf> repro --prompt-a @<a> --prompt-b @<b> --steps 16
```

The prompts are prefixes of this repository's `docs/*.md`, `README.md` and `ARCHITECTURE.md`
concatenated in that order (14,000, 220,000 and 280,000 bytes; the concat pair is bytes 0..110,000
and 110,000..220,000). Logs have home paths shortened and the printed prompt replaced by its
length.

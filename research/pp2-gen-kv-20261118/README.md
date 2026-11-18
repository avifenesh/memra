# pp2-gen-kv-20261118: `run-gen` generation on a PP-2 pair, and the prefill gate's near tie (memra#1295)

Verdict: fixed, with the pair evidence partly pending. On 2x RTX PRO 6000 Blackwell Max-Q (PP-2,
Step-3.7-Flash IQ4_XS GGUF, no MTP head) main's `run-gen` fails after its gates with
`PP cross-device tensor access at append_kv_quantized_rows.k_cache: tensor on dev0, stream on dev1`,
and its prefill gate panics on a flip it has labeled a near tie. With the fix the 8,200-token prompt
generates its 16 greedy tokens, equal to the 16 tokens `run-spec` banked for the same prompt under
memra#1275. The near-tie rows are proven on CPU; their runs on the pair, and a rerun of the pair cells
on the final commit, are listed under "Pending".

## Defect 1: the generation cache was homed on the primary

`HybridModel::generate_with` (and `generate`) allocated `Cache::new(e, ..)`: every layer's KV on the
caller's engine. With the ppN door open across devices the prime is stage-split
(`prime_chunk`'s ppN arm, the step35 batched prime): stage 1 walks its layers on its own engine and
appends their KV there, and the cross-device guard in `append_kv_quantized_rows` refuses the append.
Every other caller, `run-gen`'s own gates and the served path included, allocates through
`pp::new_cache`, which gives each stage its cache.

It does not depend on the prompt or on the passes before it:

| run (main 5a6ae2438e) | prompt | result |
|---|---|---|
| `base-24` | 24 ids | gates `MATCH`, then the cross-device error, exit 1, in 27 s |
| `base-24-nogate` | 24 ids, `MEMRA_PRIME_GATE=0` | the same error without the batched-prime pass, exit 1 |
| `base-8` | 8 ids (under `PRIME_MIN_T`, tokenwise prime) | exit 0, 16 tokens: the eager step has no such guard and reaches the home cache through peer access |

The fix: `generate` and `generate_with` take `pp::new_cache` when the model's prime is stage-split
(`HybridModel::prime_is_stage_split`, which restates the split walks' conditions). A prime that walks
the whole trunk on the caller's engine (the gemma program, the serial residual under
`MEMRA_PRIME_PP=0`) keeps the home cache it had. A stage-owned cache is decoded by the stage-split
host-logits loop: the qwen device-counter and graph doors, which walk the whole trunk on the caller's
stream, stay shut for it.

| run (9a19b18b77, the fix's first commit) | prompt | result |
|---|---|---|
| `head-8` | 8 ids | exit 0, the same 16 tokens as `base-8` |
| `head-24` | 24 ids | exit 0, gates `MATCH`, 16 tokens, the first one the prime's argmax (5466) |
| `head-8200` | 8,200 ids | exit 0, gates `MATCH` (argmax 2745), 16 tokens `[2745, 528, 855, 1309, 65, 12984, 1955, 269, 4001, 630, 65, 12984, 680, 269, 16, 7584]` |

The 8,200-token tokens equal `run-spec`'s plain greedy tokens for this prompt in
`research/step-gemm-tail-20261113/pro6000-pp2/run-084758/8200-*-gen.log`.

## Defect 2: the prefill gate panicked on a near tie

`run-gen`'s first gate compares the batched prefill (`forward_last`) with the tokenwise decode loop
at the last position and asserted equal argmax ids. On a flip it printed a label first:
`NEAR-TIE class` when the two programs' spread at the two ids was larger than the smaller of their
margins. That holds for every flip (with prefill picking i and decode picking j, the differences
p_i - d_i and d_j - p_j sum to the two margins' sum), so the label carried no information, and the
assert failed the run either way:

| run (main 5a6ae2438e) | prompt | margins (prefill, decode) | result |
|---|---|---|---|
| `base-64` | 64 ids | 0.4441, 0.0034 | `NEAR-TIE class`, then the assert, exit 101 |
| `base-4100` | 4,100 ids | 0.0382, 0.1173 | `NEAR-TIE class`, then the assert, exit 101 (the issue's row) |

The fix: `forward::prefill_gate_verdict` classes the flip on the batched-prime gate's bounds
(`MEMRA_PRIME_GATE_MARGIN`, `MEMRA_PRIME_GATE_MAXDIFF`). A flip where each program holds its token
over the other's by at most the margin bound, under drift within the maxdiff bound, prints
`FLIP-NEARTIE` and the run goes on. Any other flip prints `MISMATCH-STRUCTURED` and fails the run
(exit 1), a flip on a row with a NaN included; so does a row with no finite value, which has no
argmax. An agreeing row passes whatever its maxdiff, as before. The rule bounds both programs'
margins between the two contending ids, which
is stricter than the batched-prime gate's tokenwise top-2 margin: a prefill that holds its token by 7
logits over a decode that has the two tied is not a near tie.

CPU proof (`cargo test --release -p memra-engine --lib prefill_gate_tests`, 8 tests, no card visible),
with its red arms: the same tests against four wrong rules, each patched into
`prefill_gate_verdict_with` and reverted.

| rule in `prefill_gate_verdict_with` | tests |
|---|---|
| the fix | 8 of 8 pass |
| every flip fails (main's assert) | 2 fail: the near-tie row, the inclusive bounds |
| every flip is a near tie (main's label as the rule) | 6 fail: every row that must fail the run |
| only the decode margin is bounded (the batched-prime gate's rule) | 2 fail: wide in one program, wide between the contending ids |
| a maxdiff that drops NaN | 1 fails: the broken row |

## Conditions

- 2x RTX PRO 6000 Blackwell Max-Q, `MEMRA_PP_STAGES=2 MEMRA_PP_DEVICES=0,1 MEMRA_PARALLEL=off
  NVIDIA_TF32_OVERRIDE=0 MEMRA_NGEN=16`, sm_120a builds, 2026-11-18. Single run per cell, untimed: the
  cards were shared with other jobs, and no speed is read from these logs.
- Prompts: `BOS` plus the first N-1 tokens of `research/e2e/prompts/p4-16k.txt` under the Step-3.7
  tokenizer, passed as ids (N = 8, 24, 64, 4,100, 8,200).
- Binaries: base `run-gen` sha256 `f7a0a673ff3ab4141301beb0c2af24ffb752f052c57607e11f0457b0bd0ec98e`
  (main 5a6ae2438e), head `d54e9564e6032883d24b89cd59e79104c42e80bf54c4f40ce4700b0726876f1c`
  (9a19b18b77).
- `logs/<run>.log`: stdout and stderr of each `run-gen`, as it ran, with its exit status appended.

## Pending on a pair

The head runs above used the fix's first commit (9a19b18b77). The final commit differs from it in
which calls take the stage-owned cache (only a stage-split prime) and in the gate's rule (both
margins, rows with no finite value); on this model and placement the cache path is the same. Still
owed on the final commit, same pair and settings:

- 64 and 4,100 ids: the gate prints `FLIP-NEARTIE` and the run generates 16 tokens (4,100: expected
  `[16, 27669, 223, 18, ..]`, the tokens `run-spec` banked under memra#1275);
- 8,200, 24 and 8 ids: 16 tokens, equal to the ones above.

Two head runs were stopped by the box's schedule before they produced a result (8,200 ids after 80 s,
4,100 ids after 70 s); their logs are not kept.

## Not covered

- The gemma program on a pair, and the serial residual under `MEMRA_PRIME_PP=0`, keep their home
  cache, so `generate` and `generate_with` behave as on main there. By the code, `run-gen`'s
  batched-prime gate allocates a stage-owned cache for that whole-trunk walk and meets the same guard
  from the other side. Not run: no gemma checkpoint was on the box.
- The opt-in step35 t-row prime (`MEMRA_PRIME_TROWS=1` with `MEMRA_STEP_GEMM_PRIME=0`) walks the whole
  trunk on the caller's engine while the predicate calls the prime split. Not traced, not run.
- Two placements change decode loop with the same tokens expected: a placement that names one device
  twice, and `MEMRA_PP_SHARD=0` on a pair with a prompt under 16 tokens. Both ran the device-counter
  loop over a home cache on main and now run the stage-split loop over a stage-owned one.
- The directory-checkpoint arm of `run-gen` has its own prefill gate, which still fails on any flip.
- The batched-prime gate's own maxdiff (`prime_gate_verdict`) still drops a NaN.

# memra#641 decode-exact: pre-registration (2026-09-23, lane E)

Written and pushed before the first boot. Nothing below changes after a result is seen. A later
change of plan gets its own dated section and does not rewrite this one.

## The red

`research/prime-fairness-default-20260922/raw/gate-5090-9b-run1/` is the only divergent run. Its
non-yielding arm (`rep0-yield0`) gave peer-cold-b the bytes
`"s \t</div>\n</body>\n</html>\n```\n\n## File: src/test/resources/10000000"` (sha16
`85dbe38937836437`). Every other observation of the same prompt gave
`"s \t</div>\n</div>\n</div>\n</div>\n</div>\n</div>\n</div>\n</"` (sha16 `06bfb5126effdd4c`, called R
below): the yielding arm of run1, all later gate runs (these pin the route with
`MEMRA_SPEC_GATE_LOW=64 HIGH=65`, so their peers take the spec route and never enter a batched
prime), and all four probes in `raw/spec-vs-plain-probe/` (solo spec, solo plain, four plain copies).

The divergent run's server log shows what only it had. Peer-b was demoted to plain decode
(`K=0 source=concurrency`, wave 4) and primed in two concat batches with peer-a and the 4,096-id
peer: `[prime-batch] B=3 tokens=3072 carried=0 partial=3` (fresh, rows 0..1024 of each), then 501
ticks later `[prime-batch] B=3 tokens=3072 carried=3 partial=1` (rows 1024..2048). Peer-b then
decoded B=1 for three steps while the 4,096-id peer finished its prime in solo 1,024-row ticks, and
joined a B=2 wave from step 4. The probe `server-plain-wave4.log` batch was `carried=1`.

Code reading before any run (hypotheses, not findings):

- (a) the prime. A fresh batch of 2..=8 prompts takes the varlen FA path (`use_favl`,
  `fa_prefill_vl8`: `fa_mirror_vl` rounds the f32 K/V to bf16, then `fa_prefill_bf16kv_vl`). A solo
  prime and every carried chunk attend through the quantized cache view (`fa_prefill_view_ws`:
  q8_0 K / q5_1 V dequantized to bf16, then `fa_prefill_qw_db`). Those are two numeric programs for
  chunk 0 of the same prompt (twoprog W3, `research/twoprog-20260813/`). Secondary: the concat GEMMs
  run at m=3,072, and the batched first-token lm_head is `try_f16_gemm` against the solo m=1
  `matmul` (twoprog W4).
- (b) the B=2 mixed-context decode wave. The contract in `decode_batch.rs` says B=2..=8 rows are
  per-row identical to m=1, and seqs FA falls back per sequence when the rows disagree on
  `fa_split_keys`. `decode-batch-gate` covers it. Tested anyway.

## Artifacts

- Model: `/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`, sha256
  `52c9cceb190055e0591a9a30c21f7200572eaf3ff1c59f6e9a1eda838a8f39de`.
- Server: `target/release/memra-server` built in this worktree from origin/main `9c07b398b`, sha256
  `c0c573080301ed9486b3b3205a7850aeef3d6604cb915a7a0aa36c798c9c5259`, nvcc 13.1.
- Engine probe: `concat-prime-probe` built from this branch with the new `tickshape` mode. That
  commit changes only the probe binary, so the engine library program equals `9c07b398b`. Its
  sha256 goes into `raw/` with the first engine run.
- Prompts: `ids_for(n, seed)` = `random.Random(seed)`, `randrange(1000, 100000)` n times, as at
  `5a9fd0414`. A = `ids_for(2048, 5211)` (peer-cold-a), B = `ids_for(2048, 5212)` (peer-cold-b,
  equal to `raw/spec-vs-plain-probe/prompt.json`), C = `ids_for(4096, 5210)` (seed and peer-hit),
  long = `ids_for(131072, 521)`.
- Card: the local RTX 5090 Laptop only, under `flock /tmp/memra-5090.lock`, compute-apps empty
  before each run.

## Cell 1: engine replay (`concat-prime-probe tickshape`)

One process, `--tick 1024 --steps 32 --join 4`, prompts A, B, C. Arms, all teacher-forced on the
`ref` greedy tokens so every step compares logits for the same input:

| arm | program for B |
|---|---|
| ref | `prime_cache(B)` in one call, then `decode_step_batch` with one row, 32 steps |
| ref2 | ref again, in the same process: the determinism pin |
| tick | `prime_cache` on B[0..1024], then B[1024..2048], then B=1 decode |
| bp | `prime_cache_batch([A0, B0, C0])` (fresh), then `([A1, B1, C1])` (carried), then B=1 decode |
| bps | bp, then C's rows 2048..4096 in two solo 1,024-row `prime_cache` calls placed in the ticks of B's steps 3 and 4, B=1 for steps 1..3, then the B=2 wave [B, C] from step 4 (C fed its own argmax) |
| wave | ref's prime, C primed in one call, B=1 for steps 1..3, then the B=2 wave [B, C] |

Compared quantities, each bitwise against ref: the prime's first-token logits, `h_seed`, every
hidden row (first differing row named), a per-layer FNV-1a digest of the cache (K and V bytes up to
`len`, GDN conv and SSM state), then the per-step decode logits (first bit-different step, first
argmax flip with both margins).

Runs: N=1 with the default environment (the served program). Diagnostic only, never part of the
verdict: one run with `MEMRA_FA_VL=0` (the favl door off) to test whether the fresh-batch attention
is the whole trunk difference.

Rule: ref2 must be EXACT, or the cell is void (the program is non-deterministic in-process and that
is the finding). With ref2 EXACT, any of tick, bp, bps, wave that is not EXACT is a one-program-law
violation at that arm's shape, whether or not the greedy text flips inside 32 steps.

## Cell 2: serving repro (`repro641.py`)

The prime-fairness cell exactly as at `5a9fd0414`, no spec-gate pin (placement default LOW=2
HIGH=4). Env: every `MEMRA_*` stripped, then `MEMRA_COMPAT=openai`, `MEMRA_MODELS=gate=<model>`,
`MEMRA_ADDR=127.0.0.1:18641`, `MEMRA_MAX_SESSIONS=4`, `MEMRA_TIMEOUT_MS_MAX=600000`,
`MEMRA_TICK_TRACE=1`, `MEMRA_PRIME_YIELD=0|1`. Schedule: the seed request (C, salt `seed`)
completes; then at t=0 long (salt `long`); +2.0 s peer-cold-a (A, salt `peer-a`); +3.0 s peer-cold-b
(B, salt `peer-b`); +3.0 s peer-hit (C, salt `seed`). Each request: `prompt_ids`, `max_tokens` 32,
temperature 0, stream, `timeout_ms` 600000, `/v1/completions`.

Runs, one fresh boot each, in this order: C0 (control: peer-cold-b alone on a fresh server, yield
default), then Y0 Y1 Y0 Y1 Y0 Y1 Y0 Y0. N = 5 Y0 and 3 Y1.

On-shape (Y0 only): the log shows peer-b's `[spec-k]` line with `K=0`, exactly one
`[prime-batch] B=3 tokens=3072 carried=0 partial=3` and exactly one
`[prime-batch] B=3 tokens=3072 carried=3 partial=1`. An off-shape Y0 is kept in `raw/` and not
counted; up to 3 replacement Y0 boots are appended at the end. Fewer than 5 on-shape Y0 after
replacements is reported as "shape not reached N times", with the count.

Rule, fixed now:

- REPRODUCES if any on-shape Y0 peer-cold-b text differs from R (sha16 `06bfb5126effdd4c`).
- DOES NOT REPRODUCE if all 5 on-shape Y0 peer-cold-b texts equal R.
- C0 and every Y1 must equal R. A control that differs is reported as a control failure, and the
  Y0 verdict is then stated beside it, not instead of it.

Cell 1 carries the law verdict; cell 2 carries whether the served bytes move. A cell 2 "does not
reproduce" does not clear a cell 1 bitwise difference: a difference in logits at the serving shape
is the red, and the text flip in run1 is one draw of it.

## If it reproduces or cell 1 differs

Bisect with the probe arms first (prime versus wave), then inside the prime by door
(`MEMRA_FA_VL=0`, then the lm_head path), then by layer from the first differing hidden row and
digest. Fix per the law: make the crossing impossible or make the two programs bit-identical. The
fix lands with a serving-shape gate cell that is red on `9c07b398b` and green on the fix, both
receipts kept.

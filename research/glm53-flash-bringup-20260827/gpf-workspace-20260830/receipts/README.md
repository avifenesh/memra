# gpf-workspace gate receipts (rig 5090, flock /tmp/memra-5090.lock, NVIDIA_TF32_OVERRIDE=0)

Binary/tree SHA for every run: `BINARY-SHA` (the lane head at run time; the ppN gate
binary was rebuilt at that SHA before the batch — the rebuild-after-checkout law).
Rig law: exactness only; no timing number is read out of any of these logs.

## GREEN

| log | what | verdict |
|---|---|---|
| `chunked-prime-gpu.log` | `glm5_chunked_prime_gpu` — single-engine chunk walk vs `memra_reference` (truth) + monolithic sibling band (2e-5), depths 64/200/501 x chunks 16/32/37/128, decode continuation, teeth control | 4/4 GPU (2 CPU arms filtered by `--ignored`) |
| `moe-grouped-prefill-gpu.log` | `glm5_moe_grouped_prefill_gpu` — the grouped-vs-sequential bit-gate re-run: routing exactness, reference band, bitwise knee, `wrong_programs_fail_the_gate` red arms | 8/8 |
| `hyper-connections-gpu.log` | `hyper_connections_gpu` — the UNSPLIT hc walk vs `memra_reference` (the truth half of the ppN composition chain) | 6/6 |
| `prime-capacity.log` | `glm5_prime_capacity` — transient sub-quadratic in context + expert-residency headroom assertions | 5/5 |
| `ppn-10-n2-baseline.log` | ppN gate, untouched shape (N=2 P=6, one prime call) — regression control for the gate extension | 3 arms PASS, BIT-IDENTICAL |
| `ppn-30-n2-chunk8-p24.log` | ppN gate, CHUNKED ppN prime (MEMRA_PRIME_CHUNK=8, P=24 -> 2 calls, tail-merged) vs chunked unsplit prime, hiddens stack included | 3 arms PASS, BIT-IDENTICAL |
| `ppn-31-n3-chunk8-p24.log` | same, N=3 stages | 3 arms PASS, BIT-IDENTICAL |
| `ppn-32-n2-chunk32-p200.log` | CHUNKED ppN prime at depth (chunk=32, P=200 -> 6 calls) | 3 arms PASS, BIT-IDENTICAL |
| `ppn-33-n2-mono-control-p200.log` | monolithic control at the same depth (no chunk env -> 1 call) | 3 arms PASS, BIT-IDENTICAL |

Non-vacuity: every chunked log prints `prime schedule: N call(s)` with N > 1 and the
gate hard-asserts the split when `MEMRA_PRIME_CHUNK < P`. The prime-twin arm's 10
comparisons = prime last row + prime hiddens stack + 8 decode continuations.

The truth chain for the chunked ppN prime closes by composition, both halves above:
`hyper_connections_gpu` + `glm5_chunked_prime_gpu` anchor the unsplit/chunked walks to
`memra_reference`; the ppN arms anchor the split walk to the unsplit walk bit-for-bit
over the SAME schedule.

## RED (mutation-bound, applied -> banked -> reverted)

| log | mutation | expected | got |
|---|---|---|---|
| `ppn-90-RED-m5-hiddens-offset.log` | M5: every chunk's hiddens copied to offset 0 (`start * n_embd` -> `0`) | prime-twin FAIL on the NEW hiddens-stack compare, everything else green (the mutation is invisible to logits/decode — the reason the compare exists) | FAIL [prime-twin] 1/10, `prime hiddens stack` 3072/3072 differ; decode-serial + prefill-twin PASS; exit=1 |
| `ppn-91-RED-m6-dropped-chunk.log` | M6: the first chunk never primes (`ranges.iter().skip(1)`) | prime-twin FAIL (missing cache rows move the prime logits and the decode continuation) | FAIL [prime-twin] 10/10 mismatched (prime row, hiddens, all 8 decode steps); decode-serial + prefill-twin PASS; exit=1 |

## CPU gates (run in ordinary CI, not behind the lock)

* `glm5_admission_cost` (memra-engine): latent-plan coefficient formula on the mini
  glm5 plan, per-stage split contract, chunk-bounded charge identities — 3/3.
* `hyper_prefill_workspace_makes_admission_see_the_262k_wall` (memra-server worker
  tests): the 262k cell's own rungs (262k refused, 7,108 admitted, vendor-default
  541-token prompt on a 262k window admitted), red arm = the pre-lane 0 B/token model
  admitting 262k — part of the 473-test worker suite, all green.

## Remaining rungs (named, box work — not runnable on the rig)

1. 2-card recipe box (the 262k cell's own shape): depth ladder 8k/16k/45k/130k through
   the serving surface — over-budget rungs must 429 BEFORE prime (no `[engine-error]`
   line), in-budget rungs serve; vendor-default sampled probe per the serving law.
2. 3-card 1M arm: re-run the 1M prime with the chunked ppN walk + admission; the
   prediction (LANE.md §4) is a clean refusal at admission, not the 97.2 GiB mid-prime
   OOM.

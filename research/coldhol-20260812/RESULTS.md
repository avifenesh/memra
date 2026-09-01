# Q27 cold-prefill head-of-line batching

Date: 2026-08-12

## Verdict

The continuation-capable cross-request prime batch is a **winner under the
frozen Box1 protocol**. The N=5 first-decline clean-throughput knee moves from
**c=12 to c=16** on one RTX PRO 6000 Blackwell Server Edition. The candidate
keeps c<=12 throughput: paired median deltas are +1.65% at c8 and +0.03% at
c12. At c16 it is +2.25%, with all five paired rounds positive.

The result is an aggregate N=5 verdict, not five unanimous single-boot knees.
All five baseline boots stop at c12; candidate boots stop at c16 in rounds 1,
3, and 4 and at c12 in rounds 2 and 5. The candidate's N=5 c12 -> c16 rise is
small (+0.34%) but positive under the frozen definition; the first decline is
c16 -> c20 (-0.95%).

The winning behavior is the naked default. `MEMRA_PRIME_BATCH=1` is the
documented rollback seam and restores serialized per-session cold primes. No
board was moved by this lane.

## Frozen result

All throughput values are medians of five whole-server observations. Each
round is paired by global repetition; odd rounds run baseline then candidate,
even rounds reverse the order. The ten boots ran under one uninterrupted GPU
lock hold.

| concurrency | baseline output tok/s | candidate output tok/s | paired candidate delta | first-decline status |
|---:|---:|---:|---:|---|
| 8  | 174.900 | 178.158 | +1.645% | rise in both arms |
| 12 | **186.656** | 186.171 | +0.033% | baseline knee |
| 16 | 182.526 | **186.797** | **+2.252%** | candidate knee |
| 20 | 181.185 | 185.021 | +2.164% | first candidate decline |
| 24 | 186.046 | 189.389 | +1.878% | rebound; excluded by first-decline rule |

The baseline path rises 6.72% from c8 to c12 and falls 2.21% at c16. The
candidate rises 4.50% to c12, another 0.34% to c16, then falls 0.95% at c20.
The c24 rebound remains real, but does not rewrite the frozen knee definition.

## Cache-hit TTFT

The population cache-hit TTFT p95 is held at c8 and improves from c12 onward.
These pooled values include every hit request across all five repetitions
(N=90 at c8-c20, N=135 at c24):

| concurrency | baseline hit TTFT p95 | candidate hit TTFT p95 | delta |
|---:|---:|---:|---:|
| 8  | 588.66 ms | 590.48 ms | +0.31% |
| 12 | 609.10 ms | 559.91 ms | -8.08% |
| 16 | 610.09 ms | 578.01 ms | -5.26% |
| 20 | 608.54 ms | 580.40 ms | -4.62% |
| 24 | 920.42 ms | 595.01 ms | -35.35% |

For completeness, the median of the five per-cell hit-p95 values is noisier
because each c8 cell contains only 18 hits, making nearest-rank p95 that cell's
maximum. That view moves from 284.95 to 308.24 ms at c8 (paired median +4.52%)
against a 25.6-590.2 ms baseline range; at c12 and c16 it improves 36.83% and
5.55% by paired median. The pooled p95 above is the stable request-population
answer used for the held verdict; both views are retained in `analysis.json`.

Cold-miss TTFT also improves at the moved knee: the median c16 miss-TTFT p50 is
3,023.90 -> 2,789.08 ms, with a paired median delta of -5.72%.

## Mechanism receipt

The old binary produced zero cross-request prime-batch log lines for this
4,860-token workload. The candidate produced **90** batch calls across its five
boots:

- 81 calls left at least one request partial for a later scheduler tick;
- 79 calls carried an already-started request;
- batch widths were B=2 or B=3;
- median payload was 2,048 tokens and median call time was 544.9 ms;
- zero batch calls failed.

Every candidate boot contains partial-batch evidence (15, 20, 17, 12, and 17
calls by round). A separate c16 smoke captured the intended five-call sequence:
four partial B=2 chunks followed by the final B=2 tail, with 20/20 requests and
zero accounting drift. This confirms the performance result exercised the new
scheduler path rather than merely linking a different binary.

The scheduler admits at most one bounded chunk per eligible cold session per
tick, executes those chunks through the existing metadata-preserving
`prime_cache_batch`, and returns to decode with any remainder still queued.
Complete fresh primes, Step35, eager-only models, checkpoint/prefix boundaries,
and the existing fallback remain on their prior paths.

## Exactness and integrity

The required model-backed battery passed before and after the scheduler edit
on Box1 physical GPU0:

| gate | baseline | candidate |
|---|---|---|
| `kernel-check` | ALL GREEN (95 cells, 13 skipped) | ALL GREEN (95 cells, 13 skipped) |
| `run-gen` | prefill/decode argmax 8160 MATCH; batched/tokenwise argmax 8160 MATCH | same two MATCH checks |
| `run-spec` | self-consistency PASS at every K=1..8 | self-consistency PASS at every K=1..8 |
| carried prime batch | unchanged gate binary | B=2 uneven fresh + carried suffix argmax MATCH; ALL GREEN |

Only `memra-server` differs in the frozen binary pair: baseline SHA-256
`b5e31c8db47f2d5f04a2ffb8729c921fd4b68cb6f090819b8234eb0996385ef3`,
candidate SHA-256
`f00f1bd5d08fbf0476a540e497b51d749d813873c4b885a67fc5fce120120748`.
The four gate binaries are byte-identical across arms.

All **50/50 scored cells and 1,100/1,100 requests** passed the frozen workload
integrity checks. Each arm reconciled 33,000 output tokens, 2,405,700 cached
tokens, and 267,300 computed prompt tokens. Each arm recorded 495 hits and 55
misses, with zero cached-token, prompt-token, and prefix-hit-token drift; zero
session/VRAM admission defers; zero OOM parks; and zero required golden
failures.

Concurrent response hashes are retained as an observation, not substituted
for the repository's argmax/spec exactness gates. Across aligned A/B requests,
488/550 text hashes match and 62 differ (40 hits, 22 misses). This workload
already admits batch-composition-dependent near ties at c>1 and does not make
those text hashes a required golden. The explicit model-backed argmax,
carried-prime, and K=1..8 gates above are green.

## Sellable concurrency and $/day

At the stated $0.287/M input-token and $2.751/M output-token prices, assuming
the frozen 4,860-input/60-output mix is continuously saturated and billed at
list price:

- baseline c12 knee: 186.656 output tok/s, 3.1109 requests/s, **$419.27 gross
  usage/day/card**;
- candidate c16 knee: 186.797 output tok/s, 3.1133 requests/s, **$419.59 gross
  usage/day/card**;
- arithmetic knee-operating-point delta: **+$0.32/day/card** and **+4
  concurrent requests**.

This lane's economic win is the c12 -> c16 sellable-concurrency/headroom step,
not a large aggregate token-rate increase. At the previously overloaded c16
point specifically, the old and new medians are 182.526 and 186.797 output
tok/s; that same-width difference corresponds to about **+$9.59 gross
usage/day/card** on the frozen mix. Both figures are list-price gross usage
math, not demand, margin, or a sold-cap recommendation.

## Protocol, provenance, and thermal regime

- Baseline runtime source:
  `d2fba620031920032b253b700443af5ef1ec7866`.
- Candidate runtime source:
  `b37d77c6f6403d8b3b87099470fc3b5c2cd62cee`.
- Frozen harness source:
  `ca80b88dbe7cc74e8c3c5d31355e6bc23a500050`.
- Model: `Qwen3.6-27B-NVFP4-Q4_K_M-mtp.gguf`, SHA-256
  `d8d71c7e8a01a1c964fd904a7b496eaef19bdd66827e0949e66c723da742d517`.
- Workload lock SHA-256:
  `85597a0a28ed874f440b4a966c0b43fd3e31b94fe868266de9e299decc208c34`;
  canonical prompt-id SHA-256:
  `eba9dff66d4ebf3f40d6db80298ab9c884fa1bb2e1a6d4ca2c7dadd4f21513fb`.
- Shape: physical GPU0 only; spec off; 4,096 MB prefix cache; prefix dedup on;
  reuse pool and affinity off; max sessions 96; default decode cap and prefill
  tick; 60 output tokens; 90% full-prefix hits. Both scored arms leave
  `MEMRA_PRIME_BATCH` unset.
- One lock hold: 2026-08-12 14:23:58Z-14:42:06Z. No artificial cooldown or
  clock changes. Baseline sampled 27-59 C and candidate 34-60 C; both reached
  2,422 MHz and 100% sampled GPU-busy. The wrapper's before/after receipts and
  a live post-run `nvidia-smi` check show both cards at 0 MiB with no compute
  processes after the campaign.

## Receipt index

- Machine-readable reduction: `analysis.json`, generated by `analyze.py`.
- Design committed before code: `DESIGN.md`.
- Baseline exactness: `raw/before/`; portable manifest-file SHA-256
  `f0529529348f2af1e846cc1c4fa12bceadceeb320ffca502c8aae6c980d35c18`.
- Candidate exactness: `raw/after/`; portable manifest-file SHA-256
  `66f0f8fc573000fa8806947c3584824df363e8ba2c26b7953b1d3a1b15af6159`.
- Real-server c16 smoke: `raw/smoke-c16-candidate/`.
- Interleaved N=5 binary A/B: `raw/ab-coldchunks-n5/`; portable outer
  manifest-file SHA-256
  `6a9f50ae6969dd855fcb2641dd19198e410ffa6087c348ac5ff130bd595e70f4`.

No run used `nsys`; no origin push, merge, tag, generated perf-board update,
formatting sweep, rustup, other worktree, or verifier bypass was performed.

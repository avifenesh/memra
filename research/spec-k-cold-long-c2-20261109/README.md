# Cold long prompts at K=2 against K=3, two requests at once, on the RTX 5090 (memra#1093, 2026-11-09)

Verdict: K=2 keeps its lead at concurrency 2. Per request it decodes +5.7% (3.6k tokens), +8.5%
(5.8k) and +7.2% (1.2k) faster at T=1 and +12.2% greedy; the pair's aggregate rate is +4.0% to
+5.5% at T=1 and +8.7% greedy. TTFT is unchanged and the greedy texts are byte-equal across arms.
No class loses. This was the open condition on the cold-long flip
([`../spec-k-cold-long-20261109/`](../spec-k-cold-long-20261109/) measured one request at a time).

Model and rig: Qwen3.8-27B NVFP4 + Q5_K GGUF with its untrimmed MTP head, no DFlash drafter, RTX
5090 Laptop, context 8,192, two sessions, every switch at the device default. Binary: main
1273a16f07, the cell-evidence build (sha256 96ad59c39c4afdfb), the one the single-stream A/B ran.

## Why concurrency 2

On a single card the spec gate serves speculation at one and two live sessions and demotes from
three (`LOW=2`). At two, both sessions' rounds pool one verify forward of 2 x (K+1) rows: 8 rows at
K=3, 6 at K=2. The single-stream lead did not have to carry over.

## The A/B

`c2_ab.py` through `c2_driver.sh`:
- **Arms**, one binary: `k3`, the device default, and `k2`, `MEMRA_SPEC_K_COLD_LONG=2`.
- **Boots:** one boot per gpulock lease, k3 k2 k2 k3 three times. Boots 1 to 8 ran 20:54 to 21:20.
  A session restart then killed the driver seconds into boot 9, before its first pair finished
  (`run/killed-boot9/`, no row written). `c2_resume.sh` ran boots 9 to 12 at 21:28 to 21:37 on the
  same binary, the card in the same throttled regime (below). Other sessions' leases sat between
  some boots (boot 6 waited 8 minutes).
- **Pairs:** every boot serves six cold pairs, the two requests of a pair sent at once, pairs one
  after another: two T=1 pairs of about 3,600 tokens, one of about 5,800, one of about 1,240, one
  greedy pair of about 3,600 and one T=1 pair of 34 tokens. The short pair sits on the cold-short
  rung, K=3 on both arms: the boot's clock reference. Answers are capped at 400 tokens, thinking
  off, streamed. The second request of every pair admits at `active=2` (`[spec-k]` lines) and the
  two decode as one pooled cohort (`[spec-pool] pooled MTP cohort engaged: 2 sessions`).
- **Per request:** TTFT, decode rate after the first chunk, ITL, acceptance, sha. **Per pair:**
  completion tokens over the pair's wall, prefill included. 250 ms telemetry per boot.
- **Separation:** U counts, over the 36 pairs of a k2 boot and a k3 boot, how often the k2 boot's
  median is higher.

## Results

Per request (`run/runs.jsonl`, `run/pairs.jsonl`), 144 requests, 0 errors:

| class | arm (K) | n | TTFT median / p95 (s) | decode median, range (tok/s) | pair aggregate median (tok/s) | ITL p95 median (ms) | acceptance |
|---|---|---:|---|---|---:|---:|---:|
| T=1 1.2k | k3 (3) | 12 | 1.859 / 2.655 | 36.73, 30.20-42.27 | 64.22 | 63.1 | 0.507 |
| T=1 1.2k | k2 (2) | 12 | 1.752 / 2.596 | 39.75, 34.85-45.01 | 67.77 | 57.2 | 0.603 |
| T=1 3.6k | k3 (3) | 24 | 4.359 / 6.691 | 34.62, 24.66-44.29 | 50.88 | 58.2 | 0.495 |
| T=1 3.6k | k2 (2) | 24 | 4.169 / 6.237 | 37.86, 29.73-48.05 | 54.91 | 55.3 | 0.614 |
| T=1 5.8k | k3 (3) | 12 | 10.344 / 12.586 | 35.28, 31.43-38.95 | 36.86 | 65.5 | 0.460 |
| T=1 5.8k | k2 (2) | 12 | 10.330 / 10.880 | 38.30, 35.92-41.40 | 38.34 | 58.7 | 0.549 |
| greedy 3.6k | k3 (3) | 12 | 5.175 / 7.125 | 33.72, 25.71-43.06 | 47.00 | 59.6 | 0.466 |
| greedy 3.6k | k2 (2) | 12 | 5.086 / 7.164 | 37.45, 29.67-46.49 | 51.11 | 47.3 | 0.586 |
| short (reference) | k3 (3) | 12 | 0.195 / 0.269 | 34.61, 32.58-38.64 | 68.01 | 67.3 | 0.421 |
| short (reference) | k2 (3) | 12 | 0.192 / 0.271 | 35.70, 30.73-37.07 | 69.73 | 66.8 | 0.421 |

TTFT is bimodal inside a pair: one request's first token comes well after the other's (1.1 and
2.3 s in one 1.2k pair). The median above is over both requests.

Per boot (each boot's median; 6 boots per arm; ratio = k2 over k3; `run/summary.json`):

| class | decode ratio | U | normalized decode ratio | U normalized | pair aggregate ratio | U aggregate | TTFT ratio | U TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T=1 1.2k | 1.072 | 31/36 | 1.067 | **36/36** | 1.055 | 27/36 | 0.996 | 16/36 |
| T=1 3.6k | 1.057 | 31/36 | 1.082 | 33/36 | 1.041 | 27/36 | 1.021 | 20/36 |
| T=1 5.8k | 1.085 | 29/36 | 1.115 | 34/36 | 1.040 | 25/36 | 0.999 | 21/36 |
| greedy 3.6k | 1.122 | 31/36 | 1.092 | **36/36** | 1.087 | 27/36 | 0.979 | 17/36 |
| short (reference) | 1.026 | 19/36 | 1.000 | 21/36 | 1.025 | 19/36 | 1.001 | 18/36 |

"Normalized" divides each request's rate by its own boot's short-pair median. The card throttled
through the run (peak 88-91 °C in every boot, busy SM clock medians 1,335 to 1,732 MHz, the
reference at 30.9 to 38.4 tok/s per request), which is why the raw U counts stop at 29 to 31 of 36
while the normalized ones reach 33 to 36. The reference itself is flat between the arms (19/36),
so the lead is not clock drift. The pair aggregate includes the pair's prefill, which K does not
move, so its ratio is smaller than the decode ratio.

Greedy texts: `deserts-4k` de407fdef1983a03 and `forests-4k` e923e7db50e84c31 in all 12 boots,
both arms. At T=1 another K draws another stream, so the texts differ by arm, as the single-stream
receipt found. At about 1.2k tokens the single-stream A/B was flat; with two requests K=2 leads
there too.

Not explained here: the 5.8k pair's T=1 texts are not one text per arm. `volcanoes-6k` wrote 3
distinct texts over an arm's 6 boots and `canyons-6k` 2, on both arms alike; the six shorter T=1
prompts wrote one text per arm in every boot. It is the same on K=3 and K=2, so it does not bear
on this comparison, and its cause was not looked for.

## Limits

One model, one card, two requests started together. Three and more sessions demote to plain
batching on this card, where K does not apply.

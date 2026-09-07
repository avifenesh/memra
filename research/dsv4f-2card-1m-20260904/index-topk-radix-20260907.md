# Exact indexer top-k candidate, 2026-09-07

The actual integrated radix-cut selector passes on both RTX PRO 6000 devices:
ordered IDs match the current numeric selector and an independent raw-key CPU
oracle. The gate covers padded 2056/2064/2112/4096 sizes, signed zeros,
all-equal scores, signed NaN payloads, infinities, and real-buffer refusal at
2047 and 4097. Both cards pass memcheck. This is not a serving qualification.

The dispatch is an exclusive, default-OFF process diagnostic with no new env
flag. It requires t=1, K=512 and 2048<=N<=4096, persistent `VerifyWs` scratch,
and the already selected device numeric path. Other paths keep their current
selector. Counters increase after the real CUDA wrapper accepts a launch.
The CUDA implementation uses the current bit key, including numerical-zero
normalization, not a new score policy. No host finite filter or D2H is added.

GPU 0 warmed, three-ABBA component timings (current / candidate): N2056
63.237 / 23.435 us; N2064 tie fixture 63.243 / 32.160 us; N2112
62.939 / 24.325 us; N4096 63.211 / 30.400 us. GPU 1 independently passes.
These are selector timings, not token throughput. First-launch and sanitizer
timings are excluded. R3 calls the integrated FFI directly; R1/R2 were
standalone candidate prototypes, not integrated dispatch proof.

R3 tool source SHA256 `fdd70d8ad636eae52404fed1ccb662e82c94aba788ae7e1d771da657a7e9f007`;
binary `abcd289c8df7d22b090b26e48f9360c50f8e4735c8d497af24e2ab2d821e51b3`.
Tool includes the actual CUDA TU, compiled with CUDA 13.1, sm_120a,
`-O3 -fmad=false`. Raw paired namespaces:
`index-topk-{memcheck,rate}-20260907-r3`.

The release model gate passes build, 411 CPU tests (15 ignored), and
release lib/bin clippy with `-D warnings`. `dsv4_plain_perf_gate index-topk`
holds half2 and grouped wo_a ON in both arms, restores the same frozen
snapshot, and requires sampled tokens/final logits/committed KV identity.
At 256 tokens the selector is intentionally inert; at 8192 it requires
21 actual launches per decoded token. The receipt reader has 17 CPU tests,
including rejection of inactive long-context counters and moving baselines.

Model gate executable SHA256:
`dbeb85847a18ced7141a5bea1ca4de1abaac1ec44c92ab4fb7bd8ba35a7714af`.
## Full-model sampled verdict

KEEP the measured candidate, default OFF pending serving admission. The full
28-row gate passed, with six timed rows per arm/context, sampled T=1/p=1/k=0,
seed 20260906, 256 output tokens, and three ABBA cycles. Every token stream,
final-logit hash and committed KV hash is identical; no looped rows entered
the metrics. Both prior wins stay enabled in both arms.

| Prompt tokens | Baseline plain tok/s | Selector plain tok/s | Same-window change |
|---|---:|---:|---:|
| 256 | 33.12572 | 33.14757 | +0.066%, inert selector control |
| 8192 | 30.24389 | 30.95706 | +2.358%, actual selector win |

The candidate records exactly 5,355 selector enqueues per 255-step 8K row;
the baseline and both short-context arms record zero. Graph counters are
zero in both arms. Fullmodel log SHA256
`1afbf5b0f4540b0e90bab5807e39bbd4e86d3f7736c6191c112b6d28c97fc8e4`.
Controller finished with status zero at 04:26:33Z. Both cards additionally
passed integrated-kernel initcheck and synccheck with zero errors.

Only the same-window +2.358% is attributed to this change. Cross-run movement
from older 32.06/28.00 rows is not credited to the selector. These are native
plain decode-gate rates, not HTTP serving qualification. The 120 tok/s
objective remains unmet; no runtime default is promoted.

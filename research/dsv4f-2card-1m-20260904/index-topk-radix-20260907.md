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
The full-model sampled ABBA verdict is pending. No runtime default is promoted.

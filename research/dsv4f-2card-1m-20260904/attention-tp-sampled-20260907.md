# Attention TP2 sampled performance protocol

One model load, five eligible single-stream repeats. Each fresh state primes 256 frozen source tokens through the supported tokenwise path, then samples and forwards 256 tokens with temperature 1, top-p 1, top-k 0 and seed 20260907. Comparison sampling is pinned and reported; no speculative decoding, PP arm or concurrency substitute.

The headline `decode_wall_ns` is one `Instant` envelope starting before the entire sample-plus-forward loop and ending after its final drain. CPU `dsv4_sample_row` time is included. Summed forward durations are not a headline metric. Prompt prime is reported separately. Allocation, transcript printing, final cache/hidden digests and the canonical attention-join oracle are outside the decode envelope.

Optimized native GU-M1, GU-half2 and down-half2 controls are enabled with actual dispatch assertions. Attention TP uses the qualified per-group wo_a path explicitly, never unqualified grouped-4. Every row checks attention producer/reduction counts, zero shared refusal words, finite logits, replicated cache/hidden state and actual GPU join bits against CPU f32 addition of both partials.

EOS, short output, detected looping or profiling makes a row ineligible. All five rows must be eligible before a pooled result is printed; failed rows remain in the raw log and are not rerolled. Repeats must agree in generated tokens, final logits, cache/hidden digests and final attention join hash under the pinned program and seed.

This is an attention-arm rate measurement, not a paired comparison. Prior forward-only rates are not comparable sampled throughput. The earlier full-loop sampled EP receipt may be stated as historical context only, not as a causal speedup denominator. No performance result has been produced yet.

Run the existing `dsv4_tp_ep_sampled_perf_gate` with `MEMRA_DSV4_ATTENTION_TP_GATE=1` and `MEMRA_DSV4_SAMPLE_SORT=comparison`, retaining the ordinary plain matrix-EP gate settings. Source remains gate-only and default OFF. No local CI or GPU work is permitted; use the owned remote dev pair and exact shared GPU lock.

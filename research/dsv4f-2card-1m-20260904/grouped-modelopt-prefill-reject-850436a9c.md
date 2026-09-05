# Rejected grouped ModelOpt prefill arm

## Correction: the whole-model experiment omitted the shared expert

Source audit on 2026-09-05 found an early return at
`f218cca95:crates/memra-engine/src/dsv4_gpu.rs:10737` into
`moe_verify_f16g`, which computes routed experts only. The shared-expert chain
starts at line 10844, after that return. Consequently the enabled arm did not
compute the complete model. The historical timings, teacher-forcing drift and
cache-transparency failures below remain recorded, but cannot isolate a numeric
class effect or reject correctly implemented grouped prefill. Whole-model speed
comparisons also include missing work and are invalid as optimization evidence.
The independent split-plane mapping and shape-only microbenchmarks are separate
observables and are not invalidated by this integration defect.

The arm remains reverted. A new experiment must include the shared expert and
re-run full-path parity and sampled cache twins before a conclusion. The broad
native drift band in the old teacher-forcing gate is characterization only, not
a calibrated quality or support-admission test.

## Historical record (causal conclusions superseded by the audit above)

Date: 2026-09-05 UTC

Implementation commit: `f218cca95`  
Revert commit: `850436a9c`

The arm adapted Memra's existing single-kernel grouped f16 visitor to read the
DSV4 mint's native ModelOpt split planes directly. It did not create GGUF bytes
or a duplicate resident expert bank. A synthetic mapping gate compared the
split-plane reader with an equivalent interleaved NVFP4 bank and passed bit for
bit over 8,192 outputs.

The exact checkpoint geometry is hidden 4,096, routed intermediate 2,048, 256
experts and top-6. A first diagnostic accidentally used intermediate 18,432;
those rows were discarded before any decision. The corrected shape probe gave:

| transaction rows | routed pairs | mean rows/active expert | 128-row padding | grouped w1+w3+w2 wall/layer |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 192 | 1 | 89.33x | 3.22 ms |
| 128 | 768 | 3 | 40.50x | 5.78 ms |
| 512 | 3,072 | 12 | 10.67x | 6.14 ms |
| 1,024 | 6,144 | 24 | 5.33x | 6.52 ms |
| 2,048 | 12,288 | 48 | 2.69x | 7.77 ms |
| 4,096 | 24,576 | 96 | 1.51x | 11.14 ms |

On the exact Safetensors artifact, a 1,894-token deterministic smoke completed
in 11.0857 s with the arm versus 15.2835 s on the same binary/config except the
flag. The reasoning text diverged, so this was a performance probe rather than
an identity gate. The frozen 32,000-byte source payload
`84b570f82425c0e8984c96580d3c6a521244166453714c49dad4ed9839ef2d1b`
produced 9,909 prompt tokens and TTFT 46.4857 s, about 213.17 prompt tok/s. The
selected exact chunk-32 row on the same source bytes is about 75.51 s/131.48
prompt tok/s, but has 9,928 prompt tokens because the request wrapper differs;
the 38.4% delta is therefore a mechanism estimate, not a matched publishable A/B.

The one-load teacher-forcing gate compared exact and grouped 1,025-token cache
states for 16 shared forced steps: 7 argmaxes agreed, 9 were inside the
predeclared DSV4 native drift band, 0 were outside, and maximum absolute logit
drift was 31.32595444. This was formally in-band but as noisy as the earlier
rejected W4A8 fork.

Most importantly, the arm cannot preserve sampled cache transparency while
decode remains in the exact class:

1. Merely widening exact transaction admission beyond the proven 64-row
   ceiling caused turn-1 nondeterminism. The scheduler was fixed to keep exact
   and remainder transactions at 64 rows or fewer.
2. Using grouped math only on full 512-row transactions failed at turn 4:
   the warm state restored 510 incrementally built exact tokens while cold
   replay crossed the grouped threshold.
3. Using grouped math for every prefill transaction and restored suffix still
   failed at turn 3. Warm state included prior assistant tokens advanced by
   exact decode; cold replay processed those transcript tokens through grouped
   prefill. Fixed-seed output and DSpark telemetry diverged.

Moving decode/spec into the padded grouped f16 path would sacrifice the model's
selected exact and low-latency decode route. The arm was therefore rejected and
fully reverted. It earns no serving, cache, correctness or performance claim.

Durable extracted gate lines are in `remote-gate-lines-20260905.log`.

Hardware metadata: two RTX PRO 6000 Blackwell Workstation cards, provider
setting 500 W/card. No Nsight Compute counters were available and no power,
clock, bandwidth or compute-limit attribution is made.

# Candidate discovery and first frozen admission study

The simple highest-score candidate baseline won the first fresh admission test.
The learned per-token tables abstained; contextual regression could not be fitted
because training contained only six proposal-changing interventions. This is a
measured negative result for the tables and insufficient data for the contextual
model, not evidence of a serving speedup.

All stages used native Memra, the locked Gemma 12B/Q8 MTP artifacts, a frozen
4,608-row head, fixed K=4, zero confidence cutoff, and greedy eager decoding on
one rented RTX 5090. Each target was tested separately. No combined controller
was measured. Raw receipts, checkpoints and prompt bytes are in
`candidate-gates-receipts/`; its manifest preserves file hashes.

| Independent stage | Evidence | Result |
|---|---|---|
| Physical single-slot replay | 96 requests; 391 reached states; 7,041 interventions | Zero score error and zero nonambiguous prediction mismatches. 41 positive, 9 negative, 6,991 neutral interventions; 391 deliberate ambiguous tie fixtures. |
| Bounded 32-row discovery | 192 requests, including 48 paired cost comparisons | Found 9 of 41 oracle-repairable targets. Observer cost ratio 1.001956, bootstrap 95% interval [1.001238, 1.002661]. |
| Previous verifier suffix as candidate source | 192 requests, including 48 paired cost comparisons | Discovery rose from 9/41 to 12/41. Calibration discovery rose from 5 to 7, selecting suffix hints. Cost ratio 0.999964, interval [0.999865, 1.000063]. |
| First fresh frozen admission | 24 prompts in six correlated synthetic families; 48 paired requests; 179 reached states | Highest-score baseline: 19 positive, zero negative repairs in 24 actions. Frequency: 2 positive, zero negative in 4 actions. Both learned tables: zero actions and zero repairs. |

Cost comparisons used eight calibration code prompts, six balanced AB/BA
repetitions, and 250 ms GPU telemetry. These are observer costs with the live
head unchanged. They do not measure an online replacement policy.

The fresh checkpoint was frozen at SHA-256
`7fde7a366117ed993c0d7f97462366dece2711190204e1da7079d75f88526472`
before fresh prompt generation. Independent re-audit confirms all 48 complete
128-token outputs match, reconstructs candidate pools only from prior verifier
observations, and checks 136 comparable native scores with zero error. Earlier
abandoned verifier suffixes are candidate hints, never continuation labels.

Prompt-mean one-step correctness deltas were 0.115642 for highest-score,
0.014286 for frequency, and zero for either learned table or no-op. No learned
policy qualified for an online trial. One-step repairs do not establish later
block acceptance, latency improvement, or weight-learning efficacy.

Next registered experiment: `DENSE-ADMISSION.md`. Collect every reached block
on the original training and calibration prompts only, retain the original
model classes and selection grid, freeze a new checkpoint, and evaluate on a
second untouched set. The first fresh set remains a reported test, not training
data. Confidence adaptation and depth adaptation remain disabled throughout.

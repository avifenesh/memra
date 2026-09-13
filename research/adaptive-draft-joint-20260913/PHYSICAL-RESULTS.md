# Physical replacement qualification — 2026-09-13

**Passed for the tested one-row greedy operation.** All96 ON/OFF comparisons
preserved the complete128-token output, following21 successful two-model pilot
checks. The required missing-parent diagnostic refusal also passed.

On391 reached draft states, the shadow head executed7,041 physical replacements
at exactly4,608 rows using original Q8_0 bytes and the normal GPU projection and
argmax. Maximum score error against the frozen-state prediction was **zero**.
There were no non-ambiguous winner disagreements. The391 duplicate-winner tie
fixtures remained explicitly marked ambiguous rather than assigned an assumed
slot tie-breaking guarantee.

The interventions produced41 positive, nine negative and6,991 neutral one-step
correctness changes. These count interventions, not independent prompts or
whole-block gains. A final restoration intervention on every state restored the
original slot and winner. The live draft never consumed a shadow winner.

Native source: `c420a285f63a5d57951bce8335a675e6a15aee6b`. GPU: rented RTX5090;
the same locked Gemma12B/assistant and saturated head as the oracle. The old48
prompts were reused for numerical qualification only. No fresh policy-generalization
claim, speedup or serving promotion follows from this result.

`physical-audit.json` is the independent audit result. The next experiment measures
causal bounded candidate discovery and its full request cost separately.

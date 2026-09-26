# Qwen draft-only C/K/D v9: learned actions help, fixed C remains competitive

**Verdict: no learned-policy throughput win over the best fixed
control on the tested workload.** Learned C/D and joint policies
beat their model-running no-op twins by about 2.08% on 16 fresh
conversations, with paired intervals above zero. Each was only
about 0.30% faster than the validation-selected fixed C control,
and those intervals crossed zero. The fixed C control was itself
1.56% faster than fixed K=20/D=3/C=0 on these conversations,
with a paired interval above zero. This is a research result for
this artifact and request shape, not a serving default.

K is **MTP draft sampler top-k**; the target sampler remained
top-k=20. D is offered draft depth. C stops extending an offer
after observing its sampled proposal probability. The Qwen3.8-27B
target and full-vocabulary embedded MTP head stayed frozen. Only
small controller weights were trained.

## Measured request shape

One nonproduction Nebius RTX PRO 6000 Blackwell GPU in
`uk-south2` ran the same pinned model and binary for all arms.
Sixteen untouched eight-turn code conversations used a custom
split of Google's sanitized MBPP tasks, with one visible test
and at least two reserved tests per task. Decoding was sampled
at temperature 1.0, top-p 0.95, target top-k=20,
`max_new=4096`, and `ctx=65536`. Later turns reused native KV.
The score is **pooled returned output tokens / complete native
request seconds**; startup and post-timing code tests are outside
those request clocks. All 10 final arms had zero exact-loop turns
and passed the frozen syntax, reserved-test and cap-rate guard.
One reserved test case is a bounded check, not broad code
correctness.

| Final arm | tok/s | Reserved tests / 128 | Capped turns / 128 |
|---|---:|---:|---:|
| Fixed K=3, D=3, C=0 | 115.59 | 109 | 11 |
| Fixed K=10, D=3, C=0 | 116.73 | 109 | 11 |
| Fixed K=20, D=3, C=0 | 117.01 | 107 | 12 |
| Fixed K=20, D=3, C=(0, 0.50844) | 118.84 | 105 | 16 |
| Learned K only | 116.70 | 109 | 11 |
| K-model no-op | 116.94 | 107 | 12 |
| Learned C/D at fixed K=20 | 119.19 | 109 | 13 |
| C/D-model no-op | 116.76 | 107 | 12 |
| Joint learned K/C/D | 119.20 | 110 | 12 |
| Joint-model no-op | 116.77 | 107 | 12 |

The selected fixed C threshold came from the new training
proposal-probability quartile, not final data. It measured
**+1.56% [95% paired bootstrap +0.30%, +2.77%]** versus
K=20/D=3/C=0 on fresh conversations. Its output was 7.03%
longer and its complete request time 5.38% longer than that
control. The result supports the fixed setting on this
research shape; it does not qualify a vendor-default endpoint
or another GPU.

Learned C/D at K=20 was **+2.08% [+0.86%, +3.45%]** versus its
own no-op, and **+1.87% [+0.67%, +3.24%]** versus fixed
K=20/D=3/C=0. Against the selected fixed C point it was
**+0.30% [−0.91%, +1.74%]**. Joint learned K/C/D was
**+2.08% [+0.52%, +3.65%]** versus its no-op and
**+1.87% [+0.32%, +3.48%]** versus fixed K=20/C=0,
but **+0.30% [−1.08%, +1.54%]** versus the selected fixed C
point. Both learned arms produced fewer tokens *and* used less
time than their respective no-ops. Full request tok/s, not
accepted draft length, determines each comparison.

## What was learned and what acted

The v6 randomized receipts and all six v8 fresh conversations
became **training-only** material for v9. They supplied 192
measured K turns, 107,305 eligible D rounds and 77,095
conditional C offer labels. The new training split supplied
576 more K turns, 216,952 randomized D rounds and 424,853
observed C offers. No training task group exact-looped.
Historical K turn utility used its own RTX 5090 reference rate.
Only new-GPU randomized rounds supplied the RTX PRO 6000 C/D
time-cost fits; old acceptance observations were an explicitly
compared augmentation. Training and final task IDs were disjoint.

All six first-16, first-32 and previous-turn K fits at the frozen
penalty of 100 chose K=10 on their training inputs. The lower-penalty
first-16 training fits sometimes chose K=20, but were not native
selection arms. The final K-only and joint arms
also chose K=10 on **128/128 turns**. The K-only arm measured
116.70 tok/s versus fixed K=10's 116.73, with no useful routing
shown. The joint policy's gain over its no-op therefore came
from C/D actions on this workload, not adaptive K.

The learned C/D arm made 86,648 live C decisions with 14,371
stops. Its eligible D choices were 5,381/28,405/8,842
rounds for D=2/3/4. The joint arm made 84,389 C decisions
with 13,875 stops, and selected D=2/3/4 on
2,836/33,755/5,135 eligible rounds. Their accepted/drafted
diagnostics were about 0.634 versus 0.573 for fixed K=20/C=0.
This is evidence of actual learned actions, not an acceptance
score. The C/D controller consumed 0.290 seconds across the
16 C/D conversations; the joint K and C/D controllers together
consumed 0.330 seconds. Those times are included in requests.

The qualifier ran 22 arms. All eight model-running no-op twins
were byte-identical to fixed K=20 on its eight turns, and all
six active C/D or joint variants made both stop and continue
decisions. The final scorer additionally confirmed byte
identity for the three selected no-op twins on **every turn of
all 16 final conversations**. The sealed replay checked raw
native turns, KV receipts, code tests, no-op identity, C
engagement, score arithmetic and all archive member hashes.

## Boundary and next decision

The v9 split is not an official MBPP benchmark score. Its
reserved tests and 16 conversations bound claims to this
sampled code workload. Full sampled distribution parity and
vendor-default endpoint qualification remain separate under
Memra #673. No production setting changed.
The archived v8 source tied to this exact binary SHA appends the
sampled pick before fixed or learned C stops on sampled graph and
eager paths, and refuses positive sampled PMIN0. The chosen-pick
discard counterexample in Memra #673 applies to a different
mainline path. This source-order check does not by itself establish
full-model sampled distribution parity.

The model SHA-256 is
`1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a`;
the native binary is
`84b04b6ccccc3b64992377cef0677926cb932c7a98f063aff84d1f1f65224d7f`;
the frozen workload manifest is
`ebceeeffdf36128b98b303c124cf7459966877991d66afce15f7ecc6ffd4d6ee`.
The final corrected native archive SHA-256 is
`a914e20a4f823acbdf189202806415785d60507fa0f775ab588b10083d3c034b`
with 34,514 expanded members. Its independent replay passed.
Publication of the raw archive remains subject to the public
boundary review.

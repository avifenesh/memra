# One C/K/D policy for mixed requests

This contract was written while V11 was collecting randomized
training data, before its validation or final prompt files were
released. The V11 host reported 32/224 completed training sessions
and no validation or final workload directory at
2026-09-26T09:06:33Z.

## What V10 established

V9 selected the code-trained `joint-augmented` policy. V10 loaded the
same K router and six C/D model files for instruction following and
math; their hashes match the V9 archive byte for byte. V10 did not
ask for a topic label or switch those weights by topic. The best
fixed comparator was determined separately *in analysis* for each
topic, never selected for a served request. Code ran earlier on an
RTX PRO 6000 Blackwell; both V10 non-code topics ran on one RTX
5090. None of the three results proved a learned win over its
strongest tested fixed control.

Instruction following is a constrained task check, not a broad
open-prose check. The V10 set had 128 distinct instruction tasks,
including only two JSON-format labels, and 128 distinct math tasks.
It had no tool-call protocol score.

## Boundary of the running V11 study

V11's `joint-v11/select.py` chooses a `primary` learned arm inside a
loop over instruction and math domains. Those per-domain choices
answer which candidate performed best *within each topic*. They do
not specify one controller artifact for arbitrary requests. V11
does not have a fresh same-GPU code or open-prose final split.
Even two positive V11 domain results cannot establish a universal
serving policy. The running V11 study remains valid training and
topic-specific evaluation evidence; its pinned source and phase
commitments stay intact.

## Universal policy decision rule for a follow-up

1. Fit candidate C/K/D controllers from training data only. Each
   candidate is one immutable set of weights and one native runtime
   mode. A candidate may choose draft K, C stops and D from bounded
   prompt tokens, generated history, offered probability and prior
   rounds. It receives no code/prose/math label, user-declared topic,
   field-specific weights or field-specific threshold.
2. On one physical GPU and pinned model/binary/sampled target decode,
   freeze a mixed validation set of code, open prose and math
   continuing conversations. Select **one** policy label and model
   hash from that combined validation result. A robust selection
   objective favors pooled returned tokens per complete native
   request second only among candidates meeting every domain's
   task-quality, cap, loop and throughput no-regression guard.
   Record the selection before final prompts open. No
   `chosen_by_domain` route may affect runtime or final-arm choice.
3. Measure the same selected policy on fresh mixed final
   conversations on the same GPU and request shape, with no topic
   hint. Compare it with its byte-identical model-running no-op
   twin and the strongest **single global fixed** C/K/D setting,
   both selected without final data. Also report each domain's
   validation-selected best fixed setting as a non-deployable
   regret control. The prose leg must not lose complete-request
   tok/s or quality against its prose fixed control. Freeze the
   quality oracle and noninferiority margins before seeing final
   output; a pooled win cannot hide a prose regression.
4. Report pooled and separate code, prose and math tok/s, output
   and elapsed ratios, quality passes or blinded preferences,
   caps, loops, actual K/D/C actions, controller time, and native
   later-turn KV receipts. Acceptance is a training observation
   and diagnostic, never the performance score. A constant K
   choice does not demonstrate adaptive K.

The current seven-arm V11 fixed menu omits fixed D=1/2 and the full
K/C grid. It cannot establish superiority to every hardcoded
configuration. A future positive policy result needs stronger
training-free and fixed controls on an independently frozen final
split. Sampled-distribution parity and vendor-default endpoint
qualification remain separate before any serving change.

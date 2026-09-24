# Qwen draft-only C/K/D: learned policy did not beat fixed controls

**Verdict: no learned-policy throughput win on the tested code
workload.** K is the MTP **draft sampler top-k**; the target
sampler stayed at top-k=20. D is offered draft length and C is
an after-offer decision to draft another position. This is a
different K notation from the earlier fixed-depth studies.

The Qwen3.8 full-head target and MTP head stayed frozen. Small
K, D and C controller weights were fitted on randomized
development conversations. K used the first bounded user
tokenizer prefix and optional previous-turn acceptance; D used
committed output and measured round time; C used the sampled
proposal probability, conditional acceptance and marginal
cost. C/K/D actions were chosen live in the native request.
The [method](../joint-v6/METHOD.md) and
[v8 protocol](PROTOCOL.md) give the causal-input and source
boundary.

## Fresh sampled result

Six untouched eight-turn code conversations ran on one
nonproduction RTX 5090 with the same pinned binary, Qwen
checkpoint, prompts and seeds, target top-k=20, temperature
1.0, top-p 0.95, default thinking, `max_new=8192` and
`ctx=65536`. Later turns reused native KV. The score is
**pooled returned output tokens / complete native request
seconds**; model startup and the post-score code probes are
outside those request clocks.

| Arm | Output tok/s | Parseable code / 48 | Function cases / 48 |
|---|---:|---:|---:|
| Fixed draft K=20, D=3, C=0 | 141.29 | 48 | 48 |
| Fixed draft K=3, D=3, C=0 | 138.71 | 47 | 48 |
| Fixed draft K=10, D=3, C=0 | **142.29** | 48 | 48 |
| Development-selected fixed K=20, D=3, C=(0, 0.529) | 139.40 | 48 | 48 |
| Learned K, D=3, C=0 | 141.25 | 48 | 48 |
| K-model no-op, fixed K=20, D=3, C=0 | 141.29 | 48 | 48 |
| Joint learned K/C/D, history features | 140.75 | 48 | 48 |
| Joint-model no-op, fixed K=20, D=3, C=0 | 140.99 | 48 | 48 |
| Learned C/D at fixed draft K=20 | 142.15 | 48 | 48 |
| Learned C/D at fixed draft K=3 | 141.20 | 48 | 48 |

All ten arms had **zero exact-loop exclusions** and
42/42 later turns per arm with positive cached and new input
tokens. The one parseable-code miss in fixed draft K=3 makes
that arm ineligible under the frozen strict format gate,
despite its bounded function cases passing. A function case
is one valid-domain check per requested function; 48/48 does
not establish broad code correctness.

The joint policy was **+0.97%** against the fixed C/K/D
vector selected on development, with a six-conversation
20,000-resample bootstrap interval **[−0.89%, +2.78%]**.
Against fixed draft K=20/D=3/C=0, it was **−0.39%**
**[−2.43%, +1.33%]**. Its 38,084 returned tokens were
7.48% more than K=20/C=0's 35,432, while its 270.59
request seconds were 7.90% more than that control's
250.78 seconds. It also trailed its own no-op twin
(140.75 versus 140.99 tok/s). The learned K-only arm
measured 141.25 versus its no-op's 141.29 tok/s.

The fresh fastest quality-eligible fixed point was draft
K=10/D=3/C=0 at 142.29 tok/s, **+0.71%**
**[−0.98%, +2.85%]** versus K=20/D=3/C=0. That is a
post-score observation with a six-topic interval crossing
zero, not a qualified new default. Learned C/D at fixed
draft K=20 was 142.15 tok/s, **+0.61%**
**[−2.09%, +2.92%]** against K=20/C=0. Neither result
establishes a repeatable win.

## Actual decisions and development overfit

The joint history policy made **30,439 native C decisions**
and **3,779 stops**. It chose draft K=3 on 4/48 turns and
K=20 on 44/48. Across eligible rounds it chose D=2/3/4
on 151/7,384/5,317 rounds. The K model consumed 0.083
seconds and the C/D model 0.079 seconds across the six
complete conversations; both are included in the request
times. The joint no-op paid 24,550 C decisions with zero
stops, held K=20 and D=3, and consumed 0.081/0.071
seconds of K/C-D model time. The learned actions, not a
large controller-time bill, explain the absence of a
measured advantage in this battery.

The [development grid](../joint-v6/DEVELOPMENT.md) chose
fixed draft K=20/D=3/C=(0, 0.529) at 143.46 tok/s,
1.41% above C=0 in-sample. On fresh conversations that
same static point measured 139.40, **−1.34%**
**[−2.74%, +1.07%]** versus K=20/C=0. Its output
was 7.58% longer and its request time 9.04% longer.
Acceptance and shorter drafts were useful training and
mechanism observations, but neither selected cutoff
nor the tiny joint policy improved the fresh primary score.

## Source and qualification boundary

The model file
`Qwen3.8-27B-NVFP4-Q5K-mtp.gguf` is SHA-256
`1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a`,
revision
`0f82b27dbb264b731e7d20f576582c871ef1969c`.
The measured v8 Rust source archive is SHA-256
`7765982aacad20867b406029b945cdec9f60e5694e0e9489731e3ffc8ce24d96`;
the pod binary is
`84b04b6ccccc3b64992377cef0677926cb932c7a98f063aff84d1f1f65224d7f`.
The final input bundle is
`efe29e749de787d9eed5e4209ea45830c7f81b01c958c91201981fdb9522a529`.

V6 development training and fixed controls are
[sealed separately](parent-v6/manifest.json). V6's
attempted joint selection never made a native C decision;
its guard stopped before fresh evaluation. V7 added the
single-head C hook, but its prime-time graph had not
captured the chosen probability and passed `q=0`.
The [v7 diagnostic archive](diagnostic-v7/manifest.json)
retains that failure. V8 requests probability capture
when learned C is attached and fails closed on `q=0`.
An eight-turn v8 engagement run made 1,643 C decisions
with 192 stops and passed all code/function gates.
The final four-arm sampled qualifier made the K and joint
no-op output-token tapes byte-identical to fixed
K=20/D=3/C=0 across eight turns. Live joint C made
1,429 decisions with 178 stops; its no-op paid 1,096
decisions with no stops.

The [v8 receipt](receipts-v8/manifest.json) seals
6,666 native/source/model members, archive SHA-256
`a54b6e48e1e1067cc27e5716563be446cfda30a5c859e285af0554f321b2e55c`.
The v6 training parent seals 13,613 members, and the
v7 diagnostic seals 164. Their
[exact publication boundary](ARCHIVE-BOUNDARY.md)
is reviewed separately. Independent pod replay
verified the v8 score and code probes, parent model
and control hashes, and each archived member.
The first derived analysis JSON used integer action-map
keys that JSON reload converts to strings; its archived
copy is retained. Canonicalizing those keys changed no
paired rate, interval or quality flag, and replay then
passed.

The small synthetic code set, one-case function probes
and six paired conversations limit generalization.
Correct after-offer source order and no-op byte identity
do not establish full-model sampled distribution parity
or vendor-default endpoint qualification. Memra #673
retains that gate. **No MTP head weights, serving
default, fleet binary or customer route move from this
research.**

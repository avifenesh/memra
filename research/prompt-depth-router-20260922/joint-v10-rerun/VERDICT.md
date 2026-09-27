# Qwen C/K/D transfer to instruction following and math

**Verdict: no code-trained controller transfer win over the best fixed
control on either tested non-code workload.** The controller improved
complete native request tok/s over its byte-identical model-running
no-op on both topics. Its paired interval against the strongest
quality-eligible fixed control crossed zero on both topics. This is
bounded evidence for the pinned model, binary, sampled request shape,
and RTX 5090. It is not a serving-default or general-topic claim.

The clean rerun used Qwen3.8-27B-NVFP4-Q5K-mtp.gguf, model SHA-256
`1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a`,
and the full-vocabulary embedded MTP research binary SHA-256
`84b04b6ccccc3b64992377cef0677926cb932c7a98f063aff84d1f1f65224d7f`.
Its fresh workload manifest was
`dd9fc45646931f66fee8a3b328b404d7da227765c58ce78f57536b3605fcbeb8`.
One nonproduction Vast RTX 5090 in South Korea ran 16 eight-turn
instruction-following conversations and 16 eight-turn grade-school
math conversations, ten arms per conversation. Later turns proved
native KV reuse. Target top-k stayed 20; K changed MTP draft sampling
only. Temperature was 1, top-p 0.95, max new tokens 4096, and context
65536. These custom continuing-conversation splits are not official
standalone IFEval or GSM8K scores.

| Topic | Learned arm | Learned tok/s | Its no-op tok/s | Best fixed tok/s | Learned vs best fixed, paired 95% interval |
|---|---|---:|---:|---:|---:|
| Instruction following | joint C/K/D | 145.46 | 141.03 | 144.08, K20 with fixed C cutoff | +0.96% [-1.86%, +4.08%] |
| Grade-school math | joint C/K/D | 165.08 | 162.49 | 164.01, K20 with fixed C cutoff | +0.65% [-0.30%, +1.57%] |

The learned arm's no-op comparisons were +3.14%
[+0.95%, +5.52%] on instruction following and +1.59%
[+0.46%, +2.71%] on math. All ten arms passed the stated
task-quality and cap eligibility guard in each topic; there were
zero looped conversations. The joint arm passed 113/128 instruction
tasks and 127/128 math tasks. Both no-op twins returned byte-identical
token IDs to the fixed K20/D3/C0 reference on every held-out turn.
The fixed C cutoffs came from the earlier code training split, not
from these held-out topic results.
The paired comparison reports output-token and elapsed-time ratios
separately: 0.909 and 0.881 for instruction following, 1.046 and
1.030 for math, versus the joint arm's own no-op.

The code-trained K router chose K=10 on all 128 turns in each topic,
so these measurements do not demonstrate adaptive K selection.
Learned D chose depths 2, 3 and 4; learned C made both stop and
continue decisions. Acceptance and controller cost were retained as
diagnostics, not used as the performance score. The two learned
options across two topics were exploratory comparisons, with no
multiple-comparison adjustment. The stronger V11 design freezes a
primary option per topic on validation before any final prompts run.

The first Vast launch stopped after two qualifier arms when its math
grader was corrected to require a complete final `#### <number>`
answer line. Those first-launch outputs were excluded. The clean
rerun restarted all arms under new source and run metadata. Its
private archive SHA-256 is
`8007937a6e824d0ba10bef6481253913b9729446f790ab91d0dfc4d7bf441f1c`;
independent replay matched 20 qualifier and 320 held-out arms, quality
checks, no-op identity and complete-request timing. The instance was
destroyed and absent from a fresh Vast inventory. A verified
256-session training projection of this archive is private in
Darklanes for V11, leaving V11 validation and final tasks untouched.

V9's code result used an RTX PRO 6000 Blackwell and found no learned
win over the best fixed code control. A direct code-versus-non-code
rate comparison would mix hardware and task effects. Sampled target
distribution parity and vendor-default endpoint qualification remain
separate before any serving decision. No fleet pin, serving flag, or
customer route changed in this research lane.

# Next bounded question: where does the quality cost live?

The local component ratio clears issue #439's engineering-value test. The next
step is a quality ablation of the frozen program. This plan is prepared only;
no quality run or reduced-set implementation is included here.

## Frozen comparison

Use one immutable weight payload and the existing v3-amax multipliers. Do not
refit scales, change the tokenizer, adjust prompts, or select new evaluation
windows after seeing candidate results. Retain the original four held-out
windows at prefix lengths 1536, 3072, 9216 and 12288, each followed by the same
1024 teacher-forced positions. The existing private corpus lock and window
manifest remain the data source; keep their content private.

Use the current `qwen-nll-kl` two-pass protocol: dump reference log-softmax rows,
then compare each candidate against those rows. It scores one vocabulary row
at a time and binds the exact prefix plus scored token IDs with SHA256. Require
identical context, window length, position count and ID digest before computing
any metric. A corrupted digest must still fail as the red arm.

The diagnostic arms are:

| Arm | A4-enabled slots |
|---|---:|
| W4A8 reference | 0 |
| W4A8 repeat control | 0 |
| Full A4 negative control | 400 |
| Post-norm input group | 272 |
| ffn_down only | 64 |
| ssm_out only | 48 |
| attn_output only | 16 |

Implement class selection in a bounded diagnostic harness, with the selected
names and actual per-slot launch counts in each receipt. It must preserve the
existing arithmetic and select only the listed stamps. This is not a production
dispatch feature or a new serving flag. Load refusal for the zero-slot control
must happen before stamping, as the existing `MEMRA_A4_DISABLE` control does;
clearing only the model config is the old vacuous control.

## Budget and stopping rules

Start with the 1536-prefix window. At most seven fresh-process passes there.
Continue to the other three windows only if the reference repeat is identical,
position identity holds, every requested slot group actually executes, and the
memory envelope fits. The total maximum is **28 passes across four windows**.
Do not add models, contexts, scale searches, seeds or new classes to rescue an arm.

For the 24 GiB laptop, first account for resident weights, cache through
`ctx+1024+8`, prefill scratch and allocator reserve. Admit only a measured budget
with at least 2 GiB VRAM headroom; sample memory during the pilot. Full-vocabulary
rows are streamed to disk and candidates read the reference on the CPU. One
1024-row, 248320-token-vocabulary dump is about 0.95 GiB on disk/host memory,
not two simultaneous GPU logit matrices. Do not enable full FP16 weight mirrors.
If the complete envelope does not fit, stop and name the additional hardware
required. No rental, fallback model or offload program is authorized by this plan.

Run each finite cell through the resource controller, with a declared timeout
and no simultaneous runtimes. A timeout, nonfinite row, baseline-repeat drift,
identity mismatch or missing slot engagement is a failed prerequisite, not a
candidate quality score.

## Required report and decision

For each window and class, report paired per-position dNLL, KL(reference || arm),
top-one agreement, the repeat-control floor, intervals, artifact/source/binary
hashes and actual selected slots. Keep per-window results visible. A lower NLL
with substantial KL or top-one disagreement is distribution drift, not a win.
Do not sum per-class dNLL and assume it predicts their combination.

The old numerical bars were lane preregistrations, not standing owner quality
policy. The owner must approve the quality decision rule against the measured
repeat floor before candidate scores are interpreted.

Next owner decision: authorize this finite class-selection diagnostic and its
quality rule. A positive quality result would still need an end-to-end prefill
measurement and native gates for any runtime change. PRO 6000 transfer, serving
defaults, publication and any decision to close or kill the program remain separate.

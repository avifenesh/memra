# Reference/matrix distribution diagnostic

Protocol frozen before target results, 2026-09-05 UTC. This characterizes the
whole-request matrix realization after its phase/state/storage gate passed.
It does not replace pinned-source checkpoint qualification, task-quality
evaluation, sampled serving, or performance measurements.

`dsv4_matrix_distribution_gate` loads the same Safetensors NVFP4 checkpoint
once. Both programs retain RefFp8Round activations and consume identical forced
source tokens, including the complete prefill. Reference/matrix switching uses
the existing exclusive gate setter only after the previous state is dropped.
The drafter is primed but neither program samples its continuation inputs.

The already-frozen real-source corpus has SHA256
`f6e175a6f2588953568746fec0cd43fcd046405f74b5c71ce071fe7f37238ded`.
The tokenizer input begins `Review this engine source:\n`. Windows are defined
by token offset and conditional-prefix length: (0,160), (32768,1025), and
(131072,4097). Each scores the next 64 real tokens, 192 rows total, with chunk
32. Windows are not repeated or padded. Offsets describe source selection;
each new decode state starts at model position zero. Token ids and hashes are
banked so the input can be reconstructed exactly.

For each row, the diagnostic stores full reference and matrix F32 logits,
their hashes, vocabulary size and target id. Stable F64 log-softmax gives
target NLL, KL(reference || matrix), and total variation. Maximum raw-logit
delta, first-index argmax agreement, and a fixed-seed temperature-1/top-p-1
sample agreement are descriptive only. Source inputs are never replaced with
either program's chosen token, avoiding a divergent-history comparison.

There is deliberately no quality PASS threshold derived from these candidate
results. A successful run ends `COMPLETE distribution characterization`, not
quality admission. This small code-source panel cannot establish general
language, tool-call, long-context or checkpoint quality. Logit offset shifts
can produce a large raw delta with zero probability drift; that is why the
distribution metrics accompany the previous 1.0005016 final-row delta.

Local validation: four analytic tests passed (identical and shifted rows,
known binary distributions, extreme finite logits, and invalid-input refusal).
Strict clippy, optimized build, formatting and whitespace checks passed.
Initial diagnostic binary SHA256:
`f44d979b50e4c62c3940ec7bc1059262847b7e671cde807e362ca61117a3300f`.
No target outcome is claimed yet.

The initial target invocation subsequently failed during model loading with
CUDA OOM, before producing any scored rows. The companion private lane retains
the exact invocation, failed stderr, concurrent qualification investigation
and clean matrix profile in `matrix-profile-and-distribution-20260906.md`.
This is an uncompleted diagnostic, not a quality result; the frozen input
panel and metric tests remain ready for a coordinated retry.

## Completed retry and independent audit

The later guarded2 retry emitted all 192 rows and `COMPLETE` on 2026-09-06
UTC with the frozen `f44d979b...` binary. Its wrapper exited 75 because NVML
changed the exiting process's name to `[No data]`; all 550 recorded compute-app
rows belonged to the same PID. The original warning/exit status is preserved,
not relabelled as a clean controller pass.

`tools/dsv4-distribution-summary.py` independently verified all 384 raw-row
hashes, input/target ids, bank lengths and 960 scalar measurements using F64
log-softmax. Audited results (NLL in nats per scored token):

| Prefix | Reference NLL | Matrix NLL | Mean TV | Maximum TV |
|---|---:|---:|---:|---:|
| 160 | 2.343045 | 2.354735 | 2.9314% | 15.1663% |
| 1025 | 0.426562 | 0.436500 | 0.7010% | 8.1378% |
| 4097 | 0.000971 | 0.001830 | 0.1020% | 3.7428% |

The equal-row mean NLL increase is 0.0074961 nats/token; mean TV is 1.2448%.
The maximum raw-logit delta is 12.0403. Argmax agreement was 190/192; reported
same-seed sampled-id agreement was 169/192. Neither agreement is a general
quality score. This code-source panel is small and uneven in entropy, with
one almost deterministic window. It does not admit the matrix program for
general quality, long contexts, tools or serving. No candidate-derived quality
threshold was selected from it.

The private companion lane retains `matrix-distribution-guarded2-raw/`, its
model/controller/process logs and `matrix-distribution-audit.json`. The new
phase gate can consume the frozen window-0 reference/matrix rows to check that
the staged-chain refactor has not changed the prior programs.

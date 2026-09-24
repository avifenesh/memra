# Follow-up data ledger

The v6 randomized development archive and v8 result archive were
reclassified as **training-only** sources for v9. `measurement_rows.py`
verified the outer SHA-256 and every member it read against each sealed
manifest. It extracted:

| Decision observation | v6 | v8 | Total |
|---|---:|---:|---:|
| K measured turns | 48 | 144 | 192 |
| D eligible rounds | 41,804 | 65,501 | 107,305 |
| C observed offers | 77,095 | 0 | 77,095 |

The K totals are 63/65/64 measured turns for draft K=3/10/20. The old
v8 sessions each ran every fixed K on the same six topics; these are
training observations, not 144 independent topics. v8 selected-policy
D rows are labeled as such and cannot substitute for randomized
D outcomes. It retained no per-offer probability, so it supplies no
C labels.

`fit_k.py` fitted first-16, first-32 and prior-turn variants from those
complete-turn rewards at ridge penalties 20 and 100. All six picked
K=10 on all 192 **training** rows. This is an in-sample action audit,
not a throughput result. The original Qwen/RTX 5090 final comparison
in [v8](../joint-v8/VERDICT.md) remains the previous result.

The v9 workload freezes 24 training, 8 validation, 16 final and one
qualifier eight-turn conversations from 392 distinct sanitized tasks. Source:
Google Research `google-research` commit
`d36068b845da4c2b24927fee2cea1e6ef98dadda`,
`mbpp/sanitized-mbpp.json` SHA-256
`ca95deaa9a01ef0a6f439f88bcf0dd3db3563d22f22aad6cae04ebb9a8d8c8e9`.
This is a custom continuing-conversation workload, not an official
MBPP benchmark score. The partition is locked by `workloads.py` before
opening validation or final output.

The first native qualifier used an earlier wording with no worked
example. Its first turn reached 8,192 tokens of thinking and produced
no answer, so the run was stopped after that failure. The second
qualifier showed one test but still hit the 4,096-token cap with an
empty answer on an ambiguous task. Neither diagnostic contributes
training or evaluation observations. The v9 workload uses the
hand-verified subset, removes inert reference padding, shows one
example test and reserves at least two tests for grading. Tasks
examined in the diagnostics are excluded from all v9 splits.

The sanitized eight-turn qualifier on one nonproduction Nebius
RTX PRO 6000 (FIN-02, on-demand) completed with 3,244 returned
tokens in 27.384 complete-request seconds. All turns ended in EOS,
8/8 answers were fenced and parseable, 8/8 passed every reserved
test, zero exact loops occurred, and all seven later turns had
positive native cached and new input tokens. The binary logged
the full 248,320-row embedded MTP and fixed target top-k=20.
This is an engagement and workload check, not an evaluation score.

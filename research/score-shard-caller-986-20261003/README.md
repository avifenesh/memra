# Score-shard self-test admission

Issue #986 adds a mandatory CPU caller for the two existing score-shard self-test assertions. The self-test refuses Python `-O`, `-OO` and `PYTHONOPTIMIZE` before creating temporary shard fixtures. Ordinary score merging remains available under optimization.

The caller admits exactly 14 identified controls, with discovered, executed and successful identity sets equal. It observes both original assertion predicates and a real merge of the two fixture shards. Missing or replaced predicates, a no-op self-test, wrong merge results, removed or masked CI callers, missing or duplicate identities, absent success callbacks, skips and expected failures refuse admission.

[Admitted CPU receipt](receipts/cpu-v5/ADMITTED.json) binds the base `50f79338`, working diff and six source hashes. It records 16 actual original/current before/wrong/restore runs and ten compiling coherent mutants, each producing assertion failure with zero fixture errors. Raw logs use lossless gzip; `RAW-LOGS.json` records raw and compressed hashes. The separate 206 framework, A14 and SFT18 controls passed; all nine original SFT controls executed. Registry coverage spans 14 packages and all nine workflow files passed the duplicate-key check.

[Actual B composition](receipts/actual-B-composition/COMPOSITION.json) records a separate clean run on `985e1a76`, based on the actual #982 merge `9595cea5`. C14, B17, five original wrapper controls, A14, SFT18 with all nine original methods, and the full 206 framework controls passed. Five C code hashes and five B helper hashes remain identical. The entire B workflow is retained with the one required C step added. The original working proof and `23ce9a86` carrier above remain historical; their tuple and approval are not relabeled as this composition. Unchanged cache and native gates were not repeated locally.

The original merger snapshot has SHA-256 `dcdefb0b1a0337c1083dfb9512725c5d232e130e9d895301f5e724d9112a6d95`. Every function outside `self_test` has the same parsed code. The ordinary CLI fixture checks exact output bytes, metadata, rows, ordering and shard hashes under normal and optimized execution. Native-plan classification remains full and unqualified. This is CPU caller evidence, with no model scoring, native execution or model/runtime qualification claim.

Reproduce with:

```sh
python3 tools/merge_expert_score_shards.py --self-test
python3 tools/run_score_shard_contract.py
python3 research/score-shard-caller-986-20261003/tools/controls.py "$PWD" /path/to/owned-receipts
```

The proof owns and removes its temporary roots. Its protocol fixtures are controls for admission behavior, not evidence that production model scoring ran. Initial source-binding and fixture attempts remain retained separately and are not admitted into this receipt.

publicity: skipped: maintenance release

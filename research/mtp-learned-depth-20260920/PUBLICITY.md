# Drafts for the full-head MTP depth research publication

Owner posts these. The PR publishes research receipts and changes no serving default.

## X draft

Full-head MTP depth on Qwen3.8-27B and Gemma 4 12B. Qwen learner: -2.45% vs fixed K=3. Gemma: +7.74% vs fixed K=5, just +0.43% vs native adaptive. 64 scored native runs, 768 turns. No serving change. github.com/avifenesh/memra/pull/569

## LinkedIn draft

We measured a small depth controller while keeping the full vocabulary head and model weights fixed. Across 64 scored native runs and 768 turns, the Qwen3.8-27B controller reached 94.377 output tok/s versus 96.750 for fixed K=3, a 2.45% loss. On Gemma 4 12B, it reached 188.889 versus 175.325 for fixed K=5, a 7.74% gain. Gemma's native adaptive policy already reached 188.080 tok/s, so learning added only 0.43% against that control.

The paired audit retains exact artifact and runner identities, greedy identity gates, interrupted-set exclusions, and the raw receipts. These are complete native request rates for the frozen workload, not HTTP serving throughput. We are not changing a serving default. Results and receipts: github.com/avifenesh/memra/pull/569

## Lab blog candidate

Title: When an MTP depth learner loses to a fixed setting

Lead with Qwen's 2.45% loss and Gemma's 7.74% gain against the specified fixed controls. Show the separate comparison with native adaptation, the full-head and cold-prime controls, and the paired-set exclusion rule. End with the open test: calibrate the best fixed depth on one split and compare it with a cheaper learner on another. Link the 22 sealed archives and the exact source reconstruction from the PR.

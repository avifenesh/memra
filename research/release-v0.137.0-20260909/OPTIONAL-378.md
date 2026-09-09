# Prepared alternate, applied after owner go

Applied 2026-09-09 after #378 merged at `ff0937dd0`. The text below is the historical decision recipe; current membership and receipts are in GO-QUALIFICATION.md.

Not release membership or an enablement decision. Draft source inspected: `66b79c20ebc0422c87b6218e0e9719ff3b755416`.

Append to the README release entry and ledger only after reviewed merge: "GLM TP-2 gains an opt-in GPU sampler (#378, `MEMRA_GLM5_TP_DEVICE_SAMPLE`, default OFF, decide-by 2026-09-22). The final composition requires its exact-source correctness and sampled pair receipts; no performance claim is inferred from the draft."

Changelog alternate: "GLM TP-2 GPU sampling is available behind the default-OFF device-sampler door. The host sampler remains the default and rollback path."

On go: record the actual merge SHA and all five pair results, inspect the merged flag row and both dispatch arms, rebase, rerun affected suites and final-composition pair gates, and replace pending wording with receipt-backed scope. On no-go: omit this text from the release notes and leave #378 to a later release.

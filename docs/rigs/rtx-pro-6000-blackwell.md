# RTX PRO 6000 Blackwell

| | Recommended use |
|---|---|
| **Role** | Primary tuned and final-qualification target |
| **Build** | `sm_120a` |
| **Best fit** | Large single-card models, long context, larger prefix caches, qualified PP-2 pairs, and experimental PP-3/PP-4 work |
| **Start with** | [Cookbook](../COOKBOOK.md) configurations that name RTX PRO 6000 Blackwell |

Use this card when the model or context needs more memory than a 5090 offers, or when reproducing
a PRO-class receipt. Do not copy a 5090 default onto this card without its own measurement.
The full-power Workstation Edition has serving defaults of its own, keyed on the device name
([list](../SERVING.md#serving-defaults-keyed-to-the-rtx-pro-6000-workstation-edition-2026-11-04));
the Max-Q and Server editions keep the generic program until they carry their own receipts. The
Max-Q carries its own for the direct handoff export (2026-11-05,
`research/spill-f-20260919/pro-maxq-owed18/`).

See [models](../MODELS.md), [performance](../PERFORMANCE.md), and
[testing](../TESTING.md#target-aware-release-evidence).
The 2–4 card topology policy is in
[the multi-card decision](../decisions/PRO6000-MULTICARD.md); only PP-2 is currently qualified.

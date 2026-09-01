# RTX PRO 6000 Blackwell

| | Recommended use |
|---|---|
| **Role** | Primary tuned and final-qualification target |
| **Build** | `sm_120a` |
| **Best fit** | Large single-card models, long context, larger prefix caches, qualified PP-2 pairs, and experimental PP-3/PP-4 work |
| **Start with** | [Cookbook](../COOKBOOK.md) configurations that name RTX PRO 6000 Blackwell |

Use this card when the model or context needs more memory than a 5090 offers, or when reproducing
a PRO-class receipt. Do not copy a 5090 default onto this card without its own measurement.

See [models](../MODELS.md), [performance](../PERFORMANCE.md), and
[testing](../TESTING.md#target-aware-release-evidence).
The 2–4 card topology policy is in
[the multi-card decision](../decisions/PRO6000-MULTICARD.md); only PP-2 is currently qualified.

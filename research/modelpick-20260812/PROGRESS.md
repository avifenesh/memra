# cx-modelpick progress — 2026-08-12

## Scope locked

- Decision: rank three two-model serving configurations for one 2x RTX PRO 6000 pair.
- Evidence: live OpenRouter activity, apps, provider share/error/uptime, and effective cached pricing for every scored candidate.
- Fit: one model per 96 GB card or one PP-2 model across 192 GB; GGUF and the README support table govern implementation risk.
- Output: one-page `REPORT.md`; docs-only; no merge, tag, push, formatting pass, or runtime work.

## Current state

- Branch/worktree verified: `lane/cx-modelpick` in `/home/avifenesh/projects/wt-cx-modelpick`.
- Steering read from `~/.lanectl/inbox/cx-modelpick.md`, including the Step-3.7-Flash control economics and the revised agentic-demand/moat filter.
- Supported models verified from the generated README table.
- `PROGRESS.md` was created first and committed as `2950ff291` before the report work.
- Live candidate collection is complete across the requested Qwen, KAT, Gemma, GLM-Air, Hy3/REAP, and <=2-provider gap sets.
- `REPORT.md` ranks all three requested hardware configurations and selects Qwen3.6-35B-A3B plus Qwen3.6-27B, one per card.
- Report source URLs returned HTTP 200, local evidence links resolve, and the staged diff passes `git diff --check`.
- No runtime, generated perf surface, merge, tag, push, or formatting action was performed.

## Decision checkpoint

- Planning result: Q35 at 0.5% of its effective-price pool plus Q27 at 1.0% gives about $36/day gross.
- Those shares are deliberately below the smallest incumbent shares visible today, and both models already have memra architecture/drafter support.
- Clean few-provider gaps lost after capacity and effective-price math: Nemotron is about $5/day at 2% share; Ling's healthy leader owns 98.82%; Inkling is PP-2/new-arch; no public checkpoint was found for the exact KAT Air/Pro entries; vanilla Hy3 is Tencent-captive and REAP has no inherited route volume.
- Step plus Q27 remains viable only with idle-window arbitration; the measured active-contention result prevents additive revenue treatment.

## Evidence rules

- Record retrieval method and observation time for each OpenRouter number.
- Treat Step's owner-supplied page capture as the control, not as evidence that a fourth provider can win share.
- Use effective prompt pricing under the observed cache mix, not headline input price.
- Label estimates and do not infer provider failure causes beyond the visible metrics.

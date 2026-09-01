# Axis-B load-ceiling survey progress

- Status: complete
- Branch: `lane/cx-loadsurvey`
- Baseline: `35b285c9124f0899bcfeb4f6d010cb6ad75e3404`
- Deliverable: `research/loadsurvey-20260811/CEILINGS.md`
- Scope: read-only analysis of the existing admission/backpressure machinery and committed serve-stress receipts.
- Guardrails: documentation only; no code, GPU use, measurement runs, merge, tag, push, formatting pass, or perf-board changes.
- Evidence rule: mechanism claims will cite repository file and line; ceiling projections will be labeled as deductions or `needs-measurement` rather than reported as measurements.

Completed: traced admit-wait, step-OOM park/requeue, right-sizing, session and reserve bounds; reconciled them with the committed c=64, current-target c=1/4/8, and 262k capacity receipts in `CEILINGS.md`. No GPU work or new measurement was performed.

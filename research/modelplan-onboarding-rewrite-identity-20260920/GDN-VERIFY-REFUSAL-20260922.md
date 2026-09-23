# GDN teacher-forced verify refusal — #542

The original receipt identity criterion requires a receipt to authorize the actual
numerical program. At `07aa3af1677375a25db55f8d8098d0ced4a4284b`, the newly declared
`native-gdn-eager` could authorize public `decode_step_t` and its device/auxiliary
variants using only `DecodeEager`. These calls enter the batched GDN verify program,
whereas ordinary Eager uses `decode_step_h`. The existing `batched_serving_numeric_class`
comment records the historical distinction; this change does not invent a new numeric
failure measurement or claim that the programs have become equivalent.

Strict GDN verification now refuses before numerical/cache work in the public
entrances and the shared private funnel (including checkpoint/device-token/PP callers).
The private funnel also establishes a protected boundary, so an absent outer scope
cannot treat a stale qualified model as legacy. No Batch or Spec receipt is borrowed
to establish unmeasured equivalence. Ordinary GDN Eager, Legacy GDN verification,
non-GDN verification, Gemma arithmetic, default flags and receipt formats are unchanged.
Strict GDN Spec paths needing this verify program therefore remain explicitly refused;
a future narrowly qualified numerical program must address that dependency.

This is an admission correction for #542. The same-server bootstrap, non-circular
oracle, complete state budgets and later genuine GDN Eager/Spec qualification belong
to backlog #625; this patch implements none of that framework. `07aa` remains a
source-only declaration freeze, never a native qualification result. All `655` and
Gemma prerequisite failures remain preserved.

The CPU harness compiles the complete public `decode_step_t`, the refusal and the
canonical GDN predicate from actual source, plus the actual prefixes of private core,
device and auxiliary entrances. CUDA/math and prior identity guards are explicit
stand-ins. It checks refusal with unchanged cache/work counters at 0/1/4/15/16/17 rows
and positions 0/11, retains Legacy GDN and qualified generic/Gemma routing, and retains
missing-Eager refusal. Frozen `07aa` fails the new refusal control (2 pass, 1 fail);
the corrected source passes all three controls. This is boundary evidence, not native
arithmetic or positive qualification. Exact raw/generated/source hashes are retained
in `gdn-verify-refusal-20260922/`.

Current-source native caller and supported-program regressions remain required on
the final coordinator-selected composition. Old native receipts cannot qualify its
new ELF. No new remote job or broad main intake is part of this change.

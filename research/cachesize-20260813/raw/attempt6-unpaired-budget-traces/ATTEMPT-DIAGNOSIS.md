# Excluded attempt 6 — access traces were not paired by budget

This attempt is excluded from scoring. The first expanded-grid Q27 1,024 MiB boot was still in
progress when the control audit found that `working_set_cycle_seed` included the cache budget.
The arms had the same uniform distribution but different concrete access orders, so budget was
not the only changed variable.

The owned sweep timeout process was terminated and the fail-closed runner cleaned up its GPU0
server and samplers. GPU0 returned to 0 MiB, memra's ports cleared, and the lock released. At
that boundary an unrelated `cx-lcprestore` workload appeared on GPU1; its two process ids, memory
use, and the shared `PIX` topology are captured in `post-cleanup-unrelated-gpu1.log` and it was
left untouched.

The corrected seed depends only on model and repetition. For a given model/repetition, all six
budgets now receive the identical concurrency order, hit/cold role sequence, and working-key
permutation. No row from this attempt is used in the scored reduction.

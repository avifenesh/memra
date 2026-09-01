# Cross-lane note: the "tip binary carrying MEMRA_STEP_GEMM_PRIME_SUFFIX" does not carry it

The coordinator relayed that the affinity lane "built a tip binary carrying your
MEMRA_STEP_GEMM_PRIME_SUFFIX (`4306aecc8d`, md5 `8fd674c47d63`)" and queued a
re-confirmation of cold-vs-reused byte identity on it, offered as an independent check of
this lane's hoist.

**It is not one.** Measured on the box, 2026-08-28:

```
/root/memra-server.tip      md5=8fd674c47d63  strings MEMRA_STEP_GEMM_PRIME_SUFFIX = 0
/root/memra-server.tip      md5=8fd674c47d63  strings '[gemm-prime] WALK'           = 0
/root/memra-server.gsuffix  md5=0216d7011fb3  strings '[gemm-prime] WALK'           = 1
/root/memra-affinity/target/release/memra-server md5=8fd674c47d63 (same binary)
```

and in the repo:

```
git log --all -S 'MEMRA_STEP_GEMM_PRIME_SUFFIX'          -> no commit
git show 4306aecc8d:crates/memra-engine/src/hybrid_forward.rs | grep -c MEMRA_STEP_GEMM_PRIME_SUFFIX -> 0
```

The hoist has never been committed; it lives as a working-tree change here and as
`/root/gemmsuffix.patch` on the box (sha256 `8d1ca90e...`). `4306aecc8d` is
"MEMRA_ROWS_TAB_RESTAGE: the acceptance gate passes on the default arm" and predates it.

Consequence: the affinity lane's re-confirmation exercises the PRE-hoist walk suffix path.
Its 5/5 MATCH with 5 distinct cold shas is a strong prior for session-affinity restore
correctness at base 1440 / suffix 254 — and it is exactly the geometry where the chunk-local
`seq_end` defect lives — but it says nothing about the batched suffix arm, because that arm
is absent from the binary under test. This is the same receipt-defect class the lane has been
bitten by twice: a counter that fires only on the arm the workload does not take.

So LEG S in this lane's battery remains the only identity evidence for the hoist, and it was
not trimmed. To make their run an actual independent check, they would need to rebuild with
`/root/gemmsuffix.patch` applied and confirm `strings ... MEMRA_STEP_GEMM_PRIME_SUFFIX = 1`
plus a non-zero `[gemm-prime] ENGAGED ... base=<nonzero>` count in the server log.

## What is already on the box for them

`/root/memra-server.gsuffix` (md5 `0216d7011fb3`) IS a binary carrying the hoist: built from
`e3faf5a17c` plus `/root/gemmsuffix.patch`, receipts in `raw/gs-build.txt`, and its
fingerprint shows `MEMRA_STEP_GEMM_PRIME_SUFFIX = 1` and `[gemm-prime] WALK = 1`. It is two
commits behind the tip they used (`e3faf5a17c` vs `4306aecc8d`), and neither of those two
commits touches the prime path, so it is a usable arm for an identity re-confirmation without
a rebuild. Whatever binary they use, the check is only meaningful with a non-zero
`[gemm-prime] ENGAGED ... base=<nonzero>` count in the server log for the turns under test:
that line is what distinguishes a suffix that rode the batched prime from one that took the
walk, and its absence is what makes the current run silently pre-hoist.

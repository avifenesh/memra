# OWED 12: M1 B3 cell runner (2026-09-25)

**Landed CPU-verified against a stub; no GPU visit has run.** Tool `m1-spill-runner.py`
(`run`, `reparse`), stub `m1-stub-run-gen.py` (prints `run-gen`'s real line shapes, never a
measurement), tests `test-m1-spill-runner.py`. Registration: `CPU-PREREG.md` OWED 12 and
`M1-PREREG.md` B3; arms, order, prompt and verdict rule come from `m1-prereg/b3-arms.lock.json`.

## What a visit does

1. Re-check the proof identity triple (PRIVATE receipt, raw filesystem id) on the artifact's
   directory; any change refuses the run.
2. Apply the regime through `m1-cache-regime.py` (cold: DONTNEED plus `mincore`=0; warm: full read
   plus `mincore`=all; bounded: cold per visit under one mlocked balloon held for the whole run).
3. Start the child with the arm's env (common env, arm env, `unset_env` removed), stdout and stderr
   written straight to `run.log` (never a pipe), then the 250 ms sampler with the child's pid on
   the proof's leaf set plus top device.
4. Reap with `wait4`: the child's read/write bytes come from `ru_inblock`/`ru_oublock` (Linux
   fills them from the task's I/O accounting; `/proc/<pid>/io` is unreadable for a zombie, found
   by this suite's first run), plus CPU times and max RSS.
5. Re-check identity, record residency, validate telemetry, compute the co-tenancy share
   (device sectors minus own bytes), check drive temperatures against their hwmon max, and
   parse only the saved log.

Correctness is judged after the whole regime, against the byte oracle's tokens (the oracle may
run after an arm in round 1): gate `MATCH`, 128 tokens, token ids equal to the oracle's, the
positioned-read pool present at the arm's depth (absent for mmap arms), zero read errors and
short reads, and for `direct16` zero fallbacks and `overread_bytes == 4096 x reads`. An arm with
any failing visit is refused from timing. The verdict rule is the registered one.

## CPU gates (`owed12/test-runner.log`)

```text
Ran 7 tests in 134.186s
OK
```

| Test | What it proves |
|---|---|
| `test_full_registered_dry_run` | 10 rounds x 6 arms = 60 visits; odd rounds forward, even reversed; every pair meets 5 times in each relative order; all visits scored; stub rates 10 / 9 / 10.2 / 8 / 11 / 12 tok/s give loser, flat, loser, winner, winner against `worker16`; `wait4` read bytes cover the stub's 1 MiB read; telemetry valid on every visit; `reparse` PASS, then a tampered log gives `REPARSE MISMATCH`, exit 3 |
| `test_corrupt_arm_and_bad_overread_are_refused` | one changed token refuses `worker2`; a misreported over-read refuses `direct16`; the other arms still get verdicts |
| `test_identity_mismatch_refuses_before_any_visit` | a proof whose mount id differs refuses before any visit directory exists |
| `test_stub_bypass_refuses_a_real_binary_name` | `--stub-no-lock` refuses any binary not named `m1-stub*`; without it `--lock-fd` is required |
| `test_bounded_regime_holds_and_releases_a_balloon` | balloon LOCKED for the run and RELEASED after it |
| `VerdictRule.test_rule`, `test_contamination_edges` | 4 of 5 per order is enough, 3 of 5 is flat; 1.03 median is flat; fewer than 4 pairs per order is insufficient; more than 2 contaminated visits of an arm leaves the regime unscored |

The tests use `--contamination-limit 1.0` because this rig is shared; a real run refuses any
limit other than the registered 2% and any round count below 10.

## Box command shape (after the proof PASS; the collector holds the lock)

```sh
python3 tools/tier-battery.py --rig pro-single --timeout 7200 --external-lock --out "$R/b3-cold" \
  --execute python3 research/spill-f-20260919/m1-spill-runner.py run \
  --arms-lock research/spill-f-20260919/m1-prereg/b3-arms.lock.json --regime cold \
  --binary target/release/run-gen --artifact /scratch/spill-f/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf \
  --proof "$PRIV/m1-proof.json" --out "$R/b3-cold/visits" --rig pro-single --lock-fd @COLLECTOR_LOCK_FD@
```

Then `--regime warm` and `--regime bounded` as separate collector cells. `run-spec` K=1..8 for
`worker16` and `direct16` is its own collector cell. `direct16` enters only after OWED 7's
pinned-pool GPU test passes on the box.

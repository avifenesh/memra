# WP-B day 39: O5, the prime workspace the admission door books, measured against what a burst actually allocates

OWED.md O5. Under `MEMRA_ADMIT_BY_MEMORY` every headroom reading of the admission block is reduced by `pending_prime`,
the prime workspace still-priming sessions owe (memra#680, day 33). It is the sum over those sessions of the whole
`prefill_workspace_bytes(rows)`: `call_row_bytes x min(rows, chunk) + prompt_row_bytes x rows`. DAY24 section 2 (ii)
already named the shape this over-books: the `call_row_bytes` term is the per-call slab, which is ONE retained
per-device pool shared by every prime (`prime_slabs_get`, grow-only at the largest call), and on the batched worker
the prime calls of a tick run one after another. On the 27B that term is most of a session's 658 MB at a 1.3k prompt
(DAY33 1.1), so a 64-request burst books it 64 times. The door's ON concurrency rows (DAY36, and O3's rerun) are read
through this term, so it is corrected first, as the door's own booking measured at its best; O3's rerun then runs on
the final booking (the reason O3 follows this day, OWED.md).

Every cell is `executed-not-qualified`; the door stays OFF; no default moves; nothing outside the armed door changes.

## 1. Pre-registration

Committed and pushed before any day-39 code and before any day-39 boot. Nothing in section 1 changes after a number
is seen.

### 1.1 What a burst's primes allocate (from the code on the lane tip)

- **The slab**: `HybridModel::prime_slabs_get(t)` keeps one slab set per device, grown when a call needs more than
  its `t_cap` rows and never shrunk; every prime call (the plain prefill tick's and the MTP walker's trunk chunks)
  uses it in turn. Its owed growth is `slab_row_bytes x max(0, rows_needed - t_cap)`, once per device, where
  `rows_needed` is the largest call any still-priming session will make.
- **The per-call transients** beside the slab (the f16 activation, the pre-norm operand, the `x` and `hn` copies, the
  call's returned `[t, n_embd]` output): allocated per call from the pool and freed after it, so at most one call's
  worth is live at a time.
- **The MTP walker's whole-prompt hiddens stack** (`MtpPrimeState::hiddens`, `tp x n_embd` f32): allocated when the
  walker starts and live until the prime is consumed. Under the cooperative prime (`MEMRA_PRIME_YIELD`, default ON)
  a walker persists across ticks, so every started walker holds its stack at once. A session whose walker has not
  started yet owes it.

### 1.2 The corrected term

Under the door only (unarmed, both terms are 0 as today):

```text
pending_prime' = max(0, (call_row_bytes + prompt_row_bytes) x rows_needed - slab_bytes_resident)   one call, once
               + sum over cooperative spec sessions whose walker has not started of prompt_row_bytes x P
               + max over non-cooperative spec sessions still priming of prompt_row_bytes x P    (a stack for one call)
rows_needed    = max over still-priming sessions of min(rows_remaining, chunk_rows(P))
```

`slab_bytes_resident` is the slab set's allocated bytes on the primary device, read from the engine through a new
accessor (`HybridModel::prime_slab_bytes`). `call_row_bytes` and `prompt_row_bytes` stay the admission shape's own
numbers, so the one-call term keeps the call's extras and its returned rows. A started walker's stack is live and
already in the reading, so it books nothing; a non-cooperative walker lives for one burst call, so only the largest
one is owed. The seed term
(`pending_seed`, capped, day 35 and #705) is unchanged. The admit, defer and refuse lines keep their fields;
`pending_prime=` prints the corrected value, and a new `pending_prime_v1=` field prints day 33's value beside it, so
every line shows both.

### 1.3 Cells, arms, binaries

The day-35 cell unchanged (`day33-client.py` on the day-33 environment, `day33-run.sh` shapes), with the red and green
binaries of this day: `red` = the lane tip at the first boot minus this day's change (the day-33/35 booking), `green`
= the tip. One detached-worktree build each per card.

| shape | door | client args | card |
|---|---|---|---|
| `G2` | ON, open output 8192 | `--chars 5000 --skip-long --burst 64` | 5090 |
| `L64` | ON, open output 32768 | `--chars 5000 --skip-long --burst 64` | 5090 |
| `R64` | ON, open output 32768 | `--chars 5000 --skip-long --burst 64` | target card |
| `off` | removed | `--chars 5000 --skip-long --burst 0` | both |

5090 boots: `red-G2`, `red-L64`, `green-G2-r1`, `green-L64-r1`, `red-off`, `green-off`, `green-G2-r2`, `green-L64-r2`.
Target card: `red-R64`, `red-off`, `green-R64-r1`, `green-off`, `green-R64-r2`. The admission gate
`tools/admit-mem-burst-gate.sh` runs on green at its defaults once more at the end of the 5090 half.

### 1.4 Acceptance, per card

`day33-compare.py` unchanged (sha256 `4493164c...381f10`), its terms as DAY35.md 1.4 reads them: G-NOOM (0
`CUDA_ERROR_OUT_OF_MEMORY` lines, parks included, and 0 503s) and G-BOOK on every green burst boot, V-ID-FIX (green
ON against red ON, sequential rows), V-ID (green ON against green OFF) and V-OFF (green OFF against red OFF), card
verdict GREEN. The admission gate reads ALL GREEN on green.

Readings, no bound: the burst's 200 and 429 counts red against green per shape, and `pending_prime` against
`pending_prime_v1` at the burst's peak.

### 1.5 What a failure means

A green OOM line means the corrected term misses allocation that day 33's over-count was covering. The line and the
`nvidia-smi` state are quoted, the missing term is diagnosed from the log, and a revision is a new addendum; day 33's
term stays until one passes.

### 1.6 CPU

Unit tests: the slab owed growth (0 when the slab covers the need, the difference otherwise), one call counted once
over N sessions, a started walker's stack not booked, an unstarted one booked, the identity when unarmed, and the
admit line rendering both fields. The existing day-33 and day-35 tests stay green.

### 1.7 Addendum A (2026-09-24, before any day-39 boot): the binaries

The lane tip at the first boot is `c6f9282c2` (DAY37 addendum E's r4, after this day's `6262506fc`). Green is its
`memra-server`, the same file as DAY37's r4 lane binary (`580fe677...`); red is the same commit plus
`day39-red.patch`, the reversed crates diff of `6262506fc` (`0cef991c...`; the red binary carries no
`pending_prime_v1=` string, green does). This cell runs the pooled allocator, which addendum E does not touch. The
first builds from `6262506fc` never ran and are deleted. The target-card chain builds with the same script, from the
same source, reusing the day-37 lane file as green when its source matches.

## 2. Results

Written after the runs. Section 1 is unchanged.

# WP-B day 35: memra#680's remaining term, the post-prime prefix seed

Day 34's target-card part A (the memra#680 PRO boots on BOX4) read NOT-GREEN on one term: under the day-33 fix, a
64-request burst at open output 32768 still produced three prefill CUDA OOMs. The day-33 prefill-OOM park caught all
three, and no request answered 503 (DAY34.md 2). The lead's call: book that term under the door, pre-registered
before code. This day is authorized to change engine and server code under the default-OFF door only; door OFF stays
byte-identical. Every cell is `executed-not-qualified`; no default moves.

## 1. Pre-registration

Committed and pushed before any day-35 code and before any day-35 boot.

### 1.1 Diagnosis (from the day-34 receipts and code at main `25bbb91f5`)

- The receipt: `pro-single-day34/box/p680/boots/green-R64/server.log`. From the burst's first primes, each finished
  session seeds a prefix entry (`[prefix-cache] insert (seed): 1376 tokens, 197.8MB`, then 1344 tokens at 196.8 MB).
  The resident prefix cache grows from 7086.9 MB to 8861.0 MB in about 1.4 s, and the next three prefills OOM
  (lines 746 to 748, `[admit-mem] prefill OOM parked session back to queue ... CUDA_ERROR_OUT_OF_MEMORY`). The step-OOM
  reclaim then evicts 23 prefix entries (4316 MB), and the parked sessions are admitted again.
- The code. The day-33 booking (`pending_prime_bytes`, `worker.rs` at `25bbb91f5`) counts each still-priming
  session's `prefill_workspace_bytes(queued rows)` and nothing else. The seed entry is sized by
  `prefix_snapshot_bytes(cache)` (20236): KV rows times per-token bytes, plus the recurrent conv and SSM state in f32,
  plus any latent planes. It is allocated when the prime completes (`maybe_prefix_seed`, 20432, then
  `prefix_insert_from_session`, 20281), after the session was admitted. A session is armed to seed when
  `seed_prefix` is true and `seed_at` names its grid boundary; `maybe_prefix_seed` clears `seed_prefix` whether the
  insert lands, is refused or is skipped (20442). The admission gate sees the entry only once it exists, so a burst
  still overcommits by the seeds of every session admitted before its prime completed.
- On the 27B an entry is about 197 MB for about 1360 tokens, about 155 MB of it recurrent state. On the 9B it is
  72.2 MB for 1312 tokens.

### 1.2 The fix under test: option (a), argued

The two options the lead named:

- **(a) Book the seed.** Each still-priming session's future seed-entry bytes are counted in the booked reduction
  until the insert lands.
- **(b) Gate the seed.** The insert is deferred or skipped when booked headroom is short.

I choose (a). Under (b) the prefix cache's contents would depend on the load at the moment each prime completes. A
later request's route (a prefix restore or a cold prime, a known pair under the one-numeric-program rule) would then
depend on burst timing, and so would the numeric program it runs. Under (a) the cache holds exactly the entries it
holds without the door; only admission timing moves, and admission timing already moves under the door. (a) is also
the same mechanism as the day-33 workspace term, one more addend.

Under the door only:

- `pending_seed`: the sum, over active sessions with `seed_prefix`, `seed_at = Some(b)`, no vision input and a live
  cache, of the entry size at `b` rows. The size uses `prefix_snapshot_bytes`'s own arithmetic with the row count set
  to `b`: KV layers at `b` rows, the recurrent state as allocated, latent planes at `b` rows as an upper bound. Each
  session's term disappears when `maybe_prefix_seed` clears `seed_prefix`.
- Every headroom reading of the admission block is reduced by `pending_prime + pending_seed`.
- The admit, defer and refuse lines carry `pending_seed=`, and `device_free` is the reading after both terms.
- Unarmed, both terms are 0 and the reduction is the identity. No `[admit-mem] id=` line prints, and
  `prefix_snapshot_bytes` itself is not changed.
- CPU unit tests, one per arm (1.6).

### 1.3 Shapes, arms, binaries

Workload `day33-client.py` on the day-33 environment (DAY33.md 1.3). Binaries: `red` = main `25bbb91f5` (the day-33
tree as merged, #692), `green` = the lane commit carrying the day-35 fix. Each is built once per card in a detached
worktree.

| shape | door | client args | card |
|---|---|---|---|
| `G2` | ON, open output 8192 | `--chars 5000 --skip-long --burst 64` | 5090 |
| `L64` | ON, open output 32768 | `--chars 5000 --skip-long --burst 64` | 5090 |
| `R64` | ON, open output 32768 | `--chars 5000 --skip-long --burst 64` | BOX4 |
| `off` | removed | `--chars 5000 --skip-long --burst 0` | both |

- **Local RTX 5090** (`/tmp/memra-5090.lock`, one boot per hold, bounded idle waits). Run `red-G2`, then `red-L64`.
  The local red shape is the first of the two whose red boot reads R-OOM RED (any `CUDA_ERROR_OUT_OF_MEMORY` line,
  park receipts included, or any 503). Then run `green-<shape>-r1`, `red-off`, `green-off` and `green-<shape>-r2`.
  If neither shape is red on the day-33 tree, run the same green boots on `G2`, and section 2 records that the 5090
  has no local red for this term.
- **BOX4** (`/tmp/memra-gpu.lock` on the box, only after lane A's `LANE-A-BOX4-DONE`): `red-R64`, `red-off`,
  `green-R64-r1`, `green-off`, `green-R64-r2`.
- **The gate** `tools/admit-mem-burst-gate.sh` already counts every OOM line, parks included. If the local red shape
  is `L64`, its defaults move to `L64` (open output 32768, burst 64). It must fail once on `red` and pass twice on
  `green` at its defaults before the move is committed.

### 1.4 Readings and verdicts

`day33-compare.py` unchanged (sha256 `4493164c202e71ed091be48496518dbbea1f8f4942a916c85fb3087dc9381f10`), with the
DAY33.md 1.6 terms: R-OOM, G-NOOM, G-BOOK, V-ID-FIX, V-ID, V-OFF and the card verdict. G-NOOM counts every
`CUDA_ERROR_OUT_OF_MEMORY` line, park receipts included, as the acceptance requires. Per card, `SUMMARY.txt` and
`FAULTS.txt` (`day31-faults.py`).

Acceptance, per card: R-OOM RED on the red boot of the card's shape (for the 5090, only if a local red exists);
G-NOOM, G-BOOK, V-ID-FIX, V-ID and V-OFF PASS on both green runs; card verdict GREEN.

### 1.5 Expected readings (not rules)

- BOX4: `red-R64` shows three or more OOM lines (the three parks of day 34 part A's `green-R64`, or more). Both
  green runs show zero, with the booked seed turning those three admissions into defers or typed 429s.
- 5090: on the day-33 tree G2 read zero OOM lines four times (DAY33.md 2.3), so `red-G2` is expected NOT-RED.
  `red-L64` is uncertain: the 9B's entries are 72 MB and its card is 32 GB.

### 1.6 Unit tests (CPU)

- The seed rows owed: `Some(b)` while armed with a live cache and no vision; 0 once `seed_prefix` is cleared, when
  `seed_at` is `None`, or with vision.
- The entry size at `b` rows from layer descriptors: KV rows times per-token bytes, the recurrent state counted once.
- The booked reading with both terms: admits when the need fits, defers when the seed term alone makes it short (the
  day-34 shape), and is the identity at zero.
- The receipt renders `pending_seed=`.
- The existing day-33 tests stay green.

### 1.7 After day 35

The clean door rerun on the day-35 tree, both cards, is a fresh pre-registration: day 34's with its binary updated.
That is the owner's decision cell.

## 2. Results

Written after the runs. Section 1 is unchanged.

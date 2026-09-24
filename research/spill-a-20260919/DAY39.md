# WP-A day 39: OWED item 3, the same-tick fill (design F) on slower CPUs

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`. Rig: the local RTX 5090 Laptop GPU's host (Intel Core Ultra 9
275HX) for the survey's reading; the deciding cells need a slower-CPU host with an RTX PRO 6000 and a 9950X-class
host (the lead's rentals). Every cell `executed-not-qualified`.

## 1. The survey, pre-registered before any cell or code

**The item** (`DAY34.md` section 7 and finding 2; `DAY35.md` section 4; rulings 49, 52, 53). Design F fills the
promote's pinned staging from the resident planes (`Arc<Vec<f32>>`) in ONE host function on the copy stream,
single-threaded (`SpanFillTask::run`), then the 96 span copies follow on the copy stream. On BOX4's CPU the fill of the
27B's 156.9 MB took 11.4 ms (`156.9MB filled by the hash helper in 11.4ms`, the day-32 helper's single-thread fill of
the same bytes), and the fill plus the spans outlast the 13.1 ms from the probe's submission to the next tick top
(`polls [2] (counts [90])`), so the promote lands one tick later and F reads flat there (`promote in` 28.40 against
28.40). On the 5090 host F lands inside the probe's tick and wins one tick per promote (DAY35 section 4).

**The candidates.**

- **T, a threaded fill.** The host function splits the fill's bytes across T threads (each a contiguous share of the
  planes, the same `copy_nonoverlapping` program, the same bytes), joined before it returns; T fixed per host.
- **O, an overlapped fill.** The fill runs per plane on a separate fill stream (one host function per plane group), and
  each span copy on the copy stream waits on its own group's fill event, so the copies of early groups overlap the fill
  of later ones: the copy stream's work ends near max(fill, copies) instead of fill + copies.
- **T with O.**

**The survey cell (CPU only; a reading on this host now, the deciding reading on each rented host in its sitting).** A
detached probe (`day39-fill-survey/`): the 27B's staging shapes exactly (48 buffers of 3,145,728 B and 48 of 122,880 B,
156,893,184 B, the day-31 fill line's bytes), cached pinned destinations (`cuMemHostAlloc` flags 0, the staging set's
kind), heap `Vec<f32>` sources already touched; the fill timed at T = 1, 2, 4, 8 and 12 threads (scoped threads, each
a contiguous share of the byte list), N=5 each after one warm run, plus the host's memcpy bandwidth reading at T = 1.
The 9B's shapes (48 buffers, 52,690,944 B) the same way. Bitwise: every destination compared with its source after
each run. With `--gpu` (a card present, under its lock) the probe also times the span copies themselves: every staging
buffer to a device buffer on one stream, bracketed by events, N=5, the value the rule below subtracts.

**What the survey decides, stated now.** Nothing about slower hosts from this host: it prices the scaling on this CPU
and checks the probe. On each rented host the same probe is its first cell, and the design is chosen there by this
rule: T alone if some T at or below the host's physical cores fills the 27B's 156.9 MB in at most 13.1 ms minus the
host's measured span-copy time minus 1.0 ms of margin (the copy-stream work then ends before the next tick top on the
day-34 timeline); O with T if no T does. The chosen design is then pre-registered with its acceptance (the promote's
copy landing in the probe's tick on the slower host, `polls [1]` on at least 80 of 90 steady promotes, and F's day-35
e2e margin against the day-32 helper fill there) before its code.

**Budget.** The survey 0.05 agent-day here.

## 2. The survey on this host, as it ran (`rtx5090-day39/survey/`): a reading, deciding nothing for slower hosts

- Probe `day39-fill-survey/` (hash in `survey/binary.sha256`), in the same 5090 hold as DAY38's unit rerun (16:45:19Z),
  `--gpu`. `available_parallelism=24`. Verbatim (`survey.log`), every run `bitwise=true`:
  - 27B (156,893,184 B): `threads=1 .. median=12.677` ms (12.38 GB/s), `threads=2 .. 8.979`, `threads=4 .. 6.494`,
    `threads=8 .. 5.929` (26.46 GB/s), `threads=12 .. 5.895`; `SPANS shape=27B .. median=5.621` ms (27.91 GB/s).
  - 9B (52,690,944 B): `threads=1 .. 3.738`, `threads=4 .. 1.871`, `threads=8 .. 1.619`, `threads=12 .. 1.666`;
    `SPANS shape=9B .. median=1.913`.
- Read against section 1's rule as if this host carried the 27B: T=1 needs 12.68 ms against a budget of 13.1 - 5.62 -
  1.0 = 6.48 ms, so the single-thread fill of design F would miss the probe's tick here too; T=4 (6.49 ms) sits at the
  bound and T=8 (5.93 ms) inside it. On the 9B this host needs no thread (3.74 ms against 13.1 - 1.91 - 1.0). Neither is
  the rented hosts' reading; the probe is their first cell.

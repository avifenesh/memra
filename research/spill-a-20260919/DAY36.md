# WP-A day 36: the D2D half's restore price cell (DAY33 section 6)

Lane `lane/spill-a-20260919`, worktree `wt-spill-a`, tip `50644beeb` (integ57, PR #711, its files not touched here).
Rig: the local RTX 5090 Laptop GPU; no target card is rented. Every cell `executed-not-qualified`. Every engine push in
the announced `MEMRA_RELEASE_QUALIFICATION_MODE=development` mode.

## 0. Resync

`git fetch`: the lane equals origin at `50644beeb`; `origin/main` `fa73d0e6c`; #711 is OPEN (in CI). Main is merged
before the first push after #711 lands, or before the day's last push.

## 1. Pre-registration (committed before any day-36 code)

**The cell is DAY33 section 6's, unchanged.** A log-only field, `recurrent copy H.HHms host, G.GGms owner stream`: the
host time around the restore's recurrent copy loop (`host_restore_submit`, step 1, the `copy_into` of every layer's
conv and ssm state and its `len` mirror, on the owner stream), and the owner-stream GPU time between two timing events
recorded on the owner stream around that loop. Read on the restore arm.

**The instrument, as it will be built.** The host time goes on the `restore submitted off the tick` line (`; recurrent
copy H.HHms host`). The GPU time is read without any host wait: the two events are kept with the pending restore, and
the `restore landed off the tick` line (at the request's re-admission, one tick or more later) reads their elapsed time
only if the end event is already complete (`; recurrent copy G.GGms owner stream`, else `; recurrent copy owner stream
pending`, counted). One named departure from DAY33's wording: the field splits over the two lines, because reading the
GPU time on the submitted line would need the owner thread to wait for the owner stream to drain, which is a change to
the thing being priced. An event that cannot be created prints `n/a` and never fails the restore. No new `MEMRA_*`
name, no dispatch change, no numeric program change.

**The run.** The day-26 restore arm (`pro-single-day26/restore-arm.sh`) byte for byte in its boot and harness:
`MEMRA_PREFIX_CACHE_MB=1024 MEMRA_KV_HOST_MB=8192 MEMRA_KV_HOST_CONTRACTS=1 MEMRA_SERVE_SPEC=0 MEMRA_CTX=8192
MEMRA_MAX_SESSIONS=4`, one door-ON boot, `stall_cell.py --mode restore --n 50` (100 timed restores), the receipt
replayed. On the 5090 with the 9B NVFP4 MTP artifact under one bounded hold of `/tmp/memra-5090.lock` (60 x 120 s,
never inside another lane's hold; no compute app and at least 20000 MiB free, bounded 15 x 60 s; a foreign compute app
is handled as on day 35: a bounded 60-minute wait outside the lock, and any amendment pre-registered before a boot).
250 ms card telemetry. Receipts in `rtx5090-day36/`.

**The reading.** Over every restore of the boot that carries the field: the host median and the owner-stream GPU median
(and min, max, N), the first restore of the boot also shown alone; the count of `pending` and `n/a`.

**The decision rule, DAY33's, verbatim:** "if on the target card the owner-stream GPU median is under 0.5 ms and the host
median under 0.5 ms per restore, the D2D half closes as not worth a door (the verdict and the receipt in
`OWNER-THREAD-OFFLOAD.md` and the verdicts ledger; no code). Otherwise a restore-recurrent design is pre-registered with
its own acceptance."

**What the 5090 reading can and cannot decide, stated now.** The rule is conditioned on the target card (RTX PRO 6000,
the 27B: 96 recurrent planes, 156.9 MB). The 5090 runs the 9B (48 planes, 52.7 MB) on a different memory system
and CPU. So the 5090 cell is a reading and decides neither branch. The verdict line reads `DAY36 PRICE READING (5090,
not the rule's card)`. Closing the half, or opening a design, needs either the target-card sitting of the same cell or
the lead's ruling that the 5090 reading decides. That choice is the lead's; this lane does not relax the clause.

**Predictions.** On the 9B: 24 recurrent layers (48 `copy_into`, 52.7 MB, the day-35 receipts' `conv 24 .. ssm 24`)
and 8 KV layers (8 `set_i32_one`). GPU: 52.7 MB read plus 52.7 MB written at about 0.8 to 0.9 TB/s effective on the
laptop card, about 0.12 ms. Host: 56 calls at a few microseconds each, about 0.2 to 0.4 ms.

**Budget.** 0.25 agent-day. The card is taken by bounded waits behind the other lanes.

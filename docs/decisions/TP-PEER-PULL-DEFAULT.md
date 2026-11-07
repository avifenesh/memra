# An armed TP walk moves its hops by peer pull on sm_100a (2026-11-04)

**Chosen:** with no transport name set, an armed TP walk (today the GLM TP loader,
`MEMRA_GLM5_TP`) uses `peer-pull` on sm_100a builds, except under the glm5 symmetric posture
(amended below): every cross-rank hop is a
`cuMemcpyDtoDAsync` issued on the reading rank's stream, ordered by one publication event per
rank, with no host leg (`tp_transport::default_transport_for_build`). Arming still grants peer
access both ways and runs the byte-integrity ladder; one differing word refuses the load.

**Rejected:** `host-canonical` as the sm_100a default, where every hop is a `dtoh` with a full
stream drain followed by an `htod`. It stays the default on every other build. Its explicit
spelling stays because the `glm5-tp-gate` pins it for its banked arms.

**Measurement that settled it:** the 2x B200 pair (NVLink), darklanes
`research/glm5-b200-mint-20260904/LANE.md`, same tape, 160 greedy tokens: host-canonical 16.48
against peer-pull 19.11 tok/s (+16%), and a second reading of 15.20 against 17.81 (+17.2%,
`VERDICT:glm5-tp2-peer-pull-still-loses`, whose "loses" is TP-2 against PP-2, not pull against
host). The transports are pure movement, so the arms are byte-identical by construction and by the
gate's transport-vs-transport arm. One later sentence in that log calls the 19.11 "worse" against
its own table; the memra row had copied it until the 2026-11-03 correction. The owner's 2026-11-04
ruling makes positive-receipt winners defaults keyed to the receipt's card.

**Amended the same day (the glm5 symmetric walk restore):** `device-push` became the transport
an sm_100a glm5 TP-2 load arms with no transport named when the symmetric posture is selected,
which is the default over the expert split there (`glm5_tp::unset_transport_for`,
`docs/decisions/GLM5-B200-DOORS.md`). Not on a pull-vs-push row, which still does not exist as a
balanced interleaved A/B (the one same-window reading, tpwalk 2026-09-06, one round per arm on the
root-orchestrated walk: host-canonical 16.59, peer-pull 19.07, device-push 20.49 tok/s), but
because the posture's one-shot crossings are kernel peer stores that only the `device-push`
ladder proves, and every posture receipt ran on it. `peer-pull` keeps the walk this record's
receipt measured: the root-orchestrated walk (`MEMRA_GLM5_TP_SYMMETRIC=0` or whole experts on
sm_100a). TP-2 as a whole stayed 2-3x slower than PP-2 on the root-orchestrated walk; this record
chooses the transport for a TP walk, not TP over PP.

**Scope:** one card class, keyed on the build as the other sm_100a defaults are. The RTX PRO 6000
PCIe pairs have no peer-pull receipt and keep `host-canonical`.

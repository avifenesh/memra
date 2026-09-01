# glm5-spec-ppn-gate CROSS-DEVICE twin — box arm (2026-08-30)

The lane's named final gate: the same 23-arm battery the rig ran same-device, re-fought
with real cross-device placements (peer transport + weight sharding live). Multi-card box,
4x RTX PRO 6000 Blackwell Workstation Edition, CUDA 12.8 (nvcc r12.8/35583870), cards
0/1/2 only (`CUDA_VISIBLE_DEVICES=0,1,2`; card 3 belongs to a co-tenant server and was
never touched), TF32 off, correctness-only — no timing number is read out of these logs.

## Rebuild attribution (the rebuild-after-checkout law)

- Lane head built ON the box: `50aee8879b5c9c113e5ff9d9fcc1379d6575cb05`
  ("ppn-verify lane doc: local-ci --perf receipt ..."), fetched from origin into a
  dedicated clone (the box's shared checkout untouched).
- `cargo build --release -p memra-engine --bin glm5-spec-ppn-gate`: **4m43s real**,
  exit=0 (not a failed-checkout 0.04s "Finished"); binary 54,069,568 B.
- strings probe: `strings glm5-spec-ppn-gate | grep -c "glm5-spec-ppn gate"` = 4 (the
  gate's own verdict literals — the binary IS this lane's gate, not a stale sibling).

## Verdict: ALL 4 PLACEMENTS GREEN — 23/23 arms each, all 3 reds bite

| placement | config line (from the log) | result |
|---|---|---|
| stages=2, `MEMRA_PP_DEVICES=0,1` | devices=0,1 splits=default(even) shard=per-stage | 23/23 PASS, reds 3/3 (`20-xdev-n2-dev01.log`) |
| stages=2, `MEMRA_PP_DEVICES=0,1 MEMRA_PP_SHARD=0` | shard=OFF(bring-up placement) | 23/23 PASS, reds 3/3 (`21-xdev-n2-dev01-shard0.log`) |
| stages=3, `MEMRA_PP_DEVICES=0,1,2` | devices=0,1,2 splits=default(even) | 23/23 PASS, reds 3/3 (`22-xdev-n3-dev012.log`) |
| stages=3, `MEMRA_PP_DEVICES=0,1,2 MEMRA_PP_SPLITS=1,3` (asym) | devices=0,1,2 splits=1,3 | 23/23 PASS, reds 3/3 (`23-xdev-n3-dev012-asym.log`) |

Arms per invocation: W0 plain-ppN-decode re-pin, W1 verify-walk row bit-identity vs the
door-OFF chain, A accept-j j=0..7, E e2e tapes K=1..7 natural + forced-accept K=3/7 +
forced-rejection sweep, R1 stale-KDA red, R2 pool-key tripwire red (bites by name),
R3 rollback-disabled red. Non-vacuity asserted per invocation (hc topology, door open at
the requested stage count, Recurrent/LatentKvCache state classes split across stages —
i.e. across DEVICES here).

ON THE SPLITS=15,30 SERVING ARM, named: the literal cuts 15,30 exist only at 45 trunk
layers (the real artifact); the 4-layer fixture harness cannot take them. The serving
shape's CLASS is covered by the two stages=3 three-device arms above (even + asym
fences); the literal-cut twin belongs to the real-artifact battery (prerequisite 2),
which runs the deployed placement itself.

Timing-marker discipline: `/root/TIMING-IN-FLIGHT` checked before every arm (the runner
refuses to start an arm with a marker up); no marker existed at any point in the window.
Box left clean: build clone removed, cards 0/1/2 at 0 MiB, window START/DONE lines in the
box queue log.

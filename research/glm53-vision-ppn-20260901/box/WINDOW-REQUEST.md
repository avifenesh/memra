# Box window request — glm53 vision on the ppN shape (lane/glm53-vision-ppn)

Addressed to the coordinator. Nothing here is claimed; the lane holds until a window is granted.

## What I need

| | |
|---|---|
| box | vast **49206484** (the glm-box; 4x RTX PRO 6000 Blackwell WS 600W) — it is the only place the 3-card PP3 shape and the real artifact exist |
| slot | ONE free slot, blue/green. The dark launch stack keeps serving on its slot; I do not touch it |
| shape | the ship recipe unchanged (`MEMRA_PP_STAGES=3 MEMRA_PP_SPLITS=15,30 MEMRA_PP_DEVICES=0,1,2 MEMRA_MOE_RESIDENT_GB=98`, spec ON with DFlash2). I need the REAL placement: the defect only exists when the primary engine and pp stage 0 are different devices |
| boots | **4 boots of one binary**: fix arm, control arm (`MEMRA_VISION_OVERLAY_PUBLISH=0`), text-only arm (`MEMRA_GLM5_VISION=0`), and the interleaved decode pass re-boots the fix/text-only pair. ~38s ready each on the launch's receipt |
| wall clock | ~60-90 min including the two-pin build, install and banking |
| prerequisite | this lane merged to `main` and tagged, and the binary built by `serving/build-artifact.sh` from the TAG (a public build FATALs on `MEMRA_REQUEST_LEDGER`). Blocked on the history-rewrite freeze first |
| fixtures | the banked card3 request JSONs in darklanes `research/glm5-serving-launch-20260901/window-20260901/vision-serving-shape/` (read-only; not duplicated into memra) |

## What the window produces

The four arms in `BATTERY.md`: exact can't-hallucinate codes on greedy AND vendor-default
sampled, named refusals on the negative arms, a reproducing control arm (publish=0 refuses at
the waist), text-only byte identity with vision armed, and interleaved x3 decode rows showing no
tax. Receipts banked in-repo and on the box.

## What it unblocks

The owner's "image should be default on" order for glm5, which the launch had to park: the
launcher's `MEMRA_GLM5_VISION=0` pin comes out, and the modality fact goes from text-only to
text+image through the darklanes facts→render→gates→ripple workflow. No product surface moves
before these receipts exist.

## What I explicitly do not need

Any access to a production serving box, any change to the live dark stack, or the second card
pair for anything else. If slot B is spoken for by the cache battery, this waits — it is not
time-pressured now that shipping is held.

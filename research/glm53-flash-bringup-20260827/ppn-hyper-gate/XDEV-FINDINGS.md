# Cross-device arms — the door as a PLACEMENT (2026-08-28, lane/glm53-pp)

The arms `run-ppn-hyper-gate.sh` listed unrun. Two-card box, 2x RTX PRO 6000 Blackwell Server
96 GB, PIX (same PCIe switch), `NVIDIA_TF32_OVERRIDE=0`. Exactness only; no timing number is
read out of any run here. Driver: `run-ppn-hyper-gate-xdev.sh`.

## Verdict

**The mHC ppN walks are bit-identical cross-device.** Six runnable arms, three comparison arms
each, 18/18 PASS, on a clean exclusive box:

| log | placement | fence | result |
|---|---|---|---|
| `20-xdev-n2-dev01.log` | `0,1` N=2 | [0, 2, 4] | PASS x3 |
| `21-xdev-n2-dev01-shard0.log` | `0,1` N=2, `SHARD=0` | [0, 2, 4] | PASS x3 |
| `23-xdev-n4-dev0101.log` | `0,1,0,1` N=4 | [0, 1, 2, 3, 4] | PASS x3 |
| `24-xdev-n2-dev10.log` | `1,0` N=2 (reversed) | [0, 2, 4] | PASS x3 |
| `25-xdev-n2-dev01-split1.log` | `0,1` N=2, `SPLITS=1` | [0, 1, 4] | PASS x3 |
| `26-xdev-n2-dev01-longer.log` | `0,1` N=2, P=16 N=24 | [0, 2, 4] | PASS x3 |

This is the first evidence that peer transport, the sharded weight load and cross-device
per-stage cache placement carry the mHC stream state exactly. Same-device arms could not
have shown it.

**And the gate still binds in the new topology** — it is not passing there by construction.
Mutation M1 (dropped TX: stage 0 publishes the pre-range state) re-run cross-device:

| log | placement | result |
|---|---|---|
| `94-RED-xdev-m1-dev01.log` | `0,1` N=2 | decode-serial FAIL 14/14, prime-twin FAIL 8/9, prefill-twin PASS |
| `95-RED-xdev-m1-dev0101.log` | `0,1,0,1` N=4 | decode-serial FAIL 14/14, prime-twin FAIL 8/9, prefill-twin PASS |

The prefill-twin staying green is the arm-independence check: M1 touches only the decode walk.

## The one arm that cannot run, and why it is not a bug

`MEMRA_PP_HOST_BOUNCE=1` fails — `mla kernel 'split_latent' failed: rc 10700` — and phase
markers put the failure in `[phase] reference A: door OFF`, i.e. in the UNSPLIT walk, before a
single line of split code executes. Isolated cleanly:

| placement | bounce | result |
|---|---|---|
| `0,1,1,0` N=4 | off | PASS |
| `0,1,0` N=3 | off | PASS |
| `0,1,0` N=3 | **on** | FAIL |
| `0,1,1,0` N=4 | **on** | FAIL |

Host bounce is the single variable; placement and stage count are exonerated. The engine's own
startup log gives the mechanism:

> `[pp] cross-device transport: ... peer-pool grants bypassed; diagnostic peer access is
> removed before host-staged serving`
> `[pp] peer byte-integrity probe teardown: disabled 2 diagnostic pair(s); host-bounce serving
> has no probe-enabled peer access`

That is host bounce working as designed: the boundary bounces through pinned host memory
precisely so that no peer access is needed, and peer access is therefore revoked before
serving. But a split-vs-unsplit gate needs an UNSPLIT reference, and the unsplit walk drives
every layer from the primary engine over weights and MLA latent rows the sharded loader put on
the other card. That is a peer dereference, and with peer access correctly off it MMU-faults.

So this is **a structural limit of the gate's reference design under host bounce**, not a
defect in the ppN walks and not a defect in the engine. Corroboration: the generic `ppn-gate`
has the same reference structure, and its banked matrix
(`research/m2-pp8-20260802/run-m2-gates.sh`) contains no host-bounce arm at all. Host bounce
has never been gated this way by anyone.

How to close it properly (NOT done in this lane, named so it is not lost): the reference must
not need peer access. Give the gate a `--dump-logits` / `--against <bank>` pair, bank the
reference from a single-device door-off invocation, then compare a host-bounce cross-device
invocation against that bank. That is a two-invocation harness, and it is a STRONGER gate than
the current one — the reference stops being a sibling arm and becomes an external anchor
(GATE:pin-against-truth).

Also recorded: `22b-xdev-n2-dev01-bounce-EXPECTED-REFUSAL.log`. `MEMRA_PP_HOST_BOUNCE=1`
refuses any placement whose last stage is not on the primary device, because returned logits
and hidden state would stay peer reads. This gate's primary IS `devices[0]`, so the originally
requested `MEMRA_PP_DEVICES=0,1 x HOST_BOUNCE=1` arm is structurally refused BY DESIGN. The
refusal is banked as a receipt: a guard nobody has seen fire is a guard nobody has tested.

## Two box-level findings that nearly became false results

**1. A stale-mtime rsync silently ran the MUTATED binary as if it were clean.** Restoring
`hybrid_forward.rs` with `rsync -a` preserved the SOURCE machine's mtime, which was older than
the box's build fingerprint, so cargo reported success without rebuilding and a full matrix ran
green-labelled on a red binary — every arm "failed", which would have been reported as a
catastrophic cross-device divergence. Caught by checking the binary mtime against the source
mtime, and by the source containing zero `MUTATION M1` matches while the binary still
misbehaved. This is `rebuild-after-checkout-attribution` in a new dress: **a fast "Finished"
after a source swap is an alarm, and every cross-machine source sync must `touch` the files it
writes.** All numbers in this file were produced after that fix.

**2. The peer byte-integrity probe failed 15/15, then 0/12, and the difference was the box.**
Mid-session the probe began refusing native P2P with `boundary=0 dev1->dev0 ... 16384
mismatched byte(s)` — every byte of a 16 KiB copy — alongside `Xid 31` MMU faults
(`FAULT_PDE ACCESS_TYPE_VIRT_READ`). Two box conditions were present and neither was mine: an
unrelated `memra-server` from another working directory was holding up to 20.9 GB on card 0,
and unattended-upgrades had staged a new kernel. The box then auto-rebooted into that new
kernel while the NVIDIA DKMS module existed only for the previous one, so the driver did not
come back at all. After a `dkms autoinstall` against the running kernel restored it, the same
arm ran **12/12 PASS with zero probe failures** on the now-idle box.

The engine's behaviour through all of it was correct and is worth banking as a positive
receipt: the probe DETECTED corrupt peer bytes and REFUSED native P2P rather than serving
wrong logits, naming the boundary, the direction and the remedy. That is fail-closed working
under real adversity.

Two operational consequences, for the owner rather than for this lane:
- this box had `unattended-upgrades` able to install a kernel and reboot mid-cell, which will
  invalidate any long measurement it lands on, and which leaves the accelerator driver
  unloadable until DKMS is rebuilt for the new kernel;
- the box was described as exclusive and was not.

## Scope — what these arms still do NOT establish

Everything in the same-device receipts' scope section still applies, and one thing more must be
said plainly: **these arms do not make step 3's 26.7 ms/token and 37.4 tok/s measured.** This
gate runs a synthetic 4-layer fixture (hidden 128, vocab 32, f32 weights) and reads no clock at
all, by construction — it has to run where a stage handoff can be deliberately broken. What it
now establishes is that the split walk is bit-identical AS A PLACEMENT. Turning step 3 into a
measured number needs a different cell entirely: the real 190.7 GB artifact, expert residency
actually achieved across both cards, real prompts with `reasoning_effort` pinned, interleaved
A/B x5, vendor-default sampled rows, and the staging-subtracted decomposition METHOD.txt
defines. The artifact is now resident on this box (178 GiB), so that cell is unblocked.

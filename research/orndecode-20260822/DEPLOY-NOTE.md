# What the serving box needs to pick up the verify-graph default (owner runs this)

The engine change is merged, but the serving box is still on the v0.105-era binary
(`memra-server-live-38c85edc23e3`), which predates the flag entirely. Nothing regresses by
leaving it there; it simply keeps serving at the pre-graph rate. To collect the measured
+19.7%, the box needs a build from ≥ v0.108 and a slot restart.

Measured expectation on the serving host class: **266 → 319 tok/s** single-stream on the
cached-long shape (localhost, so the public path lands lower by its own tunnel cost). The
serving host is a 9970X (0.80 s reference loop) against the 9950X (0.66 s) these numbers came
from, so treat 319 as the shape of the win rather than its exact size there.

## Steps

1. Build the tip on the box (it already carries a CUDA 13 toolchain and the source tree):
   `cd /data/memra/memra-src && git fetch origin main && git checkout -B live origin/main && cargo build --release -j 20`
2. Install as a NEW live binary rather than overwriting the running one — the launcher takes
   `MEMRA_BIN`, and keeping the old file is what makes step 5 a file swap instead of a rebuild:
   `install -m755 target/release/memra-server /data/memra/bin/memra-server-live-<sha12>`
3. Bring it up on the IDLE slot (B: port 18192 / admin 8107) via `ops/serve-deploy.sh`, so the
   live slot keeps serving while the new one loads its 20 GB.
4. Gate the new slot before it takes traffic: `ops/serve-gate.py` against :18192 (chat,
   responses, messages, tools) and one cached-long timing probe — expect ≈319 on-box, and the
   server log should print `[spec-vg] MTP verify-graph pool ENGAGED`. If that line is absent
   the door did not arm and the deploy is pointless; stop and say so.
5. Flip the tunnel to slot B, drain slot A (`MEMRA_DRAIN_S=30` is already in the launcher).
6. Keep `MEMRA_SPEC_VERIFY_GRAPH=0` in mind as the rollback: it restores the eager walk in a
   restart, byte-identically, without reverting the binary.

## What NOT to do

Do not overwrite `memra-server-live-38c85edc23e3` in place. It is the artifact every current
receipt in this lane was measured against, and it is also the fallback the retired box still
holds. A new sha is a new row; an overwritten binary is a lost one.

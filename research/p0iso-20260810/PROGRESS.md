# P0 served-byte isolation progress (2026-08-10)

## Contract

- Branch/worktree: `lane/cx-p0iso` at base `5cd3fafd`.
- Rig: box1, two RTX PRO 6000 GPUs; every bounded GPU block holds
  `/tmp/memra-gpu.lock` for its full lifetime.
- Fixed serve shape: PP-2 on devices 0,1; context 262144; grouped MoE on;
  prefill tick 2048.
- Fixed request contract: Step-3.7 Flash IQ4_XS + MTP, identical prompt,
  `temperature=0`, `seed=3407`, concurrency 8, 64-token streaming completion.
- Golden completion: 326 bytes,
  `21b8293f2298978c74fb89f32d9b14e3ea921f39924cfce88c73b01f445bb6de`.

## Status

- Read the predecessor P0 receipt before lane work.
- Confirmed clean dedicated branch at the requested base.
- Inbox check: `~/.lanectl/inbox/cx-p0iso.md` is currently absent.
- Added and locally validated a derivative of `darktrain2`'s exact request probe. It
  records deterministic release offsets, complete response bytes, per-request hashes,
  and golden comparisons.
- Same-window cold block complete: 20/20 fresh-boot cells diverged; 20/160 requests
  returned the predecessor's exact 310-byte hash
  `7a5032f2d723e3cf9ef788fdc9d4067fe2eb909157189b666430b7997a56961f`.
- In all 20 cells the divergent request was the first request admitted. Fanout did not
  fire: every cell recorded eight prefix misses and no prefix hit, while the scheduler
  trace advanced the first row at `ready=1` before the steady `ready=8` chunk.
- Box1 was clean at acquisition and after block shutdown; all 160 requests were HTTP
  successes and no CUDA/server failure signature was captured.
- Deliberate 0-200 ms stagger block complete: 20/20 fresh-boot cells diverged,
  again exactly one known 310-byte result per cell. In every cell it was request
  index 0 and admission rank 0; every scheduler trace began `ready=1` then
  `ready=8`.
- Inbox update received after the stagger block: prioritize direct H2 controls
  (`c=2`, delayed client index 0 at `c=8`, and `c=1`) before the remaining
  dedup-off confirmation. The harness now supports those exact shapes.
- H2 `c=2` block complete: 9/10 cells split one divergent + one golden. All
  nine divergent cells began `ready=1` then `ready=2`, and the divergent row
  was admission rank 0. The one clean cell began directly at `ready=2` and
  returned 2/2 goldens.
- H2 delayed-index block complete: 7/10 cells split one divergent + seven
  goldens and began `ready=1`; three cells were 8/8 golden and began
  `ready=2`. In all divergent cells the result followed the new admission-rank-0
  request among indices 1-7. Delayed client index 0 was golden in all 10 cells.
- H2 `c=1` block complete: 10/10 cells returned a third stable 326-byte class,
  `d35be2307889b24ec1ba4361eb22fdc6ceabda65864df261bd66c08f37f192c1`.
  The three classes map to decode-width history: all-`B=1` gives `d35be230...`;
  `B=1` then `B>=2` gives `7a5032...`; batched from the first decode gives the
  predecessor golden `21b8293...`.
- Prefix-dedup-off block complete: 19/20 cells split one transition-class row
  plus seven goldens; the one clean cell began decoding directly at `ready=2`.
  This matches the scheduler-width correlation with dedup enabled and is a null
  result for H1.
- H3 is not run: the prescribed ladder allowed the trainer control only if H1/H2
  were clean, while H2 failed decisively without any trainer or co-tenant CUDA
  context.
- Reduction complete: 90 cells / 590 requests validated, with 505 golden,
  75 solo-to-batch, and 10 all-solo completions. The scheduler-history mapping
  has no exceptions.
- `RESULTS.md` written. Verdict: trial-blocking live serving bug at the PP-N
  Step3.7 eager-`B=1` to batched-`B>1` numeric-class boundary. Lane stopped;
  no runtime fix, push, merge, tag, or release was performed.

## Isolation ladder

1. H1 cold same-window burst: 20 fresh-boot cells. Complete.
2. H1 deliberate 0-200 ms arrival spread: 20 fresh-boot cells. Complete.
3. Inbox-directed H2: `c=2` barrier x10, `c=8` with client index 0 delayed
   100 ms x10, then `c=1` x10.
4. H1 prefix dedup disabled: 20 fresh-boot cells. Complete.
5. H3 exact trainer-running cell x5 only if H1/H2 are clean. Not reached: H2
   reproduced the P0 without a trainer.

Raw logs and per-request completion bytes/hashes will be retained under `raw/`.
The lane stops after `RESULTS.md`; there will be no origin push.

## Block log

- `same`, N=20 cells, completed 2026-08-10 02:46 UTC. Result: 20 divergent
  cells, exactly one divergent request per cell. Remote receipt stamp:
  `20260810T023039Z`.
- `stagger`, N=20 cells, completed 2026-08-10 03:02 UTC. Result: 20 divergent
  cells, exactly one divergent request per cell; actual release offsets tracked
  the uniform 0-200 ms schedule. Remote receipt stamp: `20260810T023039Z`.
- `h2-c2`, N=10 cells, completed 2026-08-10 03:11 UTC. Result: 9 divergent
  cells (`ready=1 -> 2`) and one clean cell (`ready=2` immediately). Remote
  receipt stamp: `20260810T023039Z`.
- `h2-first-late`, N=10 cells, completed 2026-08-10 03:19 UTC. Result: 7
  divergent cells (`ready=1`) and three clean cells (`ready=2`); delayed index
  0 stayed golden. Remote receipt stamp: `20260810T023039Z`.
- `h2-c1`, N=10 cells, completed 2026-08-10 03:27 UTC. Result: 10/10 on the
  third all-solo hash `d35be230...`; no golden or transition-class result.
  Remote receipt stamp: `20260810T023039Z`.
- `dedup-off`, N=20 cells, completed 2026-08-10 03:43 UTC. Result: 19
  transition-class cells and one clean cell; the clean cell began directly at
  `ready=2`, while every divergent cell began at `ready=1`. Remote receipt
  stamp: `20260810T023039Z`.

# tp2-battery window runbook (drives the cells; receipts are the banked outputs)

Window start protocol:
1. Wait for the decode-diet DONE line in /root/BOX-QUEUE.md. Append WINDOW START line
   (cards 0,1 [+2 for the PP-3 arms], port 18400 only for the calibration boot,
   out=/root/out-tp2, build /root/memra-tp2 @ lane/glm5-tp2-battery, wall estimate).
2. Build: `cd /root/memra-tp2 && git fetch origin lane/glm5-tp2-battery &&
   git checkout <HEAD_SHA> && nice -n 19 cargo build --release -p memra-engine
   --bin glm5-tp2-box-probe && nice -n 19 cargo build --release -p memra-server`.
   Rebuild-attribution: BUILD_START/END wall, `git log -1`, binary mtime == BUILD_END,
   sha256 of both binaries, strings probe:
   `strings target/release/glm5-tp2-box-probe | grep -c "glm5-tp-"` (expect >=4 announce
   literals: preflight/kda/mla/ep) and the same probe on memra-server (worker refusal
   string present).
3. Topology receipt: `nvidia-smi topo -m`, `nvidia-smi topo -p2p r`, `-p2p w`,
   `nvidia-smi -q -d POWER | grep -m4 "Power Limit"` -> receipts/topology/.
4. `bash mk_prompts.sh` (pools materialized), `git -C /root/memra-tp2 log -1` into the
   window log.

Cell 1 — real-artifact class gate (untimed, exactness):
  a. plain single-card reference tapes:
     `OUT=/root/out-tp2 bash probe_arm.sh plain1 tape /root/out-tp2/prompts-c1 c1ref`
  b. output-sample screen: the banked .txt files read fluent; `looplaw_screen.py` 0 flags.
  c. TP-2 boot, TEACHER-FORCED on the reference tapes:
     `BOXP_FORCE_DIR=/root/out-tp2/plain1-tape-c1ref bash probe_arm.sh tp2 tape
      /root/out-tp2/prompts-c1 c1tp` — boot log must carry [glm5-tp-preflight] armed +
      kda/mla/ep announces; ep-peer-slot-dispatches > 0.
  d. `python3 compare.py /root/out-tp2/plain1-tape-c1ref /root/out-tp2/tp2-tape-c1tp`
     Bars: step*.f32 BYTE-IDENTICAL expected (t=1 decode); if bits differ -> measure
     max_rel, calibrate band 10x worst-green, REDS MUST LAND ORDERS ABOVE (e).
     prime.f32: band class (batched GEMM m-dependence + grouped-prefill non-bit-stability);
     report measured max_rel. .ids/.txt own-vs-forced: identical expected.
  e. RED loudness: `BOXP_FORCE_DIR=... bash probe_arm.sh tp2red tape ... c1red`, compare
     vs plain — max_rel must sit orders above the green worst (rig gate: 1.4e2 class).
  f. STOP RULE: any decode-step divergence that is not orders-below the red class, any
     silent boot without the announce lines, any own-vs-forced tape fork = window STOPS,
     receipts banked, verdict SILENT-WRONG-SUSPECT.

Cell 2 — transport receipt: v1 = host-canonical only (stage-3 decision). Bank the
  topology P2P probe + the announce line `transport=host-canonical` + NAME the native-P2P
  A/B as follow-up. No code mid-window.

Cell 3 — join-cost row: from cell-4 walls: join+overhead/token =
  measured_tp2_ms − (pp3_measured_ms − launch_class_share) x bandwidth-model; decomposed
  against the decode-gap terms (15.2/2 with EP-2 1.57x haircut = 9.7ms floor + 10ms
  latency class + 4.4 drain). Report the residual as THE measured v1 join tax; size the
  ladder levers (step37: +0.6/+1.85/+2.5/+5.1 classes).

Cell 4 — bare TP-2 pricing (TIMED; touch /root/TIMING-IN-FLIGHT for the WHOLE
  interleaved window, remove after):
  Boots interleaved: pp3-1, tp2-1, pp3-2, tp2-2, pp3-3, tp2-3 — each:
  `BOXP_SAMPLED=1 BOXP_MAX_NEW=256 bash probe_arm.sh <arm> timed
   /root/out-tp2/prompts-decode t<N>` then the l3 dir (WARM + A4630) with
  BOXP_MAX_NEW=256. x5 on anomaly: within-arm decode-median rel spread > 0.5% or gap
  within 2x pooled spread. Aggregate: `python3 agg.py /root/out-tp2/*-timed-*`
  (128-token floor, named exclusions).
  Then the served calibration boot: `bash serve.sh start cal` + flip-battery run_pool
  timed x1 (reproduces the 35.41-class baseline; offset = served/engine on PP-3).
  `bash serve.sh stop`, pidfile-verified down.

Cell 5 — only if TP-2 engine-decode beats PP-3 engine-decode: the TP-2+PP-2 composition
  is REFUSED at preflight (merge-forward matrix); scope the lift honestly (multi-group
  TP runtime + PP boundary between TP roots + cache-slot moves) — if it is more than a
  gated seam lift, it is the named follow-up lane, not a mid-window hack.

Close: bank per cell (scp receipts -> rig worktree, scrub box identity, commit+push),
DONE line to BOX-QUEUE.md, /root/out-tp2 removed after final bank, cards at 1 MiB,
marker down, no processes.

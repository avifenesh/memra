# Box window runbook (pre-registered before the window opened)

Queue protocol: this window starts when the launch-diet lane writes its done-line to
/root/BOX-QUEUE.md. Box identity lives in the private ops repo, never in this file.
Cards 0/1 are the serving shape; cards 2/3 may carry other lanes' correctness co-tenants:
touch /root/TIMING-IN-FLIGHT before every timed cell (battery C2, C3; the 262k deep row)
and remove it after. Message the coordinator: (1) wall-time estimate before starting,
(2) when the FINAL cell (the 262k boot) begins, (3) done-line written.

## 0. Take the window

    cd /root/memra && git fetch origin lane/glm5-prefix-latent   # pre-fetched 2026-08-30
    git merge origin/lane/glm5-prefix-latent                     # onto the window branch
    cargo build --release -p memra-server 2>&1 | tail -3
    # binary-newer-than-sources is NOT the check (LAW:rebuild-after-checkout-attribution):
    strings target/release/memra-server | grep -c "minted before or without latent capture"  # = 1
    git log -1 --format="%H %s"                                  # goes in every receipt

## 1. Serve invocations (adapted from the parent lane's serve.sh + this box's 2-card shape)

Common env, every boot (read off this box's own prior 2-card windows, gpf-ab/l2-ab
serve scripts, adapted; serve.sh from research/prefix-restore-toolcall-20260828/ does
the PID-verified stop, never pkill):

    env MEMRA_SPILL_STATS=1 MEMRA_MOE_RESIDENT_GB=98 MEMRA_MOE_SLOTS=16 \
      MEMRA_PP_STAGES=2 MEMRA_PP_SPLITS=24 MEMRA_PP_DEVICES=0,1 \
      CUDA_VISIBLE_DEVICES=0,1 MEMRA_COMPAT=openai \
      MEMRA_MODELS="zai/glm-5.3-flash=$HOME/models/glm53-nvfp4" \
      MEMRA_ADDR=127.0.0.1:18400 MEMRA_MAX_SESSIONS=4 NVIDIA_TF32_OVERRIDE=0 \
      + the per-arm vars below (MEMRA_CTX, MEMRA_PREFIX_CACHE_MB, MEMRA_PREFIX_LATENT)

DEVIATIONS from the parent lane's env, named: MEMRA_DSA_INDEX_RING stays at DEFAULT
(the ring; the parent lane inherited =0 from a pre-drain box posture that is fixed and
this lane's snapshot design targets the shipped ring). If the launch-diet window's
receipts adopted a different residency posture (MEMRA_MOE_RESIDENT_GB/SLOTS), inherit
theirs and name it in the receipts.

    export PROMPTS_JSON=/root/memra/research/glm53-flash-bringup-20260827/decode-attribution-receipts/prompts.json
    LANE=/root/memra/research/glm5-prefix-latent-20260830

    # OFF arm: MEMRA_CTX=8192, NO MEMRA_PREFIX_LATENT, MEMRA_PREFIX_CACHE_MB=2000
    # ON  arm: MEMRA_CTX=8192, MEMRA_PREFIX_LATENT=1,  MEMRA_PREFIX_CACHE_MB=2048
    #   2048, not the DESIGN.md draft's 4096: measured plateau headroom on this shape is
    #   3.8-4.2 GiB/card (residency-cell); sessions win over cache on alloc pressure, but
    #   the battery must not manufacture eviction storms. 2048 MiB holds ~6-10 entries at
    #   battery depths (deepest ~8k-token seed ~349 MB).
    # 262k:    MEMRA_CTX=262144, NO flag, MEMRA_PREFIX_CACHE_MB=0

## 2. Cells, in order (battery.py asserts its arm's regime per row, exit 2 on violation)

    # OFF arm boot, then:
    python3 $LANE/battery.py /root/out-plx-off off        # C1 identity rows: untimed
    grep -c "snapshot failed (latent" serve-off.log       # refusals PRESENT
    # (touch /root/TIMING-IN-FLIGHT for C2/C3 — battery runs them after C1; the marker
    #  covers the whole battery run: C1 is insensitive to co-tenants either way)

    # ON arm boot, then:
    touch /root/TIMING-IN-FLIGHT
    python3 $LANE/battery.py /root/out-plx-on on
    python3 /root/memra/research/prefix-restore-toolcall-20260828/latentprobe.py /root/out-plx-zqx 4
    rm /root/TIMING-IN-FLIGHT
    grep -c "snapshot failed (latent" serve-on.log        # must be 0
    grep "insert probation" serve-on.log | head           # per-token bytes, NOT 152.6MB flat

    # 262k boot (FINAL cell — message the coordinator at its start), then:
    touch /root/TIMING-IN-FLIGHT
    python3 $LANE/ctx262k-cell.py /root/out-plx-262k 45000
    rm /root/TIMING-IN-FLIGHT

## 3. Close the window

    # PID-verified stop via serve.sh stop path; nvidia-smi all cards back to 0 MiB
    # scp /root/out-plx-* into this lane dir; scrub box identity from anything banked
    # rm -rf /root/out-plx-* on the box; write the done-line:
    echo "$(date -u +%FT%TZ) lane/glm5-prefix-latent window DONE (battery off/on + zqx + 262k cell banked); box clean" >> /root/BOX-QUEUE.md

## Pass bars (pre-registered, from DESIGN.md par.5/7)

OFF arm: guard holds (cached 0 everywhere, one greedy sha per raw prompt, refusal lines
present). ON arm: C1 ONE sha per prompt with cached_tokens == prompt_tokens on every
restored rep; zqx tool/recall/bare pass cold AND restored; C2 engagement from turn 2 with
per-turn TTFT receipts (looped rows excluded, reported separately); C3 TTFT-at-depth
receipts. 262k: VRAM at ready / +1 session / +2 sessions per card, deep prompt TTFD +
prefill tok/s through the serving surface, moe-residency evidence (nvidia-smi deltas;
moe_cache_stats through PP stages is a named counter gap).

## Wall-time estimate (sent to the coordinator before starting)

merge+rebuild+verify ~15 min; OFF arm ~25 min; ON arm ~35 min (latentprobe's 24 sampled
requests dominate); 262k cell ~25 min; teardown+bank+done-line ~10 min.
TOTAL ~1h50m, +/-30 min (unknowns: box rebuild time, expert-staging time per boot x3,
sampled decode lengths). FINAL cell = the 262k boot, announced at start (~25-30 min
before done, matching the 30-minute replacement-box warning).

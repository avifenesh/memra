#!/usr/bin/env bash
# Dev desk (memra#1173): the two CPU tests are green on the tree and each mechanism, removed,
# fails one of them.
#   green          the synced tree as is
#   red-arm        draft_state_arm answers the eager arm on the DFlash route (main's behavior)
#   red-busy       a busy DFlash request that fits without its plane waits instead of being refused
#   red-unfundable a busy DFlash request no wait can fund keeps waiting
#   red-stale      an idle card refuses the drafter on its first read, before the drain
#   red-idle       a drained idle card answers the capacity refusal instead of refusing the drafter
#   red-route      admit's route_ready no longer reads the gate's funding
#   red-funded     the gate reports every request funded
#   red-price      admission_request_may_spec no longer reads the refusal
#   red-requeue    the refusal no longer sets the sticky flag
# Each arm runs `cargo test -p memra-server --lib` on the two memra#1173 tests in a copy of the tree.
set -u
D=$HOME/.cache/memra-1173
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=$D/target
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_NVCC$\|^MEMRA_CUDA_ARCH$'); do unset "$v"; done
echo "$$" >> "$D/pids.txt"
OUT=$D/red; rm -rf "$OUT"; mkdir -p "$OUT"
W=crates/memra-server/src/worker.rs
edit() { python3 - "$W" "$1" "$2" <<'PY'
import sys
p, old, new = sys.argv[1:4]
s = open(p).read()
assert s.count(old) == 1, (old, s.count(old))
open(p, "w").write(s.replace(old, new))
PY
}
for arm in green red-arm red-busy red-unfundable red-stale red-idle red-route red-funded red-price red-requeue; do
  rm -rf "$D/tree-red"; cp -a "$D/tree" "$D/tree-red"; cd "$D/tree-red" || exit 1
  case $arm in
    red-arm) edit "        (_, true, _) => Some(DraftStateArm::RefuseDrafter)," \
                  "        (_, true, gate) => gate.then_some(DraftStateArm::EagerCapture)," ;;
    red-busy) edit "        (Some(DraftStateArm::RefuseDrafter), false) if fits_without => {
            GateVerdict::RefuseDrafter(RefusalArm::BusyFits)
        }" "        (Some(DraftStateArm::RefuseDrafter), false) if fits_without => GateVerdict::Defer," ;;
    red-unfundable) edit "        (Some(DraftStateArm::RefuseDrafter), false) if !fundable => {
            GateVerdict::RefuseDrafter(RefusalArm::BusyUnfundable)
        }" "        (Some(DraftStateArm::RefuseDrafter), false) if !fundable => GateVerdict::Defer," ;;
    red-stale) edit "        (_, true) => GateVerdict::Drain," \
                    "        (Some(DraftStateArm::RefuseDrafter), true) if fits_without => {
            GateVerdict::RefuseDrafter(RefusalArm::IdleFits)
        }
        (_, true) => GateVerdict::Drain," ;;
    red-idle) edit "        Some(DraftStateArm::RefuseDrafter) => GateVerdict::RefuseDrafter(RefusalArm::IdleShort)," \
                   "        Some(DraftStateArm::RefuseDrafter) => GateVerdict::Reject," ;;
    red-route) edit "        route_ready: dflash_route_open && req_drafter_funded," \
                    "        route_ready: dflash_route_open," ;;
    red-funded) edit "            req.drafter_funded = estimate_spec && dspark_drafts.contains_key(&model_key);" \
                     "            req.drafter_funded = true;" ;;
    red-price) edit "        || req.drafter_refused
    {" "    {" ;;
    red-requeue) edit "                        req.drafter_refused = true;
" "" ;;
  esac
  diff -u "$D/tree/$W" "$W" > "$OUT/$arm.patch"
  { date -Is; sha256sum "$W"; } > "$OUT/$arm.log"
  nice -n 15 cargo test --release -p memra-server --lib -j 14 -- \
    the_vram_gate_never_admits_a_dflash_request_without_its_draft_plane the_drafter_refusal_is_wired_from_the_gate_to_every_dflash_route >> "$OUT/$arm.log" 2>&1
  echo "$arm rc=$? $(grep -E '^test result' "$OUT/$arm.log" | tail -n 1)" | tee -a "$OUT/summary.txt"
done
cd "$D" && rm -rf "$D/tree-red"
echo "RED-DONE $(date -Is)" | tee -a "$OUT/summary.txt"

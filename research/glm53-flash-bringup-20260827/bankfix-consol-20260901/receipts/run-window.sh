#!/usr/bin/env bash
# The whole bankfix box-B window, in the one order that makes the receipts mean anything.
#
#   1. THE ORACLE WITH A CORRUPTION ARM — first, on a clean tree, before any other build touches
#      the release tree. Corrupted state must FAIL 4/4; the true fix must PASS 4/4; the two binary
#      shas and the two nvcc object md5s must differ. If this step fails, STOP: every later gate
#      would be reporting on a build tree we have not established tracks its own kernel source.
#   2. build the debug artifacts the suites and the ladders need (test bins + gate bins).
#   3. the standing GPU battery (27 suites, ship arm + compose arms + tp-gate).
#   4. the PPN ladders.
#
# Nothing here is timed and nothing here is a perf row: the box is being used for shapes the rig
# cannot hold (real geometry, four real cards), not for walls.
set -u
cd "$(dirname "$0")/../../../.."
OUT="${BANKFIX_OUT:-/root/out-bankfix}"
mkdir -p "$OUT"
R="$OUT/WINDOW.log"
step() { echo "" | tee -a "$R"; echo "########## $* ##########" | tee -a "$R"; }

echo "=== bankfix window start $(date -u +%Y-%m-%dT%H:%M:%SZ) sha=$(git rev-parse HEAD) ===" | tee "$R"
nvidia-smi --query-gpu=index,name,memory.used,power.max_limit --format=csv,noheader | tee -a "$R"
git log -1 --format='commit=%H%nsubject=%s' | tee -a "$R"
if [ -n "$(git status --porcelain)" ]; then
    echo "REFUSED: dirty tree at window start" | tee -a "$R"; git status --porcelain | tee -a "$R"; exit 1
fi

step "1/4 oracle corruption arm"
research/glm53-flash-bringup-20260827/bankfix-consol-20260901/receipts/run-oracle-corruption-arm.sh \
    >>"$R" 2>&1
oracle_rc=$?
echo "oracle_rc=$oracle_rc" | tee -a "$R"
grep -E "^(OK|FAIL|ORACLE|.*binary sha256|.*kernel \.o md5)" "$OUT/oracle-corruption-arm.log" | tee -a "$R"
if [ "$oracle_rc" -ne 0 ]; then
    echo "STOPPING: the oracle receipt is invalid, so no later gate on this tree is evidence." | tee -a "$R"
    exit 1
fi

step "2/4 build debug test + gate binaries"
nice -n 5 cargo test -p memra-engine --no-run -j 160 >"$OUT/build-tests.log" 2>&1
bt=$?
echo "build_tests_rc=$bt $(sed -n 's/^ *Finished.*in //p' "$OUT/build-tests.log" | tail -1)" | tee -a "$R"
nice -n 5 cargo build -p memra-engine --bins -j 160 >"$OUT/build-bins.log" 2>&1
bb=$?
echo "build_bins_rc=$bb $(sed -n 's/^ *Finished.*in //p' "$OUT/build-bins.log" | tail -1)" | tee -a "$R"
if [ "$bt" -ne 0 ] || [ "$bb" -ne 0 ]; then
    echo "STOPPING: debug build failed" | tee -a "$R"; tail -30 "$OUT/build-tests.log" "$OUT/build-bins.log" | tee -a "$R"; exit 1
fi
for b in glm5-tp-gate glm5-spec-ppn-gate glm5-hyper-ppn-gate glm5-hyper-batch-gate; do
    if [ ! -x "target/debug/$b" ]; then echo "STOPPING: target/debug/$b missing" | tee -a "$R"; exit 1; fi
    echo "bin $b sha256=$(sha256sum "target/debug/$b" | cut -c1-16)" | tee -a "$R"
done

step "3/4 standing GPU battery"
research/glm53-flash-bringup-20260827/bankfix-consol-20260901/receipts/run-battery-box.sh \
    >"$OUT/battery.out" 2>&1
bat_rc=$?
echo "battery_rc=$bat_rc" | tee -a "$R"
tail -40 "$OUT/battery.out" | tee -a "$R"

step "4/4 PPN ladders"
research/glm53-flash-bringup-20260827/bankfix-consol-20260901/receipts/run-matrices-box.sh \
    >"$OUT/matrices.out" 2>&1
mat_rc=$?
echo "matrices_rc=$mat_rc" | tee -a "$R"
grep -E "^(PASS|FAIL|bankfix matrices)" "$OUT/matrices.out" | tail -45 | tee -a "$R"

step "window verdict"
echo "oracle=$oracle_rc battery=$bat_rc matrices=$mat_rc" | tee -a "$R"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tee -a "$R"
echo "=== bankfix window end $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$R"
[ "$oracle_rc" -eq 0 ] && [ "$bat_rc" -eq 0 ] && [ "$mat_rc" -eq 0 ]

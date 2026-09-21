#!/usr/bin/env bash
while ! grep -q PROVISION_DONE /root/lane/provision.log 2>/dev/null; do sleep 15; done
echo "provision done $(date -u +%T)"
cd /data/hy3 && sha256sum -c --quiet SHA256SUMS > /root/lane/sha-verify.log 2>&1; echo "sha rc=$? ($(grep -c FAILED /root/lane/sha-verify.log) failed: $(grep FAILED /root/lane/sha-verify.log | tr "\n" " "))"
if grep FAILED /root/lane/sha-verify.log | grep -qv README.md; then echo WEIGHTS_BAD; exit 1; fi
bash /root/lane/build.sh; tail -1 /root/lane/build-base.log; tail -1 /root/lane/build-lane.log; tail -1 /root/lane/build-companion.log
grep -q BASE_OK /root/lane/build-base.log && grep -q LANE_OK /root/lane/build-lane.log && grep -q COMPANION_OK /root/lane/build-companion.log || { echo BUILD_BAD; tail -20 /root/lane/build-lane.log; exit 1; }
echo "== cpu_native_check on the box (lane binary, lane companion)"; MEMRA_CPU_EXPERT_LIB=/root/lane/libmemra-cpu-experts.so /root/lane/target-lane/release/cpu_native_check 2>&1 | grep -E 'raw|ALL GREEN|FAIL|rror' | head -8
bash /root/lane/ab-rows.sh
echo "CHAIN_DONE $(date -u +%T)"

#!/usr/bin/env bash
# provision -> verify weights -> build probe -> run the #577 matrix, unattended.
while ! grep -q PROVISION_DONE /root/lane/provision.log 2>/dev/null; do sleep 15; done
echo "provision done $(date -u +%T)"
cd /data/hy3 && sha256sum -c --quiet SHA256SUMS > /root/lane/sha-verify.log 2>&1; echo "sha rc=$? ($(grep -c FAILED /root/lane/sha-verify.log) failed: $(grep FAILED /root/lane/sha-verify.log | tr "\n" " "))"
if grep FAILED /root/lane/sha-verify.log | grep -qv README.md; then echo "WEIGHTS_BAD"; exit 1; fi
bash /root/lane/build.sh; tail -1 /root/lane/build-probe.log; tail -1 /root/lane/build-companion.log
grep -q PROBE_OK /root/lane/build-probe.log || { echo BUILD_BAD; tail -30 /root/lane/build-probe.log; exit 1; }
bash /root/lane/ab577.sh
echo "CHAIN_DONE $(date -u +%T)"

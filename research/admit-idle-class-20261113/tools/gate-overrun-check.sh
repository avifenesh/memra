#!/usr/bin/env bash
# memra#1304 third review fold, CPU only: the check the two serving gates gained
# (tools/spec-on-cache-hit-gate.sh in entry_census, tools/kv-host-spill-identity-gate.sh after the
# ON boot), read against this record's stored server logs and against a copy of one with the
# server's warning appended. No server is started; the gates themselves were not rerun.
# usage: gate-overrun-check.sh RECORD_DIR SCRATCH_DIR
set -u
R=$1; S=$2
PAT="its count missed a plane"
mkdir -p "$S"
n=0; hits=0
while IFS= read -r log; do
  n=$((n + 1))
  if grep -q "$PAT" "$log"; then hits=$((hits + 1)); echo "WARNING present: ${log#"$R"/}"; fi
done < <(find "$R/gb10" "$R/rtx5090" -name '*server.log' | sort)
echo "stored server logs read: $n, with the warning: $hits"
grep -c "$PAT" "$R/../../tools/spec-on-cache-hit-gate.sh" "$R/../../tools/kv-host-spill-identity-gate.sh" | sed "s|$R/../../||"
cp "$R/gb10/host-gate-dflash/ev/host-on-server.log" "$S/host-on-server.log"
echo "[prefix-cache] WARNING: a dspark-boundary publication holds 161415168 bytes, 4096 more than it reserved (161411072 to allocate, 0 held): its count missed a plane" >> "$S/host-on-server.log"
absent() { ! grep -q "$1" "$2"; }
if absent "$PAT" "$R/gb10/host-gate-dflash/ev/host-on-server.log"; then echo "ok: the stored ON boot passes the check"; else echo "FAIL: the stored ON boot"; fi
if absent "$PAT" "$S/host-on-server.log"; then echo "NOT-RED: the log with the warning passes"; else echo "red as stated: the log with the warning fails the check"; fi

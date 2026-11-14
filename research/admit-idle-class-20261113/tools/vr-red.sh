#!/usr/bin/env bash
# The verifier's red arm (memra#1304 review 2, item 3): the kv phase branches on the capacity the
# server logs again (as at ccdd965593). tools/test_choice_verifier.py must fail on the reviewer's
# probes P5 and P6 (the one test is run: the scratch copy has no receipts for the others). Runs on a
# scratch copy of tools/*.py; the worktree is not touched.
set -u
WT=$HOME/.cache/memra-1304/wt
S=$HOME/.cache/memra-1304/fold/vr
OUT=$WT/research/admit-idle-class-20261113/cpu-fold/round2/RV2-verifier-branches-on-the-logged-capacity.log
mkdir -p "$S/tools"
cp "$WT"/tools/*.py "$S/tools/"
python3 - "$S/tools/replay_native_choices.py" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
old = "    if sizes['required']<=boot_free:\n        require(record['status']==503 and error.get('code')=='overloaded' and sizes['logged']=='short' and sizes['required']<=sizes['capacity'],"
new = "    if sizes['required']<=sizes['capacity']:\n        require(record['status']==503 and error.get('code')=='overloaded' and sizes['logged']=='short',"
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
PY
{ echo "red arm: check_idle_refusal branches on the logged capacity (sizes['required']<=sizes['capacity'])"; (cd "$S" && python3 tools/test_choice_verifier.py Controls.test_idle_refusal_follows_the_request_size); echo "rc=$?"; } > "$OUT" 2>&1
sed -i "s|/home/[a-z]*/|~/|g" "$OUT"
grep -E '^(FAIL|ERROR|rc=|Ran|FAILED|OK)|AssertionError' "$OUT"

#!/usr/bin/env bash
# Live agentic round-trip receipt for the GLM-5.3-Flash standard surface.
#
# WHAT IT PROVES, and how the chain closes:
#
#   The two prompts POSTed here are the VERBATIM bytes of
#   surface-fixtures/{21-roundtrip-turn1-ask,22-roundtrip-turn2-after-result}/expected.txt.
#   Those bytes are what the shipped Rust arm renders — asserted, not assumed, by
#   `glm5_fixtures_match_the_vendor_jinja` (memra-server), which replays every fixture's
#   input.json through the REAL request pipeline and compares byte-for-byte. So a
#   /v1/completions round-trip on those bytes exercises the exact prompt a
#   /v1/chat/completions (or /v1/messages, or /v1/responses) tools request produces, against
#   the live model, WITHOUT needing the fixed binary deployed first.
#
#   Turn 1: tool declared -> the model must emit the NATIVE call wire
#           `<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>`
#           and stop (`<|observation|>` is a declared eos id).
#   Turn 2: the tool result rendered into an `<|observation|>` block -> a final answer that
#           uses the result (21C, sunny) and stops on `<|user|>`.
#
#   `finish_reason` must be "stop" on both: a "length" means the stop set is not being
#   honoured (the multi-EOS defect this lane already fixed).
#
# WHAT IT DOES NOT PROVE: the server-side wiring (caps probe, parser selection, response
# rendering) on the deployed binary. That needs the fixed build serving and is the separate
# post-deploy battery.
#
# Usage (from the box, or through an ssh tunnel):
#   ENDPOINT=http://127.0.0.1:18400 MODEL=zai/glm-5.3-flash OUT=/tmp/glm-rt bash run-roundtrip.sh
#
# Bank the OUT directory into this receipts dir AFTER redacting build fingerprints:
#   sed -i 's/memra-[0-9a-f]\{12,\}/memra-<redacted-build-fingerprint>/g' *.json
# (`live_fingerprint` is a sev1 pattern in the repo's pre-push boundary scanner.)
set -u

ENDPOINT=${ENDPOINT:-http://127.0.0.1:18400}
MODEL=${MODEL:-zai/glm-5.3-flash}
OUT=${OUT:-/tmp/glm-rt}
HERE=$(cd "$(dirname "$0")" && pwd)
FIX="$HERE/../surface-fixtures"
mkdir -p "$OUT"

turn() { # name fixture
  local name=$1 fixture=$2
  python3 - "$FIX/$fixture/expected.txt" "$MODEL" > "$OUT/roundtrip-$name.request.json" <<'PY'
import json, sys
prompt = open(sys.argv[1], encoding="utf-8").read()
json.dump({"model": sys.argv[2], "prompt": prompt, "max_tokens": 400, "stream": False},
          sys.stdout, ensure_ascii=False)
PY
  curl -s -m 600 "$ENDPOINT/v1/completions" -H 'content-type: application/json' \
    --data-binary "@$OUT/roundtrip-$name.request.json" > "$OUT/roundtrip-$name.response.json"
  echo "== $name"
  python3 - "$OUT/roundtrip-$name.response.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1], encoding="utf-8"))
c = r.get("choices", [{}])[0]
print("  finish_reason:", c.get("finish_reason"))
print("  text:", repr(c.get("text", ""))[:600])
PY
}

turn turn1-call 21-roundtrip-turn1-ask
turn turn2-final 22-roundtrip-turn2-after-result

echo
echo "receipts in $OUT (redact memra-<hex> before banking)"

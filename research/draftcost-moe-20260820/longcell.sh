#!/usr/bin/env bash
# Long-prompt spec identity cell (the replay pin's named gate, serving shape):
# ~15k-token prompt, greedy 256, spec-on (new replay-free default) vs spec-off.
set -uo pipefail
cd "$HOME/models/ornith15"
BINS=$HOME/memra-src/target/release
M=$PWD/Ornith-1.5-35B-A3B-NVFP4-MTP-v2.gguf

python3 - <<'PYEOF'
import json
rows = [json.loads(l) for l in open('mtp-train/corpus.jsonl')][:40]
text = '\n\n'.join((r['reasoning'] or '') + (r['content'] or '') for r in rows)
open('/tmp/longprompt.txt', 'w').write(
    'Here is a long document:\n' + text[:60000] + '\nSummarize the main themes in five bullets.')
PYEOF

run_arm() {
  env $2 MEMRA_MODELS=m="$M" MEMRA_ADDR=127.0.0.1:8099 "$BINS/memra-server" > /dev/null 2>&1 &
  local pid=$!
  for _ in $(seq 240); do curl -sf http://127.0.0.1:8099/health >/dev/null 2>&1 && break; sleep 2; done
  python3 - "$1" <<'PYEOF'
import json, urllib.request, sys, time
p = open('/tmp/longprompt.txt').read()
body = {"model": "m", "messages": [{"role": "user", "content": p}],
        "max_tokens": 256, "temperature": 0}
req = urllib.request.Request("http://127.0.0.1:8099/v1/chat/completions",
                             json.dumps(body).encode(), {"Content-Type": "application/json"})
t0 = time.time(); r = json.load(urllib.request.urlopen(req, timeout=900)); dt = time.time() - t0
m = r["choices"][0]["message"]
u = r["usage"]
json.dump({"text": (m.get("reasoning") or "") + "|" + (m.get("content") or ""),
           "prompt_tokens": u["prompt_tokens"], "spec": u.get("spec"), "dt": round(dt, 2)},
          open(f"mtp-train/longcell-{sys.argv[1]}.json", "w"))
print(sys.argv[1], "prompt_toks", u["prompt_tokens"], "dt", round(dt, 2), "spec", u.get("spec"))
PYEOF
  kill $pid 2>/dev/null; wait $pid 2>/dev/null || true
}
run_arm spec ""
run_arm plain "MEMRA_SERVE_SPEC=0"

python3 - <<'PYEOF'
import json
a = json.load(open('mtp-train/longcell-spec.json'))
b = json.load(open('mtp-train/longcell-plain.json'))
print('LONG-CELL', 'IDENTICAL' if a['text'] == b['text'] else 'MISMATCH',
      '| prompt_toks', a['prompt_tokens'], '| spec dt', a['dt'], 'plain dt', b['dt'],
      '| spec', a['spec'])
PYEOF
rm -f /tmp/longprompt.txt

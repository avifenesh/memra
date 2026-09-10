set -eu
cd /root/dp
arm=$1; round=$2; shift 2
tag="${arm}-b${round}"
out=/root/dp/out
mkdir -p "$out"
nohup bash scripts/serve.sh "$arm" > "$out/$tag-server.log" 2>&1 < /dev/null &
srv=$!
for i in $(seq 1 240); do
  if curl --fail --silent http://127.0.0.1:8080/health > "$out/$tag-health.json" 2>/dev/null; then break; fi
  if ! kill -0 $srv 2>/dev/null; then echo "SERVER DIED"; tail -40 "$out/$tag-server.log"; exit 3; fi
  sleep 5
done
# Arm identity is read off /proc, not asserted: pid, start ticks, exe, binary sha, and the
# MEMRA_* environment the process was actually booted with. A health 200 proves nothing
# about which arm answered it.
real=$(pgrep -f '/root/dp/memra/target/release/memra-server' | head -1)
python3 - "$real" "$tag" "$out" "$arm" <<'PY'
import hashlib, json, os, sys
pid, tag, out, arm = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
env = dict(p.split('=', 1) for p in open(f'/proc/{pid}/environ').read().split('\0') if '=' in p)
exe = os.readlink(f'/proc/{pid}/exe')
h = hashlib.sha256()
with open(exe.replace(' (deleted)', ''), 'rb') as f:
    for b in iter(lambda: f.read(8 << 20), b''):
        h.update(b)
memra_env = {k: v for k, v in env.items() if k.startswith('MEMRA') or k == 'PORT'}
nonce = {'pid': int(pid), 'arm': arm,
         'start_ticks': open(f'/proc/{pid}/stat').read().split(') ')[1].split()[19],
         'exe': exe, 'binary_sha256': h.hexdigest(), 'env': memra_env,
         'devpenalty_env': memra_env.get('MEMRA_SERVE_DEVPENALTY', '<unset>')}
json.dump(nonce, open(f'{out}/{tag}-boot-nonce.json', 'w'), indent=1)
expect = '1' if arm == 'on' else '<unset>'
assert nonce['devpenalty_env'] == expect, ('ARM IDENTITY MISMATCH', arm, nonce['devpenalty_env'])
print('boot nonce', tag, pid, 'ticks', nonce['start_ticks'], h.hexdigest()[:16],
      'MEMRA_SERVE_DEVPENALTY=' + nonce['devpenalty_env'], flush=True)
PY
curl --fail --silent http://127.0.0.1:8080/v1/models > "$out/$tag-models.json"
python3 scripts/runner.py \
  --model "$(python3 -c "import json;print(json.load(open('$out/$tag-models.json'))['data'][0]['id'])")" \
  --tag "$tag" --arm "$arm" --outdir "$out" "$@"
kill "$real"
for i in $(seq 1 60); do kill -0 "$real" 2>/dev/null || break; sleep 1; done
kill -0 "$real" 2>/dev/null && kill -9 "$real" || true
sleep 3
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits > "$out/$tag-vram-after-stop.txt"
echo "BLOCK_DONE $tag"

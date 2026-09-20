#!/usr/bin/env python3
"""CPU-only native compile/test phase, AFTER patch application in owned remote scratch.

No GPU cell is launched by this script. The caller must use the collector for full
server tests or model gates; selected filters below were inspected as CPU-only.
"""
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if ROOT != Path('/root/wt-b'):
    raise SystemExit('refuse: native patch checks belong only in owned /root/wt-b scratch')
OUT = ROOT / sys.argv[1]
OUT.mkdir(parents=True, exist_ok=False)
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
patch = ROOT / 'research/spill-b-20260919/HOSTPREFIX-PATCH.diff'
subprocess.run(['git', 'apply', '--reverse', '--check', str(patch)], cwd=ROOT, check=True)
identity = {'source': head, 'patch_sha256': hashlib.sha256(patch.read_bytes()).hexdigest(),
            'kind': 'native-compile-and-cpu-tests', 'gpu_executed': False}
(OUT / 'source.json').write_text(json.dumps(identity, indent=2) + '\n')
# All six filters instantiate CPU-only metadata and empty payload constructors; no Engine::new.
commands = [
    ('server-build', ['cargo', 'build', '--release', '-p', 'memra-server', '-j', '16']),
    ('server-test-compile', ['cargo', 'test', '--release', '-p', 'memra-server', '--no-run', '-j', '16']),
    ('gate-build', ['cargo', 'build', '--release', '-p', 'memra-engine', '--bin', 'kv_tier_gate', '-j', '16']),
    ('gate-cli-tests', ['cargo', 'test', '--release', '-p', 'memra-engine', '--bin', 'kv_tier_gate', '-j', '16']),
]
for case in ['prefix_cache_', 'host_cache_', 'host_purge_', 'host_image_', 'host_handoff_', 'prefix_restore_plane_preflight']:
    commands.append((case, ['cargo', 'test', '--release', '-p', 'memra-server', '--lib', '-j', '16', case, '--', '--test-threads=1']))
rows = []
for name, command in commands:
    raw = OUT / (name + '.log')
    with raw.open('wb') as log:
        proc = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(proc.stdout.readline, b''):
            log.write(line)
            log.flush()
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
        code = proc.wait()
    if name not in ['server-build', 'server-test-compile', 'gate-build'] and code == 0:
        if not re.search(rb'test result: ok\. [1-9][0-9]* passed', raw.read_bytes()):
            code = 97  # A mistyped filter running zero tests is not a pass.
    rows.append({'name': name, 'command': command, 'exit': code,
                 'raw': raw.name, 'sha256': hashlib.sha256(raw.read_bytes()).hexdigest()})
    (OUT / 'commands.json').write_text(json.dumps(rows, indent=2) + '\n')
    if code:
        raise SystemExit(code)

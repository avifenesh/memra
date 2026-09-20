#!/usr/bin/env python3
"""Scoped day-ten CPU checks with raw logs; no hardware qualification."""
import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=False)
commands = [
    ['cargo', 'fmt', '--all', '--', '--check'],
    ['python3', '-B', '-m', 'unittest', 'discover', '-s', 'crates/memra-tier/tests/battery', '-p', 'test_*.py'],
    ['python3', '-m', 'py_compile', *sorted(str(f.relative_to(ROOT)) for f in (ROOT/'tools').glob('tier-*.py')),
     *sorted(str(f.relative_to(ROOT)) for f in (ROOT/'crates/memra-tier/tests/battery').glob('*.py')),
     'research/spill-d-20260919/validate-day8-archives.py', 'research/spill-d-20260919/run-g2.py', str(Path(__file__).relative_to(ROOT))],
    *[['bash', '-n', path] for path in ['tools/tier-rig-bootstrap.sh', 'tools/kv-host-spill-identity-gate.sh',
                                      'tools/kv-host-spill-failure-gate.sh']],
    ['shellcheck', 'tools/tier-rig-bootstrap.sh'],
    ['git', 'diff', '--check'],
    ['bash', 'tools/check-flags.sh'],
    ['python3', '-c', 'import re,subprocess; from pathlib import Path; '
     'blocks=re.findall(r"```sh\\n(.*?)```",Path("research/spill-d-20260919/DAY8-CELLS.md").read_text(),re.S); '
     '[subprocess.run(["bash","-n"],input=b,text=True,check=True) for b in blocks]; '
     'print("RUNBOOK SYNTAX MATCH:",len(blocks),"shell blocks")'],
]
source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
rows = []
for i, command in enumerate(commands):
    start = time.monotonic()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        result = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        code, raw = result.returncode, result.stdout
    except (OSError, subprocess.TimeoutExpired) as error:
        code, raw = 125, str(error).encode()
    name = f'check-{i:02}.log.gz'
    (a.out/name).write_bytes(gzip.compress(raw, mtime=0))
    rows.append({'command': command, 'source_commit': source, 'started_utc': started,
                 'elapsed_seconds': time.monotonic()-start, 'exit_code': code,
                 'raw_gzip': name, 'raw_sha256': hashlib.sha256(raw).hexdigest(), 'qualification': False})
    (a.out/'checks.json').write_text(json.dumps(rows, indent=2)+'\n')
    print(f'{i}: exit={code} '+ ' '.join(command), flush=True)
    print(raw.decode(errors='replace'), flush=True)
sys.exit(0 if all(r['exit_code'] == 0 for r in rows) else 1)

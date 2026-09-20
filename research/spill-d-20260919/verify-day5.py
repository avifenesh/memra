#!/usr/bin/env python3
"""Bounded offline checks; retain raw output before parsing. Never a GPU receipt."""
import argparse
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--out', type=Path, required=True, help='new receipt directory')
a = p.parse_args()
out = a.out.resolve()
out.mkdir(parents=True, exist_ok=False)
revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
status = subprocess.check_output(['git', 'status', '--short'], cwd=ROOT, text=True)
commands = [
    ['cargo', 'fmt', '--all', '--', '--check'],
    ['cargo', 'check', '-p', 'memra-tier', '-p', 'memra-kv', '--offline', '--all-targets'],
    ['cargo', 'check', '-p', 'memra-tier', '-p', 'memra-kv', '--target', 'x86_64-unknown-linux-gnu', '--offline', '--all-targets'],
    ['cargo', 'test', '-p', 'memra-tier', '--offline', '--no-fail-fast'],
    ['cargo', 'test', '-p', 'memra-kv', '--offline', '--no-fail-fast'],
    ['cargo', 'clippy', '-p', 'memra-tier', '--offline', '--all-targets', '--', '-D', 'warnings'],
    ['python3', '-B', '-m', 'unittest', 'discover', '-s', 'crates/memra-tier/tests/battery', '-p', 'test_*.py'],
    ['python3', '-m', 'py_compile', *sorted(str(f.relative_to(ROOT)) for f in (ROOT/'tools').glob('tier-*.py'))],
    ['bash', '-n', 'tools/tier-rig-bootstrap.sh'],
    ['python3', '-c', 'from pathlib import Path; lines=Path("tools/tier-rig-bootstrap.sh").read_text().splitlines(); begin=next(i for i,line in enumerate(lines) if line.startswith("exec python3"))+1; compile(chr(10).join(lines[begin:-1]),"bootstrap-heredoc","exec"); print("bootstrap Python syntax PASS")'],
    ['git', 'apply', '--check', 'research/spill-d-20260919/C-RUNNER-GOLDENS.diff'],
    ['python3', 'tools/tier-battery.py', '--validate', 'research/spill-lead-20260919/rented-5090-20260919/receipts'],
    ['python3', 'tools/tier-battery.py', '--validate', 'research/spill-d-20260919/day2-dry-run/runs.jsonl'],
    ['python3', 'tools/tier-battery.py', '--validate', 'research/spill-d-20260919/day2-dry-run/telemetry/correctness-control-off.jsonl'],
    ['python3', '-c', "import json,subprocess; from pathlib import Path; actual=json.loads(subprocess.check_output(['python3','tools/tier-battery.py','--first-hour'])); assert actual==json.loads(Path('research/spill-d-20260919/FIRST-HOUR-PLAN.json').read_text()); print('FIRST-HOUR MATCH: AB/BA correctness only; no perf medians')"],
    ['git', 'diff', '--check'],
    ['bash', 'tools/check-flags.sh'],
    ['python3', '-B', 'crates/memra-tier/tests/contracts/fixture_reference.py', '--check'],
    ['python3', '-B', 'tools/tier-battery.py', '--validate-campaign', 'research/spill-d-20260919/day2-dry-run'],
]
commands.append(['python3', '-c',
    "import json,subprocess; m=json.loads(subprocess.check_output(['cargo','metadata','--offline','--locked','--no-deps','--format-version','1'])); "
    "e=next(p for p in m['packages'] if p['name']=='memra-engine'); "
    "assert 'memra-tier' in {d['name'] for d in e['dependencies']}; "
    "assert {'storage-bench','pp-transport-smoke','qwen4exp_gpu_gate','run-gen','run-spec'} <= {t['name'] for t in e['targets']}; "
    "s=next(p for p in m['packages'] if p['name']=='memra-server'); assert 'memra-server' in {t['name'] for t in s['targets']}; "
    "print('METADATA MATCH: engine tier dependency and all first-hour binary targets registered; NOT native compile')"])
if shutil.which('shellcheck'):
    commands.append(['shellcheck', 'tools/tier-rig-bootstrap.sh'])
else:
    (out/'shellcheck-unavailable.txt').write_text('NOT RUN: shellcheck executable absent\n')
paths = [*sorted((ROOT/'tools').glob('tier-*')),
         *sorted((ROOT/'crates/memra-tier/tests/battery').glob('*.py')),
         *sorted((ROOT/'crates/memra-tier/tests/battery').glob('*.json')),
         ROOT/'research/spill-d-20260919/RIG-DAY1.md', Path(__file__).resolve()]
(out/'source-manifest.json').write_text(json.dumps({str(path.relative_to(ROOT)):
    hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}, indent=2)+'\n')
records = []
for i, command in enumerate(commands):
    path = out/f'check-{i:02d}.log'
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    monotonic_start = time.monotonic()
    with path.open('xb') as raw:
        try:
            result = subprocess.run(command, cwd=ROOT, stdout=raw, stderr=subprocess.STDOUT,
                                    timeout=180, check=False)
            code = result.returncode
        except subprocess.TimeoutExpired:
            raw.write(b'ERROR: CPU verification timeout (180 seconds)\n')
            code = 124
    data = path.read_bytes()
    compressed = path.with_suffix('.log.gz')
    compressed.write_bytes(gzip.compress(data, mtime=0))
    path.unlink()
    records.append({'schema_version': 1, 'kind': 'cpu-verification', 'source_commit': revision,
                    'worktree_status_at_start': status, 'command': command, 'started_utc': started,
                    'ended_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'elapsed_seconds': time.monotonic()-monotonic_start,
                    'exit_code': code, 'raw_gzip': compressed.name,
                    'raw_sha256': hashlib.sha256(data).hexdigest(), 'qualification': False})
    # Persist incrementally so an interrupted session retains every completed check.
    (out/'checks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    print(f'{i:02d} exit={code}: '+ ' '.join(command), flush=True)
    print(data.decode(errors='replace'), end='', flush=True)
sys.exit(0 if all(r['exit_code'] == 0 for r in records) else 1)

#!/usr/bin/env python3
"""Bounded offline CPU verification, preserving raw output before interpreting status."""
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
LANE = ROOT / 'research/spill-d-20260919'
RAW = LANE / 'raw'
revision = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
commands = [
    ['cargo','fmt','--all','--','--check'],
    ['cargo','check','-p','memra-tier','--offline','--all-targets'],
    ['cargo','test','-p','memra-tier','--offline'],
    ['python3','-B','-m','unittest','discover','-s','crates/memra-tier/tests/battery','-p','test_*.py'],
    ['python3','-m','py_compile',*sorted(str(p.relative_to(ROOT)) for p in (ROOT/'tools').glob('tier-*.py'))],
    ['git','diff','--check'],
    ['bash','tools/check-flags.sh'],
    ['python3','-B','crates/memra-tier/tests/contracts/fixture_reference.py','--check'],
    ['python3','-B','tools/tier-battery.py','--validate','research/spill-d-20260919/day2-dry-run/runs.jsonl'],
    ['python3','-B','tools/tier-battery.py','--validate-campaign','research/spill-d-20260919/day2-dry-run'],
    ['cargo','clippy','-p','memra-tier','--offline','--all-targets','--','-D','warnings'],
]
records=[]
for i,command in enumerate(commands):
    path=RAW/f'day2-check-{i:02d}.log'
    started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with path.open('wb') as raw:
        try:
            result=subprocess.run(command,cwd=ROOT,stdout=raw,stderr=subprocess.STDOUT,timeout=180,check=False)
            code=result.returncode
        except subprocess.TimeoutExpired:
            code=124
    data=path.read_bytes() # raw complete before inspecting status/content
    compressed=path.with_suffix('.log.gz')
    compressed.write_bytes(gzip.compress(data,mtime=0)); path.unlink()
    records.append({'schema_version':1,'kind':'cpu-verification','source_commit':revision,
                    'command':command,'started_utc':started,'exit_code':code,
                    'raw_gzip':str(compressed.relative_to(ROOT)),
                    'raw_sha256':hashlib.sha256(data).hexdigest()})
    print(f'{i:02d} exit={code}: '+ ' '.join(command))
    print(data.decode(errors='replace'),end='')
(LANE/'day2-checks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
sys.exit(0 if all(r['exit_code']==0 for r in records) else 1)

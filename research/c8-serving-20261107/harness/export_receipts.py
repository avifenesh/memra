"""Copy the lane's runs and harness into research/c8-serving-20261107/, public form (idempotent).

- Local absolute paths become placeholders (`<lane>`, `<bins>`, `<models>`, ...).
- Agentic rows keep the per-turn numbers and the sha256 of each output (they never held texts); the
  private session file name is cut to its first 8 characters, as the prefix-budget lane did.

  python3 export_receipts.py DEST_RESEARCH_DIR
"""
import json
import shutil
import sys
from pathlib import Path

SRC = Path('<lane>')
dest = Path(sys.argv[1])
SUBS = [
    ('<harness>', '<harness>'),
    ('<bins>', '<bins>'),
    ('<runs>', '<runs>'),
    ('<tree>', '<tree>'),
    ('<lane>', '<lane>'),
    ('<agentic-workload>', '<agentic-workload>'),
    ('<models>', '<models>'),
    ('<tmp>', '<tmp>'),
    ('~', '~'),
    ('<tmp>', '<tmp>'),
]
KEEP_HARNESS = ['arm2.sh', 'ts.py', 'agentic2.py', 'mixed.py', 'analyze.py', 'exact_cmp.py', 'score.py',
                'sched_sim.py', 'tick_sim.py', 'agentic-approx.json', 'phaseR.sh', 'phaseH.sh', 'phaseE.sh',
                'phaseE2.sh', 'phaseE3.sh', 'phaseX.sh', 'phaseS8.sh', 'phaseSM.sh', 'phaseS4.sh',
                'phaseS1.sh', 'phaseF.sh', 'phaseT.sh', 'phaseT2.sh', 'split_t.py', 'tables.py', 'split_p2a.py', 'patch_guard.py', 'export_receipts.py', 'build.sh']


def scrub_text(t):
    for a, b in SUBS:
        t = t.replace(a, b)
    return t


def copy_tree(src, dst):
    n = 0
    for p in src.rglob('*'):
        if not p.is_file():
            continue
        q = dst / p.relative_to(src)
        q.parent.mkdir(parents=True, exist_ok=True)
        if p.name == 'rows.json':
            j = json.loads(p.read_text())
            rows = j['rows'] if isinstance(j, dict) else j
            for r in rows:
                if isinstance(r.get('session'), str):
                    r['session'] = r['session'][:8]
            q.write_text(scrub_text(json.dumps(j, indent=1)))
        else:
            try:
                q.write_text(scrub_text(p.read_text()))
            except UnicodeDecodeError:
                shutil.copy(p, q)
        n += 1
    return n


h = dest / 'harness'
h.mkdir(parents=True, exist_ok=True)
for name in KEEP_HARNESS:
    p = SRC / 'harness' / name
    if not p.exists():
        p = SRC / name
    if p.exists():
        (h / name).write_text(scrub_text(p.read_text()))
total = copy_tree(SRC / 'runs', dest / 'runs')
print(f'exported {total} run files and the harness to {dest}')

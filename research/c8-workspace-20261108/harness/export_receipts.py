"""Copy the lane's runs and harness into research/c8-workspace-20261108/, public form (idempotent).

- Server logs and telemetry are gzipped (`serve.log.gz`, `gpu-*.csv.gz`, `memavail.txt.gz`).
- Local absolute paths become placeholders (`<lane>`, `<bins>`, `<models>`, ...).
- Agentic rows keep the per-turn numbers and the sha256 of each output (they never held texts); the
  private session file name is cut to its first 8 characters, as the prefix-budget lane did.

  python3 export_receipts.py DEST_RESEARCH_DIR
"""
import gzip
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
    ('<main-tree>', '<main-tree>'),
    ('<growrewind-harness>', '<growrewind-harness>'),
    ('<main-tree>-main', '<main-tree>'),
    ('<agentic-workload>', '<agentic-workload>'),
    ('<lane>', '<lane>'),
    ('<agentic-workload>', '<agentic-workload>'),
    ('<agentic-workload-le8k>', '<agentic-workload-le8k>'),
    ('<models-9b>', '<models-9b>'),
    ('<models>', '<models>'),
    ('<tmp>', '<tmp>'),
    ('~', '~'),
    ('<tmp>', '<tmp>'),
]
GZIP_NAMES = {'serve.log', 'gpu-tel.csv', 'gpu-250ms.csv', 'memavail.txt'}
KEEP_HARNESS = ['arm2.sh', 'ts.py', 'agentic2.py', 'mixed.py', 'analyze.py', 'exact_cmp.py', 'score.py',
                'tables.py', 'export_receipts.py', 'build.sh', 'build3.sh', 'phase.sh', 'campaign.sh',
                'oom_boots.sh', 'probe_run.sh', 'ws_probe_patch.py', 'ws-probe.patch', 'ws_probe_read.py',
                'census_check.py', 'spark_census.sh', 'spark_census_run.sh', 'unit.sh', 'checks.sh', 'group.sh',
                'campaign3.sh', 'serving_table.py', 'results_tables.py', 'grow_ab.sh', 'agent_turns.py',
                'summarize.py', 'trace_requests.py', 'group2.sh', 'campaign5.sh', 'campaign6.sh',
                'campaign7.sh', 'campaign8.sh', 'campaign9.sh', 'campaign10.sh', 'campaign11.sh',
                'campaign12.sh', 'grow_check.sh', 'prime_tail.py', 'grow_floor.py', 'grow_table.py']


def scrub_text(t):
    for a, b in SUBS:
        t = t.replace(a, b)
    # Trailing blanks (an empty `env: ` line) would fail `git diff --check`.
    return '\n'.join(line.rstrip() for line in t.split('\n'))


def copy_tree(src, dst):
    n = 0
    for p in src.rglob('*'):
        if not p.is_file() or p.name.startswith('.group'):
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
        elif p.name in GZIP_NAMES:
            data = scrub_text(p.read_text(errors='replace')).encode()
            with gzip.GzipFile(q.with_name(q.name + '.gz'), 'wb', mtime=0) as z:
                z.write(data)
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

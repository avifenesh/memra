"""Export small public engineering receipts; keep full profiles/raw archives private."""
import argparse
import gzip
import hashlib
import json
import pathlib
import re

p = argparse.ArgumentParser()
p.add_argument('root')
p.add_argument('--out', required=True)
a = p.parse_args()
root, out = pathlib.Path(a.root), pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)
manifest = {}
for run in sorted(root.iterdir()):
    if not run.is_dir() or not (run/'cleanup.json').exists():
        continue
    if not ((run/'stage0-summary.json').exists() or (run/'summary.json').exists()):
        continue
    identity = json.loads((run/'identity.json').read_text())
    profile = json.loads((run/'profile.json').read_text())
    cleanup = json.loads((run/'cleanup.json').read_text())
    summary_name = 'stage0-summary.json' if (run/'stage0-summary.json').exists() else 'summary.json'
    receipt = {
        'boot': run.name,
        'identity': {k: v for k, v in identity.items() if k in (
            'boot_nonce', 'pid', 'server_pid', 'start_ticks', 'server_start_ticks',
            'binary_sha256', 'sha256')},
        'settings': {k: v for k, v in profile.items() if k in (
            'MEMRA_PRIME_YIELD', 'MEMRA_PRIME_CHUNK', 'MEMRA_MAX_SESSIONS',
            'MEMRA_SPEC_GATE_LOW', 'MEMRA_SPEC_GATE_HIGH', 'MEMRA_CTX',
            'MEMRA_TICK_TRACE', 'MEMRA_ALLOC_TRACE')},
        'summary': json.loads((run/summary_name).read_text()), 'cleanup': cleanup,
        'request_sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                           for f in sorted(run.glob('*-request.json'))},
    }
    if (run/'requests.jsonl').exists():
        receipt['requests'] = [json.loads(line) for line in (run/'requests.jsonl').read_text().splitlines()
                               if json.loads(line).get('stage') == 'stage0']
    target = out/(run.name+'.json')
    target.write_text(json.dumps(receipt, indent=2)+'\n')
    # Log fields are selected explicitly; model pricing and launcher secrets stay out.
    log = '\n'.join(line for line in (run/'server.log').read_text().splitlines()
                    if re.match(r'\[(prime-chunk|prime-yield|prime-finalize|ttft|spec-acc|dspark-acc|mtp-prime-oracle|dflash-oracle|tick-spec)\]', line))+'\n'
    trace = out/(run.name+'.trace.gz')
    trace.write_bytes(gzip.compress(log.encode(), mtime=0))
    for artifact in (target, trace):
        manifest[artifact.name] = {'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
                                   'bytes': artifact.stat().st_size}
(out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print('BANKED', len(manifest)//2, 'completed boots')

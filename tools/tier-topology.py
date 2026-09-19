#!/usr/bin/env python3
"""Read-only topology census. Capability queries are NOT direct-route qualification.

No driver, clocks, power, ACS, IOMMU or pool settings are changed. Keep live JSON
machine-local: raw hardware identifiers must not be committed to public receipts.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess

COMMANDS = [
    ['nvidia-smi', 'topo', '-m'],
    ['nvidia-smi', '-q'],
    ['nvidia-smi', 'topo', '-p2p', 'r'],
    ['nvidia-smi', 'topo', '-p2p', 'w'],
    ['lscpu', '--extended=CPU,NODE,SOCKET'],
    ['numactl', '-H'],
]


def probe(fixture=None):
    captures = []
    for command in COMMANDS:
        if fixture is not None:
            value = fixture.get(' '.join(command))
            if value is None:
                raise ValueError('fixture missing command: ' + ' '.join(command))
            result = dict(value)
        else:
            try:
                p = subprocess.run(command, text=True, capture_output=True, timeout=20, check=False)
                result = {'exit_code':p.returncode, 'stdout':p.stdout, 'stderr':p.stderr, 'timed_out':False}
            except FileNotFoundError as e:
                result = {'exit_code':None, 'stdout':'', 'stderr':str(e), 'timed_out':False}
            except subprocess.TimeoutExpired as e:
                def text(b):
                    return b.decode(errors='replace') if isinstance(b,bytes) else (b or '')
                result = {'exit_code':None, 'stdout':text(e.stdout), 'stderr':text(e.stderr), 'timed_out':True}
        captures.append({'command':command, **result})
    return {'schema_version':1, 'kind':'cpu-fixture' if fixture is not None else 'read-only-inventory',
            'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'route_qualification':False, 'target':'PCIe Gen5 only; no NVLink assumption',
            'warning':'Capability queries do not prove context/pool grants, direct DMA, active link health or bandwidth.',
            'commands':captures}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dry-run', type=Path, metavar='FIXTURE_JSON')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    fixture = json.loads(a.dry_run.read_text()) if a.dry_run else None
    report = probe(fixture)
    with a.out.open('x') as out:
        out.write(json.dumps(report,indent=2)+'\n')
    print(f"{report['kind']}: {len(report['commands'])} commands; direct P2P remains unqualified")


if __name__ == '__main__':
    main()

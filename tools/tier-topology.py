#!/usr/bin/env python3
"""Read-only topology census. Capability queries are NOT direct-route qualification.

No driver, clocks, power, ACS, IOMMU or pool settings are changed. Keep live JSON
machine-local: raw hardware identifiers must not be committed to public receipts.
"""
import argparse
import datetime
import json
import re
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


STORAGE_COMMANDS = [
    ['lscpu'], ['df', '-hT'], ['lsblk', '-J', '-o', 'NAME,TYPE,SIZE,ROTA,TRAN,MOUNTPOINTS'],
    ['free', '-g'], ['findmnt', '-J', '-T', '/scratch'],
]


def pcie_links(text):
    """Separate observed generation from host/device ceilings; never infer bandwidth."""
    links = []
    for block in text.split('GPU Link Info')[1:]:
        match = re.search(r'PCIe Generation\s*\n(.*?)Link Width\s*\n(.*?)(?=\n\s*Bridge Chip|\Z)', block, re.S)
        if not match:
            continue
        def fields(section, width=False):
            return {key.strip(): int(value) for key, value in re.findall(
                r'^\s*(Max|Current|Device Current|Device Max|Host Max)\s*:\s*(\d+)' + ('x' if width else '') + r'\s*$',
                section, re.M)}
        gen, width = fields(match[1]), fields(match[2], True)
        links.append({'device_ordinal': len(links),
                      'generation_current': gen.get('Current'), 'generation_max': gen.get('Max'),
                      'device_generation_max': gen.get('Device Max'), 'host_generation_max': gen.get('Host Max'),
                      'width_current': width.get('Current'), 'width_max': width.get('Max'),
                      'bandwidth_measured': False})
    return links


def probe(fixture=None, storage=False):
    captures = []
    for command in COMMANDS + (STORAGE_COMMANDS if storage else []):
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
            'pcie_links': pcie_links(next((c['stdout'] for c in captures
                                            if c['command'] == ['nvidia-smi', '-q'] and c['exit_code'] == 0), '')),
            'commands':captures}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dry-run', type=Path, metavar='FIXTURE_JSON')
    p.add_argument('--storage', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    fixture = json.loads(a.dry_run.read_text()) if a.dry_run else None
    report = probe(fixture, a.storage)
    with a.out.open('x') as out:
        out.write(json.dumps(report,indent=2)+'\n')
    print(f"{report['kind']}: {len(report['commands'])} commands; direct P2P remains unqualified")


if __name__ == '__main__':
    main()

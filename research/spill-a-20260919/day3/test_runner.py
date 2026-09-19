#!/usr/bin/env python3
"""Dry-run command-routing tests. No GPU command, lock, build, or NVMe writes."""
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / 'research/spill-a-20260919/rig-cells-a.sh'
with tempfile.TemporaryDirectory(prefix='runner-stub-', dir=Path(__file__).parent) as tmp:
    tmp = Path(tmp)
    record = tmp / 'calls.jsonl'
    stub = tmp / 'stub'
    stub.write_text('#!/usr/bin/env python3\nimport json,sys\n'
                    f'with open({str(record)!r}, "a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n')
    stub.chmod(0o700)
    for _ in range(2):
        result = subprocess.run(['bash', str(RUNNER), '/never-created nvme', '--dry-run', '--stub', str(stub)],
                                cwd=ROOT, capture_output=True, text=True, check=True)
        assert 'flock -x /tmp/memra-5090.lock (entire window)' in result.stdout
        assert 'DRY-RUN ONLY' in result.stdout
        assert not Path('/never-created nvme').exists()
    calls = [json.loads(line) for line in record.read_text().splitlines()]
    assert len(calls) == 70, len(calls)
    assert calls[:35] == calls[35:]
    names = [row[0] for row in calls[:35]]
    assert len(set(names)) == 35
    assert sum(name.startswith('roundtrip-') for name in names) == 12
    assert sum(name.startswith('restore-') for name in names) == 12
    assert all(name in names for name in ['a1-sharded-catalog', 'a1-gc', 'a1-telemetry-join', 'a2-existing-worker', 'a2-byte-roundtrip-unimplemented',
                                          'm1-row-unimplemented', 'm1-bulk-unimplemented', 'm1-mixed-unimplemented'])
    assert all(row[1:3] == ['timeout', '5'] for row in calls[:35] if row[0].endswith('-unimplemented'))
    for args in [[], ['/not-created', '--stub', str(stub)], ['/not-created', '--unknown']]:
        result = subprocess.run(['bash', str(RUNNER), *args], cwd=ROOT, capture_output=True)
        assert result.returncode == 2
print('RUNNER_DRY_PASS: two identical 35-command plans; 24 filesystem + 3 CPU conformance + existing A2 + 4 blocked probes; no hardware ran')

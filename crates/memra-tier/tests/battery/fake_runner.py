#!/usr/bin/env python3
"""Deterministic opaque runner; no model computation or hardware access."""
import argparse
import json
from pathlib import Path
import struct
import sys
p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
p.add_argument('--arm', choices=['on','off'], required=True)
p.add_argument('--fail', action='store_true')
a = p.parse_args()
print('CPU FAKE: no model/GPU execution', flush=True)
if a.fail:
    print('ERROR: injected read failure (CPU fixture)', file=sys.stderr, flush=True)
    sys.exit(9)
for name, data in [('state',b'\x01\x02\x03'),('logits',struct.pack('<f',1)),('tokens',struct.pack('<I',1))]:
    (a.out / f'{name}.bin').write_bytes(data)
print('RESULT '+json.dumps({'arm':a.arm,'migrated_bytes':3 if a.arm=='on' else 0,'duration_ns':500_000_000}))

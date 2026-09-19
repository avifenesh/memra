#!/usr/bin/env python3
"""Synthetic default-SLRU oracle, transcribed from moe_cache.rs (see SLRU.md).
No model routes or GPU timings. Fixed seed; --check refuses source/fixture drift.
"""
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
CLASSES = [(32, 3), (64, 2), (128, 1)]

class Oracle:
    def __init__(self):
        self.classes = []
        self.occupant = {}
        self.pending = {}
        self.table = {}
        self.slot_class = {}
        start = 0
        for cap, count in CLASSES:
            self.classes.append(dict(cap=cap, free=list(reversed(range(start, start+count))), probation=[], protected=[], protected_cap=max(1, int(count*0.8))))
            for slot in range(start, start+count):
                self.slot_class[slot] = len(self.classes)-1
            start += count

    def publish(self, key):
        slot = self.pending.pop(key)
        self.table[key] = slot
        self.occupant[slot] = key
        self.classes[self.slot_class[slot]]['probation'].append(slot)
        return slot

    def step(self, op, key, size, keep):
        result, slot, victim = 'noop', None, None
        if op == 'abort':
            if key in self.pending:
                slot = self.pending.pop(key)
                self.classes[self.slot_class[slot]]['free'].append(slot)
                result = 'aborted'
        elif op == 'complete':
            if key in self.pending:
                slot, result = self.publish(key), 'published'
        elif op == 'demand' and key in self.table:
            slot, result = self.table[key], 'hit'
            c = self.classes[self.slot_class[slot]]
            if not c['free']:
                for q in ['probation', 'protected']:
                    if slot in c[q]:
                        c[q].remove(slot)
                c['protected'].append(slot)
                while len(c['protected']) > c['protected_cap']:
                    c['probation'].append(c['protected'].pop(0))
        elif op == 'demand' and key in self.pending:
            slot, result = self.publish(key), 'published'
        elif key not in self.table and key not in self.pending:
            for c in self.classes:
                if c['cap'] >= size and c['free']:
                    slot = c['free'].pop()
                    break
            if slot is None:
                for c in self.classes:
                    if c['cap'] < size:
                        continue
                    choices = c['probation'] + c['protected']
                    eligible = [s for s in choices if self.occupant[s] not in keep]
                    if eligible:
                        slot = eligible[0]
                        for q in ['probation', 'protected']:
                            if slot in c[q]:
                                c[q].remove(slot)
                        victim = self.occupant.pop(slot)
                        del self.table[victim]
                        break
            if slot is None:
                result = 'capacity'
            else:
                self.pending[key] = slot
                result = 'reserved'
                if op == 'demand':
                    self.publish(key)
                    result = 'admitted'
        return dict(op=op, key=key, size=size, keep=keep, result=result, slot=slot, victim=victim,
                    orders=[[c['free'][:], c['probation'][:], c['protected'][:]] for c in self.classes])

r = random.Random(20260919)
o = Oracle()
rows = []
# Initial hits while free, full promotion/demotion, then delayed/cancelled prefetch.
ops = [('demand', k, []) for k in [0, 0, 4, 8, 0, 4, 8, 12, 0]]
ops += [('prefetch', 1, [0, 4, 8]), ('prefetch', 5, [0, 4, 8]), ('demand', 1, []), ('abort', 5, [])]
ops += [(r.choices(['demand', 'prefetch', 'complete', 'abort'], [6, 3, 1, 1])[0], r.randrange(24), r.sample(range(24), r.randrange(7))) for _ in range(2000)]
for op, key, keep in ops:
    rows.append(o.step(op, key, [16, 40, 80, 256][key % 4], keep if op == 'prefetch' else []))
serial = Oracle()
serial_rows = [serial.step('demand', k, [16, 40, 80, 256][k % 4], []) for k in [(i*7+i//11) % 24 for i in range(256)]]
result = dict(label='SYNTHETIC CPU decision trace, not model or GPU evidence',
              source='crates/memra-engine/src/moe_cache.rs',
              source_sha256=hashlib.sha256((ROOT/'crates/memra-engine/src/moe_cache.rs').read_bytes()).hexdigest(),
              classes=CLASSES, serial_rows=serial_rows, rows=rows)
path = ROOT/'research/spill-c-20260919/fixtures/slru-synthetic.json'
header = {k: v for k, v in result.items() if k not in ('rows', 'serial_rows')}
text = json.dumps(header, indent=2)[:-2] + ',\n'
for key, values in [('serial_rows', serial_rows), ('rows', rows)]:
    text += '  '+json.dumps(key)+': [\n'+',\n'.join('    '+json.dumps(row, separators=(',', ':')) for row in values)+'\n  ]'
    text += ',\n' if key == 'serial_rows' else '\n}\n'
if '--check' in sys.argv:
    assert path.read_text() == text, 'trace or native-source drift'
else:
    path.write_text(text)
print(f'SYNTHETIC SLRU decisions: {len(rows)}; source/trace matched' if '--check' in sys.argv else f'wrote {len(rows)} synthetic decisions')

#!/usr/bin/env python3
"""Replay all raw G2 observations and collector evidence before emitting medians."""
import argparse
from collections import defaultdict
import csv
import datetime
import importlib.util
import json
import math
from pathlib import Path
import statistics

spec = importlib.util.spec_from_file_location('g2', Path(__file__).with_name('run-g2.py'))
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)


def summarize(root):
    G.B.validate_cell(root/'collector/CELL.jsonl')
    capture = json.loads((root/'collector/command.capture.json').read_text())
    G.B.require(capture['exit_code'] == 0 and capture['result']['samples'] == 200,
                'campaign command did not complete')
    telemetry = capture['gpu_telemetry']
    G.B.require(telemetry['status'] == 'captured-unvalidated' and telemetry['interval_ms'] == 250,
                'telemetry not continuously captured')
    visits = root/'visits'
    rows = [json.loads(s) for s in (visits/'samples.jsonl').read_text().splitlines()]
    scored = [r for r in rows if r['phase'] == 'scored']
    expected = [f'score-{size}-{pair}-{order}' for size in G.SIZES for pair in range(5) for order in ('ab', 'ba')]
    G.B.require([r['visit'] for r in scored[::4]] == expected and len(scored) == 200,
                'incomplete/reordered scored campaign')
    calibrated = json.loads((visits/'calibration.json').read_text())
    grouped = defaultdict(list)
    for name in dict.fromkeys(r['visit'] for r in rows):
        recorded = [r for r in rows if r['visit'] == name]
        first = recorded[0]
        raw = visits/first['raw_log']
        samples = G.check_samples(raw.read_text(), first['bytes'], first['copies'], first['order'],
                                  first['phase'] == 'scored')
        G.B.require(len(recorded) == 4 and not (visits/(name+'.compute.log')).read_text().strip(),
                    'incomplete visits or competing GPU process')
        for sample, record in zip(samples, recorded):
            G.B.require(all(record[k] == v for k, v in sample.items()) and
                        record['raw_sha256'] == G.B.digest(raw), 'raw sample/hash mismatch')
            if record['phase'] == 'scored':
                G.B.require(record['copies'] == calibrated[str(record['bytes'])], 'copy count changed')
                grouped[(record['bytes'], record['direction'], record['arm'])].append(record)
    with (root/'collector'/telemetry['raw_csv']['path']).open() as stream:
        gpu = list(csv.DictReader(stream, skipinitialspace=True))
    G.B.require(len(gpu) >= capture['elapsed_seconds'] / 0.5, 'insufficient 250ms telemetry coverage')
    def numbers(column):
        values = [float(r[column].split()[0]) for r in gpu]
        G.B.require(all(math.isfinite(v) for v in values), 'nonfinite telemetry')
        return values
    G.B.require(set(numbers('power.limit [W]')) == {600} and
                set(numbers('power.max_limit [W]')) == {600}, 'power envelope changed')
    stamp = [datetime.datetime.strptime(r['timestamp'], '%Y/%m/%d %H:%M:%S.%f') for r in gpu]
    gaps = [(b-a).total_seconds()*1000 for a, b in zip(stamp, stamp[1:])]
    G.B.require(all(0 < v <= 1000 for v in gaps), 'missing/nonmonotonic telemetry interval')
    thermal = {'regime': 'sequential calibration-warmed; no steady-state soak; whole-campaign telemetry',
               'samples': len(gpu), 'interval_target_ms': 250,
               'interval_median_ms': statistics.median(gaps), 'interval_max_ms': max(gaps)}
    for column in ['temperature.gpu', 'clocks.current.sm [MHz]', 'clocks.current.memory [MHz]',
                   'power.draw [W]', 'pcie.link.gen.current', 'pcie.link.width.current']:
        vals = numbers(column)
        thermal[column] = {'min': min(vals), 'max': max(vals)}
    result = []
    for (size, direction, arm), samples in sorted(grouped.items()):
        G.B.require(len(samples) == 10 and all(sum(r['order'] == o for r in samples) == 5 for o in ('ab', 'ba')),
                    'requires N=5 in each order')
        result.append({'bytes': size, 'direction': direction, 'arm': arm, 'n': 10,
                       'ab_n': 5, 'ba_n': 5, 'copies_per_visit': samples[0]['copies'],
                       'median_us_per_copy': statistics.median(r['wall_ns']/r['copies']/1000 for r in samples),
                       'median_GiB_s': statistics.median(r['completed_bytes']/(r['wall_ns']/1e9)/(1<<30) for r in samples),
                       'minimum_visit_ms': min(r['wall_ns']/1e6 for r in samples)})
    G.B.require(len(result) == 20, 'incomplete size/direction/arm matrix')
    return {'kind': 'G2-single-target-class-development', 'qualification': False,
            'capture_seconds': capture['elapsed_seconds'], 'scored_samples': len(scored),
            'calibration_samples': len(rows)-len(scored), 'thermal': thermal, 'medians': result}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    a = p.parse_args()
    print(json.dumps(summarize(a.root), indent=2))

"""Recompute component results from banked raw files, with timing/thermal context."""
import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root):
    raw = root/'raw/isolated'
    rows = [json.loads(x) for x in (raw/'runs.jsonl').read_text().splitlines()]
    spec = importlib.util.spec_from_file_location('isolated', Path(__file__).with_name('isolated.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for row in rows:
        assert digest(raw/(row['id']+'.log')) == row['raw_sha256']
        assert row['status'] == 'pass' and row['exit_code'] == 0
        if row['phase'] == 'timed':
            assert 'MEMRA_GEMMA_DRAFT_TRACE' not in row['config']
        if row.get('trace_sha256'):
            assert digest(raw/(row['id']+'.trace.jsonl')) == row['trace_sha256']
    summaries = {}
    for component in ['head', 'confidence', 'depth']:
        subset = [r for r in rows if r['component'] == component]
        if not (raw/f'{component}-summary.json').exists():
            continue
        result = mod.summarize(subset)
        assert result == json.loads((raw/f'{component}-summary.json').read_text())
        seal = json.loads((raw/f'{component}-seal.json').read_text())
        assert all(digest(raw/name) == sha for name, sha in seal.items())
        assert result['independent_prompt_groups'] == 8 and result['paired_repeats'] == 48
        timed = [r for r in subset if r['phase'] == 'timed']
        prompts = sorted({r['prompt'] for r in timed})
        assert len(timed) == 96
        arms = {}
        for arm in ['A', 'B']:
            rr = [r for r in timed if r['arm'] == arm]
            repeats = {p: [r['request_seconds'] for r in rr if r['prompt'] == p] for p in prompts}
            spreads = {p: (max(v)-min(v))/statistics.median(v) for p, v in repeats.items()}
            arms[arm] = {'request_seconds_total': sum(r['request_seconds'] for r in rr),
                         'emitted_tokens_total': sum(r['emitted'] for r in rr),
                         'request_tokens_per_second': sum(r['emitted'] for r in rr)/sum(r['request_seconds'] for r in rr),
                         'decode_seconds_total': sum(r['decode_seconds'] for r in rr),
                         'drafted_per_round': sum(r['drafted'] for r in rr)/sum(r['rounds'] for r in rr),
                         'accepted_fraction': sum(r['accepted'] for r in rr)/sum(r['drafted'] for r in rr),
                         'learned_rows_min_max': [min(r['learned_rows'] for r in rr), max(r['learned_rows'] for r in rr)],
                         'within_prompt_repeat_relative_range': spreads}
        result['arms'] = arms
        result['repeat_geomean_request_ratio_b_over_a'] = {
            str(i): statistics.geometric_mean(p['request_ratio_b_over_a'] for p in result['pairs'] if p['repeat'] == i)
            for i in range(6)}
        summaries[component] = result
    telemetry = []
    with (raw/'telemetry.csv').open() as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            try:
                stamp = dt.datetime.strptime(row['timestamp'], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=dt.timezone.utc)
                telemetry.append((stamp.timestamp(), row))
            except ValueError:
                continue
    # Telemetry window includes plain reference and startup; this is a hardware
    # regime receipt, not an energy attribution to the timed spec section.
    for component, result in summaries.items():
        rr = [r for r in rows if r['component'] == component and r['phase'] == 'timed']
        begin = min(dt.datetime.fromisoformat(r['utc']).timestamp() for r in rr)
        end = max(dt.datetime.fromisoformat(r['utc']).timestamp()+r['wall_seconds'] for r in rr)
        samples = [r for t, r in telemetry if begin <= t <= end]
        stats = {}
        for column in ['temperature.gpu', 'clocks.current.sm [MHz]', 'power.draw [W]', 'utilization.gpu [%]']:
            values = []
            for r in samples:
                try:
                    values.append(float(r[column].split()[0]))
                except (KeyError, ValueError):
                    pass
            if values:
                stats[column] = {'min': min(values), 'median': statistics.median(values), 'max': max(values)}
        result['campaign_telemetry'] = {'samples': len(samples), 'stats': stats}
    return {'verified_raw_runs': len(rows), 'components': summaries}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bank', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.bank)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({'verified_raw_runs': result['verified_raw_runs'],
                      'components': {k: {x: v[x] for x in ['paired_repeats', 'request_ratio_b_over_a', 'bootstrap_95_percentile']} for k, v in result['components'].items()}}, indent=2))

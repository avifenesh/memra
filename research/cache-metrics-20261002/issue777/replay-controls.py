#!/usr/bin/env python3
"""Rejudge the live receipt, then restore the stale cache law as an offline red control."""
import importlib.util
import json
from pathlib import Path
from unittest import mock

repo = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('q35_gate', repo / 'tools/q35-cold-mixed-gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
harness = gate.load_sellgate(repo / 'research/sellgate-20260812/sellgate_replay.py')
workload = gate.load_frozen_workload(harness, repo)
log = out / 'live/gate.jsonl'
green = gate.replay(harness, workload, 32, log)
with mock.patch.object(harness, 'expected_hit_cached', lambda prompt, grid: prompt):
    red = gate.replay(harness, workload, 32, log)
(out / 'live/replay-green.json').write_text(json.dumps(green, indent=2) + '\n')
(out / 'live/replay-stale-expectation.json').write_text(json.dumps(red, indent=2) + '\n')
print(json.dumps({'green': green['verdict'], 'red': red['verdict'], 'red_failure_classes': red['failure_classes']}))
assert green['verdict'] == 'PASS', green
assert red['verdict'] == 'FAIL', red
assert 'usage_mismatch' in red['failure_classes'], red
assert 'token_regression' not in red['failure_classes'], red
assert red['short_or_non_length'] == [], red

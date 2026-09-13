import hashlib
import json
from pathlib import Path
import runpy

lane = Path(__file__).resolve().parent
bank = lane/'row-oracle-receipts'
sealed = bank/'raw/compiled-source/research/adaptive-draft-joint-20260913/audit_row_oracle.py'
original = json.loads(json.dumps(runpy.run_path(str(sealed))['analyze'](bank)))
current = json.loads((lane/'row-oracle-audit.json').read_text())
def same(old, new):
    if isinstance(old, dict):
        assert old.keys() <= new.keys()
        for k, v in old.items():
            same(v, new[k])
    else:
        assert old == new, (old, new)
same(original, current)
manifest = json.loads((lane/'row-oracle-receipt-hashes.json').read_text())
for name, digest in manifest['files'].items():
    assert hashlib.sha256((bank/name).read_bytes()).hexdigest() == digest, name
seed = json.loads((bank/'checkpoints/row-oracle/seed.json').read_text())
for item in seed['sources']:
    assert hashlib.sha256((bank/'checkpoints/row-oracle/seed-prompts'/item['prompt']).read_bytes()).hexdigest() == item['sha256']
runs = [json.loads(x) for x in (bank/'raw/row-oracle/runs.jsonl').read_text().splitlines()]
cost = [x for x in runs if x['phase'] == 'cost']
orders = {}
for i in range(0, len(cost), 2):
    a, b = cost[i:i+2]
    assert a['prompt'] == b['prompt'] and a['repeat'] == b['repeat'] and a['probe'] != b['probe']
    orders.setdefault(a['prompt'], []).append(a['probe'])
assert len(orders) == 8 and all(len(v) == 6 and sum(v) == 3 for v in orders.values())
proof = {'original_preregistered_auditor_sha256': hashlib.sha256(sealed.read_bytes()).hexdigest(),
    'original_metrics_match_extended_auditor': True, 'receipt_files_verified': len(manifest['files']),
    'original_seed_prompt_files_verified': len(seed['sources']),
    'six_repetitions_three_pairs_each_order_per_prompt': True,
    'extended_auditor_sha256': hashlib.sha256((lane/'audit_row_oracle.py').read_bytes()).hexdigest()}
(lane/'row-analysis-proof.json').write_text(json.dumps(proof, indent=2))
print(json.dumps(proof, indent=2))

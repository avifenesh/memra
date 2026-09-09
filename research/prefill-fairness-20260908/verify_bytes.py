"""Compare saved GPU oracles and HTTP bytes. Run on the authorized box CPU."""
import argparse
import collections
import json
import pathlib

p = argparse.ArgumentParser()
p.add_argument('root')
p.add_argument('--out', required=True)
a = p.parse_args()
root = pathlib.Path(a.root)
receipt = {}

def read(run, name):
    return json.loads((root / run / name).read_text())

def oracles(run, prefix):
    return [line for line in (root / run / 'server.log').read_text().splitlines()
            if line.startswith(prefix)]

for model, prefix, route in [('ornith', '[mtp-prime-oracle]', 'served_spec'),
                              ('qwen', '[dflash-oracle]', 'served_dspark')]:
    chains = []
    for turn in range(1, 5):
        off = read(f'{model}-chain-off', f'turn{turn}-result.json')
        on = read(f'{model}-chain-on', f'turn{turn}-result.json')
        assert [off['content'], off['reasoning']] == [on['content'], on['reasoning']]
        chains.append({'turn': turn, 'output_sha256': off['output_sha256'],
                       'cached_tokens': on['usage']['prompt_tokens_details']['cached_tokens']})
    pair = read(f'{model}-pair-on', 'summary.json')
    profile = read(f'{model}-pair-on', 'profile.json')
    assert profile['MEMRA_SPEC_GATE_LOW'] == '64'
    assert profile['MEMRA_SPEC_GATE_HIGH'] == '65'
    assert pair['yield_count'] > 0
    assert pair['metric_delta'][route] == 2 and pair['metric_delta']['served_plain'] == 0
    pairs = []
    for request in ('long', 'small'):
        off = read(f'{model}-solo-off', f'{request}-result.json')
        on = read(f'{model}-pair-on', f'{request}-result.json')
        assert [off['content'], off['reasoning']] == [on['content'], on['reasoning']]
        pairs.append({'request': request, 'output_sha256': off['output_sha256'],
                      'c1_ttft_s': off['ttft_s'], 'c2_ttft_s': on['ttft_s']})
    # Completion order changes in the concurrent pair; compare the full multiset.
    comparisons = []
    for off, on in [('chain-off', 'chain-on'), ('solo-off', 'pair-on')]:
        expected = oracles(f'{model}-{off}', prefix)
        actual = oracles(f'{model}-{on}', prefix)
        assert len(expected) >= 2
        assert collections.Counter(expected) == collections.Counter(actual), (model, off, on)
        comparisons.append({'off': off, 'on': on, 'equal_oracle_lines': expected})
    receipt[model] = {'pass': True, 'chains': chains, 'pairs': pairs,
                      'pair_yields': pair['yield_count'], 'spec_gate_low_high': [64, 65],
                      'pair_routes': pair['metric_delta'], 'boundary_comparisons': comparisons}

pathlib.Path(a.out).write_text(json.dumps(receipt, indent=2) + '\n')
print('BYTE_AND_BOUNDARY_GATES_PASS')

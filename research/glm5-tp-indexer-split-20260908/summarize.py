#!/usr/bin/env python3
"""Render receipt rows and require each recorded OFF/ON pair to have identical ids."""
import hashlib
import json
import pathlib
import statistics

root = pathlib.Path(__file__).parent / 'receipts'
rows = []
for path in sorted(root.glob('*/rows.jsonl')):
    cell = path.parent
    if (cell / 'exit').read_text().strip() != '0':
        raise SystemExit(f'{cell.name}: nonzero exit')
    env = dict(line.split('=', 1) for line in (cell / 'env.txt').read_text().splitlines())
    for line in path.read_text().splitlines():
        row = json.loads(line)
        suffix = '-vendor' if row['arm'] == 'vendor' else ''
        ids_path = cell / (row['tag'] + suffix + '.ids')
        ids = ids_path.read_bytes()
        row.update(cell=cell.name, ids_sha256=hashlib.sha256(ids).hexdigest(),
                   split=int(env['MEMRA_GLM5_TP_INDEXER_SPLIT_PRIME']
                             if 'MEMRA_GLM5_TP_INDEXER_SPLIT_PRIME' in env
                             else env['MEMRA_GLM5_TP_INDEXER_SPLIT']),
                   tc=int(env['MEMRA_DSA_SCORE_TC']),
                   profile=env['BOXP_PROFILE_PHASE'])
        if len(ids.split()) != row['out_tokens']:
            raise SystemExit(f'{cell.name}: ids count mismatch')
        rows.append(row)

comparisons = []
for path in sorted(root.glob('pair*-off')):
    twin = path.with_name(path.name[:-3] + 'on')
    if not twin.exists() or not (twin / 'exit').exists():
        continue
    for ids_path in sorted(path.glob('*.ids')):
        other = twin / ids_path.name
        exact = ids_path.read_bytes() == other.read_bytes()
        comparisons.append(dict(off=path.name, on=twin.name, file=ids_path.name,
                                byte_identical=exact,
                                sha256=hashlib.sha256(ids_path.read_bytes()).hexdigest()))
        if not exact:
            raise SystemExit(f'ID DIVERGENCE: {ids_path} vs {other}')

summary = dict(rows=rows, comparisons=comparisons)
(root / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('| Cell | Sampling | Prompt tokens | Prime s | Decode tok/s | IDs SHA256 |')
print('|---|---|---:|---:|---:|---|')
for r in rows:
    print(f"| {r['cell']} | {r['arm']} | {r['prompt_tokens']} | {r['prime_s']:.4f} | "
          f"{r['decode_tok_s']:.3f} | {r['ids_sha256']} |")
for context in ('128k', '1m'):
    for split in (0, 1):
        group = [r for r in rows if r['cell'].startswith('pair') and f'-{context}-' in r['cell']
                 and r['arm'] == 'greedy' and r['split'] == split]
        if group:
            print(f"{context} split={split} N={len(group)} median prime="
                  f"{statistics.median(r['prime_s'] for r in group):.4f} s median decode="
                  f"{statistics.median(r['decode_tok_s'] for r in group):.3f} tok/s")
print(f'{len(comparisons)} full-file identity comparisons passed')

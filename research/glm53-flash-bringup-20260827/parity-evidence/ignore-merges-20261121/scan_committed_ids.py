
import json, subprocess, sys, os, re
from tokenizers import Tokenizer
repo, gd = sys.argv[1], sys.argv[2]
tok = Tokenizer.from_file(gd + '/tokenizer.json')
raw = json.load(open(gd + '/tokenizer.json')); raw['model']['ignore_merges'] = False
old = Tokenizer.from_str(json.dumps(raw))
files = subprocess.run(['git','-C',repo,'ls-files','research','crates','docs','tools'],capture_output=True,text=True).stdout.split('\n')
cand = [f for f in files if (re.search(r'glm', f, re.I) and re.search(r'\.(ids|tsv|jsonl|json|txt)$', f)) or f.endswith('.rs')]
print('candidates', len(cand))
def idlist(s):
    s = s.strip()
    if not s: return None
    parts = re.split(r'[,\s]+', s)
    if len(parts) < 4 or not all(p.isdigit() for p in parts): return None
    ids = [int(p) for p in parts]
    return ids if max(ids) < 154856 else None
res = {'canon':0,'old_only':[], 'neither':0}
seen = 0
for f in cand:
    p = os.path.join(repo, f)
    if os.path.getsize(p) > 40_000_000: continue
    try: text = open(p, encoding='utf-8').read()
    except Exception: continue
    seqs = []
    if f.endswith('.ids'):
        ids = idlist(text)
        if ids: seqs.append(ids)
    elif f.endswith('.tsv'):
        for line in text.split('\n'):
            for col in line.split('\t'):
                ids = idlist(col) if ',' in col else None
                if ids: seqs.append(ids)
    if f.endswith(('.json','.jsonl','.txt','.rs')):
        for m in re.finditer(r'\[\s*(\d+(?:\s*,\s*\d+){3,})\s*\]', text):
            ids = idlist(m.group(1))
            if ids: seqs.append(ids)
    for ids in seqs:
        seen += 1
        t = tok.decode(ids, skip_special_tokens=False)
        new = tok.encode(t, add_special_tokens=False).ids
        o = old.encode(t, add_special_tokens=False).ids
        if new == o: continue      # the flag does not matter for this text
        if ids == new: res['canon'] += 1
        elif ids == o: res['old_only'].append(f)
        else: res['neither'] += 1
print('id sequences', seen, 'flag-sensitive: canonical', res['canon'], 'old-encoding', len(res['old_only']), 'neither', res['neither'])
for f in sorted(set(res['old_only'])): print('OLD', f)

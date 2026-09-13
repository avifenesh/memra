"""Versioned code-workload prompts; split by source file before generation."""
import hashlib
import json
from pathlib import Path
import re
import sys

repo = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=True)
limits = {'train':192, 'calibration':24, 'heldout':24}
counts = dict.fromkeys(limits, 0)
records = []
files = sorted((repo/'crates').rglob('*.rs'), key=lambda p: hashlib.sha256(p.relative_to(repo).as_posix().encode()).hexdigest())
for path in files:
    relative = path.relative_to(repo).as_posix()
    if '/bin/' in relative or '/tests/' in relative or 'test' in path.name:
        continue
    bucket = int(hashlib.sha256(relative.encode()).hexdigest()[:8],16) % 10
    split = 'train' if bucket < 6 else 'calibration' if bucket < 8 else 'heldout'
    if counts[split] >= limits[split]:
        continue
    source = path.read_text(encoding='utf-8', errors='strict')
    match = re.search(r'(?m)^(?:pub(?:\([^)]*\))? )?(?:unsafe )?fn [a-zA-Z_][a-zA-Z_0-9]*', source)
    if not match:
        continue
    excerpt = '\n'.join(source[match.start():].splitlines()[:36])[:2600]
    prompt = ('Review this Rust source excerpt. Explain its purpose, inputs, outputs and state changes, '
              'then identify two important edge cases and propose focused tests. If surrounding definitions '
              'are missing, state the assumption instead of inventing behavior.\n\n```rust\n'+excerpt+'\n```')
    identifier = hashlib.sha256(relative.encode()).hexdigest()[:16]
    folder = out/split
    folder.mkdir(exist_ok=True)
    destination = folder/f'{identifier}.txt'
    destination.write_text(prompt, encoding='utf-8', newline='\n')
    records.append({'id':identifier,'split':split,'source_file':relative,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()})
    counts[split] += 1
(out/'manifest.json').write_text(json.dumps({'workload':'native-source-code-review-pilot','split_unit':'source_file','source_commit':'3bb21381848067d922dec1320261846f99ceb29a','counts':counts,'records':records},indent=2), encoding='utf-8')
assert all(counts[s] >= min(limits[s],12) for s in limits), counts
print(json.dumps(counts))

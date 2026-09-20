"""Verify saved oracle artifacts with HF only. No Memra implementation is loaded."""
from pathlib import Path
import hashlib,json
from tokenizers import Tokenizer
import tokenizers
assert tokenizers.__version__=='0.22.2'
root=Path(__file__).resolve().parent
summary=json.loads((root/'summary.json').read_text())
for name,digest in summary['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
fixtures=json.loads((root/'fixtures.json').read_text());count=0
for variant in fixtures['variants']:
    path=root/variant['tokenizer_json']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==variant['tokenizer_json_sha256']
    t=Tokenizer.from_file(str(path))
    for case in variant['cases']:
        for mode in [False,True]:
            assert t.encode(case['raw'],add_special_tokens=mode).ids==case['expected'][str(mode).lower()]['ids'],case['name']
        count+=1
matrix=json.loads((root/'special-mode-matrix.json').read_text());special=0
for variant in matrix['variants']:
    t=Tokenizer.from_file(str(root/variant['tokenizer_json']))
    for case in variant['cases']:
        for parse in [False,True]:
            t.encode_special_tokens=not parse
            for mode in [False,True]:
                assert t.encode(case['raw'],add_special_tokens=mode).ids==case['expected_by_parse_special'][str(parse).lower()][str(mode).lower()]['ids']
        special+=1
baseline=json.loads((root/'baseline-551.json').read_text());t=Tokenizer.from_file(str(root/'pinned/tokenizer.json'))
for case in baseline['cases']:
    assert t.normalizer.normalize_str(case['raw'])==case['nfc']
    for mode in [False,True]:
        assert t.encode(case['raw'],add_special_tokens=mode).ids==case['hf_declared_nfc'][str(mode).lower()]['ids']
print(json.dumps({'oracle':'tokenizers 0.22.2','baseline_cases_verified':len(baseline['cases']),
                  'serialized_variant_cases_verified':count,'special_recognition_cases_verified':special,
                  'all_hashes_match':True,'Memra_or_GPU_executed':False},indent=2))

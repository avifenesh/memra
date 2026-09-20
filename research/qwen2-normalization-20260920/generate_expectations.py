"""Offline independent NFC oracle for issue 554; no native implementation is imported.
Run with this directory's .venv/bin/python (tokenizers==0.22.2).
All inputs come from frozen #540 fixtures and locally preserved tokenizer-only artifacts.
"""
import copy, gzip, hashlib, json, struct, subprocess, sys
from pathlib import Path
import tokenizers
from tokenizers import AddedToken, Tokenizer
from tokenizers import normalizers

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
COMMIT = 'fdb781362c8366e11db492cba14bfe1dc6bb6941'
assert tokenizers.__version__ == '0.22.2'

def sha(data): return hashlib.sha256(data).hexdigest()
def save(name, obj):
    path = ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
def frozen(path):
    return subprocess.check_output(['git', '-C', str(REPO), 'show', COMMIT + ':' + path])
def snapshot(tok, text):
    result = {}
    for mode in (False, True):
        e = tok.encode(text, add_special_tokens=mode)
        result[str(mode).lower()] = {
            'ids': e.ids, 'tokens': e.tokens, 'offsets': e.offsets,
            'special_tokens_mask': e.special_tokens_mask,
            'decode_keep_special': tok.decode(e.ids, skip_special_tokens=False),
            'decode_skip_special': tok.decode(e.ids, skip_special_tokens=True),
        }
    return result

tiny_bytes = frozen('crates/memra-tokenizer/tests/fixtures/qwen2-tokenizer.json')
tiny = json.loads(tiny_bytes)
(ROOT / 'tiny-base-tokenizer.json').write_bytes(tiny_bytes)
old_oracle = json.loads(frozen('crates/memra-tokenizer/tests/fixtures/qwen2-oracle.json'))
manifest = json.loads(frozen('research/qwen2-tokenizer-20260920/manifest.json'))
for name, digest in manifest['hashes'].items():
    assert sha((ROOT / 'pinned' / name).read_bytes()) == digest, name
pinned_bytes = (ROOT / 'pinned/tokenizer.json').read_bytes()
full = Tokenizer.from_str(pinned_bytes.decode())
plain_doc = json.loads(pinned_bytes); plain_doc['normalizer'] = None
plain = Tokenizer.from_str(json.dumps(plain_doc))
raw_corpus = gzip.decompress(frozen('research/qwen2-tokenizer-20260920/corpus.tsv.gz'))
normalized_corpus = gzip.decompress(frozen('research/qwen2-tokenizer-20260920/corpus-normalized.tsv.gz'))
reference = gzip.decompress(frozen('research/qwen2-tokenizer-20260920/reference.tsv.gz'))
refs = {r.split('\t')[0]: r.split('\t')[1:] for r in reference.decode().splitlines()}
norm_inputs = {name: bytes.fromhex(value).decode() for name, value in (r.split('\t', 1) for r in normalized_corpus.decode().splitlines())}
baseline = []
for i, line in enumerate(raw_corpus.decode().splitlines()):
    name, value = line.split('\t', 1); text = bytes.fromhex(value).decode()
    normalized = full.normalizer.normalize_str(text)
    assert normalized == norm_inputs[name]
    expected = snapshot(full, text)
    without = snapshot(plain, text)
    for mode, column in [('true', 0), ('false', 1)]:
        old_ids = [int(v) for v in refs[name][column].split(',') if v]
        assert expected[mode]['ids'] == old_ids
        assert expected[mode]['ids'] == plain.encode(normalized, add_special_tokens=(mode == 'true')).ids
    baseline.append({'name': name, 'category': 'original-39' if i < 39 else 'seeded-512',
        'raw': text, 'raw_utf8_hex': value, 'nfc': normalized,
        'nfc_changes_text': text != normalized, 'hf_declared_nfc': expected,
        'identity_normalizer_ids': {mode: v['ids'] for mode, v in without.items()}})
assert len(baseline) == 551
changed = sum(r['nfc_changes_text'] for r in baseline)
changed_ids = sum(r['hf_declared_nfc']['false']['ids'] != r['identity_normalizer_ids']['false'] for r in baseline)
assert changed == changed_ids == 250
save('baseline-551.json', {'schema': 'memra-independent-nfc-oracle-v1', 'source_commit': COMMIT,
    'oracle': 'Hugging Face tokenizers 0.22.2', 'pinned_tokenizer': manifest,
    'corpus_sha256': sha(raw_corpus), 'normalized_corpus_sha256': sha(normalized_corpus),
    'reference_sha256': sha(reference), 'cases': baseline,
    'summary': {'cases': 551, 'changed_by_nfc': changed, 'ids_changed_without_nfc': changed_ids,
                'unchanged': 301, 'original_reference_parity': '551/551 both modes'}})

unicode_texts = [
    '', 'e\u0301', 'é', 'cafe\u0301', 'e\u0301\u0323', 'e\u0323\u0301',
    'a\u0315\u0300', 'a\u0301\u0300', 'a\u0300\u0301', '\u0315\u0300a',
    '\u0344', '\u212b', '\u2126', 'Ａ ﬁ ①', 'א\u05b8\u05b9', 'ש\u05c1\u05b8',
    'م\u0651\u064e', 'ا\u0654', 'क\u093c', '\u0958', 'বাংলা', '가', '각',
    '가', '각', '👩\u200d💻', 'a\x00e\u0301b', '\t e\u0301\r\n', '\u00a0e\u0301\u00a0',
    'x\u0301y', '中文かな', 'a\u034f\u0301', 'a\ufe0f\u0301',
]
variants = []
def added(content, normalized, special=False, **kwargs):
    return dict(content=content, normalized=normalized, special=special,
                single_word=kwargs.get('single_word', False), lstrip=kwargs.get('lstrip', False),
                rstrip=kwargs.get('rstrip', False))
def variant(name, additions, texts, normalizer='NFC', note='', split_pattern=None):
    doc = copy.deepcopy(tiny)
    if split_pattern is not None:
        doc['pre_tokenizer']['pretokenizers'][0]['pattern']['Regex'] = split_pattern
    if normalizer == 'missing': doc.pop('normalizer', None)
    else: doc['normalizer'] = None if normalizer is None else {'type': normalizer}
    tok = Tokenizer.from_str(json.dumps(doc))
    if additions: tok.add_tokens([AddedToken(**a) for a in additions])
    serialized = json.loads(tok.to_str())
    # Preserve the exact importer declaration being tested for absent vs null.
    if normalizer == 'missing': serialized.pop('normalizer', None)
    elif normalizer is None: serialized['normalizer'] = None
    relative = f'variants/{name}/tokenizer.json'
    save(relative, serialized)
    save(f'variants/{name}/generation_config.json', {'eos_token_id': old_oracle['eos_token_id']})
    cases = []
    for i, text in enumerate(texts):
        cases.append({'name': f'{name}-{i:03d}', 'raw': text, 'raw_utf8_hex': text.encode().hex(),
            'whole_string_nfc_for_diagnostic_only': normalizers.NFC().normalize_str(text),
            'expected': snapshot(tok, text)})
    variants.append({'name': name, 'tokenizer_json': relative,
        'tokenizer_json_sha256': sha((ROOT / relative).read_bytes()),
        'normalizer_declaration': 'absent' if normalizer == 'missing' else serialized['normalizer'],
        'requested_added_tokens': additions,
        'added_token_ids': {a['content']: tok.token_to_id(a['content']) for a in additions},
        'note': note, 'cases': cases})

variant('nfc-unicode', [], unicode_texts, note='NFC only: canonical composition/reordering, not compatibility or case folding.')
variant('identity-missing', [], unicode_texts, normalizer='missing')
variant('identity-null', [], unicode_texts, normalizer=None)
variant('original-39-identity', [], [c['text'] for c in old_oracle['cases']], normalizer=None,
        note='Must preserve every original #540 split/BPE golden exactly.')
variant('original-39-nfc', [], [c['text'] for c in old_oracle['cases']],
        note='Same #540 vocabulary and texts; only the NFC declaration changes.')
qwen2_pattern=tiny['pre_tokenizer']['pretokenizers'][0]['pattern']['Regex']
qwen35_pattern=qwen2_pattern.replace(r']?\p{L}+',r']?[\p{L}\p{M}]+').replace(r'[^\s\p{L}\p{N}]+',r'[^\s\p{L}\p{M}\p{N}]+')
glm_pattern=qwen2_pattern.replace(r'|\p{N}|',r'|\p{N}{1,3}|')
assert qwen35_pattern != qwen2_pattern and glm_pattern != qwen2_pattern
for name,pattern in [('qwen35-identity-control',qwen35_pattern),('glm-identity-control',glm_pattern)]:
    variant(name,[],['e\u0301','x\u0301y','مُحَمَّد','1234567','שָׁלוֹם','a\t\tb'],normalizer='missing',
            split_pattern=pattern,note='No normalizer declaration: preserve the existing family-specific split; never infer NFC from the pretokenizer.')
marker_texts = ['<e\u0301>', '<é>', 'x<e\u0301>y', '<e\u0301><é>', ' <e\u0301> ', '<e\u0301>\u0323']
for normalized in [False, True]:
    for special in [False, True]:
        variant(f'decomposed-n{int(normalized)}-s{int(special)}',
                [added('<e\u0301>', normalized, special)], marker_texts,
                note='Content normalized=true is matched after NFC; special input recognition still occurs with add_special_tokens=false.')
        variant(f'raw-mark-n{int(normalized)}-s{int(special)}',
                [added('\u0301', normalized, special)],
                ['e\u0301','é','a\u0301\u0323','a\u0323\u0301','x\u0301y','\u0301e\u0301'],
                note='Raw added marks partition input before normalization; whole-string NFC first is wrong.')
    variant(f'composed-n{int(normalized)}', [added('<é>', normalized)], marker_texts)
    variant(f'single-word-n{int(normalized)}', [added('e\u0301', normalized, single_word=True)],
            ['e\u0301','é','xe\u0301','e\u0301x','_e\u0301',' e\u0301 ', ',e\u0301!', 'אe\u0301', 'e\u0301\u0308'])
    variant(f'strip-n{int(normalized)}', [added('<e\u0301>', normalized, lstrip=True, rstrip=True)],
            ['A \t<e\u0301> \n B','A\u00a0<e\u0301>\u00a0B',' <é> ','<e\u0301><e\u0301>'])
raw_prefix = [added('<R>', False), added('<R>é', True)]
for name, adds in [('raw-prefix',raw_prefix),('raw-prefix-reverse',list(reversed(raw_prefix))),
                   ('normalized-longest',[added('<R>', True),added('<R>é', True)]),
                   ('raw-suffix',[added('>',False),added('<e\u0301>',True)])]:
    variant(name, adds, ['<R>e\u0301','<R>é','x<R>e\u0301y','<e\u0301>','<é>'],
            note='Raw stage precedes normalized stage; longest match only competes within its stage.')
variant('raw-longest', [added('<R>',False),added('<R>e\u0301',False)], ['<R>e\u0301','<R>é'])
variant('raw-jamo', [added('ᅡ', False)], ['가','가','각'])
for order in [False,True]:
    aliases=[added('<e\u0301>',True),added('<é>',True)]
    if order: aliases.reverse()
    variant(f'normalized-collision-{int(order)}',aliases,['<e\u0301>','<é>'],
            note='Diagnostic collision/registration-order case. Native must explicitly decide whether to preserve this accepted HF behavior or refuse an ambiguous supported-contract extension.')
variant('literal-special-modes', [added('<e\u0301>',False,True),added('<N:e\u0301>',True,True),added('<x>',False)],
        ['<|endoftext|>','<e\u0301>','<é>','<N:e\u0301>','<N:é>','x<|endoftext|>y','<e\u0301><x>','e\u0301<x>e\u0301'],
        note='add_special_tokens toggles post-processing, not recognition of literal special added tokens. This fixture has no automatic BOS/EOS insertion.')
variant('special-mark-and-ordinary', [added('\u0301',False,True),added('<x>',False)],
        ['e\u0301','e\u0301<x>e\u0301','é<x>'],
        note='Turning special recognition off removes the raw special-token barrier, while ordinary added tokens still match.')
# Strong comparisons make an oracle that accidentally normalizes globally fail generation.
byname={v['name']:v for v in variants}
ids=lambda n,i=0:byname[n]['cases'][i]['expected']['false']['ids']
assert ids('raw-mark-n0-s0') != ids('raw-mark-n1-s0')
assert ids('raw-prefix') != ids('normalized-longest')
assert ids('decomposed-n0-s0',1) != ids('decomposed-n1-s0',1)
assert ids('nfc-unicode',1) != ids('identity-null',1)
assert ids('qwen35-identity-control') == old_oracle['qwen35_combining_ids']
assert ids('glm-identity-control',3) != ids('original-39-identity',23)
for c, old in zip(byname['original-39-identity']['cases'],old_oracle['cases']):
    assert c['expected']['false']['ids'] == old['ids']
for v in variants:
    for c in v['cases']:
        assert c['expected']['true']['ids'] == c['expected']['false']['ids']
save('fixtures.json', {'schema':'memra-independent-nfc-added-token-fixtures-v1','oracle':'tokenizers 0.22.2',
    'source_commit':COMMIT,'base_fixture_sha256':sha(tiny_bytes),'base_vocab_size':len(tiny['model']['vocab']),
    'offset_units':'HF offsets are Python character positions, not UTF-8 byte offsets; diagnostic only unless native API exposes them.',
    'normalization_note':'whole_string_nfc_for_diagnostic_only is not the normalization oracle when raw added tokens partition the string.',
    'native_test_priority':['IDs and loader acceptance','no global-NFC-before-raw-stage shortcut','NFC idempotence and unchanged default','special recognition independent of add_special_tokens'],
    'variants':variants})

special_modes=[]
for name in ['literal-special-modes','special-mark-and-ordinary','pinned-real-specials']:
    if name=='pinned-real-specials':
        tok=Tokenizer.from_str(pinned_bytes.decode())
        texts=['<|endoftext|>','<|im_start|>user\ne\u0301<|im_end|>',
               'e\u0301<|im_start|>e\u0301<|im_end|>','x<|im_end|>y']
        tokenizer_file='pinned/tokenizer.json'
    else:
        v=byname[name];tok=Tokenizer.from_file(str(ROOT/v['tokenizer_json']))
        texts=[c['raw'] for c in v['cases']];tokenizer_file=v['tokenizer_json']
    cases=[]
    for text in texts:
        result={}
        for parse_special in [True,False]:
            tok.encode_special_tokens=not parse_special
            result[str(parse_special).lower()]=snapshot(tok,text)
        cases.append({'raw':text,'expected_by_parse_special':result})
    special_modes.append({'name':name,'tokenizer_json':tokenizer_file,'cases':cases})
assert special_modes[1]['cases'][0]['expected_by_parse_special']['true']['false']['ids'] != special_modes[1]['cases'][0]['expected_by_parse_special']['false']['false']['ids']
save('special-mode-matrix.json',{'schema':'memra-independent-special-recognition-v1',
    'HF_to_native_mapping':'HF encode_special_tokens = !native parse_special; add_special_tokens is a separate post-processing switch.',
    'scope':'These Qwen2 fixtures do not insert BOS/EOS automatically; add_special_tokens true/false therefore keep identical input-recognition IDs. This does not claim all postprocessors behave that way.',
    'variants':special_modes})

programs=[
 ('missing', 'missing', 'accept_identity'), ('null',None,'accept_identity'),
 ('nfc',{'type':'NFC'},'accept_nfc'),
 ('nfd',{'type':'NFD'},'explicit_refusal_required_outside_NFC_scope'),
 ('nfkc',{'type':'NFKC'},'explicit_refusal_required_outside_NFC_scope'),
 ('nfkd',{'type':'NFKD'},'explicit_refusal_required_outside_NFC_scope'),
 ('lowercase',{'type':'Lowercase'},'explicit_refusal_required_outside_NFC_scope'),
 ('strip',{'type':'Strip','strip_left':True,'strip_right':True},'explicit_refusal_required_outside_NFC_scope'),
 ('strip-accents',{'type':'StripAccents'},'explicit_refusal_required_outside_NFC_scope'),
 ('bert',{'type':'BertNormalizer','clean_text':True,'handle_chinese_chars':True,'strip_accents':None,'lowercase':True},'explicit_refusal_required_outside_NFC_scope'),
 ('prepend',{'type':'Prepend','prepend':'X'},'explicit_refusal_required_outside_NFC_scope'),
 ('replace',{'type':'Replace','pattern':{'String':'é'},'content':'E'},'explicit_refusal_required_outside_NFC_scope'),
 ('mixed-sequence',{'type':'Sequence','normalizers':[{'type':'NFC'},{'type':'Lowercase'}]},'explicit_refusal_required_outside_NFC_scope'),
 ('sequence-nfc',{'type':'Sequence','normalizers':[{'type':'NFC'}]},'contract_choice_support_exactly_or_refuse_explicitly'),
 ('sequence-empty',{'type':'Sequence','normalizers':[]},'contract_choice_support_identity_or_refuse_explicitly'),
 ('unknown',{'type':'UnknownNormalizer554'},'refuse_malformed_or_unknown'),
 ('empty-object',{},'refuse_malformed_or_unknown'), ('string','NFC','refuse_malformed_or_unknown'),
 ('array',[],'refuse_malformed_or_unknown'),('boolean',True,'refuse_malformed_or_unknown'),
 ('missing-sequence-list',{'type':'Sequence'},'refuse_malformed_or_unknown'),
 ('null-type',{'type':None},'refuse_malformed_or_unknown'),
]
imports=[]
for name,decl,policy in programs:
    d=copy.deepcopy(tiny)
    if decl=='missing':d.pop('normalizer',None)
    else:d['normalizer']=decl
    item={'name':name,'normalizer_declaration':decl,'native_acceptance_expectation':policy}
    try:
        t=Tokenizer.from_str(json.dumps(d));item['hf_accepts']=True
        item['hf_normalized_examples']={text: t.normalizer.normalize_str(text) if t.normalizer else text for text in [' E\u0301 ','Ａ ﬁ ①','ש\u05c1\u05b8']}
    except Exception as e:item.update(hf_accepts=False,hf_error=str(e))
    imports.append(item)
save('normalizer-import-contract.json',{'schema':'memra-nfc-only-import-contract-v1',
    'note':'HF acceptance of another normalizer is diagnostic, not a request to implement it. Missing/null identity and direct NFC are required. Optional empty/NFC-only Sequence wrappers must be supported exactly or refused explicitly; never drop unknown programs.',
    'cases':imports})

# Prove the preserved metadata-only GGUF itself declares no normalization program.
b=(ROOT/'pinned/tokenizer.gguf').read_bytes();pos=0
assert b[:4]==b'GGUF';version,tensors,meta=struct.unpack_from('<IQQ',b,4);pos=24
assert tensors==0
sizes={0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}
def string():
    global pos
    n=struct.unpack_from('<Q',b,pos)[0];pos+=8;value=b[pos:pos+n].decode();pos+=n;return value
def skip(kind):
    global pos
    if kind==8:return string()
    if kind==9:
        sub,n=struct.unpack_from('<IQ',b,pos);pos+=12
        for _ in range(n):skip(sub)
    else:pos+=sizes[kind]
keys=[];scalars={}
for _ in range(meta):
    key=string();kind=struct.unpack_from('<I',b,pos)[0];pos+=4;value=skip(kind);keys.append(key)
    if value is not None:scalars[key]=value
assert not any('normaliz' in k.lower() for k in keys)
selected=[baseline[i] for i in [1,2,5,9,10,11,12,14]]
save('gguf-import-contract.json',{'schema':'memra-explicit-normalizer-import-contract-v1',
    'fixture':'pinned/tokenizer.gguf','fixture_sha256':sha(b),'model_tensors':0,'metadata_keys':keys,'metadata_strings':scalars,
    'declared_normalization':'none','family_inference_permitted':False,
    'default_expected_behavior':'identity; preserve existing #540 literal split/BPE IDs on raw input even when the source HF tokenizer independently declares NFC',
    'explicit_import_requirement':'Only a documented, explicitly preserved normalization declaration may activate NFC. Do not invent an existing GGUF standard key, infer from qwen2/qwen35/family, or quietly discard a declared unsupported program. A conversion promising exact parity but unable to represent declared NFC must refuse. Existing GGUF artifacts with no normalization declaration remain identity.',
    'NFC_field_name':'schema choice for parent; no implicit field guessed by this review',
    'same_vocabulary_raw_cases':[{'name':c['name'],'raw':c['raw'],'default_no_declaration_ids':c['identity_normalizer_ids'],
        'explicit_NFC_reference_ids':{mode:v['ids'] for mode,v in c['hf_declared_nfc'].items()}} for c in selected]})
outputs=['baseline-551.json','fixtures.json','special-mode-matrix.json','normalizer-import-contract.json','gguf-import-contract.json']
summary={'oracle':'Hugging Face tokenizers 0.22.2','python':sys.version.split()[0], 'source_commit':COMMIT,
    'baseline_cases':551,'raw_NFC_vs_identity_ID_deltas':250,'added_token_unicode_variants':len(variants),
    'variant_case_count':sum(len(v['cases']) for v in variants),'normalizer_import_cases':len(imports),
    'special_recognition_cases':sum(len(v['cases']) for v in special_modes),
    'artifact_inputs':manifest,'files':{n:sha((ROOT/n).read_bytes()) for n in outputs},
    'no_native_implementation_imported':True,'model_weights_or_GPU_used':False}
save('summary.json',summary)
print(json.dumps(summary,indent=2))

"""Regenerate pinned tokenizer-only parity inputs; no model weights or GPU work."""
import hashlib, json, random, struct, urllib.request
from pathlib import Path
from tokenizers import Tokenizer
import tokenizers

assert tokenizers.__version__ == '0.22.2'
repo=Path(__file__).resolve().parents[2]
root=Path(__file__).parent/'artifacts'
root.mkdir(exist_ok=True)
model='Qwen/Qwen2.5-0.5B-Instruct'
def get(url):
    with urllib.request.urlopen(url,timeout=20) as r:return r.read()
revision='7ae557604adf67be50417f59c2c2f167def9a775'
hashes={}
for name in ['tokenizer.json','tokenizer_config.json','generation_config.json']:
    data=get(f'https://huggingface.co/{model}/resolve/{revision}/{name}')
    (root/name).write_bytes(data)
    hashes[name]=hashlib.sha256(data).hexdigest()
tok=Tokenizer.from_file(str(root/'tokenizer.json'))
fixture=json.loads((repo/'crates/memra-tokenizer/tests/fixtures/qwen2-oracle.json').read_text())
texts=[x['text'] for x in fixture['cases']]
rng=random.Random(540)
alphabet="aAbCxyz 0129\t\r\n\u00a0é\u0301\u0308\u017f'!?-🙂אבד\u05b8\u05b9\u05c1محمد\u064f\u064e\u0651नमस्ते中文かな①Ⅸ\x00"
texts += [''.join(rng.choice(alphabet) for _ in range(rng.randrange(1,101))) for _ in range(512)]
with (root/'corpus.tsv').open('w') as corpus,(root/'reference.tsv').open('w') as refs:
    for i,text in enumerate(texts):
        name=f'case-{i:04d}'
        corpus.write(name+'\t'+text.encode().hex()+'\n')
        fields=[','.join(map(str,tok.encode(text,add_special_tokens=mode).ids)) for mode in [True,False]]
        refs.write(name+'\t'+'\t'.join(fields)+'\n')

raw=json.loads((root/'tokenizer.json').read_text())
vocab=tok.get_vocab(with_added_tokens=True)
tokens=['']*(max(vocab.values())+1)
for word,i in vocab.items():tokens[i]=word
assert all(tokens)
types=[1]*len(tokens)
for entry in raw['added_tokens']:
    if entry['special']:types[entry['id']]=3
merges=[' '.join(x) if isinstance(x,list) else x for x in raw['model']['merges']]
config=json.loads((root/'tokenizer_config.json').read_text())
eos=tok.token_to_id(config['eos_token'])
def string(s):
    data=s.encode();return struct.pack('<Q',len(data))+data
def array(kind,items,encode):return struct.pack('<IQ',kind,len(items))+b''.join(encode(v) for v in items)
kv=[
    ('tokenizer.ggml.model',8,string('gpt2')),
    ('tokenizer.ggml.pre',8,string('qwen2')),
    ('tokenizer.ggml.tokens',9,array(8,tokens,string)),
    ('tokenizer.ggml.token_type',9,array(5,types,lambda v:struct.pack('<i',v))),
    ('tokenizer.ggml.merges',9,array(8,merges,string)),
    ('tokenizer.ggml.eos_token_id',4,struct.pack('<I',eos)),
    ('tokenizer.ggml.add_bos_token',7,b'\x00'),
]
gguf=b'GGUF'+struct.pack('<IQQ',3,0,len(kv))
gguf+=b''.join(string(key)+struct.pack('<I',kind)+value for key,kind,value in kv)
gguf+=b'\x00'*((-len(gguf))%32)
(root/'tokenizer.gguf').write_bytes(gguf)
manifest={'model':model,'revision':revision,'tokenizers_version':tokenizers.__version__,
          'hashes':hashes,'cases':len(texts),'random_seed':540,
          'gguf':'metadata-only wrapper of the same pinned HF vocabulary/merges, not a model weight artifact'}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))

# Isolate pretokenizer/BPE parity from the independently tracked NFC omission.
with (root/'corpus-normalized.tsv').open('w') as corpus,(root/'reference-normalized.tsv').open('w') as refs:
    for i,text in enumerate(texts):
        text=tok.normalizer.normalize_str(text)
        name=f'case-{i:04d}'
        corpus.write(name+'\t'+text.encode().hex()+'\n')
        fields=[','.join(map(str,tok.encode(text,add_special_tokens=mode).ids)) for mode in [True,False]]
        refs.write(name+'\t'+'\t'.join(fields)+'\n')

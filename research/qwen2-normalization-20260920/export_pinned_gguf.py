"""Export a tokenizer-only test artifact with the explicit Memra input program.

This uses the pinned Qwen2.5 vocabulary, not model weights or a general model converter.
"""
import argparse, hashlib, json, struct
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('directory', type=Path)
parser.add_argument('output', type=Path)
a=parser.parse_args()
raw=(a.directory/'tokenizer.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539'
tj=json.loads(raw);cfg=json.loads((a.directory/'tokenizer_config.json').read_text())
vocab=tj['model']['vocab'];added=tj['added_tokens'];n=max(list(vocab.values())+[t['id'] for t in added])+1
pieces=['']*n;types=[1]*n
for text,token_id in vocab.items():pieces[token_id]=text
for token in added:pieces[token['id']]=token['content'];types[token['id']]=3 if token['special'] else 4
assert all(pieces)
merges=[' '.join(x) if isinstance(x,list) else x for x in tj['model']['merges']]
eos=next(t['id'] for t in added if t['content']==cfg['eos_token'])
def string(s):
 b=s.encode();return struct.pack('<Q',len(b))+b
def array(kind,values,encode):return struct.pack('<IQ',kind,len(values))+b''.join(encode(x) for x in values)
program={'version':1,'normalizer':tj.get('normalizer'),'added_tokens':added}
metadata=[('tokenizer.ggml.model',8,string('gpt2')),('tokenizer.ggml.pre',8,string('qwen2')),
 ('tokenizer.ggml.tokens',9,array(8,pieces,string)),('tokenizer.ggml.token_type',9,array(5,types,lambda x:struct.pack('<i',x))),
 ('tokenizer.ggml.merges',9,array(8,merges,string)),('tokenizer.ggml.eos_token_id',4,struct.pack('<I',eos)),
 ('tokenizer.ggml.add_bos_token',7,b'\0'),('tokenizer.memra.input_program',8,string(json.dumps(program,ensure_ascii=False)))]
b=b'GGUF'+struct.pack('<IQQ',3,0,len(metadata))
b+=b''.join(string(key)+struct.pack('<I',kind)+value for key,kind,value in metadata);b+=b'\0'*((-len(b))%32)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(b)
print(json.dumps({'artifact':a.output.name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'model_tensors':0,'input_program_version':1},indent=2))

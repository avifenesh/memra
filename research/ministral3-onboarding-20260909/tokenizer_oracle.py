"""Remote-only frozen 2,000-line Tekken token-id oracle, Hebrew/English/code."""
import json,random,sys,hashlib
from pathlib import Path
from tokenizers import Tokenizer
root=Path(sys.argv[1]); out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
tok=Tokenizer.from_file(str(root/'tokenizer.json'))
rng=random.Random(20260909)
he=['שלום עולם','שָׁלוֹם לְכֻלָּם','נא להכין סיכום של הישיבה','המסמך נשלח ביום ראשון','מה מצב ההזמנה מספר','בדוק את לוח השנה והחזר תשובה בעברית','סה״כ לתשלום','הרחוב נקרא על שם סופר','מחר נשלח את הדוח למנהלת']
en=['HTTPServer','getURLValue','Hello WORLD','I\'m sure we\'ll test it','camelCase PascalCase UPPERCASE','Read the document and summarize it','APIResponse JSONParser UTF8']
code=['fn main() { println!("שלום"); }','SELECT id, name FROM users WHERE active = true;','def add(a, b):\n    return a + b','const result = await tool({city: "תל אביב"});','/* comment */ // trailing/\n','x = {"text": "[TOOL_CALLS]", "n": 12345}']
ws=[' ','  ','\t','\n','\r\n','\u00a0','\u2009','\u2028']
lines=[]
for i in range(2000):
 groups=[he,en,code];g=groups[i%3]
 text=rng.choice(ws).join([rng.choice(g),str(i),rng.choice(g)])
 if i%7==0:text+=rng.choice(ws)+rng.choice(['אבְג','ǅǆ','ΑΒΓΔabc','Ⅳ²۳','emoji🙂🏳️‍🌈','\u0000','e\u0301','क्','１２３'])
 lines.append((f'case-{i:04d}',text))
(out/'corpus.tsv').write_text(''.join(n+'\t'+t.encode().hex()+'\n' for n,t in lines))
(out/'reference-ids.tsv').write_text(''.join(n+'\t'+','.join(map(str,tok.encode(t,add_special_tokens=True).ids))+'\t'+','.join(map(str,tok.encode(t,add_special_tokens=False).ids))+'\n' for n,t in lines))
(out/'tokenizer-oracle.json').write_text(json.dumps({'rows':len(lines),'source_tokenizer_sha256':hashlib.sha256((root/'tokenizer.json').read_bytes()).hexdigest(),'seed':20260909,'categories':['Hebrew','English','code'],'normalization':'none','add_special_tokens':[True,False]},indent=2))
print('tokenizer oracle rows',len(lines),flush=True)

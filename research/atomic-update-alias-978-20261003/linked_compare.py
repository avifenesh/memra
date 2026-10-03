"""Inspect actual linked witnesses. Never discard an unresolved target or byte."""
from pathlib import Path
import struct, json, re, subprocess, hashlib, difflib
R=Path(__file__).resolve().parent
O=R/'linked-proof';O.mkdir(exist_ok=True)
def h(b):return hashlib.sha256(b).hexdigest()
def identity(s):return re.sub(r'\.llvm\.\d+', '.llvm.LOCAL', s)
class Elf:
 def __init__(self,arm):
  self.arm=arm;self.path=R/f'emit-{arm}-server/server-tests-{arm}';self.b=self.path.read_bytes()
  e=struct.unpack_from('<16sHHIQQQIHHHHHH',self.b);assert e[0][:6]==b'\x7fELF\x02\x01'
  self.secs=[struct.unpack_from('<IIQQQQIIQQ',self.b,e[6]+i*e[11]) for i in range(e[12])]
  self.symbols={};self.byid={};self.reloc={};self.external={};self.instances={};self.all=[]
  for s in self.secs:
   if s[1] not in (2,11):continue
   st=self.secs[s[6]];strings=self.b[st[4]:st[4]+st[5]]
   for off in range(s[4],s[4]+s[5],s[9]):
    n,info,other,ndx,val,size=struct.unpack_from('<IBBHQQ',self.b,off)
    name=strings[n:strings.find(b'\0',n)].decode()
    if val and size and name:
     row=(name,val,size,info&15);self.symbols[name]=row;self.byid.setdefault(identity(name),[]).append(row);self.instances.setdefault(name,[]).append(row);self.all.append(row)
  for s in self.secs:
   if s[1]!=4:continue
   symsec=self.secs[s[6]];strsec=self.secs[symsec[6]];strings=self.b[strsec[4]:strsec[4]+strsec[5]]
   for off in range(s[4],s[4]+s[5],s[9]):
    addr,info,add=struct.unpack_from('<QQq',self.b,off)
    if info&0xffffffff==8:self.reloc[addr]=add
    elif info&0xffffffff in (1,6,7):
     entry=symsec[4]+(info>>32)*symsec[9];n=struct.unpack_from('<I',self.b,entry)[0]
     self.external[addr]=(info&0xffffffff,strings[n:strings.find(b'\0',n)].decode(),add)
 def mem(self,addr,n):
  for s in self.secs:
   if s[3]<=addr and addr+n<=s[3]+s[5]:
    return b'\0'*n if s[1]==8 else self.b[s[4]+addr-s[3]:s[4]+addr-s[3]+n]
  raise ValueError(('unmapped',hex(addr),n))
 def resolve(self,addr,hint):
  matches=[r for r in self.instances.get(hint,[]) if r[1]<=addr<r[1]+r[2]]
  if not matches:matches=[r for r in self.all if r[1]<=addr<r[1]+r[2]]
  if not matches:return None
  return min(matches,key=lambda r:r[2])
 def asm(self,row):
  name,addr,size,kind=row
  p=subprocess.run(['llvm-objdump','-d','--no-show-raw-insn',f'--start-address={addr}',f'--stop-address={addr+size}',str(self.path)],text=True,capture_output=True,check=True)
  (O/f'{h(name.encode())[:16]}-{self.arm}.asm').write_text(p.stdout+p.stderr)
  out=[]
  for line in p.stdout.splitlines():
   m=re.match(r'^\s*([0-9a-f]+):\s+(.+)$',line)
   if m:out.append((int(m[1],16)-addr,m[2]))
  assert out,(name,p.stdout);return out
L=Elf('before');U=Elf('after');rows=[];todo=[];seen=set();fail=[]
def pair(name):
 a=L.byid[identity(name)];b=U.byid[identity(name)];assert len(a)==len(b)==1,(name,a,b);return a[0],b[0]
def target_data(la,ua,n=64,depth=0):
 a=L.mem(la,n);b=U.mem(ua,n)
 diffs=[]
 for i in range(n):
  if a[i]!=b[i]:diffs.append(i)
 # Actual RELATIVE relocation slots are resolved, not blanked.
 aa=bytearray(a);bb=bytearray(b);resolved=[]
 for off in sorted(set(x-la for x in L.external if la<=x and x+8<=la+n)):
  assert L.external[la+off]==U.external.get(ua+off),('external relocation mismatch',hex(la+off),hex(ua+off))
  resolved.append({'offset':off,'external_relocation':L.external[la+off]})
 for off in sorted(set(x-la for x in L.reloc if la<=x and x+8<=la+n)):
  assert ua+off in U.reloc,('missing relocation',hex(la+off),hex(ua+off))
  av=L.reloc[la+off];bv=U.reloc[ua+off]
  x,y=L.resolve(av,''),U.resolve(bv,'')
  if x and y and x[3]==y[3]==2:
   assert identity(x[0])==identity(y[0]) and av-x[1]==bv-y[1],('code relocation target changed',x,y)
   todo.append((x,y));witness={'equal':True,'kind':'actual code target, followed as complete function','before_symbol':x[0],'after_symbol':y[0],'before_address':hex(av),'after_address':hex(bv),'offset':av-x[1]}
  else:
   assert depth<2,('relocation recursion',hex(av),hex(bv))
   size=min(64,x[2]-(av-x[1]),y[2]-(bv-y[1])) if x and y else 64
   witness=target_data(av,bv,size,depth+1);assert witness['equal'],witness
  aa[off:off+8]=bb[off:off+8]=b'RELOCEQ!';resolved.append({'offset':off,'before_addend':hex(av),'after_addend':hex(bv),'target':witness})
 return {'equal':aa==bb,'bytes':n,'before':hex(la),'after':hex(ua),'before_sha256':h(a),'after_sha256':h(b),'actual_relative_relocations':resolved,'unresolved_byte_offsets':[i for i in range(n) if aa[i]!=bb[i]]}
def normalize(op,elf,row,other_row):
 # Every referenced address keeps its resolved symbol+offset, or its exact BB offset.
 base=row[1];end=base+row[2];refs=[]
 def repl(m):
  address=int(m[1],16);symbol=m[2]
  if base<=address<end:return 'BB+'+hex(address-base)
  stem=re.sub(r'\+0x[0-9a-f]+$','',symbol);actual=elf.resolve(address,stem)
  if actual:stem=actual[0]
  offset=address-actual[1] if actual else None
  refs.append((address,stem,offset));return 'TARGET['+identity(stem)+(('+'+hex(offset)) if offset else '')+']'
 op=re.sub(r'0x([0-9a-f]+) <([^>]+)>',repl,op)
 # RIP displacement is represented by the actual resolved target above, never erased alone.
 if refs and '%rip' in op:op=re.sub(r'-?0x[0-9a-f]+\(%rip\)','RESOLVED(%rip)',op)
 return op,refs
initial=json.loads((R/'emit-before-server/actual-site-assembly.json').read_text())
roots={identity(w['symbol']) for w in initial.values()}
for name,w in initial.items():todo.append(pair(w['symbol']))
while todo:
 a,b=todo.pop(0);key=identity(a[0])
 instance=(key,a[1],b[1])
 if instance in seen:continue
 seen.add(instance);assert len(seen)<150,'closure expanded beyond bound'
 left=L.asm(a);right=U.asm(b);assert len(left)==len(right),(a[0],len(left),len(right))
 witness={'symbol':key,'before_address':hex(a[1]),'after_address':hex(b[1]),'instruction_count':len(left),'references':[],'different_instructions':[]}
 for (lo,lop),(uo,uop) in zip(left,right):
  ln,lr=normalize(lop,L,a,b);un,ur=normalize(uop,U,b,a)
  if lo!=uo or ln!=un:witness['different_instructions'].append({'before_offset':lo,'after_offset':uo,'before':lop,'after':uop,'normalized_before':ln,'normalized_after':un})
  assert len(lr)==len(ur),(lop,uop)
  for (la,ls,lx),(ua,us,ux) in zip(lr,ur):
   ref={'offset':lo,'before_target':hex(la),'after_target':hex(ua),'before_symbol':ls,'after_symbol':us,'symbol_equal':identity(ls)==identity(us),'symbol_offset_equal':lx==ux}
   x,y=L.resolve(la,ls),U.resolve(ua,us)
   if x and y and x[3]==y[3]==2:
    raw_equal=x[2]==y[2] and L.mem(x[1],x[2])==U.mem(y[1],y[2]);ref['callee_raw_equal']=raw_equal
    ref['callee_before_sha256']=h(L.mem(x[1],x[2]));ref['callee_after_sha256']=h(U.mem(y[1],y[2]))
    if key in roots and (not raw_equal or 'end_request' in ls or 'PendingAdmissionGuard' in ls):todo.append((x,y));ref['callee_followed']=True
    elif not raw_equal:ref['callee_identity_only']=identity(ls)==identity(us) and lx==ux
   else:
    try:
     index=next(i for i,v in enumerate(left) if v[0]==lo)
     nearby='\n'.join(v[1] for v in left[max(0,index-8):index+5])
     if not x and not y and 'leaq' in lop and key.endswith('InMemoryJobStoreNtNtB6_8metering8JobStore3get') and all(s in nearby for s in ['movl\t$0x5, %ecx','movslq\t(%rdx,%rcx,4), %rcx','jmpq\t*%rcx']):
      # Six serde_json::Value discriminants, established by the actual selector above.
      rawleft=L.mem(la,24);rawright=U.mem(ua,24)
      ltargets=[la+v-a[1] for v in struct.unpack('<6i',rawleft)];utargets=[ua+v-b[1] for v in struct.unpack('<6i',rawright)]
      assert all(v in {off for off,_ in left} for v in ltargets) and all(v in {off for off,_ in right} for v in utargets)
      ref['data']={'equal':ltargets==utargets,'kind':'actual six-entry signed-relative jump table','bytes':24,'before_address':hex(la),'after_address':hex(ua),'before_raw_sha256':h(rawleft),'after_raw_sha256':h(rawright),'before_BB_offsets':ltargets,'after_BB_offsets':utargets,'selector_instruction_witness':nearby}
     else:
      size=min(64,x[2]-lx,y[2]-ux) if x and y else 8
      assert size>0
      ref['data']=target_data(la,ua,size)
    except Exception as e:ref['data']={'equal':False,'error':str(e)}
   witness['references'].append(ref)
 witness['equal']=not witness['different_instructions'] and all(r['symbol_equal'] and r['symbol_offset_equal'] and (r.get('callee_followed') or r.get('callee_identity_only') or r.get('callee_raw_equal') or r.get('data',{}).get('equal')) for r in witness['references'])
 rows.append(witness)
 if not witness['equal']:fail.append(key)
result={'before_binary_sha256':h(L.b),'after_binary_sha256':h(U.b),'rows':rows,'failed':fail,'equal':not fail,'qualification':False,'limits':'Actual complete three linked test functions and their changed direct callees, with 64-byte referenced data witnesses. Further callees retain resolved identity+offset and raw hashes, not recursively asserted code equality. Raw-equal callees are hashed. No stripping of target identities, BB offsets, constants, orderings or instructions.'}
(O/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'functions':len(rows),'failed':fail,'equal':not fail}))
assert not fail

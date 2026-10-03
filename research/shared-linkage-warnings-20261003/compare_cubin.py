#!/usr/bin/env python3
"""Bound this lane's measured three local-name changes, never generic ELF normalization."""
import hashlib,json,struct,sys
from pathlib import Path
if not __debug__:
 raise RuntimeError('proof readers require Python assertion checks')
PAIRS={'$str$9':'$str$1','$str$10':'$str$2','$str$11':'$str$3'}
def sha(b):return hashlib.sha256(b).hexdigest()
def parse(path):
 b=Path(path).read_bytes();assert b[:6]==b'\x7fELF\x02\x01'
 h=struct.unpack_from('<16sHHIQQQIHHHHHH',b);assert h[11]==64 and h[12]>0
 rows=[struct.unpack_from('<IIQQQQIIQQ',b,h[6]+i*64) for i in range(h[12])]
 nr=rows[h[13]];names=b[nr[4]:nr[4]+nr[5]]
 def label(pool,off):return pool[off:pool.index(b'\0',off)].decode()
 sections=[]
 for i,x in enumerate(rows):
  name=label(names,x[0]);payload=b'' if x[1]==8 else b[x[4]:x[4]+x[5]]
  s={'index':i,'name':name,'type':x[1],'flags':x[2],'address':x[3],'offset':x[4],'size':x[5],'link':x[6],'info':x[7],'alignment':x[8],'entry_size':x[9],'payload':payload}
  if name in ['.symtab','.nv.merc.symtab']:
   assert x[9]==24;poolrow=rows[x[6]];pool=b[poolrow[4]:poolrow[4]+poolrow[5]];syms=[]
   for j,pos in enumerate(range(0,len(payload),24)):
    sn,info,other,sec,val,size=struct.unpack_from('<IBBHQQ',payload,pos);syms.append([label(pool,sn),info,other,sec,val,size])
   s['symbols']=syms
  if name=='.strtab':s['strings']=[v.decode() for v in payload.split(b'\0')]
  sections.append(s)
 ph=[list(struct.unpack_from('<IIQQQQQQ',b,h[5]+i*h[9])) for i in range(h[10])]
 return b,h,sections,ph

def attrs(s,*,offset=True,size=True):
 return {k:v for k,v in s.items() if k not in ['payload','symbols','strings'] and (offset or k!='offset') and (size or k!='size')}
def memberships(s,ph):
 return [[i,p[1]] for i,p in enumerate(ph) if p[0]==1 and p[2]<=s['offset'] and s['offset']+s['size']<=p[2]+(p[6] if s['type']==8 else p[5])]
def compare(left,right,target):
 lb,lh,ls,lp=parse(left);rb,rh,rs,rp=parse(right);assert lh==rh
 assert len(ls)==len(rs)
 label_changes={};physical=[];different=[]
 for a,b in zip(ls,rs):
  assert attrs(a,offset=False,size=a['name']!='.strtab')==attrs(b,offset=False,size=a['name']!='.strtab'),a['name']
  if a['offset']!=b['offset']:physical.append({'index':a['index'],'name':a['name'],'before':a['offset'],'after':b['offset']})
  assert a['offset']%max(a['alignment'],1)==b['offset']%max(b['alignment'],1)==0
  if a['flags']&2:assert memberships(a,lp)==memberships(b,rp),a['name']
  if a['payload']==b['payload']:continue
  name=a['name'];different.append(name)
  if name in ['.symtab','.nv.merc.symtab']:
   assert len(a['symbols'])==len(b['symbols']);changes=[]
   for i,(x,y) in enumerate(zip(a['symbols'],b['symbols'])):
    if x==y:continue
    assert PAIRS.get(x[0])==y[0] and x[1:]==y[1:],(name,x,y)
    assert x[1:3]==([1,0] if name=='.symtab' else [13,32]),(name,x,y)
    sa=ls[x[3]];sb=rs[y[3]];expected='.nv.global.init' if name=='.symtab' else '.nv.merc.nv.global.init'
    assert sa['name']==sb['name']==expected and sa['payload']==sb['payload']
    assert attrs(sa,offset=False)==attrs(sb,offset=False)
    assert x[4]+x[5]<=sa['size']
    changes.append({'index':i,'before':x,'after':y,'backing_section_index':x[3],'backing_sha256':sha(sa['payload'])})
   assert len(changes)==3
   label_changes[name]=changes
  elif name=='.strtab':
   assert a['size']-b['size']==2
   assert [PAIRS.get(v,v) for v in a['strings']]==b['strings']
  else:raise AssertionError('unknown section delta '+name)
 assert set(different)==({'.strtab','.symtab','.nv.merc.symtab'} if target in ['100a','120a','120','120a-r4','120-r4'] else {'.strtab','.symtab'})
 padding=[];phchanges=[]
 if target in ['89','120a-r4','120-r4']:
  assert len(physical)==(819 if target=='89' else 779) and all(x['after']-x['before']==-8 and not x['name'].startswith('.text') for x in physical)
  assert len(lp)==len(rp)
  for i,(x,y) in enumerate(zip(lp,rp)):
   if x==y:continue
   assert x[3:5]==y[3:5]==[0,0] and x[7]==y[7]==8 and x[2]-y[2]==8
   if target=='89':
    assert i==1 and x[0:2]==y[0:2]==[1,5]
    assert y[5]-x[5]==y[6]-x[6]==8 and x[2]+x[5]==y[2]+y[5]
    assert x[5]==x[6] and y[5]==y[6]
    start_a=next(s for s in ls if s['name']=='.nv.constant3');start_b=rs[start_a['index']]
    assert x[2]==start_a['offset'] and y[2]==start_b['offset']
   else:
    assert i==2 and x[0:2]==y[0:2]==[1,4] and x[3:]==y[3:] and x[5]==x[6]==128
    assert lb[x[2]:x[2]+x[5]]==rb[y[2]:y[2]+y[5]]
   phchanges.append({'index':i,'before':x,'after':y})
  assert len(phchanges)==1
  ordered=sorted([(a,b) for a,b in zip(ls,rs) if a['type']!=8 and a['size']],key=lambda pair:pair[0]['offset'])
  for (a,b),(aa,bb) in zip(ordered,ordered[1:]):
   if a['offset']-b['offset']==8 and aa['offset']==bb['offset']:
    before=lb[a['offset']+a['size']:aa['offset']];after=rb[b['offset']+b['size']:bb['offset']]
    assert a['name']==('.nv.constant0.dsv4_nvfp4_deq_bf16_kernel' if target=='89' else '.nv.merc.nv.constant.pic') and aa['name'].startswith('.text.')
    if target!='89':assert aa['name']=='.text._ZN21dsv4_dense_tc_gate_r434bf16_mma_output_neuron_k128_packedEPKhPKaPKtPK13__nv_bfloat16Pfiii'
    assert len(before)==(108 if target=='89' else (104 if target=='120-r4' else 96)) and len(after)==(116 if target=='89' else (112 if target=='120-r4' else 104)) and not any(before) and not any(after)
    padding.append({'before_section':a['name'],'next_section':aa['name'],'before_zero_bytes':len(before),'after_zero_bytes':len(after)})
  assert len(padding)==1
 else:assert not physical and lp==rp
 return {'target':target,'left_sha256':sha(lb),'right_sha256':sha(rb),'section_count':len(ls),'allocated_sections_exact':sum(bool(s['flags']&2) for s in ls),'same_segment_memberships_and_permissions':True,'full_header_identical':True,'program_headers_before':lp,'program_headers_after':rp,'symbol_label_changes':label_changes,'physical_offset_changes':physical,'padding':padding,'instruction_data_shared_abi_equal':True,'representation_equal':False,'generic_loader_or_hardware_qualification':False,'sections':[{'index':a['index'],'name':a['name'],'before':attrs(a),'after':attrs(b),'left_payload_sha256':sha(a['payload']),'right_payload_sha256':sha(b['payload'])} for a,b in zip(ls,rs)]}
if __name__=='__main__':
 d=compare(sys.argv[1],sys.argv[2],sys.argv[3]);Path(sys.argv[4]).write_text(json.dumps(d,indent=2)+'\n');print(d['target'],'instruction/data/shared/ABI exact, representation deltas bounded')

#!/usr/bin/env python3
"""Admit only the measured fatbin-dependent link representation changes."""
import importlib.util,json,re,sys
from pathlib import Path
if not __debug__:
 raise RuntimeError('proof readers require Python assertion checks')
sp=importlib.util.spec_from_file_location('elf',Path(__file__).with_name('compare_cubin.py'));elf=importlib.util.module_from_spec(sp);sp.loader.exec_module(elf)
sp=importlib.util.spec_from_file_location('unwind',Path(__file__).with_name('compare_host_unwind.py'));unwind=importlib.util.module_from_spec(sp);sp.loader.exec_module(unwind)
MOVED={'.eh_frame_hdr','.eh_frame','.gcc_except_table','.note.gnu.property','.note.ABI-tag'}
def file_name(s):return re.sub(r'^tmpxft_[0-9a-f]+_[0-9a-f]+-[0-9]+_', 'tmpxft_RANDOM_', s)
def compare(a,b):
 lb,lh,ls,lp=elf.parse(a);rb,rh,rs,rp=elf.parse(b);assert lh==rh and len(ls)==len(rs)
 sa={s['name']:s for s in ls};sb={s['name']:s for s in rs};delta=sb['.nv_fatbin']['size']-sa['.nv_fatbin']['size'];changes=[]
 for x,y in zip(ls,rs):
  assert x['name']==y['name'];name=x['name'];aa=elf.attrs(x);bb=elf.attrs(y)
  if name in MOVED:
   assert y['address']-x['address']==y['offset']-x['offset']==delta
   aa['address']=bb['address'];aa['offset']=bb['offset']
  elif name=='.nv_fatbin':aa['size']=bb['size']
  assert aa==bb,name
  if x['payload']==y['payload']:continue
  changes.append(name)
  if name=='.note.gnu.build-id':
   assert x['payload'][:16]==y['payload'][:16]==bytes.fromhex('040000001400000003000000474e5500') and len(x['payload'])==len(y['payload'])==36
   continue
  if name in ['.nv_fatbin','.eh_frame_hdr','.eh_frame','.gcc_except_table']:continue
  if name=='.strtab':assert [file_name(v) for v in x['strings']]==[file_name(v) for v in y['strings']]
  elif name=='.symtab':
   assert len(x['symbols'])==len(y['symbols'])
   for u,v in zip(x['symbols'],y['symbols']):
    u=u.copy();v=v.copy()
    if u[1]&15==4:u[0]=file_name(u[0]);v[0]=file_name(v[0])
    if u!=v:
     assert u[0]==v[0] and u[0] in ['__abi_tag','__FRAME_END__','__GNU_EH_FRAME_HDR']
     assert u[:4]==v[:4] and u[5:]==v[5:]
     assert u[3]<len(ls) and ls[u[3]]['name'] in MOVED
     assert u[4]-ls[u[3]]['address']==v[4]-rs[v[3]]['address']
  else:raise AssertionError('unknown host payload delta '+name)
 assert unwind.unwind(a)==unwind.unwind(b)
 ph=[]
 for i,(x,y) in enumerate(zip(lp,rp)):
  if x==y:continue
  if x[:2]==[1,4]:
   assert x[:5]==y[:5] and x[7]==y[7] and y[5]-x[5]==y[6]-x[6]==delta
  else:
   assert x[0] in [4,0x6474e553,0x6474e550] and x[:2]==y[:2] and x[5:]==y[5:]
   assert y[2]-x[2]==y[3]-x[3]==y[4]-x[4]==delta
   assert any(s['offset']==x[2] and s['address']==x[3] and s['name'] in MOVED for s in ls)
  ph.append({'index':i,'before':x,'after':y})
 return {'left_sha256':elf.sha(lb),'right_sha256':elf.sha(rb),'fatbin_size_delta':delta,'bounded_payload_changes':changes,'program_header_changes':ph,'entrypoint_identical':True,'host_code_data_exports_relocations_exact':True,'symbol_indices_preserved':True,'decoded_unwind_exception_abi_equal':True,'sections':[{'index':x['index'],'name':x['name'],'before':elf.attrs(x),'after':elf.attrs(y),'left_payload_sha256':elf.sha(x['payload']),'right_payload_sha256':elf.sha(y['payload'])} for x,y in zip(ls,rs)]}
if __name__=='__main__':
 d=compare(sys.argv[1],sys.argv[2]);Path(sys.argv[3]).write_text(json.dumps(d,indent=2)+'\n');print('host code/data/export/relocation/unwind ABI exact; representation deltas bounded',d['fatbin_size_delta'])

#!/usr/bin/env python3
"""Decode GNU relative unwind/LSDA pointers instead of stripping moved bytes."""
import importlib.util,json,struct,sys
from pathlib import Path
if not __debug__:
 raise RuntimeError('proof readers require Python assertion checks')
sp=importlib.util.spec_from_file_location('elf',Path(__file__).with_name('compare_cubin.py'));elf=importlib.util.module_from_spec(sp);sp.loader.exec_module(elf)
def leb(b,p,signed=False):
 n=0;shift=0
 while True:
  v=b[p];p+=1;n|=(v&127)<<shift;shift+=7;assert shift<=64
  if not v&128:break
 if signed and v&64:n-=1<<shift
 return n,p

def pointer(b,p,encoding,address):
 assert encoding!=255
 kind=encoding&15;application=encoding&112
 formats={0:('<Q',8),3:('<I',4),4:('<Q',8),11:('<i',4),12:('<q',8)}
 assert kind in formats,(kind,encoding)
 fmt,size=formats[kind];value=struct.unpack_from(fmt,b,p)[0];position=p;p+=size
 if value==0:return None,p
 if application==16:value+=address+position
 else:assert application==0,(application,encoding)
 return value,p

def unwind(path):
 _,_,sections,_=elf.parse(path);ss={s['name']:s for s in sections};eh=ss['.eh_frame'];b=eh['payload'];records=[];cies={};lsdas=set();p=0
 while p<len(b):
  start=p;length=struct.unpack_from('<I',b,p)[0];p+=4
  if length==0:assert not any(b[start:]);break
  assert length<0xffffffff and p+length<=len(b);end=p+length;ci_pos=p;cid=struct.unpack_from('<I',b,p)[0];p+=4
  if cid==0:
   version=b[p];p+=1;assert version in [1,3];q=b.index(b'\0',p);aug=b[p:q].decode();p=q+1;ca,p=leb(b,p);da,p=leb(b,p,True);rr,p=leb(b,p)
   r={'offset':start,'length':length,'version':version,'augmentation':aug,'code_alignment':ca,'data_alignment':da,'return_register':rr};meta={'pc':None,'lsda':255,'z':aug.startswith('z')}
   if meta['z']:
    count,p=leb(b,p);ae=p+count
    for char in aug[1:]:
     if char=='P':
      encoding=b[p];p+=1;target,p=pointer(b,p,encoding,eh['address']);r['personality']=[encoding,target]
     elif char=='L':meta['lsda']=b[p];p+=1;r['lsda_encoding']=meta['lsda']
     elif char=='R':meta['pc']=b[p];p+=1;r['pc_encoding']=meta['pc']
     else:raise AssertionError(aug)
    assert p==ae
   assert meta['pc'] is not None;cies[start]=meta;r['instructions']=b[p:end].hex();r['kind']='CIE'
  else:
   co=ci_pos-cid;meta=cies[co];pc,p=pointer(b,p,meta['pc'],eh['address']);size,p=pointer(b,p,meta['pc']&15,0);r={'offset':start,'length':length,'kind':'FDE','cie_offset':co,'pc':pc,'range':size}
   if meta['z']:
    count,p=leb(b,p);ae=p+count
    if count and meta['lsda']!=255:
     target,p=pointer(b,p,meta['lsda'],eh['address']);assert target is not None;off=target-ss['.gcc_except_table']['address'];assert 0<=off<ss['.gcc_except_table']['size'];lsdas.add(off);r['lsda_offset']=off
    assert p==ae
   r['instructions']=b[p:end].hex()
  records.append(r);p=end
 hdr=ss['.eh_frame_hdr'];b=hdr['payload'];assert b[:4]==bytes([1,27,3,59]);ep,pos=pointer(b,4,27,hdr['address']);assert ep==eh['address'];count=struct.unpack_from('<I',b,pos)[0];pos+=4;table=[]
 for i in range(count):
  fun,fde=struct.unpack_from('<ii',b,pos);pos+=8;table.append([hdr['address']+fun,hdr['address']+fde-eh['address']])
 assert pos==len(b)
 # GNU LSDA call-site actions identify every referenced reverse type-table entry.
 section=ss['.gcc_except_table'];b=section['payload'];decoded=[];normalized=bytearray(b)
 for start in sorted(lsdas):
  p=start;lp=b[p];p+=1;assert lp==255;tt=b[p];p+=1;base=None
  if tt!=255:offset,p=leb(b,p);base=p+offset;assert tt in [155,27]
  cs=b[p];p+=1;assert cs==1;length,p=leb(b,p);cend=p+length;action_start=cend;indices=set()
  while p<cend:
   _,p=leb(b,p);_,p=leb(b,p);_,p=leb(b,p);action,p=leb(b,p)
   if not action:continue
   ap=action_start+action-1;seen=set()
   while True:
    assert ap not in seen;seen.add(ap);filter_value,np=leb(b,ap,True);displacement,after=leb(b,np,True);assert filter_value>=0
    if filter_value:indices.add(filter_value)
    if not displacement:break
    ap=np+displacement
  assert p==cend
  if indices:
   assert base is not None
   for index in sorted(indices):
    off=base-4*index;assert cend<=off<base and off+4<=len(b);target,end=pointer(b,off,tt,section['address']);assert end==off+4
    decoded.append({'lsda':start,'type_index':index,'encoding':tt,'position':off,'target':target});normalized[off:end]=b'\0'*4
 return {'frames':records,'header_table':table,'lsda_pointer_records':decoded,'lsda_other_bytes':bytes(normalized).hex()}
if __name__=='__main__':
 a=unwind(sys.argv[1]);b=unwind(sys.argv[2]);assert a==b
 Path(sys.argv[3]).write_text(json.dumps({'decoded_equal':True,'frame_records':len(a['frames']),'header_records':len(a['header_table']),'lsda_type_pointers':len(a['lsda_pointer_records']),'canonical':a},indent=2)+'\n');print('decoded unwind/exception ABI exact',len(a['frames']),len(a['lsda_pointer_records']))

from pathlib import Path
import json,hashlib,re,sys,subprocess
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));import llvm_compare as c
O=R/'site-map';O.mkdir(exist_ok=True)
old=R/'emit-before-server/memra_server-4929ad54b672e180.ll';new=R/'emit-after-server/memra_server-4929ad54b672e180.ll'
f,a,m,t=c.read_functions(old)
symbols=[s for s,b in f.items() if 'PendingAdmissionGuard' in s and 'cmpxchg weak' in b]
assert len(symbols)==3
extra=c.compare(old,new,symbols,O);assert all(x['equal'] for x in extra)
(O/'ADDITIONAL-IR-PAIRS.json').write_text(json.dumps(extra,indent=2)+'\n')
tier=json.loads((R/'COMPARE-tier-kv/RESULTS.json').read_text())['actual_emitted_function_rows']
server=json.loads((R/'COMPARE-server/RESULTS.json').read_text())['actual_emitted_function_rows']
linked=json.loads((R/'linked-proof/RESULTS.json').read_text());assert linked['equal']
def ir(rows,match,dir):
 candidates=[x for x in rows if match in x['name']];assert candidates,(match,rows)
 x=candidates[0];p=dir/(x['canonical_pair_prefix']+'-before.ll');b=p.read_text();ops=[{'line':i+1,'instruction':v} for i,v in enumerate(b.splitlines()) if 'cmpxchg weak' in v]
 assert ops,('no actual atomic operation',x['name'])
 return {'kind':'actual emitted IR','function':x['name'],'symbol':x['symbol'],'canonical_pair':str(p.relative_to(R)).replace('-before.ll',''),'equal':x['equal'],'atomic_operations':ops,'canonical_sha256':x['before_sha256']}
def asm(match):
 x=next(x for x in linked['rows'] if match in x['symbol']);assert x['equal']
 return {'kind':'actual linked test code','symbol':x['symbol'],'before_address':x['before_address'],'after_address':x['after_address'],'instruction_count':x['instruction_count'],'equal':x['equal'],'receipt':'linked-proof/RESULTS.json'}
inventory=json.loads((R/'atomic978-seams-readonly.json').read_text())['calls'];rows=[]
matches=['BankService<memra_tier::contracts::ExpertDomain','ExpertBankOwner>::register','object_store::next_transaction_id','LeaseIssuer as core::default::Default','QwenMaterializer>::new','PackedMaterializer>::new']
for i,site in enumerate(inventory):
 if i<6:w=ir(tier,matches[i],R/'COMPARE-tier-kv')
 elif i==6:w=asm('RouteHealth11end_request')
 elif i==7:w=ir(server,'RouteTicket as core::ops::drop::Drop',R/'COMPARE-server')
 elif i in (8,9):
  w=ir(server,'EventSender>::send',R/'COMPARE-server');w['selected_source_atomic_operation']=w['atomic_operations'][i-8]
  text=(R/(w['canonical_pair']+'-before.ll')).read_text();assert ('i64 %v45, 256' if i==8 else '8388609') in text
 elif i==10:w=ir(extra,'PendingAdmissionGuard as core::ops::drop::Drop',O)
 else:w=asm('pending_admission_reservation_is_atomic_and_rolls_back_on_drop')
 rows.append({'site':i,'file':site['file'],'line':site['line'],'emitted_operation_witness':w,'copied_expression_boundary_control':'boundary-controls/SOURCE-MAP.json#site'+str(i),'real_package_tests':'baseline-tier-kv/actual-successful-identities.json' if i<6 else 'server-controls-before/RESULTS.json','limits':'Callees and constants are kept by exact target identity and independently compared direct witnesses. Actual operations are not inferred from Arc-support rows.'})
assert len(rows)==12 and all(x['emitted_operation_witness']['equal'] for x in rows)
(O/'ALL12.json').write_text(json.dumps({'source':'89507cd5addc3ebd756d35f959842849731d7e37','rows':rows,'actual_IR_pairs_total':42,'all12_mapped':True,'qualification':False},indent=2)+'\n')
print('All12 actual operations mapped; three additional real guard IR pairs equal (42 total).')

"""Copy all twelve actual atomic expressions, without reimplementing their closures."""
from pathlib import Path
import subprocess,json,hashlib,re
R=Path(__file__).resolve().parent;WT=Path('/home/avifenesh/projects/wt-memra-atomic-update-alias-978')
O=R/'boundary-controls';O.mkdir(exist_ok=True)
sites=json.loads((R/'atomic978-seams-readonly.json').read_text())['calls']
rows=[]
for arm,rev,method in [('before','9595cea51efe1fb70b9f922acf5798a0584ce08d','fetch_update'),('after','89507cd5addc3ebd756d35f959842849731d7e37','try_update')]:
 funcs=[];tests=[];diagnostic=[]
 for i,site in enumerate(sites):
  text=subprocess.check_output(['git','show',rev+':'+site['file']],cwd=WT,text=True)
  # The line pointer is fixed by the historical census; no second search can pick a different call.
  lines=text.splitlines(keepends=True);start=sum(len(x) for x in lines[:site['line']-1]);pos=text.index(method+'(',start);assert pos<start+len(lines[site['line']-1])
  depth=1;end=pos+len(method)+1
  while depth:
   c=text[end];depth+=int(c=='(')-int(c==')');end+=1
  expression=text[pos:end];kind='u64' if i<6 else 'usize';atype='AtomicU64' if i<6 else 'AtomicUsize'
  args=', retained: usize' if i==9 else ''
  funcs.append(f'fn site{i}(counter: &{atype}{args}) -> Result<{kind},{kind}> {{ counter.{expression} }}')
  diagnostic.append('pub '+funcs[-1])
  if i<6:
   rest=text[end:];adapter_name='map_err' if i<3 else 'expect'
   match=re.match(r'\s*\.'+adapter_name+r'\(',rest);assert match,(i,rest[:100])
   adend=match.end();addepth=1
   while addepth:
    ch=rest[adend];addepth+=int(ch=='(')-int(ch==')');adend+=1
   adapter=rest[:adend]
   ret='Result<u64,Error>' if i<3 else 'u64'
   funcs.append(f'fn site{i}_adapted(counter: &{atype}) -> {ret} {{ counter.{expression}{adapter} }}')
   if i<3:tests.append(f'#[test] fn site{i}_overflow_error_adapter() {{let c={atype}::new(u64::MAX);assert_eq!(site{i}_adapted(&c),Err(Error::Overflow));assert_eq!(c.load(Ordering::Relaxed),u64::MAX);}}')
   else:
    tests.append(f'''#[test] fn site{i}_overflow_panic_adapter() {{let c={atype}::new(u64::MAX);let p=std::panic::catch_unwind(||site{i}_adapted(&c)).expect_err("must retain exhausted panic");let msg=p.downcast_ref::<String>().map(String::as_str).or_else(||p.downcast_ref::<&str>().copied()).unwrap();assert!(msg.contains({rest[match.end():adend-1].strip()}));assert_eq!(c.load(Ordering::Relaxed),u64::MAX);}}''')
  rows.append({'arm':arm,'revision':rev,'site':i,'file':site['file'],'line':site['line'],'source_sha256':hashlib.sha256(text.encode()).hexdigest(),'exact_atomic_expression':expression,'expression_sha256':hashlib.sha256(expression.encode()).hexdigest(),'receiver_adapter':'local same-width Atomic reference; expression/closures/orderings copied exactly','return_adapter':'raw std Result before unchanged map_err/expect/unwrap_or adapters'})
  call=f'site{i}(&c'+(', 1)' if i==9 else ')')
  if i<6:
   tests.append(f'''#[test] fn site{i}_overflow_and_contention() {{
let c={atype}::new({kind}::MAX-1); assert_eq!({call},Ok({kind}::MAX-1)); assert_eq!({call},Err({kind}::MAX)); assert_eq!(c.load(Ordering::Relaxed),{kind}::MAX);
let c={atype}::new(0); std::thread::scope(|s|{{for _ in 0..4{{s.spawn(||{{for _ in 0..256{{site{i}(&c).unwrap();}}}});}}}}); assert_eq!(c.load(Ordering::Relaxed),1024);
}}''')
  elif i in [6,7,10]:
   tests.append(f'''#[test] fn site{i}_underflow_and_contention() {{let c={atype}::new(0);assert_eq!({call},Err(0));assert_eq!(c.load(Ordering::Acquire),0);c.store(1024,Ordering::Release);std::thread::scope(|s|{{for _ in 0..4{{s.spawn(||{{for _ in 0..256{{site{i}(&c).unwrap();}}}});}}}});assert_eq!(c.load(Ordering::Acquire),0);assert_eq!({call},Err(0));}}''')
  elif i==8:
   tests.append(f'''#[test] fn site{i}_queue_bound_and_contention() {{let c={atype}::new(MAX_EVENT_QUEUE_EVENTS-1);assert_eq!({call},Ok(MAX_EVENT_QUEUE_EVENTS-1));assert_eq!({call},Err(MAX_EVENT_QUEUE_EVENTS));assert_eq!(c.load(Ordering::Acquire),MAX_EVENT_QUEUE_EVENTS);c.store(0,Ordering::Release);std::thread::scope(|s|{{for _ in 0..4{{s.spawn(||{{for _ in 0..128{{let _=site{i}(&c);}}}});}}}});assert_eq!(c.load(Ordering::Acquire),MAX_EVENT_QUEUE_EVENTS);}}''')
  elif i==9:
   tests.append(f'''#[test] fn site{i}_byte_bound_and_arithmetic_overflow() {{let c={atype}::new(MAX_EVENT_QUEUE_BYTES-1);assert_eq!({call},Ok(MAX_EVENT_QUEUE_BYTES-1));assert_eq!({call},Err(MAX_EVENT_QUEUE_BYTES));assert_eq!(c.load(Ordering::Acquire),MAX_EVENT_QUEUE_BYTES);c.store(usize::MAX,Ordering::Release);assert_eq!({call},Err(usize::MAX));assert_eq!(c.load(Ordering::Acquire),usize::MAX);assert_eq!(site{i}(&c,0),Err(usize::MAX));}}''')
  else:tests.append(f'''#[test] fn site{i}_reset_returns_previous_value() {{for n in [0,1,usize::MAX] {{let c={atype}::new(n);assert_eq!({call},Ok(n));assert_eq!(c.load(Ordering::Acquire),0);}}}}''')
 worker=subprocess.check_output(['git','show',rev+':crates/memra-server/src/worker.rs'],cwd=WT,text=True)
 constants='\n'.join(re.findall(r'^const MAX_EVENT_QUEUE_(?:EVENTS|BYTES): usize = .+;',worker,re.M));assert len(constants.splitlines())==2
 source='use std::sync::atomic::{AtomicU64,AtomicUsize,Ordering};\n#[derive(Debug,PartialEq)] enum Error {Overflow}\n'+constants+'\n'+'\n'.join(funcs+tests)+'\n'
 (O/f'{arm}.rs').write_text(source)
 (O/f'{arm}-diagnostic.rs').write_text('use std::sync::atomic::{AtomicU64,AtomicUsize,Ordering};\n'+constants+'\n'+'\n'.join(diagnostic)+'\n')
 assert len(funcs)==18
(O/'SOURCE-MAP.json').write_text(json.dumps({'sites':rows,'controls_per_arm':18,'limits':'Copied exact atomic expressions on local same-width counters, plus copied six overflow map_err/expect adapters. Error::Overflow is a local same-name stand-in; surrounding constructors are not exercised here. Complements actual 530 per-arm package tests and real site-linked code. No runtime/model qualification.'},indent=2)+'\n')

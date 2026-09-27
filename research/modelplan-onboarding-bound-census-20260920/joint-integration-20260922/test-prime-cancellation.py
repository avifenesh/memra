"""Actual composed Eager-loop + typed cancellation/taint code; device/math providers are stand-ins."""
import argparse,importlib.util,pathlib,subprocess,hashlib,json
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
path=a.root/'research/modelplan-onboarding-rewrite-identity-20260920/run-prime-dispatch-tests.py';spec=importlib.util.spec_from_file_location('existing',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
source=(a.root/'crates/memra-engine/src/hybrid_forward.rs').read_text();progress=(a.root/'crates/memra-engine/src/progress.rs').read_text()
cancel=progress[progress.index('thread_local! {\n    static PRIME_CANCEL'):progress.index("/// The odometer's observable state.")]
stubs=m.STUBS.replace('__ACTUAL_PRIME_CANCEL_BOUNDARY__','use std::cell::RefCell;\n'+cancel)
guard=source[source.index('struct CacheTaintGuard {'):source.index('#[derive(Debug, Clone, Copy, PartialEq, Eq)]\nstruct PrimePpWaveSlot')]
tests=r'''
#[test] fn partial_stop_occurs_after_second_completed_row_and_taints() {
 let calls=std::rc::Rc::new(Cell::new(0));let seen=calls.clone();
 let _scope=progress::PrimeCancelScope::install(Box::new(move||{seen.set(seen.get()+1);seen.get()==2}));
 let m=HybridModel::new(false);let mut c=Cache::at(7);
 let err=m.prime_cache_eager(&Engine::default(),&[3,4,5,6],&mut c).expect_err("cancelled prime must stop");
 let typed=err.downcast_ref::<progress::PrimeCancelled>().expect("typed cancellation");
 assert_eq!((typed.chunk,typed.rows_done,typed.rows_total),(1,2,4));
 assert_eq!(*m.calls.borrow(),vec![(7,3),(8,4)]);assert_eq!(c.pos,9);assert!(c.tainted);
 assert!(m.prime_cache_eager(&Engine::default(),&[9],&mut c).is_err());assert_eq!(m.calls.borrow().len(),2);
}
#[test] fn final_completed_row_returns_normally_without_calling_predicate() {
 let calls=std::rc::Rc::new(Cell::new(0));let seen=calls.clone();
 let _scope=progress::PrimeCancelScope::install(Box::new(move||{seen.set(seen.get()+1);true}));
 let m=HybridModel::new(false);let mut c=Cache::at(7);
 let result=m.prime_cache_eager(&Engine::default(),&[3],&mut c).unwrap();
 assert_eq!(calls.get(),0);assert_eq!(c.pos,8);assert!(!c.tainted);assert_eq!(result.1,result.2);
}
'''
records={}
for name,text in [('composed',source),('frozen-7236',subprocess.check_output(['git','show','7236d6368563bdce844583fc8d108950c441009b:crates/memra-engine/src/hybrid_forward.rs'],cwd=a.root,text=True))]:
 body=m.function(text,'prime_cache_eager');program=stubs+guard+'\nimpl HybridModel {\n'+body+'\n}\n'+tests;src=a.out/(name+'.rs');src.write_text(program);binary=a.out/name
 build=subprocess.run(['rustc','--edition=2024','--test',str(src),'-o',str(binary)],capture_output=True);(a.out/(name+'-build.log')).write_bytes(build.stdout+build.stderr);assert build.returncode==0
 run=subprocess.run([str(binary),'--test-threads=1','--nocapture'],capture_output=True);raw=run.stdout+run.stderr;(a.out/(name+'.log')).write_bytes(raw)
 assert (run.returncode==0 and b'2 passed; 0 failed' in raw) if name=='composed' else (run.returncode!=0 and b'1 passed; 1 failed' in raw)
 records[name]={'source_sha256':hashlib.sha256(text.encode()).hexdigest(),'body_sha256':hashlib.sha256(body.encode()).hexdigest(),'program_sha256':hashlib.sha256(program.encode()).hexdigest(),'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'exit':run.returncode}
(a.out/'result.json').write_text(json.dumps({'scope':__doc__,'progress_sha256':hashlib.sha256(progress.encode()).hexdigest(),'results':records},indent=2)+'\n');print(json.dumps(records,indent=2))

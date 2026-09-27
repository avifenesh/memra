#!/usr/bin/env python3
"""CPU-only controller contract tests. Simulated ELF/build/lease are never native evidence."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import os
import signal
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import release_qualification as q
import release_input_view as view

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value
    spec.loader.exec_module(value);return value

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')

class Fixture:
    def __init__(self,base,fixtures):
        base=base.resolve()
        self.base=base;self.repo=base/'repo';self.evidence=base/'evidence'
        self.fixtures=base/'fixtures';self.evidence.mkdir();self.repo.mkdir()
        self.prefix=HERE.relative_to(ROOT)
        self.binding_module=module(HERE/'native_binding.py','fixture_binding_source')
        for name in (*self.binding_module.HELPERS,str(self.prefix/'run-native.py'),str(self.prefix/'generate-fixtures.py')):
            dst=self.repo/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dst)
        (self.repo/'Cargo.lock').write_text('# CPU-only simulated checkout\n')
        def git(*args):subprocess.run(['git','-C',str(self.repo),*args],check=True,stdout=subprocess.DEVNULL)
        git('init','-q');git('config','user.name','CPU control');git('config','user.email','cpu@example.invalid')
        git('config','core.hooksPath','/dev/null');git('add','.');git('commit','-qm','CPU controller fixture')
        self.source=q.source_snapshot(self.repo);write(self.evidence/'source.json',self.source)
        shutil.copytree(fixtures,self.fixtures)
        self.binary=self.evidence/'target/release/deps/bound_loader_gpu-cpu_fixture'
        self.binary.parent.mkdir(parents=True)
        self.binary.write_bytes(b'\x7fELF\x02\x01'+bytes(12)+b'\x3e\0'+b'CPU CONTRACT FIXTURE NOT A NATIVE EXECUTABLE')
        self.binary.chmod(0o700)
        self.artifact={'path':str(self.binary.relative_to(self.evidence)),**q.file_identity(self.binary)}
        for name in ('build.log','fetch.log','test.stderr'):(self.evidence/name).write_text('CPU simulated build, no compiler\n')
        self.stock={'schema':'memra-native-build-v3','exit_code':0,'source':self.ref('source.json'),
            'source_before':self.source['inputs_sha256'],'source_after':self.source['inputs_sha256'],
            'input_view_before':view.identity(self.source),'input_view_after':view.identity(self.source),
            'cuda_arch':'120a','docs_rs':False,'cuda_visible_devices':'','rustc':'CPU fixture','nvcc':'CPU fixture',
            'platform':{'machine':'x86_64','profile':'CPU fixture','glibc':'CPU fixture'},
            'command':['CPU-BWRAP','/toolchain/bin/cargo','build','--release','--locked'],
            'log':self.ref('build.log'),'fetch_log':self.ref('fetch.log'),
            'compiler_environment':{k:q.digest(v.encode()) for k,v in {'CUDA_VISIBLE_DEVICES':'',
                'MEMRA_CUDA_ARCH':'120a','CARGO_HOME':'/cargo','CARGO_TARGET_DIR':'/target','RUSTC':'/toolchain/bin/rustc'}.items()},
            'recipe':{'policy':'controlled-cargo-v3','cargo_home':'fresh-config-free','checkout':view.POLICY,
                'build_source':'fingerprinted-input-view','cargo_config':'tracked-jobs-only',
                'sandbox':{'policy':view.SANDBOX_POLICY,'version':'bubblewrap CPU fixture',
                    'executable':{'bytes':32,'sha256':'e'*64}},
                'compilers':{k:{'bytes':32,'sha256':'c'*64} for k in ('cargo','rustc','nvcc')}},
            'binaries':{k:{'bytes':32,'sha256':'d'*64,'format':'ELF-x86_64'} for k in q.BINARIES}}
        write(self.evidence/'stock.json',self.stock)
        self.events=[{'reason':'compiler-artifact','target':{'name':'bound_loader_gpu','kind':['test']},
            'profile':{'test':True},'fresh':False,'executable':'/target/release/deps/'+self.binary.name},
            {'reason':'build-finished','success':True}]
        self.put_events()
        self.test={k:copy.deepcopy(self.stock[k]) for k in ('recipe','compiler_environment','source_before','source_after',
            'input_view_before','input_view_after','cuda_arch','docs_rs','cuda_visible_devices')}
        self.test.update({'schema':'memra-bound-loader-test-build-v1','source':self.ref('source.json'),
            'stock_build':self.ref('stock.json'),'exit_code':0,'started_utc':'2026-09-22T00:00:00Z',
            'completed_utc':'2026-09-22T00:00:01Z','command':['CPU-BWRAP','/toolchain/bin/cargo','test','--locked',
                '--release','--no-run','--message-format=json','-p','memra-engine','--test','bound_loader_gpu'],
            'artifact':self.artifact,'events':self.ref('events.jsonl'),'stderr':self.ref('test.stderr')})
        write(self.evidence/'test.json',self.test)
        self.selection={'schema':'memra-bound-loader-selection-v1','source_commit':self.source['commit'],
            'source_tree':self.source['tree'],'inputs_sha256':self.source['inputs_sha256'],
            'controller_sha256':digest(self.repo/self.prefix/'run-native.py'),
            'generator_sha256':digest(self.repo/self.prefix/'generate-fixtures.py'),
            'helpers':{name:digest(self.repo/name) for name in self.binding_module.HELPERS},
            'source':self.ref('source.json'),'stock_build':self.ref('stock.json'),'test_build':self.ref('test.json'),
            'binary':self.artifact,'fixtures_sha256':digest(self.fixtures/'cases.json')}
        self.selection_path=self.evidence/'selection.json';self.seal()
        self.runner=module(self.repo/self.prefix/'run-native.py','cpu_runner')
        self.original_bootstrap=self.runner.bootstrap
        self.launches=[];self.lease={'scope':'CPU fake lease, no GPU authorization'}
        self.real_popen=subprocess.Popen

    def ref(self,name):return {'path':name,'sha256':digest(self.evidence/name)}
    def seal(self):
        write(self.selection_path,self.selection);self.selection_sha=digest(self.selection_path)
    def put_events(self):
        (self.evidence/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in self.events))
    def update_test(self):
        write(self.evidence/'test.json',self.test);self.selection['test_build']=self.ref('test.json');self.seal()
    def mutate_manifest(self,change):
        m=json.loads((self.fixtures/'cases.json').read_text());change(m)
        write(self.fixtures/'cases.json',m);self.selection['fixtures_sha256']=digest(self.fixtures/'cases.json');self.seal()
    def binding(self):
        return self.original_bootstrap(SimpleNamespace(selection=self.selection_path,selection_sha256=self.selection_sha,
            evidence_root=self.evidence,binary=self.binary,fixtures=self.fixtures))
    def launch(self,command,**kwargs):
        if command[0]!=str(self.binary):return self.real_popen(command,**kwargs)
        env=kwargs['env'];log=kwargs['stdout'];self.launches.append(command)
        if self.mode=='signal':
            code='import os,time;from pathlib import Path;Path('+repr(str(self.base/'child.pid'))+').write_text(str(os.getpid()));time.sleep(60)'
            return self.real_popen([sys.executable,'-B','-c',code],**kwargs)
        m=json.loads((self.fixtures/'cases.json').read_text())
        c=next(c for c in m['cases'] if c['id']==env['MEMRA_BOUND_CASE'])
        out=Path(env['MEMRA_BOUND_OUT']);out.mkdir()
        if c['expect_error']:(out/'error.txt').write_text(c['expected_error_fragment'])
        else:
            raw=(struct.pack('<ee',.5,.125)+bytes([1])*12+bytes([0x55])*32+bytes([0x22])*128)*64 if c['id']=='gguf-q5_head' else struct.pack('<f',.25 if c['expected_owner']=='token_embd.weight' else .5)*(256*64)
            (out/'output.bytes').write_bytes(raw)
        text='BOUND_LOADER_BEGIN '+c['id']+' '+env['MEMRA_BOUND_ROOT']+' unqualified_fixture\n'
        if not c['expect_error'] or self.mode=='malformed-alloc':text+='[alloc-trace] CPU simulated\n[dtoh-trace] CPU simulated\n'
        text+='BOUND_LOADER_END '+c['id']+' '+env['MEMRA_BOUND_ROOT']+' '+('contextual_refusal' if c['expect_error'] else 'accepted_unqualified_fixture')+'\n'
        text+='test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n'
        if self.mode=='skip':text='test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out\n'
        if self.mode=='setup-failure':text='CPU simulated setup failure\n'
        log.write(text.encode());log.flush()
        if len(self.launches)==1:
            if self.mode=='binary-drift':self.binary.write_bytes(self.binary.read_bytes()+b'drift')
            if self.mode=='fixture-drift':
                p=self.fixtures/c['path'];p.write_bytes(p.read_bytes()+b'drift')
            if self.mode=='source-drift':(self.repo/'Cargo.lock').write_text('# changed source\n')
            if self.mode=='lease-drift':self.lease={'scope':'different CPU lease'}
        code=17 if self.mode=='bad-exit' else 101 if self.mode=='setup-failure' else 0
        return SimpleNamespace(pid=999999999,returncode=code,poll=lambda:code,wait=lambda **_:code)

    def run(self,mode='normal',output=None):
        self.mode=mode;out=output or self.base/'out'
        def boot(args):
            b=self.original_bootstrap(args)
            b.admission.verify_lease=lambda:dict(self.lease)
            return b
        args=['--binary',str(self.binary),'--fixtures',str(self.fixtures),'--output',str(out),
            '--selection',str(self.selection_path),'--selection-sha256',self.selection_sha,
            '--evidence-root',str(self.evidence),'--wall-seconds','120']
        original_writer=self.runner.write_json
        def writer(path,value):
            original_writer(path,value)
            if mode=='publish-signal' and path.name=='result.json.pending' and value['status']=='passed':
                os.kill(os.getpid(),signal.SIGTERM)
        with patch.object(self.runner,'bootstrap',side_effect=boot),patch.object(subprocess,'Popen',side_effect=self.launch),patch.object(self.runner,'write_json',side_effect=writer),patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            code=self.runner.main(args)
        result=json.loads((out/'result.json').read_text()) if (out/'result.json').exists() else None
        if os.environ.get('NATIVE_RUNNER_CPU_EVIDENCE'):
            saved=Path(os.environ['NATIVE_RUNNER_CPU_EVIDENCE'])/self.base.name
            saved.mkdir(parents=True,exist_ok=False)
            write(saved/'control.json',{'scope':'CPU controller control; no native execution',
                'mode':mode,'optimize':sys.flags.optimize,'exit_code':code,'launches':len(self.launches),'result':result})
            for name in ('selection.json','source.json','stock.json','test.json','events.jsonl'):
                shutil.copyfile(self.evidence/name,saved/name)
            if out.is_dir() and not out.is_symlink():
                for log in out.glob('*.log'):shutil.copyfile(log,saved/log.name)
        return code,result

class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared=tempfile.TemporaryDirectory(prefix='541-real-fixtures-')
        cls.fixtures=Path(cls.shared.name)/'fixtures'
        subprocess.run([sys.executable,'-B',str(HERE/'generate-fixtures.py'),str(cls.fixtures)],check=True,stdout=subprocess.DEVNULL)
    @classmethod
    def tearDownClass(cls):cls.shared.cleanup()
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='541-controller-control-')
        self.f=Fixture(Path(self.tmp.name),self.fixtures)
    def tearDown(self):self.tmp.cleanup()
    def reject(self,mode='normal',before_launch=False):
        code,result=self.f.run(mode)
        self.assertEqual(code,1);self.assertEqual(result['status'],'failed')
        if before_launch:self.assertEqual(len(self.f.launches),0)
        else:self.assertGreater(len(self.f.launches),0,result)
        reasons={'bad-exit':'native process failed: 17','skip':'missing executed test result',
            'setup-failure':'native process failed: 101','malformed-alloc':'malformed load reached device wrappers',
            'binary-drift':'binary bytes changed','fixture-drift':'fixture input bytes drifted',
            'source-drift':'actual tracked input differs','lease-drift':'lease changed after case'}
        if mode in reasons:self.assertIn(reasons[mode],result['error'])
        return result
    def test_complete_real_fixture_closure_and_simulated_40_case_schedule(self):
        self.assertEqual(digest(self.fixtures/'cases.json'),'12d936f70fed949e5bb493e67f4ac4558dfe18f0545fbf1a4e7508c82c8f0d45')
        code,result=self.f.run();self.assertEqual(code,0,result);self.assertEqual(result['status'],'passed')
        self.assertEqual(len(result['results']),40);self.assertEqual(len(self.f.launches),40)
    def test_empty_manifest(self):
        self.f.mutate_manifest(lambda m:m.update(cases=[],files=[]));self.reject(before_launch=True)
    def test_duplicate_case(self):
        self.f.mutate_manifest(lambda m:m['cases'].__setitem__(1,m['cases'][0]));self.reject(before_launch=True)
    def test_wrong_count(self):
        self.f.mutate_manifest(lambda m:m['cases'].pop());self.reject(before_launch=True)
    def test_wrong_valid_count(self):
        self.f.mutate_manifest(lambda m:m['cases'][0].update(expect_error=True,expected_error_fragment='unexpected'));self.reject(before_launch=True)
    def test_bad_exit(self):self.reject('bad-exit')
    def test_skipped_test(self):self.reject('skip')
    def test_setup_failure(self):self.reject('setup-failure')
    def test_malformed_device_activity(self):self.reject('malformed-alloc')
    def test_binary_drift(self):self.reject('binary-drift')
    def test_fixture_drift(self):self.reject('fixture-drift')
    def test_source_drift(self):self.reject('source-drift')
    def test_lease_drift(self):self.reject('lease-drift')
    def test_signal_during_success_publication(self):
        result=self.reject('publish-signal');self.assertIn('publication_error',result)
    def test_external_selection_hash(self):self.f.selection_sha='0'*64;self.reject(before_launch=True)
    def test_wrong_controller(self):self.f.selection['controller_sha256']='0'*64;self.f.seal();self.reject(before_launch=True)
    def test_wrong_generator(self):self.f.selection['generator_sha256']='0'*64;self.f.seal();self.reject(before_launch=True)
    def test_wrong_source_selection(self):self.f.selection['source_commit']='0'*40;self.f.seal();self.reject(before_launch=True)
    def test_wrong_input_selection(self):self.f.selection['inputs_sha256']='0'*64;self.f.seal();self.reject(before_launch=True)
    def test_wrong_binary_selection(self):self.f.selection['binary']={**self.f.artifact,'sha256':'0'*64};self.f.seal();self.reject(before_launch=True)
    def test_changed_build_record(self):(self.f.evidence/'test.json').write_text('{}');self.reject(before_launch=True)
    def test_changed_source_on_disk(self):(self.f.repo/'Cargo.lock').write_text('# drift');self.reject(before_launch=True)
    def test_docs_rs_build(self):self.f.test['docs_rs']=True;self.f.update_test();self.reject(before_launch=True)
    def test_different_test_recipe(self):self.f.test['recipe']['cargo_home']='ambient';self.f.update_test();self.reject(before_launch=True)
    def test_failed_test_build(self):self.f.test['exit_code']=1;self.f.update_test();self.reject(before_launch=True)
    def test_wrong_test_command(self):self.f.test['command'][-1]='other';self.f.update_test();self.reject(before_launch=True)
    def test_stale_cargo_artifact(self):
        self.f.events[0]['fresh']=True;self.f.put_events();self.f.test['events']=self.f.ref('events.jsonl');self.f.update_test();self.reject(before_launch=True)
    def test_failed_cargo_completion(self):
        self.f.events[-1]['success']=False;self.f.put_events();self.f.test['events']=self.f.ref('events.jsonl');self.f.update_test();self.reject(before_launch=True)
    def test_absolute_case_path(self):self.f.mutate_manifest(lambda m:m['cases'][0].update(path='/tmp/outside'));self.reject(before_launch=True)
    def test_parent_case_path(self):self.f.mutate_manifest(lambda m:m['cases'][0].update(path='../outside'));self.reject(before_launch=True)
    def test_escape_output_id(self):self.f.mutate_manifest(lambda m:m['cases'][0].update(id='../escaped'));self.reject(before_launch=True)
    def test_unlisted_case_files(self):self.f.mutate_manifest(lambda m:m.update(files=[]));self.reject(before_launch=True)
    def test_incomplete_hf_closure(self):
        self.f.mutate_manifest(lambda m:m.update(files=[f for f in m['files'] if f['path']!='hf-separate/config.json']));self.reject(before_launch=True)
    def test_unlisted_hf_shard(self):(self.f.fixtures/'hf-separate/extra.safetensors').write_text('unlisted');self.reject(before_launch=True)
    def test_symlinked_fixture_file(self):
        p=self.f.fixtures/'gguf-separate.gguf';outside=self.f.base/'outside';p.rename(outside);p.symlink_to(outside);self.reject(before_launch=True)
    def test_symlinked_hf_directory(self):
        p=self.f.fixtures/'hf-separate';outside=self.f.base/'outside';p.rename(outside);p.symlink_to(outside);self.reject(before_launch=True)
    def test_output_overlap(self):
        code,result=self.f.run(output=self.f.fixtures/'out');self.assertEqual((code,result),(1,None));self.assertEqual(self.f.launches,[])
    def test_output_symlink_to_inputs(self):
        out=self.f.base/'out';out.symlink_to(self.f.fixtures)
        code,result=self.f.run(output=out);self.assertEqual((code,result),(1,None));self.assertEqual(self.f.launches,[])
    def test_real_termination_signals_kill_and_reap_separate_session_child(self):
        for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
            with self.subTest(signal=sig),tempfile.TemporaryDirectory(prefix='541-real-signal-') as temp:
                base=Path(temp);log=(base/'worker.log').open('wb')
                command=[sys.executable,'-B',*(['-O'] if sys.flags.optimize else []),str(Path(__file__).resolve()),'--signal-worker',str(base),str(self.fixtures)]
                p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                child=None
                try:
                    deadline=time.monotonic()+30
                    while not (base/'child.pid').exists() and p.poll() is None and time.monotonic()<deadline:time.sleep(.02)
                    self.assertTrue((base/'child.pid').exists(),(base/'worker.log').read_text())
                    child=int((base/'child.pid').read_text());self.assertEqual(os.getpgid(child),child)
                    os.kill(p.pid,sig);self.assertEqual(p.wait(timeout=10),1)
                    result=json.loads((base/'out/result.json').read_text());self.assertEqual(result['status'],'failed')
                    self.assertTrue(any(c.get('pid')==child and c.get('reaped') for c in result['cleanup']))
                    with self.assertRaises(ProcessLookupError):os.kill(child,0)
                    with self.assertRaises(ProcessLookupError):os.killpg(child,0)
                    if os.environ.get('NATIVE_RUNNER_CPU_EVIDENCE'):
                        write(Path(os.environ['NATIVE_RUNNER_CPU_EVIDENCE'])/('signal-'+str(int(sig))+'.json'),
                            {'signal':int(sig),'controller_pid':p.pid,'controller_exit':p.returncode,
                             'child_pid':child,'child_group':child,'child_absent':True,'group_absent':True,
                             'result':result,'scope':'real CPU child cleanup only; no GPU lease or execution'})
                finally:
                    if p.poll() is None:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=5)
                    if child:
                        try:os.killpg(child,signal.SIGKILL)
                        except ProcessLookupError:pass
                    log.close()

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--signal-worker':
        f=Fixture(Path(sys.argv[2]),Path(sys.argv[3]));raise SystemExit(f.run('signal')[0])
    unittest.main(verbosity=2)

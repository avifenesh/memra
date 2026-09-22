#!/usr/bin/env python3
"""Pinned native loader regression; ordinary provenance checks, never qualification approval."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import time
import types
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BINDING_PATH = 'research/modelplan-onboarding-bound-census-20260920/native-acceptance-20260922/native_binding.py'

def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)

def unique(pairs):
    out = {}
    for key, value in pairs:
        require(key not in out, 'duplicate JSON field: '+key)
        out[key] = value
    return out

def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()

def regular(path):
    require(stat.S_ISREG(path.lstat().st_mode), 'nonregular or symlinked path: '+str(path))

def bootstrap(args):
    regular(args.selection)
    raw = args.selection.read_bytes()
    require(re.fullmatch('[0-9a-f]{64}', args.selection_sha256) is not None and
            hashlib.sha256(raw).hexdigest() == args.selection_sha256,
            'externally selected binding digest mismatch')
    selected = json.loads(raw, object_pairs_hook=unique)
    require(sha(Path(__file__)) == selected['controller_sha256'], 'controller selection mismatch')
    path = ROOT/BINDING_PATH
    regular(path)
    data = path.read_bytes()
    require(hashlib.sha256(data).hexdigest() == selected['helpers'][BINDING_PATH], 'binding helper mismatch')
    module = types.ModuleType('native_loader_binding')
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(data,str(path),'exec'),module.__dict__)
    return module.Binding(args.selection,args.selection_sha256,args.evidence_root,args.binary,args.fixtures)

def stop_owned(process):
    """Existing owned-session kill/reap pattern, with a finite reap limit."""
    if process is None or process.poll() is not None:
        return {'needed':False,'reaped':True}
    try:
        os.killpg(process.pid,signal.SIGKILL)
    except (ProcessLookupError,PermissionError):
        process.kill()
    process.wait(timeout=5)
    return {'needed':True,'pid':process.pid,'returncode':process.returncode,'reaped':True}

def write_json(path, value):
    data=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC|getattr(os,'O_NOFOLLOW',0),0o600)
    with os.fdopen(fd,'wb') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())

def output_root(args):
    parent=args.output.parent.resolve(strict=True)
    require(not args.output.exists() and not args.output.is_symlink(), 'output must be a fresh directory')
    out=parent/args.output.name
    require(args.output.name not in ('','.', '..'), 'invalid output directory')
    for path in (ROOT,args.fixtures.resolve(strict=True),args.evidence_root.resolve(strict=True)):
        require(not out.is_relative_to(path) and not path.is_relative_to(out), 'input/output overlap')
    out.mkdir(mode=0o700)
    return out

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','fixtures','output','selection','evidence-root'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--selection-sha256',required=True)
    parser.add_argument('--wall-seconds',type=int,required=True)
    args=parser.parse_args(argv)
    started=time.monotonic();deadline=started+args.wall_seconds
    previous={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    def interrupted_before_launch(signum,_frame):
        raise RuntimeError('controller interrupted before launch by signal '+str(signum))
    for sig in previous:signal.signal(sig,interrupted_before_launch)
    out=None;process=None;binding=None
    result={'schema':'memra-native-bound-loader-regression-v2','status':'failed','results':[],
            'scope':'unqualified tiny-container loading/readback/refusal only; no model/rewrite/support approval',
            'cleanup':[]}
    try:
        require(1<=args.wall_seconds<=1800,'invalid bounded wall limit')
        require(not any(k in os.environ for k in ('DOCS_RS','MEMRA_ARTIFACT_LOCK','MEMRA_REWRITE_BUNDLE')),
                'fixture gate refuses documentation stubs or qualification context')
        out=output_root(args)
        output_stat=out.stat()
        def output_unchanged():
            now=out.lstat()
            require(stat.S_ISDIR(now.st_mode) and (now.st_dev,now.st_ino)==(output_stat.st_dev,output_stat.st_ino),
                    'output root replaced or escaped')
        binding=bootstrap(args)
        ctl=binding.controller
        with ctl.cleanup_signals():
            try:
                result['binding']=binding.verify()
                lease=binding.admission.verify_lease();result['lease']=lease
                ctl.check_deadline(deadline)
                cases=binding.manifest['cases']
                for case in cases:
                    for loader in ('dense','hybrid'):
                        ctl.check_deadline(deadline);output_unchanged()
                        require(binding.verify()==result['binding'],'binding changed before case')
                        require(binding.admission.verify_lease()==lease,'lease changed before case')
                        tag=case['id']+'-'+loader
                        case_out=out/tag
                        require(not case_out.exists() and not case_out.is_symlink(),'case output already exists')
                        log_path=out/(tag+'.log')
                        env=os.environ.copy()
                        env.update({'MEMRA_ALLOC_TRACE':'1','MEMRA_DTOH_TRACE':'1','MEMRA_BOUND_CASE':case['id'],
                            'MEMRA_BOUND_CASE_PATH':str(binding.fixtures/case['path']),'MEMRA_BOUND_FORMAT':case['format'],
                            'MEMRA_BOUND_ROOT':loader,'MEMRA_BOUND_OUT':str(case_out),
                            'MEMRA_BOUND_EXPECT_ERROR':case['expected_error_fragment'],
                            'MEMRA_BOUND_OUTPUT_SHA256':case['expected_output_sha256'] or '',
                            'MEMRA_BOUND_OUTPUT_KIND':case['expected_output_kind'],
                            'MEMRA_BOUND_NVFP4_QUERY':'1' if case['nvfp4_query'] else '0'})
                        command=[str(binding.binary),'--exact','native_bound_loader_case',
                                 '--ignored','--nocapture','--test-threads=1']
                        fd=os.open(log_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
                        with os.fdopen(fd,'wb') as log:
                            process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,
                                                     start_new_session=True)
                            result['active_child']={'pid':process.pid,'case':case['id'],'root':loader}
                            case_deadline=min(deadline,time.monotonic()+90)
                            while process.poll() is None:
                                ctl.check_deadline(case_deadline)
                                require(binding.admission.verify_lease()==lease,'lease changed during case')
                                time.sleep(.05)
                            code=process.wait(timeout=5)
                        ctl.check_deadline(deadline);output_unchanged()
                        require(binding.verify()==result['binding'],'binding changed after case')
                        require(binding.admission.verify_lease()==lease,'lease changed after case')
                        regular(log_path)
                        raw=log_path.read_text()
                        require(code==0,tag+': native process failed: '+str(code))
                        require(len(re.findall(r'^test result: ok\. 1 passed; 0 failed; 0 ignored;.*$',raw,re.M))==1,
                                tag+': missing executed test result')
                        before='BOUND_LOADER_BEGIN '+case['id']+' '+loader+' unqualified_fixture'
                        after='BOUND_LOADER_END '+case['id']+' '+loader+' '+(
                            'contextual_refusal' if case['expect_error'] else 'accepted_unqualified_fixture')
                        require(raw.count(before)==raw.count(after)==1 and raw.index(before)<raw.index(after),
                                tag+': missing, duplicated or reordered load markers')
                        section=raw.split(before,1)[1].split(after,1)[0]
                        alloc=section.count('[alloc-trace]');readback=section.count('[dtoh-trace]')
                        require(case_out.is_dir() and not case_out.is_symlink() and case_out.resolve().parent==out,
                                tag+': missing or escaped case output')
                        if case['expect_error']:
                            require(alloc==0 and readback==0,tag+': malformed load reached device wrappers')
                            output=case_out/'error.txt';regular(output)
                            require(case['expected_error_fragment'] in output.read_text(),tag+': wrong contextual refusal')
                        else:
                            require(alloc>0 and readback>0,tag+': observation channel not active')
                            output=case_out/'output.bytes';regular(output)
                            require(output.stat().st_size==case['expected_output_bytes'] and
                                    sha(output)==case['expected_output_sha256'],tag+': wrong readback bytes')
                        result['results'].append({'case':case['id'],'root':loader,'pid':process.pid,
                            'exit_code':code,'allocations_in_load':alloc,'readbacks_in_load':readback,
                            'status':'passed','log_sha256':sha(log_path),'output_sha256':sha(output),
                            'binding':result['binding']})
                        result['cleanup'].append(stop_owned(process));process=None
                        result.pop('active_child',None)
                require(len(result['results'])==40,'incomplete native case schedule')
                result['status']='passed'
            except BaseException as error:
                result['status']='failed';result['error']=type(error).__name__+': '+str(error)
            finally:
                try:
                    result['cleanup'].append(stop_owned(process));process=None
                except BaseException as error:
                    result['status']='failed';result['cleanup_error']=str(error)
                result['elapsed_seconds']=time.monotonic()-started
            def final_checks():
                ctl.check_deadline(deadline);output_unchanged()
                require(binding.verify()==result['binding'],'final binding drift')
                require(binding.admission.verify_lease()==result['lease'],'final lease drift')
            try:
                if result['status']=='passed':final_checks()
                output_unchanged()
                binding.finalization.publish_result(out,result,writer=write_json,check_cancelled=final_checks)
            except BaseException as error:
                result['status']='failed';result['publication_error']=str(error)
                output_unchanged();write_json(out/'result.json',result)
    except BaseException as error:
        result['status']='failed';result['error']=type(error).__name__+': '+str(error)
        result['elapsed_seconds']=time.monotonic()-started
        if out is not None:
            output_unchanged();write_json(out/'result.json',result)
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
    return 0 if result['status']=='passed' else 1

if __name__=='__main__':
    raise SystemExit(main())

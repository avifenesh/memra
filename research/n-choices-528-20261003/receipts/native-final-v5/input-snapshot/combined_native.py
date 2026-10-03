#!/usr/bin/env python3
"""Frozen broker-only composite coordinator. Failures remain in the inventory."""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def digest(path):return hashlib.file_digest(Path(path).open('rb'),'sha256').hexdigest()

def main():
    cli=argparse.ArgumentParser()
    for key in ['observer','stock','baseline','model','out','source','tools','baseline_receipt']:
        cli.add_argument('--'+key.replace('_','-'),required=True)
    cli.add_argument('--replay',action='store_true')
    args=cli.parse_args();out=Path(args.out);tools=Path(args.tools)
    if not args.replay:out.mkdir()
    report=out/'replay-v5' if args.replay else out
    if args.replay:report.mkdir()
    sys.path.insert(0,str(tools))
    from replay_native_choices import verify,cell
    from choice_verifier import check_completed,check_identity,check_stream_twin,check_repeat,check_tools,Invalid,require
    context={'source':args.source,'binary_sha256':digest(args.observer),'stock_sha256':digest(args.stock),'baseline_sha256':digest(args.baseline),'model_sha256':digest(args.model),'helper_sha256':{n:digest(tools/n) for n in ['native_choices.py','choice_verifier.py','replay_native_choices.py']}}
    if args.replay:
        context=json.loads((out/'expected-context.json').read_text())
        require(context['source']==args.source and context['binary_sha256']==digest(args.observer) and context['stock_sha256']==digest(args.stock) and context['model_sha256']==digest(args.model),'context_binding','replayed native tuple matches original frozen inputs')
    else:save(out/'expected-context.json',context)
    results={};failures=[]
    for index,phase in enumerate(['baseline','candidate','deadline','slots','budget','kv']):
        binary=args.stock if phase=='baseline' else args.observer
        cmd=[sys.executable,str(tools/'native_choices.py'),'--binary',binary,'--binary-sha',digest(binary),'--model',args.model,'--model-sha',context['model_sha256'],'--source',args.source,'--out',str(out/phase),'--phase',phase,'--port',str(18270+index),'--context','4096','--sessions','4' if phase=='slots' else '8']
        if phase=='budget':cmd+=['--output-budget','128']
        if args.replay:
            require(json.loads((out/phase/'manifest.json').read_text())['status']=='complete','collection',phase)
            rc=0
        else:
            with (out/(phase+'.log')).open('w') as log:
                rc=subprocess.call(cmd,stdout=log,stderr=subprocess.STDOUT,pass_fds=(9,))
        results[phase]={'exit':rc}
        save(report/'phase-results.json',results)
        if rc:failures.append(phase)
    require(not failures,'collection','failed native phases '+str(failures))
    # Baseline manifests are real stock servers, without observer callbacks.
    for directory in [Path(args.baseline_receipt),out/'baseline']:
        manifest=json.loads((directory/'manifest.json').read_text())
        require(manifest['status']=='complete' and len(manifest['results'])==2 and all(r['status']==200 for r in manifest['results']),'n1_identity','two actual stock endpoint controls')
        for name,expected in manifest['files'].items():
            require(digest(directory/name)==expected,'retained_hash',str(directory/name))
    current=json.loads((out/'baseline/manifest.json').read_text())
    old=json.loads((Path(args.baseline_receipt)/'manifest.json').read_text())
    for before,after in zip(old['results'],current['results']):
        require(before['name']==after['name'],'n1_identity','same endpoint controls')
        check_identity(before['body'],after['body'])
    for phase in ['candidate','deadline','slots','budget','kv']:
        manifest=json.loads((out/phase/'manifest.json').read_text())
        expected={k:context[k] for k in ['source','binary_sha256','model_sha256','helper_sha256']}
        expected['hardware']=json.loads((out/'candidate/manifest.json').read_text())['hardware']
        result=verify(out/phase,Path(args.baseline_receipt) if phase=='candidate' else None,expected)
        results[phase]['replay']=result
    native=json.loads((out/'candidate/manifest.json').read_text());records={r['name']:r for r in native['results']}
    bad_expected={**expected,'source':'0'*40}
    try:verify(out/'candidate',Path(args.baseline_receipt),bad_expected)
    except Invalid as error:require(error.edge=='context_binding','context_binding',str(error))
    else:raise AssertionError('changed source context unexpectedly passed')
    for route in ['completions','chat-completions']:
        for n in [2,4,8]:
            record=records[route+f'-greedy-n{n}'];rows=record['worker_rows']
            require(len({r['token_sha256'] for r in rows})==1,'greedy_identity','every greedy row retains the same exact token sequence')
    sample=cell(out/'candidate',records['chat-completions-sampled']);stream=cell(out/'candidate',records['chat-completions-stream'])
    controls={'changed-context':{'passed':True,'edge':'context_binding'}}
    def red(name,edge,mutate,target=sample):
        bad=deepcopy(target);mutate(bad)
        try:check_completed(bad)
        except Invalid as error:require(error.edge==edge,'control.'+name,str(error));controls[name]={'passed':True,'edge':edge};return
        raise AssertionError('control.'+name+' unexpectedly passed')
    red('missing-choice','indexed_termination',lambda x:x['body']['choices'].pop())
    red('duplicate-choice','indexed_termination',lambda x:x['body']['choices'].__setitem__(1,deepcopy(x['body']['choices'][0])))
    red('premature-done','premature_done',lambda x:x['packets'].insert(0,'[DONE]'),stream)
    red('shared-rng','rng_isolation',lambda x:x['worker_rows'][1].update(seed=x['seed']))
    red('output-sum','output_sum',lambda x:x['worker_rows'][1].update(output=x['worker_rows'][1]['output']+1))
    red('missing-fork','shared_prefill',lambda x:x.update(forks=[]))
    red('follower-prime','shared_prefill',lambda x:x.update(follower_prime_segments=1))
    red('duplicate-prompt','prompt_once',lambda x:x['callbacks'].append(deepcopy(next(c for c in x['callbacks'] if c['kind']=='prompt'))))
    red('prompt-multiplied','reservation',lambda x:x['reserve'].update(prompt=x['reserve']['prompt']*x['n']))
    red('output-unscaled','reservation',lambda x:x['reserve'].update(output=x['resolved_output_bound']))
    red('duplicate-terminal','callback_lifetime',lambda x:x['callbacks'].append(deepcopy(next(c for c in x['callbacks'] if c['kind']=='terminal'))))
    twin=cell(out/'candidate',records['chat-completions-stream-twin']);bad_wire=deepcopy(stream['body']);bad_wire['choices'][0]['message']['content']='lost or substituted wire bytes'
    try:check_stream_twin(bad_wire,twin['body'])
    except Invalid as error:require(error.edge=='stream_content_identity','control.wire-content',str(error));controls['wire-content']={'passed':True,'edge':error.edge}
    else:raise AssertionError('wire-content substitution unexpectedly passed')
    other=cell(out/'candidate',records['chat-completions-sampled-repeat']);bad=deepcopy(other);bad['worker_rows'][1]['token_sha256']='0'*64
    try:check_repeat(sample,bad)
    except Invalid as error:require(error.edge=='token_identity','control.token-hash',str(error));controls['token-hash']={'passed':True,'edge':error.edge}
    else:raise AssertionError('control.token-hash unexpectedly passed')
    context['replay_helper_sha256']={n:digest(tools/n) for n in ['choice_verifier.py','replay_native_choices.py']}
    context.update(model='Qwen3.5-9B Q8_0',artifact=context['model_sha256'],hardware=native['hardware'],numeric_program='existing GGUF forward/sampler and ordinary leader prefix_snapshot/prefix_restore',binary=context['binary_sha256'],request_shape='chat+text n1/2/3/4/8; sampled+greedy+SSE; constrained N2; isolated deadline and capacity boots')
    spec=importlib.util.spec_from_file_location('coverage',tools/'validation_coverage.py');coverage=importlib.util.module_from_spec(spec);spec.loader.exec_module(coverage)
    edges=['n1_identity','greedy_row_identity','seeded_rng_isolation','shared_prefill','indexed_termination','stream_content_identity','prompt_once_output_sum','reservation','n_slots','kv_exhaustion','prepaid_exhaustion','cancel_all_and_recover','bare_defaults','deadline_partial','constrained_choices']
    inputs={n:digest(tools/n) for n in ['native_choices.py','choice_verifier.py','replay_native_choices.py','test_choice_verifier.py','validation_coverage.py']}
    tests=[{'id':'native-composite','kind':'gpu','cost':600,'covers':edges,'inputs':inputs,'scope':context,'controls':list(controls),'mandatory':True}]+[{'id':n,'kind':'cpu','cost':1,'covers':['control.'+n],'inputs':inputs,'mandatory':True} for n in controls]
    selected=coverage.select(edges+['control.'+n for n in controls],tests,context,tools)
    common={'contract_id':selected['contract_id'],'context':context,'status':'passed','skipped':0}
    admitted={'native-composite':{**common,'executed':len(edges),'edges':{e:'passed' for e in edges}}}
    admitted.update({n:{**common,'executed':1,'edges':{'control.'+n:'passed'}} for n in controls})
    save(report/'coverage-plan.json',selected);save(report/'coverage-results.json',admitted)
    proof=coverage.validate_results(selected,admitted,context,tools)
    save(report/'phase-results.json',results);save(report/'controls.json',controls);save(report/'coverage-admission.json',proof);save(report/'execution-context.json',context)
    print(json.dumps({'passed':True,'native_edges':len(edges),'coherent_red_controls':len(controls),'qualification':False}))

if __name__=='__main__':main()

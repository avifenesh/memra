#!/usr/bin/env python3
"""Replay complete native receipts without starting a server or reading a model."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from replay_native_choices import verify, cell
from choice_verifier import check_completed, check_identity, check_greedy, check_repeat, check_stream_twin, require, Invalid

def hashes(root,manifest):
    for name,expected in manifest['files'].items():
        p=(root/name).resolve()
        require(p.is_relative_to(root.resolve()) and hashlib.sha256(p.read_bytes()).hexdigest()==expected,'retained_hash',name)

def replay(root):
    expected=json.loads((root/'expected-context.json').read_text())
    before=root/'before';current=root/'baseline';candidate=root/'candidate'
    for folder in [before,current]:
        manifest=json.loads((folder/'manifest.json').read_text());hashes(folder,manifest)
        require(manifest['status']=='complete' and len(manifest['results'])==2 and all(r['status']==200 for r in manifest['results']),'n1_identity','both real stock endpoints completed')
    b=json.loads((before/'manifest.json').read_text());c=json.loads((current/'manifest.json').read_text())
    for old,new in zip(b['results'],c['results']):
        require(old['name']==new['name'],'n1_identity','same endpoint controls');check_identity(old['body'],new['body'])
    context={k:expected[k] for k in ['source','binary_sha256','model_sha256','helper_sha256','hardware']}
    phases={phase:verify(root/phase,before if phase=='candidate' else None,context) for phase in ['candidate','deadline','slots','budget','kv']}
    cache_context={k:expected[k] for k in ['source','binary_sha256','model_sha256','hardware']}
    phases['cache']=verify(root/'cache',None,cache_context)
    cached=json.loads((root/'cache/manifest.json').read_text());by_name={r['name']:r for r in cached['results']}
    count=by_name['hit']['body'].get('cached_tokens',0)
    require(count>0 and by_name['group']['body']['usage']['prompt_tokens_details']['cached_tokens']==count,'cached_tokens','real nonzero cached count is copied once to the parent')
    check_greedy(by_name['hit']['body'],by_name['group']['body'])
    # Coherent semantic mutations hold producer/callback witnesses fixed.
    records={r['name']:r for r in json.loads((candidate/'manifest.json').read_text())['results']}
    sample=cell(candidate,records['chat-completions-sampled']);stream=cell(candidate,records['chat-completions-stream'])
    controls={}
    def red(name,edge,mutate,target=sample):
        bad=deepcopy(target);mutate(bad)
        try:check_completed(bad)
        except Invalid as error:require(error.edge==edge,'control.'+name,str(error));controls[name]=True;return
        raise Invalid('control.'+name,'coherent mutation passed')
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
    other=cell(candidate,records['chat-completions-sampled-repeat']);bad=deepcopy(other);bad['worker_rows'][1]['token_sha256']='0'*64
    try:check_repeat(sample,bad)
    except Invalid as error:require(error.edge=='token_identity','control.token-hash',str(error));controls['token-hash']=True
    else:raise Invalid('control.token-hash','producer token mutation passed')
    twin=cell(candidate,records['chat-completions-stream-twin']);bad=deepcopy(stream['body']);bad['choices'][0]['message']['content']='lost wire bytes'
    try:check_stream_twin(bad,twin['body'])
    except Invalid as error:require(error.edge=='stream_content_identity','control.wire-content',str(error));controls['wire-content']=True
    else:raise Invalid('control.wire-content','wire substitution passed')
    try:verify(candidate,before,{**context,'source':'0'*40})
    except Invalid as error:require(error.edge=='context_binding','control.changed-context',str(error));controls['changed-context']=True
    else:raise Invalid('control.changed-context','wrong execution source passed')
    return {'passed':True,'phases':len(phases),'completed_requests':sum(len(v['completed']) for v in phases.values()),'cached_tokens':count,'coherent_controls':len(controls),'qualification':False}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    print(json.dumps(replay(parser.parse_args().root),indent=2))

#!/usr/bin/env python3
"""Join retained raw HTTP and independent native callback/producer witnesses."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from choice_verifier import check_completed, check_identity, check_greedy, check_seeded, check_refusal, check_repeat, check_tools, require

TAG={'CHOICE_OPEN':'open','CHOICE_PROMPT':'prompt','CHOICE_TOKEN':'token','CHOICE_TERMINAL':'terminal','CHOICE_DROP':'drop'}

def cell(root,record):
    request=json.loads((root/record['name']/'request.json').read_text())
    n=request.get('n',1)
    packets=json.loads((root/record['name']/'packets.json').read_text()) if (root/record['name']/'packets.json').exists() else None
    body=record.get('body')
    if packets is not None:
        texts={i:'' for i in range(n)};reasons={};usage=None;calls={i:[] for i in range(n)};reasoning={i:'' for i in range(n)}
        for packet in packets:
            if not isinstance(packet,dict):continue
            if packet.get('usage'):usage=packet['usage']
            for c in packet.get('choices',[]):
                i=c['index']; d=c.get('delta',{})
                texts[i]+=c.get('text','') or d.get('content','') or ''
                reasoning[i]+=d.get('reasoning','') or ''
                calls[i].extend({k:v for k,v in call.items() if k!='index'} for call in d.get('tool_calls',[]))
                if c.get('finish_reason') is not None:reasons[i]=c['finish_reason']
        body={'choices':[{'index':i,'text':texts[i],'finish_reason':reasons.get(i)} for i in range(n)],'usage':usage}
        if 'messages' in request:
            choices=[]
            for i in range(n):
                message={'role':'assistant','content':None if not texts[i] and calls[i] else texts[i]}
                if reasoning[i]:message.update(reasoning=reasoning[i],reasoning_details=[{'type':'reasoning.text','text':reasoning[i]}])
                if calls[i]:message['tool_calls']=calls[i]
                choices.append({'index':i,'message':message,'finish_reason':reasons.get(i)})
            body['choices']=choices
    if n==1 and 'choices' not in body:
        body={**body,'choices':[{'index':0,'text':body['text'],'finish_reason':'length' if body.get('stop_reason')=='MaxNew' else 'stop'}],
              'usage':{'prompt_tokens':body['prompt_tokens'],'completion_tokens':body['n_tokens'],
                       'total_tokens':body['prompt_tokens']+body['n_tokens'],'prompt_tokens_details':{'cached_tokens':body['cached_tokens']}}}
    callbacks=[]
    for item in record['callbacks']:
        if item['tag'] not in TAG:continue
        value=dict(item['value'])
        if item['tag']=='CHOICE_TERMINAL':value['outcome']=value['kind']
        value['kind']=TAG[item['tag']];callbacks.append(value)
    forks=[]
    for line in record.get('fork_lines',[]):
        found=dict(re.findall(r'(choices|copies|leader|prompt|cached|bytes)=(\d+)',line))
        forks.append({k:int(v) for k,v in found.items()})
    events=json.loads((root/'callback-events.json').read_text())
    opened=next(c for c in callbacks if c['kind']=='open')
    # All positive requests in this cell are serialized; pair the last reserve preceding this exact open.
    index=next(i for i,e in enumerate(events) if e['tag']=='CHOICE_OPEN' and e['value']['id']==record['id'])
    reserves=[e['value'] for e in events[:index] if e['tag']=='CHOICE_RESERVE']
    reserve=reserves[-1] if reserves else None
    primes=record.get('prime_events',[])
    output_bound=request.get('max_tokens', (opened.get('reserved_ctx',0)-reserve['prompt'])//n if reserve else None)
    return {'id':record['id'],'n':n,'seed':request.get('seed',next((r['seed'] for r in record.get('worker_rows',[]) if r['index']==0),None)),'body':body,'callbacks':callbacks,'worker_rows':record.get('worker_rows',[]),'forks':forks,'leader_prime_segments':sum(e['index']==0 for e in primes),'follower_prime_segments':sum(e['index']!=0 for e in primes),'reserve':reserve,'resolved_output_bound':output_bound,'packets':packets}

def verify(root,baseline=None,expected=None):
    manifest=json.loads((root/'manifest.json').read_text())
    require(manifest['status']=='complete','manifest','native invocation must complete')
    require(manifest['support_promotion'] is False,'manifest','no support-state promotion')
    if expected is not None:
        require(all(manifest.get(k)==v for k,v in expected.items()),'context_binding','source/ELF/model/hardware/helper context')
    for name,digest in manifest['files'].items():
        path=(root/name).resolve()
        require(path.is_relative_to(root.resolve()) and hashlib.sha256(path.read_bytes()).hexdigest()==digest,'retained_hash',name)

    records={r['name']:r for r in manifest['results']}
    require(len(records)==len(manifest['results']) and records,'required_cells','nonempty unique request records')
    logs=(root/'server.log').read_text().splitlines()
    events=[]
    for index,line in enumerate(logs):
        if line.startswith('CHOICE_'):
            tag,payload=line.split(' ',1)
            events.append({'tag':tag,'value':json.loads(payload),'line':index})
    require(events==json.loads((root/'callback-events.json').read_text()),'raw_binding','callback inventory is derived from retained server bytes')
    for name,r in records.items():
        folder=root/name
        require(r==json.loads((folder/'client.json').read_text()),'raw_binding','manifest result agrees with retained client '+name)
        if 'body' in r:require(r['body']==json.loads((folder/'wire.json').read_text()),'raw_binding','body agrees with actual HTTP bytes '+name)
        require(r['callbacks']==[e for e in events if e['value'].get('id')==r['id']],'raw_binding','one parent callback history '+name)
        require(r['worker_rows']==[e['value'] for e in events if e['tag']=='CHOICE_WORKER_ROW' and e['value'].get('group')==r['id']],'raw_binding','independent producer history '+name)
        if (folder/'packets.json').exists():
            raw=[line[5:].strip() for line in (folder/'wire.sse').read_text().splitlines() if line.startswith('data:')]
            packets=['[DONE]' if value=='[DONE]' else json.loads(value) for value in raw]
            require(packets==json.loads((folder/'packets.json').read_text()),'raw_binding','packets agree with actual SSE bytes '+name)
    positive={name for name in records if '-refuse-' not in name and name not in {'cancel','deadline-partial','slots-contender','budget','kv'}}
    require(all(records[name]['status']==200 and not records[name]['disconnected'] for name in positive),'required_cells','every intended successful request completed')
    completed=[]
    for name,r in records.items():
        if '-refuse-' in name:
            check_refusal(r,name)
        elif r['status']==200 and not r['disconnected'] and name!='deadline-partial':
            c=cell(root,r);check_completed(c);completed.append(c)
    if manifest['phase']=='candidate':
        for route in ['completions','chat-completions']:
            reference=records[route+'-n1']['body']
            check_identity(reference,records[route+'-n1-after']['body'])
            if baseline:
                b=json.loads((baseline/'manifest.json').read_text())
                before=next(r['body'] for r in b['results'] if r['name']=='v1-'+route+'-n1')
                check_identity(before,reference)
            for n in [2,4,8]:check_greedy(reference,records[route+f'-greedy-n{n}']['body'])
            check_seeded([records[route+f'-seed-{i}']['body'] for i in range(3)],records[route+'-sampled']['body'])
            check_repeat(cell(root,records[route+'-sampled']),cell(root,records[route+'-sampled-repeat']))
            check_identity(cell(root,records[route+'-stream'])['body'],cell(root,records[route+'-stream-twin'])['body'])
        check_tools(records['constrained-choices']['body'])
        check_tools(cell(root,records['constrained-choices-stream'])['body'])
        require(all(records[name]['masked_steps']>0 for name in ['constrained-choices','constrained-choices-stream']),'constrained_choices','native grammar mask engagement')
        cancel=records['cancel']
        require(cancel['disconnected'],'cancel','real connection reset required')
        terminal=[e for e in cancel['callbacks'] if e['tag']=='CHOICE_TERMINAL']
        dropped=[e['value'] for e in cancel['callbacks'] if e['tag']=='CHOICE_DROP']
        require(not terminal and len(dropped)==1 and dropped[0]['terminal'] is False and 0<dropped[0]['output']<2048,'cancel','one partial handler drop')
        require(records['cancel-recovery']['status']==200,'recovery','ordinary request succeeds after cancellation')
        bare=json.loads((root/'bare-default'/'request.json').read_text())
        n_only=json.loads((root/'n-only-default'/'request.json').read_text())
        require(set(bare)=={'model','messages'} and set(n_only)=={'model','messages','n'},'bare_defaults','no decoder/mode knobs')
        vendor={'temperature':1.,'top_p':.95,'top_k':20,'min_p':0.,'presence_penalty':1.5,'repetition_penalty':1.}
        for name in ['n-only-default','n-only-default-fresh']:
            for row in records[name]['worker_rows']:
                require(row['think']=='Default' and all(abs(row['sampling'][key]-value)<1e-6 for key,value in vendor.items()),'bare_defaults','actual row receives full pinned vendor profile and default thinking mode')
        require(bool(records['bare-default']['body']['choices'][0]['message'].get('reasoning')),'bare_defaults','bare n1 default thinking path engaged')
    elif manifest['phase']=='deadline':
        partial=records['deadline-partial']
        require(partial['status']==200 and partial['body'].get('error',{}).get('code')=='deadline_exceeded','deadline_partial','actual native partial deadline required')
        require(any(c['finish_reason']=='error' for c in partial['body']['choices']),'deadline_partial','unfinished row is explicit')
        calls=[e['value'] for e in partial['callbacks'] if e['tag']=='CHOICE_TERMINAL']
        require(len(calls)==1 and calls[0]['kind']=='deadline_partial' and calls[0]['output']==calls[0]['observed_output']==partial['body']['usage']['completion_tokens']>0,'deadline_partial','observed count unchanged by buffer flush')
        require(records['deadline-recovery']['status']==200,'deadline_partial','all group slots recover')
    elif manifest['phase']=='budget':
        require(records['budget-singleton']['status']==200 and records['budget']['status']==402,'prepaid_exhaustion','same per-row bound passes alone and fails after N multiplication')
        require(not records['budget'].get('prime_events') and not records['budget'].get('worker_rows'),'prepaid_exhaustion','refusal precedes native work')
    elif manifest['phase']=='kv':
        r=records['kv']
        pressure=json.loads((root/'pressure.json').read_text())
        require(pressure['allocation_code']==pressure['free_code']==pressure['own_context_reset_code']==0 and pressure['allocated']>0 and pressure['keep_free']>=1024**3,'kv_exhaustion','real lane-owned allocation is released with its own context')
        require(records['kv-warm']['status']==records['kv-recovery']['status']==200,'kv_exhaustion','same N8 states recover after owned pressure is removed')
        require(r['status']==400 and r.get('body',{}).get('error',{}).get('code')=='context_length_exceeded','kv_exhaustion','idle native state admission uses the existing non-retryable capacity contract')
        require(not r.get('prime_events') and not r.get('fork_lines') and not r.get('worker_rows'),'kv_exhaustion','native allocation refused before priming')
    elif manifest['phase']=='slots':
        require(records['slots-contender']['status'] in [408,429],'n_slots','four active choices prevent extra immediate admission')
        require(not records['slots-contender'].get('worker_rows') and not records['slots-contender'].get('prime_events'),'n_slots','contender never reached generation')
        occupier=json.loads((root/'occupier/client.json').read_text())
        require(occupier['status']==200 and occupier['disconnected'] and any(e['tag']=='CHOICE_DROP' and not e['value']['terminal'] for e in occupier['callbacks']),'n_slots','retained real overlap and group reset')
        require(records['slots-recovery']['status']==200,'n_slots','all four slots become usable after group reset')
    return {'pass':True,'completed':completed,'scope':'this exact native receipt and declared assertion edges; no support promotion'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--baseline',type=Path);p.add_argument('--expected',type=Path)
    args=p.parse_args();expected=json.loads(args.expected.read_text()) if args.expected else None
    print(json.dumps(verify(args.root,args.baseline,expected),indent=2))

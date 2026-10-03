"""Replay indexed-choice native evidence. Synthetic controls never qualify a model."""
from collections import Counter
import hashlib
import json
from pathlib import Path

class Invalid(AssertionError):
    def __init__(self, edge, detail):
        self.edge = edge
        super().__init__(f"{edge}: {detail}")

def require(value, edge, detail):
    if not value: raise Invalid(edge, detail)

def normalized(body):
    if isinstance(body, dict):
        return {k:normalized(v) for k,v in body.items() if k not in {'id','created','system_fingerprint','elapsed_s'}}
    if isinstance(body, list): return [normalized(v) for v in body]
    return body

def text(choice):
    return choice.get('text') if 'text' in choice else choice['message']

def check_completed(cell):
    n = cell['n']; body = cell['body']; identity = cell['id']
    choices = body['choices']
    require(len(choices)==n and Counter(c['index'] for c in choices)==Counter(range(n)), 'indexed_termination', identity)
    require(all(c['finish_reason'] in ('stop','length','tool_calls') for c in choices), 'indexed_termination', identity)
    callbacks = cell['callbacks']
    opens = [c for c in callbacks if c['kind']=='open']
    terminals = [c for c in callbacks if c['kind']=='terminal']
    drops = [c for c in callbacks if c['kind']=='drop']
    tokens = [c['output'] for c in callbacks if c['kind']=='token']
    require(len(opens)==len(terminals)==len(drops)==1, 'callback_lifetime', identity)
    require(all(c['id']==identity for c in callbacks), 'callback_identity', identity)
    require(tokens==list(range(1,len(tokens)+1)), 'callback_sequence', identity)
    terminal=terminals[0]
    require(terminal['outcome']=='complete' and drops[0]['terminal'] is True, 'callback_lifetime', identity)
    usage=body['usage']
    require(usage['prompt_tokens']==terminal['prompt']==terminal['observed_prompt'] and
            usage['prompt_tokens_details']['cached_tokens']==terminal['cached']==terminal['observed_cached'] and
            usage['completion_tokens']==terminal['output']==terminal['observed_output']==len(tokens) and
            usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens'], 'accounting', identity)
    if n>1:
        row_events=cell['worker_rows']
        require(len(row_events)==n and Counter(r['index'] for r in row_events)==Counter(range(n)), 'producer_rows', identity)
        require(all(r['group']==identity for r in row_events), 'producer_rows', identity)
        require(sum(r['output'] for r in row_events)==len(tokens), 'output_sum', identity)
        require(all(r['prompt']==usage['prompt_tokens'] and r['cached']==terminal['cached'] for r in row_events), 'prompt_once', identity)
        require(all(r['seed']==(cell['seed']+r['index'])%(1<<64) for r in row_events), 'rng_isolation', identity)
        forks=cell['forks']
        require(len(forks)==1 and forks[0]['choices']==n and forks[0]['copies']==n-1 and forks[0]['leader']==0, 'shared_prefill', identity)
        require(cell['leader_prime_segments']>=1 and cell['follower_prime_segments']==0, 'shared_prefill', identity)
        reserve=cell['reserve']
        require(reserve['prompt']==usage['prompt_tokens'] and reserve['output']==n*cell['resolved_output_bound'], 'reservation', identity)
    if cell.get('packets') is not None:
        ended=[]; done=0
        for packet in cell['packets']:
            if packet=='[DONE]':
                done+=1
                require(Counter(ended)==Counter(range(n)), 'premature_done', identity)
            else:
                for choice in packet.get('choices',[]):
                    require(choice['index'] in range(n), 'stream_index', identity)
                    if choice.get('finish_reason') is not None: ended.append(choice['index'])
        require(done==1 and Counter(ended)==Counter(range(n)), 'indexed_termination', identity)
    return {'id':identity,'n':n,'output':len(tokens),'pass':True}

def check_identity(before, after):
    require(normalized(before)==normalized(after), 'n1_identity', 'only identity/time fields are normalized')

def check_greedy(reference, group):
    expected=text(reference['choices'][0]) if 'choices' in reference else reference['text']
    require(all(text(c)==expected for c in group['choices']), 'greedy_identity', 'every row matches the n1 control')

def check_seeded(singletons, group):
    require(all(text(choice)==text(singletons[choice['index']]['choices'][0] if 'choices' in singletons[choice['index']] else {'text':singletons[choice['index']]['text']}) for choice in group['choices']), 'rng_isolation', 'row i matches its independent seed+i request')

if __name__=='__main__':
    import argparse
    cli=argparse.ArgumentParser();cli.add_argument('receipt',type=Path)
    args=cli.parse_args(); receipt=json.loads(args.receipt.read_text())
    print(json.dumps([check_completed(cell) for cell in receipt['completed']],indent=2))

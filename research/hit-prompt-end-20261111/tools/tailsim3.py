"""Toy model: prefix cache under the frozen mixed90 sequence (8 hot templates round robin, 2 colds
per 20, seeded once), tail policies for memra#1233. Units MB (decimal)."""
import random
BASE, TAIL = 300.4, 158.0
def run(budget, strict, cold_attach, attach_mode, reps=5, seed=3407):
    t = [0]
    ent = {}
    st = {'hit_tail':0,'hit_base':0,'miss_hot':0}
    tot = lambda: sum(BASE + (TAIL if e['tail'] is not None else 0) for e in ent.values())
    def tick(): t[0]+=1; return t[0]
    def room(need, protect, entries=True, tails=True):
        while tot()+need > budget:
            items=[]
            for n,e in ent.items():
                if n==protect: continue
                if entries: items.append((e['use'],0,n))
                if tails and e['tail'] is not None: items.append((e['tail'],1,n))
            if strict=='hybrid':
                unproven=[i for i in items if i[1]==1 and not ent[i[2]].get('proven')]
                if unproven: items=unproven
            elif strict and any(k==1 for _,k,_ in items): items=[i for i in items if i[1]==1]
            if not items: return False
            _,k,n=min(items)
            if k==1: ent[n]['tail']=None
            else: del ent[n]
        return True
    def insert(n):
        room(BASE,n); ent[n]={'use':tick(),'tail':None}
    proven_on_attach=[False]
    def attach(n):
        if attach_mode=='none' or n not in ent: return
        ok = tot()+TAIL<=budget if attach_mode=='free' else room(TAIL,n,entries=(attach_mode=='item'))
        if ok: ent[n]['tail']=tick(); ent[n]['use']=tick(); ent[n]['proven']=proven_on_attach[0]
    for h in range(8):
        insert(f'h{h}')
        if cold_attach: attach(f'h{h}')
    for rep in range(1,reps+1):
        roles=['hit']*18+['miss']*2
        random.Random(seed+rep*1009+4*9173).shuffle(roles)
        hi=0
        for i,r in enumerate(roles):
            if r=='hit':
                n=f'h{hi%8}'; hi+=1
                if n not in ent:
                    st['miss_hot']+=1; insert(n)
                    if cold_attach: attach(n)
                elif ent[n]['tail'] is not None:
                    st['hit_tail']+=1; ent[n]['tail']=tick(); ent[n]['use']=tick(); ent[n]['proven']=True
                else:
                    st['hit_base']+=1; ent[n]['use']=tick(); proven_on_attach[0]=True; attach(n); proven_on_attach[0]=False
            else:
                n=f'c{rep}-{i}'; insert(n)
                if cold_attach: attach(n)
    return st
print('budget strict cold_attach attach_mode -> stats')
for b in (2936, 4295):
    print(b,'baseline(no tails)',run(b,True,False,'none'))
    for strict in (False,True,'hybrid'):
        for ca in (True,False):
            for am in ('item','tails','free'):
                print(b,strict,ca,am,run(b,strict,ca,am))

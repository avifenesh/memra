import json, subprocess, sys, time, statistics, hashlib, os
from concurrent.futures import ThreadPoolExecutor
LABEL = sys.argv[1]
CONCS = [int(x) for x in sys.argv[2:]] or [1,4,8,16,24,32]
K = subprocess.run(["secret-tool","lookup","service","cx-connect","provider","battery","kind","gate"],
                   capture_output=True,text=True).stdout.strip()
URL="https://api.tiyuvta.ai/v1/chat/completions"
PARA=("Distributed inference systems must balance prefill throughput against decode latency, and the "
      "prefix cache is the only mechanism that removes prefill work entirely rather than making it faster. ")
SHARED = PARA*151                      # ~4,860 tokens: the sold shape, and the cached prefix
OUT=60
def call(sysmsg, user, max_tokens=OUT, temp=0):
    body=json.dumps({"model":"qwen3.6-35b-a3b",
                     "messages":[{"role":"system","content":sysmsg},{"role":"user","content":user}],
                     "max_tokens":max_tokens,"temperature":temp})
    t0=time.time()
    r=subprocess.run(["curl","-sS","--max-time","600","-o","-","-w","\n%{http_code}",
                      "-H",f"Authorization: Bearer {K}","-H","Content-Type: application/json",
                      "-d",body,URL],capture_output=True,text=True)
    dt=time.time()-t0
    parts=r.stdout.rsplit("\n",1)
    code=parts[-1].strip() if len(parts)>1 else "000"
    try: d=json.loads(parts[0])
    except Exception: d={}
    u=d.get("usage") or {}
    return code, dt, u.get("prompt_tokens",0), (u.get("prompt_tokens_details") or {}).get("cached_tokens",0), u.get("completion_tokens",0)

# warm the shared prefix so the 90%-hit mix is real
call(SHARED, "warm")
time.sleep(1)
rows=[]
for c in CONCS:
    lat=[];codes=[];cached=0;prompt=0;comp=0;walls=[]
    for rnd in range(3):                                  # N=3 rounds per width, medians reported
        def one(i):
            miss = (i % 10 == 0)                          # 10% forced miss -> 90% cache hit
            sysmsg = SHARED + (f" unique-{c}-{rnd}-{i}" if miss else "")
            return call(sysmsg, f"Summarize in one sentence. req {i}")
        t0=time.time()
        with ThreadPoolExecutor(max_workers=c) as ex: res=list(ex.map(one, range(c)))
        walls.append(time.time()-t0)
        for code,dt,p,ca,co in res:
            codes.append(code); lat.append(dt); prompt+=p; cached+=ca; comp+=co
        time.sleep(2)
    ok=codes.count("200"); rl=codes.count("429"); other=len(codes)-ok-rl
    lat.sort(); wall=statistics.median(walls)
    hit = cached/prompt if prompt else 0
    rows.append(dict(c=c, ok=ok, rl=rl, other=other, wall=round(wall,2),
                     p50=round(statistics.median(lat),3), p95=round(lat[max(0,int(len(lat)*0.95)-1)],3),
                     rps=round(ok/3/wall,2), out_tps=round(comp/3/wall,1), hit=round(hit,4)))
    print(f"{LABEL} c={c:<3} 200={ok:<4} 429={rl:<3} other={other:<3} wall={wall:6.2f}s "
          f"p50={rows[-1]['p50']:6.3f} p95={rows[-1]['p95']:6.3f} req/s={rows[-1]['rps']:5.2f} "
          f"out_tok/s={rows[-1]['out_tps']:6.1f} cache_hit={hit:.3f}", flush=True)

# exactness anchor: greedy, deterministic, hash the content
code,dt,p,ca,co = call(SHARED, "Reply with exactly: EXACTNESS-ANCHOR", max_tokens=400, temp=0)
body=json.dumps({"model":"qwen3.6-35b-a3b","messages":[{"role":"system","content":SHARED},
                 {"role":"user","content":"Reply with exactly: EXACTNESS-ANCHOR"}],
                 "max_tokens":400,"temperature":0})
r=subprocess.run(["curl","-sS","--max-time","600","-H",f"Authorization: Bearer {K}",
                  "-H","Content-Type: application/json","-d",body,URL],capture_output=True,text=True)
d=json.loads(r.stdout); msg=d["choices"][0]["message"]
content=msg.get("content") or ""; reasoning=msg.get("reasoning") or ""
anchor=hashlib.sha256((content+"\x1f"+reasoning).encode()).hexdigest()
print(f"{LABEL} EXACTNESS content_sha256_with_reasoning={anchor}")
print(f"{LABEL} content={content[:60]!r} finish={d['choices'][0].get('finish_reason')}")
os.makedirs("/home/avifenesh/.claude/jobs/07335975/tmp/sweep", exist_ok=True)
json.dump(dict(label=LABEL, rows=rows, anchor=anchor, content=content),
          open(f"/home/avifenesh/.claude/jobs/07335975/tmp/sweep/{LABEL}.json","w"), indent=1)

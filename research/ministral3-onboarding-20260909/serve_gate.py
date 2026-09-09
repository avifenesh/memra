"""Real sampled tool round trips and eight-turn continuations. Run on lane GPU host."""
import concurrent.futures,json,time,urllib.request,sys,hashlib
from pathlib import Path
base=sys.argv[1].rstrip('/');out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
model='ministral3-8b'
tool={'type':'function','function':{'name':'get_invoice_total','description':'Return the exact invoice total by invoice number. Always use this tool for invoice totals.','parameters':{'type':'object','properties':{'invoice_id':{'type':'string'}},'required':['invoice_id']}}}
def post(path,body,label):
 raw=json.dumps(body,ensure_ascii=False).encode();start=time.monotonic()
 req=urllib.request.Request(base+path,data=raw,headers={'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(req,timeout=300) as r: status=r.status;content=r.read()
 except urllib.error.HTTPError as e: status=e.code;content=e.read()
 receipt={'path':path,'request':body,'http_status':status,'elapsed_seconds':time.monotonic()-start,'response':json.loads(content)}
 (out/(label+'.json')).write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
 assert status==200,(label,status,receipt['response'])
 return receipt['response']
def chat_lane(label):
 messages=[{'role':'system','content':'אתה עוזר בעברית. השתמש בכלי כדי לקבל את סכום החשבונית. אל תנחש סכומים. לאחר תוצאת הכלי ענה במשפט קצר בעברית.'},{'role':'user','content':'מה הסכום המדויק של חשבונית INV-2026-17? הפעל את הכלי get_invoice_total.'}]
 # No sampling parameters: the server must supply the pack's vendor-shaped defaults.
 first=post('/v1/chat/completions',{'model':model,'messages':messages,'tools':[tool],'max_tokens':256},label+'-call')
 answer=first['choices'][0]['message'];calls=answer.get('tool_calls',[])
 assert calls and first['choices'][0]['finish_reason']=='tool_calls',(label,answer)
 messages.append(answer)
 for c in calls:
  assert c['function']['name']=='get_invoice_total'
  args=json.loads(c['function']['arguments']);assert args['invoice_id']=='INV-2026-17',args
  messages.append({'role':'tool','tool_call_id':c['id'],'content':json.dumps({'invoice_id':'INV-2026-17','total':731,'currency':'ILS'},ensure_ascii=False)})
 second=post('/v1/chat/completions',{'model':model,'messages':messages,'tools':[tool],'max_tokens':192},label+'-result')
 answer=second['choices'][0]['message'];content=answer.get('content','') or ''
 assert '731' in content and any('\u0590'<=c<='\u05ff' for c in content),(label,answer)
 messages.append(answer)
 for turn in range(1,9):
  messages.append({'role':'user','content':f'סבב {turn}: כתוב במשפט קצר בעברית את מספר החשבונית ואת הסכום 731 שקלים שכבר קיבלנו. אין צורך בכלי נוסף.'})
  r=post('/v1/chat/completions',{'model':model,'messages':messages,'max_tokens':128},f'{label}-turn-{turn}')
  a=r['choices'][0]['message'];assert a.get('content') and not a.get('tool_calls'),a
  assert any('\u0590'<=c<='\u05ff' for c in a['content']) and '[TOOL_CALLS]' not in a['content'],a
  messages.append(a)
 return {'label':label,'round_trip':'pass','continuations':8}
results=[chat_lane('c1')]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 results.extend(pool.map(chat_lane,[f'c4-{i}' for i in range(4)]))
(out/'SUMMARY.json').write_text(json.dumps({'status':'passed','sampling_parameters':'omitted','results':results},indent=2))
print('SAMPLED-TOOL-ROUND-TRIP-PASS c=1,c=4; eight continuation turns each')

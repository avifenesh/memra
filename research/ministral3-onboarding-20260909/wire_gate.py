"""Sampled Anthropic and Responses tool round trips through native serving."""
import json,sys,urllib.request
from pathlib import Path
base=sys.argv[1].rstrip('/');out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
model='ministral3-8b';prompt='מה הסכום המדויק של חשבונית INV-2026-17? הפעל את הכלי get_invoice_total ואחר כך ענה בעברית.'
schema={'type':'object','properties':{'invoice_id':{'type':'string'}},'required':['invoice_id']}
def post(path,body,label):
 req=urllib.request.Request(base+path,data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json','anthropic-version':'2023-06-01'})
 try:
  with urllib.request.urlopen(req,timeout=300) as r:code=r.status;data=json.load(r)
 except urllib.error.HTTPError as e:code=e.code;data=json.load(e)
 (out/(label+'.json')).write_text(json.dumps({'request':body,'status':code,'response':data},ensure_ascii=False,indent=2));assert code==200,(code,data)
 return data
result=json.dumps({'invoice_id':'INV-2026-17','total':731,'currency':'ILS'})
tool={'name':'get_invoice_total','description':'Return exact invoice total. Use this for invoice totals.','input_schema':schema}
messages=[{'role':'user','content':prompt}]
a=post('/v1/messages',{'model':model,'max_tokens':256,'system':'ענה בעברית. אל תנחש סכומים. השתמש בכלי.','messages':messages,'tools':[tool]},'anthropic-call')
calls=[c for c in a['content'] if c['type']=='tool_use'];assert calls,a
messages.append({'role':'assistant','content':a['content']})
messages.append({'role':'user','content':[{'type':'tool_result','tool_use_id':c['id'],'content':result} for c in calls]})
b=post('/v1/messages',{'model':model,'max_tokens':192,'messages':messages,'tools':[tool]},'anthropic-result')
text=''.join(c.get('text','') for c in b['content']);assert '731' in text and any('\u0590'<=c<='\u05ff' for c in text),b
rt={'type':'function','name':'get_invoice_total','description':'Return exact invoice total. Use this for invoice totals.','parameters':schema}
items=[{'role':'user','content':prompt}]
a=post('/v1/responses',{'model':model,'input':items,'tools':[rt],'max_output_tokens':256},'responses-call')
calls=[c for c in a['output'] if c['type']=='function_call'];assert calls,a
items.extend(a['output']);items.extend({'type':'function_call_output','call_id':c['call_id'],'output':result} for c in calls)
b=post('/v1/responses',{'model':model,'input':items,'tools':[rt],'max_output_tokens':192},'responses-result')
text=''.join(c.get('text','') for m in b['output'] for c in m.get('content',[]));assert '731' in text and any('\u0590'<=c<='\u05ff' for c in text),b
print('ANTHROPIC-RESPONSES-TOOL-ROUND-TRIP-PASS')

"""Generate canonical template bytes remotely from pinned Jinja with HF's tojson law."""
import json,sys
from pathlib import Path
from jinja2.sandbox import ImmutableSandboxedEnvironment
root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
env=ImmutableSandboxedEnvironment(trim_blocks=True,lstrip_blocks=True)
env.filters['tojson']=lambda x,**kw:json.dumps(x,ensure_ascii=False,**kw)
def fail(message):raise ValueError(message)
env.globals['raise_exception']=fail
t=env.from_string((root/'chat_template.jinja').read_text())
cases={
 'plain':([{'role':'user','content':'שלום'}],[]),
 'system':([{'role':'system','content':'ענה בעברית'},{'role':'user','content':'Hello!'}],[]),
 'aggregate':([{'role':'user','content':'א'},{'role':'user','content':''},{'role':'user','content':'ב'}],[]),
 'multiple_calls':([{'role':'system','content':'עברית'},{'role':'user','content':'בדוק'},{'role':'assistant','content':'','tool_calls':[{'function':{'name':'clock','arguments':'{ "zone" : "Asia/Jerusalem" }'}},{'function':{'name':'count','arguments':''}}]},{'role':'tool','content':'12:00'},{'role':'tool','content':'7'},{'role':'assistant','content':'השעה שתים עשרה.'}],[]),
 'tools':([{'role':'system','content':'עברית'},{'role':'user','content':'בדוק'}],[{'type':'function','function':{'name':'clock','parameters':{'type':'object','properties':{}}}}]),
}
for name,(messages,tools) in cases.items():
 result=t.render(messages=messages,tools=tools,bos_token='<s>',eos_token='</s>',add_generation_prompt=True)
 (out/(name+'.txt')).write_text(result)
 print(name,len(result.encode()))

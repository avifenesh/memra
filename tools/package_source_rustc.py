#!/usr/bin/python3 -I
"""Finite prepared Linux/Cargo compiler-input receiver; no Cargo recursion."""
import hashlib,importlib.util,json,os,re,stat,subprocess,sys
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).absolute().parent.parent
spec=importlib.util.spec_from_file_location('source_io',Path(__file__).absolute().parent/'package_source_identity.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def fail(reason):raise m.Refused(reason)
def checked(root,path):
 p=Path(path).absolute();return m.regular(root,p.relative_to(root).as_posix())
def fresh():return m.package_capsule(ROOT)

def val(args,key,default=None):
 found=[]
 for index,arg in enumerate(args):
  if arg==key:
   m.require(index+1<len(args),'missing compiler option value');found.append(args[index+1])
  elif arg.startswith(key+'='):found.append(arg[len(key)+1:])
 m.require(len(found)<=1,'duplicate compiler option: '+key)
 return found[0] if found else default

def codes(args,key):
 found=[]
 for index,arg in enumerate(args):
  if arg=='-C' and index+1<len(args):item=args[index+1]
  elif arg.startswith('-C'):item=arg[2:]
  else:continue
  if item.startswith(key+'='):found.append(item[len(key)+1:])
 m.require(len(found)<=1,'duplicate compiler code option')
 return found[0] if found else ''

def cfg_options(args):
 found=[]
 for index,arg in enumerate(args):
  if arg=='--cfg':
   m.require(index+1<len(args),'missing compiler cfg value');found.append(args[index+1])
  elif arg.startswith('--cfg='):found.append(arg[len('--cfg='):])
 m.require(len(found)==len(set(found)),'duplicate compiler cfg')
 return found

def record_path(artifact):
 path=Path(artifact)
 members=[path.with_suffix('.rlib'),path.with_suffix('.rmeta')] if path.suffix in ('.rlib','.rmeta') else [path]
 bundle={member.suffix or 'binary':m.regular(member.parent,member.name) for member in members}
 return Path(str(path)+'.'+m.digest(bundle)+'.memra-source.json')
def artifact_record(path,cap):
 path=Path(path).absolute();out=Path(cap['output']);m.require(path.is_relative_to(out),'unmatched extern outside owned output')
 record=m.owned_json(record_path(path))
 fields={'schema','source_seal','package','manifest_root','features','argv','artifact','generated','input_seal','input_tuple','custody_key','compiler_unit','artifact_bundle','artifact_role','seal'}
 m.require(type(record) is dict and set(record)==fields and record['schema']=='memra-package-compiler-unit-v1','unknown artifact receipt shape')
 body={k:v for k,v in record.items() if k!='seal'};m.require(m.digest(body)==record['seal'],'artifact receipt seal differs')
 tuple_fields={'domain','source_seal','admission','package','features','cfgs','dependencies','own_generated','nonidentity_env','codegen'}
 m.require(type(record['input_tuple']) is dict and set(record['input_tuple'])==tuple_fields and record['input_seal']==m.digest(record['input_tuple']),'compiled input tuple seal differs')
 m.require(record['compiler_unit']==m.digest({'inputs':record['input_tuple'],'argv':record['argv']}),'compiler unit context seal differs')
 key=m.digest({'inputs':record['input_tuple'],'argv':record['argv'],'artifact':str(path),'produced':record['artifact_bundle']})
 m.require(record['custody_key']==key,'producer observation key differs')
 expected_path=Path(cap['expectations'])/(key+'.json')
 m.require(expected_path.exists(),'external producer observation expectation missing: '+key)
 m.require(m.owned_json(expected_path)=={'receipt_seal':record['seal']},'external producer custody differs')
 m.require(record['package'] in cap['roots'] and record['source_seal']==m.digest(cap['snapshot']['payload']['packages'][record['package']]),'stale artifact source seal')
 current_admission=m.digest({'recipe':cap['recipe'],'supplementary':{k:v for k,v in cap['supplementary'].items() if v['owner']==record['package']}})
 m.require(record['input_tuple']['admission']==current_admission,'stale compiler recipe admission')
 m.require(record['manifest_root']==cap['roots'][record['package']],'artifact manifest differs')
 m.require(checked(out,path)==record['artifact'],'extern artifact bytes/mode differ')
 m.require(record['artifact_role'] in record['artifact_bundle'] and record['artifact_bundle'][record['artifact_role']]==record['artifact'],'consumed compiler unit member differs')
 for role,entry in record['artifact_bundle'].items():
  m.require(role in ('rlib','rmeta','so','binary'),'unknown produced compiler member role')
  member=path if role=='binary' else path.with_suffix('.'+role)
  m.require(checked(out,member)==entry,'compiler unit bundle member bytes/mode differ')
 for name,row in record['generated'].items():
  p=Path(name);m.require(p.is_absolute() and p.is_relative_to(out),'generated input outside owned output')
  m.require(checked(out,p)==row,'generated input bytes/mode differ')
 return record

def dependencies(args,cap,owner=None):
 pairs=[];own=[]
 for index,arg in enumerate(args):
  if arg=='--extern':m.require(index+1<len(args),'missing extern value');item=args[index+1]
  elif arg.startswith('--extern='):item=arg[len('--extern='):]
  else:continue
  m.require('=' in item,'unmatched extern without explicit artifact')
  alias,path=item.split('=',1);m.require(alias and path,'invalid extern binding')
  record=artifact_record(path,cap)
  if record['package']==owner:
   own.append(record);continue
  pairs.append({'alias':alias,'package':record['package'],'artifact_bundle':record['artifact_bundle'],'compiler_unit':record['compiler_unit'],'input_seal':record['input_seal']})
 return sorted(pairs,key=lambda x:(x['alias'],x['package'])),own

def dep_inputs(depfile,package_root,output,source_files):
 # Pilot supports normal Cargo dep-info in owned paths without whitespace.
 raw=m.regular(depfile.parent,depfile.name,contents=True).decode()
 files=set()
 for line in raw.replace('\\\n','').splitlines():
  if not line or line.startswith('#'):continue
  m.require(': ' in line or line.endswith(':'),'unknown dep-info form')
  if ': ' not in line:continue
  for word in line.split(': ',1)[1].split():
   m.require('\\' not in word,'escaped dep-info path unsupported in prepared recipe')
   p=Path(word);p=p if p.is_absolute() else Path.cwd()/p;files.add(p.absolute())
 generated={}
 for path in sorted(files):
  if path.is_relative_to(package_root):
   m.require(path.relative_to(package_root).as_posix() in source_files,'compiler source was excluded from capsule')
   checked(package_root,path)
  elif path.is_relative_to(output):generated[str(path)]=checked(output,path)
  else:fail('compiler consumed unbound generated/source input: '+str(path))
 return generated

def own_inputs(cap,key,crate):
 generated={};physical={}
 if crate!='build_script_build':
  root=Path(os.environ.get('OUT_DIR','')).absolute();expected=sorted(cap['recipe']['generated'][key])
  m.require(root.is_relative_to(Path(cap['output'])),'unknown own OUT_DIR binding')
  found=[]
  for parent,dirs,files in os.walk(root,followlinks=False):
   for name in dirs:
    fd,_=m.directory(Path(parent)/name);os.close(fd)
   found.extend((Path(parent)/name).relative_to(root).as_posix() for name in files)
  m.require(sorted(found)==expected,'unknown generated OUT_DIR membership')
  for name in expected:
   generated[name]=m.regular(root,name);physical[str(root/name)]=generated[name]
 names=cap['recipe']['env'][key] if crate!='build_script_build' else []
 env={name:{'present':name in os.environ,'sha256':hashlib.sha256(os.environ[name].encode()).hexdigest() if name in os.environ else None} for name in names}
 return generated,physical,env

def checked_env_reads(depfile,declared,generated_env):
 raw=m.regular(depfile.parent,depfile.name,contents=True).decode()
 for line in raw.splitlines():
  if not line.startswith('# env-dep:'):continue
  item=line[len('# env-dep:'):];name,separator,value=item.partition('=')
  if name in m.METADATA_ENV:continue
  m.require(name in declared,'unknown compiler env read: '+name)
  m.require(declared[name]['present']==bool(separator) and (not separator or hashlib.sha256(value.encode()).hexdigest()==declared[name]['sha256']),'actual compiled env binding differs')

def compiler_fds():
 pairs=re.findall(r'--jobserver-(?:auth|fds)=(\d+),(\d+)',os.environ.get('CARGO_MAKEFLAGS',''))
 if not pairs:return ()
 m.require(len(set(pairs))==1,'ambiguous inherited Cargo jobserver descriptors')
 descriptors=tuple(int(x) for x in pairs[0])
 m.require(len(set(descriptors))==2 and all(2<x<65536 for x in descriptors),'invalid Cargo jobserver descriptors')
 for descriptor in descriptors:m.require(stat.S_ISFIFO(os.fstat(descriptor).st_mode),'jobserver descriptor is not a pipe')
 return descriptors

def receiver(argv):
 if argv==['--fresh']:
  cap=fresh();expected=ROOT/cap['recipe']['wrapper'];actual=os.environ.get('RUSTC_WRAPPER','')
  m.require(actual and Path(actual).absolute()==expected,'effective wrapper missing or unmatched')
  print(cap['snapshot']['sha256']);return 0
 m.require(argv,'missing rustc executable')
 compiler,args=argv[0],argv[1:]
 m.require(not any(arg=='--test' or arg.startswith('--test=') for arg in args),'unadmitted implicit compiler test mode')
 if ('--crate-name' not in args and not any(x.startswith('--crate-name=') for x in args)) or ('-' in args and val(args,'--crate-name')=='___' and any(x.startswith('--print') for x in args) and val(args,'--out-dir') is None):
  return subprocess.run([compiler,*args],check=False,pass_fds=compiler_fds()).returncode
 cap=fresh()
 m.require(type(cap['compiler']) is dict and set(cap['compiler'])=={'path','file'} and Path(compiler).absolute()==Path(cap['compiler']['path']) and m.regular(Path(compiler).absolute().parent,Path(compiler).name)==cap['compiler']['file'],'actual compiler differs from supported builder')
 manifest=Path(os.environ.get('CARGO_MANIFEST_DIR','')).absolute()
 keys=[key for key,rel in cap['roots'].items() if ROOT/rel==manifest]
 m.require(len(keys)==1,'actual compiler manifest/source root is unbound')
 key=keys[0];row=cap['snapshot']['payload']['packages'][key]
 m.require(os.environ.get('CARGO_PKG_NAME')==row['name'] and os.environ.get('CARGO_PKG_VERSION')==row['version'],'actual compiler package identity differs')
 cfgs=cfg_options(args)
 m.require(all(x.startswith('feature=') or x in cap['recipe']['cfgs'][key] for x in cfgs),'unknown actual compiler cfg outside prepared recipe')
 features=sorted(x.split('=',1)[1].strip('"') for x in cfgs if x.startswith('feature='))
 m.require(set(features)<=set(row['features']),'actual compiler features outside admitted source graph')
 crate=val(args,'--crate-name');kind=val(args,'--crate-type');out=Path(val(args,'--out-dir','')).absolute();output=Path(cap['output'])
 m.require(out.is_relative_to(output),'compiler output outside owned role')
 selected_codegen={}
 for index,arg in enumerate(args):
  if arg=='-C':m.require(index+1<len(args),'missing codegen option');option=args[index+1]
  elif arg.startswith('-C'):option=arg[2:]
  else:continue
  name,separator,value=option.partition('=')
  m.require(separator and name,'invalid codegen option')
  if name in ('metadata','extra-filename','incremental'):continue
  m.require(name in cap['recipe']['codegen'] and value in cap['recipe']['codegen'][name],'unknown compiler profile/default')
  m.require(name not in selected_codegen,'duplicate nonidentity compiler codegen choice')
  selected_codegen[name]=value
 extra=codes(args,'extra-filename');m.require(kind in ('lib','bin','proc-macro'),'unsupported compiler crate type')
 m.require(val(args,'--target',cap['recipe']['target'])==cap['recipe']['target'],'compiler target differs')
 if key==cap['snapshot']['payload']['entry']:m.require(crate in cap['recipe']['targets'] or crate=='build_script_build','unknown root Cargo target')
 m.require('link' in val(args,'--emit','').split(','),'metadata-only compiler mode unsupported in prepared recipe')
 input_files=[Path(x).absolute() for x in args if x.endswith('.rs') and not x.startswith('-')]
 m.require(len(input_files)==1 and input_files[0].is_relative_to(manifest),'primary compiler source is unbound')
 checked(manifest,input_files[0])
 deps,own=dependencies(args,cap,key)
 source=cap['source_seal']
 own_generated,declared_generated,declared_env=own_inputs(cap,key,crate)
 admission = m.digest({'recipe':cap['recipe'],'supplementary':{k:v for k,v in cap['supplementary'].items() if v['owner']==key}})
 input_payload={'domain':'memra-package-input-tuple-v2','source_seal':m.digest(row),'admission':admission,'package':key,'features':features,'cfgs':sorted(cfgs),'dependencies':deps,'own_generated':own_generated,'nonidentity_env':declared_env,'codegen':selected_codegen}
 input_seal=m.digest(input_payload)
 id_value=m.identity('memra-package-compiled-input-v1',input_payload)
 m.require(all(x['input_seal']==input_seal for x in own),'own library foreign input identity differs')
 env=dict(os.environ);metadata={'MEMRA_BUILD_ID':id_value,'MEMRA_BUILD_ID_SRC':'package-source-v1','MEMRA_BUILD_ID_NOTE':'package source='+m.identity('memra-package-source-v1',source)+'; checked compiler inputs, not binary recipe or native qualification','MEMRA_BUILD_SHA':os.environ.get('MEMRA_BUILD_SHA','unknown')}
 if key==cap['snapshot']['payload']['entry'] and crate!='build_script_build':env.update({'MEMRA_BUILD_ID':os.environ.get('MEMRA_BUILD_ID','000000000000'),'MEMRA_BUILD_ID_SRC':'degraded','MEMRA_BUILD_ID_NOTE':'package compiler read bootstrap; not admitted','MEMRA_BUILD_SHA':'unknown'})
 changed_env_keys=sorted(k for k in set(env)|set(os.environ) if env.get(k)!=os.environ.get(k))
 m.require(set(changed_env_keys)<=set(metadata),'unexpected compiler env mutation')
 forward=[compiler,*args];code=subprocess.run(forward,env=env,check=False,pass_fds=compiler_fds()).returncode
 if code:return code
 m.require(fresh()==cap,'package capsule/source changed during compiler execution')
 # Recheck inputs after actual compiler returns. No passing record after drift.
 dependencies(args,cap,key)
 stem=crate+extra;paths=[out/(('lib'+stem+'.rlib') if kind=='lib' else ('lib'+stem+'.so') if kind=='proc-macro' else stem)]
 if kind=='lib' and (out/('lib'+stem+'.rmeta')).exists():paths.append(out/('lib'+stem+'.rmeta'))
 depfile=out/(stem+'.d');generated=dep_inputs(depfile,manifest,output,row['files'])
 if crate!='build_script_build':
  m.require(all(name in declared_generated and declared_generated[name]==entry for name,entry in generated.items()),'actual generated input binding differs')
  checked_env_reads(depfile,declared_env,declared_generated)
 if key==cap['snapshot']['payload']['entry'] and crate!='build_script_build':
  # First compiler result is explicitly unaccepted. Only a checked complete
  # read set permits the intended identity metadata on the second pass.
  env.update(metadata);code=subprocess.run(forward,env=env,check=False,pass_fds=compiler_fds()).returncode
  if code:return code
  m.require(fresh()==cap,'package capsule/source changed during compiler execution');dependencies(args,cap,key)
  m.require(own_inputs(cap,key,crate)==(own_generated,declared_generated,declared_env),'own compiler inputs changed')
  m.require(all(name in declared_generated and declared_generated[name]==entry for name,entry in dep_inputs(depfile,manifest,output,row['files']).items()),'actual generated input binding differs')
  checked_env_reads(depfile,declared_env,declared_generated)
 artifact_bundle={('rlib' if x.suffix=='.rlib' else 'rmeta' if x.suffix=='.rmeta' else 'so' if x.suffix=='.so' else 'binary'):checked(output,x) for x in paths}
 compiler_unit=m.digest({'inputs':input_payload,'argv':forward})
 for artifact in paths:
  body={'schema':'memra-package-compiler-unit-v1','source_seal':m.digest(row),'package':key,'manifest_root':cap['roots'][key],'features':features,'argv':forward,'artifact':checked(output,artifact),'generated':generated,'input_seal':input_seal,'input_tuple':input_payload,'compiler_unit':compiler_unit,'artifact_bundle':artifact_bundle,'artifact_role':'rlib' if artifact.suffix=='.rlib' else 'rmeta' if artifact.suffix=='.rmeta' else 'so' if artifact.suffix=='.so' else 'binary'}
  custody_key=m.digest({'inputs':input_payload,'argv':forward,'artifact':str(artifact),'produced':artifact_bundle})
  body['custody_key']=custody_key
  record={**body,'seal':m.digest(body)}
  expected_path=Path(cap['expectations'])/(custody_key+'.json')
  expectation={'receipt_seal':record['seal']}
  if expected_path.exists():m.require(m.owned_json(expected_path)==expectation,'external producer custody differs')
  else:m.immutable_json(expected_path,expectation)
  m.immutable_json(record_path(artifact),record)
 if key==cap['snapshot']['payload']['entry'] and crate!='build_script_build':
  bindings=[]
  for index,arg in enumerate(args):
   item=args[index+1] if arg=='--extern' else arg[len('--extern='):] if arg.startswith('--extern=') else None
   if item is None:continue
   record=artifact_record(item.split('=',1)[1],cap)
   if record['package']==key:continue
   path=Path(item.split('=',1)[1])
   for role,file in record['artifact_bundle'].items():bindings.append({'path':str(path.with_suffix('.'+role)),'file':file})
  bindings.extend({'path':name,'file':file} for name,file in declared_generated.items())
  value={'source_seal':source,'tuple':input_payload,'bindings':sorted(bindings,key=lambda x:x['path'])}
  m.immutable_json(output/('identity-'+id_value+'.json'),value)
  m.immutable_json(Path(cap['expectations'])/('identity-'+id_value+'.json'),{'sha256':m.digest(value)})
 return 0
if __name__=='__main__':
 try:sys.exit(receiver(sys.argv[1:]))
 except (m.Refused,OSError,ValueError,KeyError) as error:print('package compiler refused: '+str(error),file=sys.stderr);sys.exit(86)

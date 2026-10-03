"""Compare actual emitted functions while retaining operative IR and metadata."""
import hashlib,json,re,subprocess
from pathlib import Path

def read_functions(path):
    source=Path(path).read_text()
    attributes={m.group(1):m.group(2) for m in re.finditer(r'^attributes #(\d+) = (.*)$',source,re.M)}
    metadata={m.group(1):m.group(2) for m in re.finditer(r'^!(\d+) = (.*)$',source,re.M)}
    types=set(re.findall(r'^(%[^ ]+) = type ',source,re.M))
    functions={}
    header=None;body=[]
    for line in source.splitlines(keepends=True):
        if line.startswith('define '):
            header=line;body=[line];continue
        if header is not None:
            body.append(line)
            if line=='}\n':
                symbol=re.search(r'@([^\s(]+)',header).group(1)
                functions[symbol]=''.join(body)
                header=None;body=[]
    return functions,attributes,metadata,types

def canonical(body,attributes,metadata,types):
    # Comments and debug attachment IDs are non-operative. All other metadata stays.
    lines=[]
    for line in body.splitlines():
        if line.lstrip().startswith(';'):continue
        line=re.sub(r', !dbg !\d+','',line)
        line=re.sub(r' ;.*$','',line)
        if line.strip():lines.append(line.rstrip())
    text='\n'.join(lines)
    values={}
    def value(name):
        if name in types:return name
        return values.setdefault(name,'%v'+str(len(values)))
    # Bare block definitions and their %branch uses share one SSA namespace.
    text=re.sub(r'(?m)^([A-Za-z0-9_.$-]+):',lambda m:value('%'+m.group(1))[1:]+':',text)
    text=re.sub(r'%[A-Za-z0-9_.$-]+',lambda m:value(m.group()),text)
    text=re.sub(r'#(\d+)',lambda m:'ATTR'+attributes[m.group(1)],text)
    nodes={}
    def node(number):
        if number in nodes:return 'META'+str(nodes[number])
        index=len(nodes);nodes[number]=index
        raw=metadata[number]
        resolved=re.sub(r'!(\d+)',lambda m:node(m.group(1)),raw)
        return 'META'+str(index)+'('+resolved+')'
    text=re.sub(r'!(\d+)',lambda m:node(m.group(1)),text)
    return text

def name(symbol):
    return subprocess.check_output(['llvm-cxxfilt',symbol],text=True).strip()

def compare(before,after,symbols,out):
    old,oa,om,ot=read_functions(before)
    new,na,nm,nt=read_functions(after)
    output=Path(out);output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for symbol in symbols:
        assert symbol in old and symbol in new,('emitted symbol absent',symbol)
        left=canonical(old[symbol],oa,om,ot);right=canonical(new[symbol],na,nm,nt)
        short=hashlib.sha256(symbol.encode()).hexdigest()[:16]
        (output/(short+'-before.ll')).write_text(left+'\n')
        (output/(short+'-after.ll')).write_text(right+'\n')
        rows.append({'symbol':symbol,'name':name(symbol),'equal':left==right,
                     'before_sha256':hashlib.sha256(left.encode()).hexdigest(),
                     'after_sha256':hashlib.sha256(right.encode()).hexdigest(),
                     'canonical_pair_prefix':short})
    return rows

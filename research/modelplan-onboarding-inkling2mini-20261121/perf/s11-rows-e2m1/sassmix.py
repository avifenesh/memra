import sys,re,collections
lines=open(sys.argv[1]).read().split('\n')
start=[i for i,l in enumerate(lines) if 'Function :' in l and sys.argv[2] in l][0]
end=next((i for i in range(start+1,len(lines)) if 'Function :' in lines[i]), len(lines))
c=collections.Counter(); tot=0
for l in lines[start:end]:
    m=re.match(r'\s+/\*[0-9a-f]+\*/\s+(@!?U?P\w+\s+)?([A-Z0-9_]+)',l)
    if m: c[m.group(2)]+=1; tot+=1
print('total',tot); print(c.most_common(25))

import struct,math,sys
def load(p):
    v={}
    for line in open(p):
        f=line.rstrip('\n').split('\t')
        if f[0]=='logit': v[int(f[1])]=struct.unpack('>f',bytes.fromhex(f[2]))[0]
    return [v[i] for i in range(len(v))]
def top(x,k=8):
    idx=sorted(range(len(x)),key=lambda i:-x[i])[:k]
    return [(i,round(x[i],3)) for i in idx]
a=load(sys.argv[1]); b=load(sys.argv[2])
print(sys.argv[1],'top8',top(a))
print(sys.argv[2],'top8',top(b))
d=[x-y for x,y in zip(a,b)]
print('max_abs %.4f mean_abs %.4f'%(max(abs(x) for x in d),sum(abs(x) for x in d)/len(d)))
num=sum(x*y for x,y in zip(a,b)); na=math.sqrt(sum(x*x for x in a)); nb=math.sqrt(sum(y*y for y in b))
print('cosine %.6f  |a| %.3f |b| %.3f'%(num/(na*nb),na,nb))

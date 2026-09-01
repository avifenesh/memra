import os, struct, sys
def load(p):
    b=open(p,"rb").read(); n=len(b)//4
    return struct.unpack(f"<{n}f", b)
a_dir,b_dir=sys.argv[1],sys.argv[2]
files=sorted(f for f in os.listdir(a_dir) if f.endswith(".f32") and os.path.exists(os.path.join(b_dir,f)))
print("file\tmax_abs_diff\tnorm_rel(=max_abs/scale)\targmax_same\tmargin_a(top1-top2)\ttop1_b_rank_in_a")
worst=0.0
for f in files:
    va=load(os.path.join(a_dir,f)); vb=load(os.path.join(b_dir,f))
    scale=max(abs(x) for x in va)
    mad=max(abs(x-y) for x,y in zip(va,vb))
    nr=mad/scale
    ia=max(range(len(va)),key=lambda i:va[i]); ib=max(range(len(vb)),key=lambda i:vb[i])
    sa=sorted(va,reverse=True); margin=sa[0]-sa[1]
    # rank of b-argmax inside a
    rank=sum(1 for x in va if x>va[ib])
    print(f"{f}\t{mad:.6e}\t{nr:.3e}\t{ia==ib}\t{margin:.6f}\t{rank}")
    worst=max(worst,nr)
print(f"WORST norm_rel={worst:.3e}")

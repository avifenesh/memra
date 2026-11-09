"""Make the lever-6 P2 probe TU: the tree's mmq_nvfp4_w4a8.cu with a mode-3 switch in the v3 mainloop.
With f8f4v3_mode3 set, the CTA loads its first STAGES stages and then computes every k step on those
resident slots: no waits after the prologue, no refills, so the time is the stage compute and the
epilogue without the feed (the output is wrong by design; timing only).
usage: make_v3m3.py IN OUT"""
import sys
src, out = sys.argv[1], sys.argv[2]
s = open(src).read()
def sub1(a, b):
    global s
    if s.count(a) != 1:
        sys.exit(f"anchor count {s.count(a)}: {a[:70]!r}")
    s = s.replace(a, b)
sub1("struct F8F4V3Maps { CUtensorMap a, b, sa, sb; };",
     "struct F8F4V3Maps { CUtensorMap a, b, sa, sb; };\n__device__ int f8f4v3_mode3_d = 0;   // probe switch (mode 3)")
sub1('''        const int slot = kt % STAGES;
        f8f4v3_wait(bars + 8u * slot, (unsigned) (kt / STAGES) & 1u);''',
'''        const int slot = kt % STAGES;
        const bool m3 = f8f4v3_mode3_d != 0;
        if (!m3 || kt < STAGES) { f8f4v3_wait(bars + 8u * slot, (unsigned) (kt / STAGES) & 1u); }''')
sub1('''        if (tid == 0) {
            const int nx = kt + STAGES;
            if (nx < nk) {''',
'''        if (tid == 0 && !m3) {
            const int nx = kt + STAGES;
            if (nx < nk) {''')
open(out, "w").write(s)
print("ok")

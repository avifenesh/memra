from pathlib import Path
p=Path(__file__).resolve().parent
s=(p.parent.parent/'crates/memra-engine/cu/mmq_nvfp4_w4a8.cu').read_text()
s=s.replace('#include "mmq_common.cuh"','#include "../../crates/memra-engine/cu/mmq_common.cuh"')
a=s.index('// ======================= mul_mat_q_process_tile PIPELINED')
b=s.index('// ======================= mul_mat_q (conventional',a)
part=s[a:b]
needle='        dequant_tiles_nvfp4_w4a8<mmq_y, need_check, is_rp>(xr, tile_x, tile_x_max_i);'
assert part.count(needle)==1
part=part.replace(needle,'''#if ABLATION == 1
        if (kb0 == kb0_start)
#endif
'''+needle)
# Stage both activation planes once, then omit only subsequent global->shared staging.
needle='    for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {'
assert part.count(needle)==1
part=part.replace(needle,'''#if ABLATION == 3
    pipe_stage_y<mmq_x, nwarps>(tile_y1, y + ncols_y * (kb0_start * qk / ne_block + 1) * sz);
    pipe_commit();
    pipe_wait<0>();
#endif
'''+needle)
for line in ['        pipe_stage_y<mmq_x, nwarps>(tile_y1, y + ncols_y * (kchunk + 1) * sz);','            pipe_stage_y<mmq_x, nwarps>(tile_y0, y + ncols_y * (kchunk + 2) * sz);']:
 assert part.count(line)==1
 part=part.replace(line,'#if ABLATION != 3\n'+line+'\n#endif')
s=s[:a]+part+s[b:]
s='// Generated research-only component-removal control. Numerically invalid when ABLATION != 0.\n'+s
(p/'ablation-kernel.cu').write_text(s)
# Simple identical-input ABI comparison/timer, numerical changes explicitly diagnostic.
s=(p/'int8-tiles.cu').read_text()
s=s.replace('#include "../../crates/memra-engine/cu/mmq_nvfp4_w4a8.cu"','#include <cuda_runtime.h>')
a=s.index('template<int X,int Y,int Pipe>');b=s.index('int main(){',a)
s=s[:a]+'''extern "C" size_t base_bytes(int,int);
using Gemm=int(*)(const void*,const float*,float*,int,int,int,void*,void*,float,int);
extern "C" int base_gemm(const void*,const float*,float*,int,int,int,void*,void*,float,int);
extern "C" int unpack_gemm(const void*,const float*,float*,int,int,int,void*,void*,float,int);
extern "C" int fold_gemm(const void*,const float*,float*,int,int,int,void*,void*,float,int);
extern "C" int act_gemm(const void*,const float*,float*,int,int,int,void*,void*,float,int);
Gemm funcs[]={base_gemm,unpack_gemm,fold_gemm,act_gemm};
'''+s[b:]
a=s.index('    resource<');b=s.index('    for(int shape',a);s=s[:a]+s[b:]
s=s.replace('memra_mmq_nvfp4_w4a8_act_bytes(k,m)','base_bytes(k,m)')
a=s.index('            switch(arm)');b=s.index('        };',a)
s=s[:a]+'''            int rc=funcs[arm](dw,dx,dy,k,n,m,scratch,nullptr,0.37f,1);
            if(rc){fprintf(stderr,"arm %d rc=%d\\n",arm,rc);exit(3);}
'''+s[b:]
s=s.replace('arm<7','arm<4').replace('pos<7','pos<4').replace('6-pos','3-pos')
(p/'ablation-bench.cu').write_text(s)

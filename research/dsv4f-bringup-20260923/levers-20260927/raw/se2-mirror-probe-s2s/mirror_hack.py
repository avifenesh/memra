import sys
p, mode = sys.argv[1], sys.argv[2]
s = open(p).read()
gu_old = '''    const bool bad = dsv4_moe_fused_mirror<2 * WP>(xf + (size_t)(p / slots_per_row) * in_f,
                                                    in_f, As, s_scale, &s_rs, warp, lane);'''
gu_new = '''    // TIMING PROBE ONLY (box-local, never committed): no x mirror, zero activations.
    for (int i = threadIdx.y * 32 + threadIdx.x; i < in_f / 2; i += 2 * WP * 32) As[i] = 0u;
    if (threadIdx.x == 0 && threadIdx.y == 0) s_rs = 1.0f;
    __syncthreads();
    const bool bad = false;'''
dn_old = '''    const bool bad = dsv4_moe_fused_mirror<WARPS>(H + (size_t)p * in_f, in_f, As, s_scale, &s_rs,
                                                  warp, lane);'''
dn_new = '''    // TIMING PROBE ONLY (box-local, never committed): no h mirror, zero activations.
    for (int i = threadIdx.y * 32 + threadIdx.x; i < in_f / 2; i += WARPS * 32) As[i] = 0u;
    if (threadIdx.x == 0 && threadIdx.y == 0) s_rs = 1.0f;
    __syncthreads();
    const bool bad = false;'''
assert s.count(gu_old) == 1 and s.count(dn_old) == 1
s = s.replace(gu_old, gu_new)
if mode == 'both':
    s = s.replace(dn_old, dn_new)
open(p, 'w').write(s)

// Included after the existing exact GU helpers. GU only, same A/B orientation,
// m16n8k16 f32 accumulation and complete ascending K chain. N64's two valid
// warps become two N32 CTAs, each with one MMA warp and one loading warp.
#pragma once

static __global__ void __launch_bounds__(64)
dsv4_gu_n32_kernel(
        const unsigned long long* __restrict__ table, int n_expert,
        const int* __restrict__ ex_ids, long row_bytes,
        const __half* __restrict__ A,
        float* __restrict__ H, const float* __restrict__ row_scale,
        const float* __restrict__ macro_g, const float* __restrict__ macro_u,
        const float* __restrict__ route_w,
        const int* __restrict__ ex_off, int n_active,
        int in_f, int out_f, int total_tiles, float limit, unsigned* visits = nullptr){
    constexpr int QT = QT_NVFP4_MODELOPT;
    constexpr bool PackedStore = true;
    constexpr int BN = 32;
    __shared__ int s_pre[SK_MAX_G + 1];
    __shared__ __align__(16) __half As[SKT_STAGES][16][SKT_STRIDE];
    __shared__ __align__(16) __half Bs[BN][SKT_STRIDE];
    __shared__ uint32_t s_cb[KQ_CB_WORDS(QT)];
    extern __shared__ uint32_t packed_h2[];
    const int ntx = (out_f + BN - 1) / BN;
    kq_stage_codebook<QT>(s_cb);
    kq_stage_half2_lut<PackedStore>(packed_h2);
    // m_e=1 only.  Empty and larger groups are left for the rollback visitor.
    sk_tile_prefix(s_pre, ex_off, n_active, ntx, SK_BM, 1, 2);
    if(total_tiles < 0) total_tiles = s_pre[n_active];

    const int lane = threadIdx.x, warp = threadIdx.y;
    const int tid  = warp * 32 + lane;
    const int nkb  = in_f / SKT_BK;
    const int wm = 0, wn = 0;
    const int brow = tid >> 1, bc0 = (tid & 1) * 32;

    for(int t = blockIdx.x; t < total_tiles; t += gridDim.x){
        const int g  = sk_tile_group(s_pre, n_active, t);
        const int lo = ex_off[g], m_e = ex_off[g+1] - lo;
        if(m_e != 1) continue;
        if(visits && tid == 0) atomicAdd(visits, 1u);
        const int local = t - s_pre[g];
        const int m0 = (local / ntx) * SK_BM;
        const int n0 = (local % ntx) * BN;
        const __half* Ag = A + (size_t)lo * in_f;
        const int eid = ex_ids[g];
        const uint8_t* Wg = (const uint8_t*)table[(size_t)0 * n_expert + eid];
        const uint8_t* Sg = (const uint8_t*)table[(size_t)1 * n_expert + eid];
        const uint8_t* Wu = (const uint8_t*)table[(size_t)4 * n_expert + eid];
        const uint8_t* Su = (const uint8_t*)table[(size_t)5 * n_expert + eid];
        const int bn = min(n0 + brow, out_f - 1);
        const uint8_t* gwrow = Wg + (size_t)bn * row_bytes;
        const uint8_t* gsrow = Sg + (size_t)bn * (in_f / 16);
        const uint8_t* uwrow = Wu + (size_t)bn * row_bytes;
        const uint8_t* usrow = Su + (size_t)bn * (in_f / 16);

        const __half* agp[2]; int asr[2], asc[2];
        #pragma unroll
        for(int i = 0; i < 2; i++){
            const int c = tid + i * 64;
            asr[i] = c >> 3; asc[i] = (c & 7) * 8;
            const int am = min(m0 + asr[i], m_e - 1);
            agp[i] = Ag + (size_t)am * in_f + asc[i];
        }
        #define N32_LOAD_A(st, k0) do { \
            _Pragma("unroll") \
            for(int i = 0; i < 2; i++) \
                sk_cp16(&As[st][asr[i]][asc[i]], agp[i] + (k0)); \
            asm volatile("cp.async.commit_group;"); \
        } while(0)

        N32_LOAD_A(0, 0);
        if(nkb > 1) N32_LOAD_A(1, SKT_BK);
        KqRaw gb0 = kq_fetch<QT>(gwrow, gsrow, bc0, s_cb, in_f);
        KqRaw gb1 = kq_fetch<QT>(gwrow, gsrow, bc0 + 16, s_cb, in_f);
        KqRaw ub0 = kq_fetch<QT>(uwrow, usrow, bc0, s_cb, in_f);
        KqRaw ub1 = kq_fetch<QT>(uwrow, usrow, bc0 + 16, s_cb, in_f);
        float ag[4][4] = {}, au[4][4] = {};
        for(int kb = 0; kb < nkb; kb++){
            const int cur = kb % SKT_STAGES;
            if(kb + 2 < nkb) N32_LOAD_A((kb + 2) % SKT_STAGES, (kb + 2) * SKT_BK);

            // Gate B tile and its next fetch.  The A tile is shared by both
            // projections; this is the only new fusion relative to the tail.
            kq_store_variant<QT, PackedStore>(gb0, &Bs[brow][bc0], s_cb, packed_h2);
            kq_store_variant<QT, PackedStore>(gb1, &Bs[brow][bc0 + 16], s_cb, packed_h2);
            if(kb + 1 < nkb){
                gb0 = kq_fetch<QT>(gwrow, gsrow, (kb + 1) * SKT_BK + bc0, s_cb, in_f);
                gb1 = kq_fetch<QT>(gwrow, gsrow, (kb + 1) * SKT_BK + bc0 + 16, s_cb, in_f);
            }
            if(kb + 2 < nkb)      asm volatile("cp.async.wait_group 2;");
            else if(kb + 1 < nkb) asm volatile("cp.async.wait_group 1;");
            else                  asm volatile("cp.async.wait_group 0;");
            __syncthreads();
            if(warp == 0) {
                #pragma unroll
                for(int kk = 0; kk < 4; kk++){
                    unsigned a[4], b0[4], b1[4];
                    sk_ldm16x16(a,  &As[cur][wm][kk*16],  SKT_STRIDE);
                    sk_ldm16x16(b0, &Bs[wn][kk*16],       SKT_STRIDE);
                    sk_ldm16x16(b1, &Bs[wn + 16][kk*16],  SKT_STRIDE);
                    sk_mma(ag[0], a, b0[0], b0[2]);
                    sk_mma(ag[1], a, b0[1], b0[3]);
                    sk_mma(ag[2], a, b1[0], b1[2]);
                    sk_mma(ag[3], a, b1[1], b1[3]);
                }
            }
            __syncthreads();

            // Up uses the same B tile and A tile after gate has consumed it.
            kq_store_variant<QT, PackedStore>(ub0, &Bs[brow][bc0], s_cb, packed_h2);
            kq_store_variant<QT, PackedStore>(ub1, &Bs[brow][bc0 + 16], s_cb, packed_h2);
            if(kb + 1 < nkb){
                ub0 = kq_fetch<QT>(uwrow, usrow, (kb + 1) * SKT_BK + bc0, s_cb, in_f);
                ub1 = kq_fetch<QT>(uwrow, usrow, (kb + 1) * SKT_BK + bc0 + 16, s_cb, in_f);
            }
            __syncthreads();
            if(warp == 0) {
                #pragma unroll
                for(int kk = 0; kk < 4; kk++){
                    unsigned a[4], b0[4], b1[4];
                    sk_ldm16x16(a,  &As[cur][wm][kk*16],  SKT_STRIDE);
                    sk_ldm16x16(b0, &Bs[wn][kk*16],       SKT_STRIDE);
                    sk_ldm16x16(b1, &Bs[wn + 16][kk*16],  SKT_STRIDE);
                    sk_mma(au[0], a, b0[0], b0[2]);
                    sk_mma(au[1], a, b0[1], b0[3]);
                    sk_mma(au[2], a, b1[0], b1[2]);
                    sk_mma(au[3], a, b1[1], b1[3]);
                }
            }
            __syncthreads();
        }
        #undef N32_LOAD_A

        const int r0 = m0 + wm + lane / 4;
        const int cb = n0 + wn + (lane % 4) * 2;
        const int pair = lo + r0;
        const float rs = (r0 < m_e) ? row_scale[pair] : 0.0f;
        const float mg = (r0 < m_e) ? macro_g[pair] : 0.0f;
        const float mu = (r0 < m_e) ? macro_u[pair] : 0.0f;
        const float rw = (r0 < m_e) ? route_w[pair] : 0.0f;
        float* hrow = H + (size_t)pair * out_f;
        #pragma unroll
        for(int nb = 0; nb < 4; nb++){
            const int c = cb + nb * 8;
            if(warp == 0 && r0 < m_e){
                #define N32_STORE(col, ga, ua) do { \
                    if((col) < out_f){ \
                        float g = __fmul_rn(__fmul_rn((ga), rs), mg); \
                        float u = __fmul_rn(__fmul_rn((ua), rs), mu); \
                        u = fminf(fmaxf(u, -limit), limit); \
                        g = fminf(g, limit); \
                        float hv = __fmul_rn(__fmul_rn(g, kq_gu_sigmoid(g)), u); \
                        hrow[(col)] = __fmul_rn(hv, rw); \
                    } \
                } while(0)
                N32_STORE(c,     ag[nb][0], au[nb][0]);
                N32_STORE(c + 1, ag[nb][1], au[nb][1]);
                #undef N32_STORE
            }
        }
    }
}


// Untimed fixture acquisition only. The caller explicitly opts in and runs the
// existing eager control, never a retained graph. Payloads contain only active
// real routed operands and selected packed planes, not model/runtime addresses.
static int dsv4_gu_n32_capture_fixture(
        const unsigned long long* table, int ne, const int* ids, const int* off,
        const void* act, const float* rs, const float* mg, const float* mu,
        const float* rw, const float* md, const int* pairs, int k, int n,
        float limit, void* stream_v, const char* directory) {
    static std::mutex lock;
    static int seen[2] = {};
    std::lock_guard<std::mutex> hold(lock);
    int dev = -1;
    if(cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev > 1) return 40080;
    if(seen[dev] >= 8) return 0;
    if(ne != 128 || k != 4096 || n != 2048 || !directory || !*directory) return 40080;
    cudaStream_t stream = (cudaStream_t)stream_v;
    cudaStreamCaptureStatus capture;
    if(cudaStreamIsCapturing(stream, &capture) != cudaSuccess || capture != cudaStreamCaptureStatusNone)
        return 40080;
    auto read = [stream](void* out, const void* in, size_t bytes) {
        if(bytes == 0) return true;
        return cudaMemcpyAsync(out,in,bytes,cudaMemcpyDeviceToHost,stream)==cudaSuccess
            && cudaStreamSynchronize(stream)==cudaSuccess;
    };
    std::vector<int> offsets(ne+1), expert_ids(ne);
    std::vector<unsigned long long> pointers(6*ne);
    if(!read(offsets.data(),off,offsets.size()*4) || !read(expert_ids.data(),ids,ne*4)
       || !read(pointers.data(),table,pointers.size()*8)) return 40080;
    const int live=offsets[ne];
    if(offsets[0] != 0 || live<0 || live>6) return 40080;
    for(int g=0;g<ne;++g)
        if(offsets[g+1]<offsets[g] || offsets[g+1]-offsets[g]>1) return 40080;
    char path[4096];
    if(snprintf(path,sizeof(path),"%s/rank%d-call%02d.gun32",directory,dev,seen[dev]) >= (int)sizeof(path)) return 40080;
    FILE* file=fopen(path,"wbx");
    if(!file) return 40080;
    bool ok=true;
    auto write=[&](const void* data,size_t bytes){ if(ok && fwrite(data,1,bytes,file)!=bytes) ok=false; };
    uint32_t header[8]={0x4e333247,1,(uint32_t)dev,(uint32_t)seen[dev],(uint32_t)ne,(uint32_t)k,(uint32_t)n,(uint32_t)live};
    write(header,sizeof(header));write(&limit,4);
    for(int g=0;g<ne && ok;++g) if(offsets[g+1]!=offsets[g]) {
        const int row=offsets[g],eid=expert_ids[g];
        if(eid<0 || eid>=ne) {ok=false;break;}
        int pair=-1;float scalars[5];
        std::vector<uint16_t> activation(k);
        ok=read(&pair,pairs+row,4) && pair>=0 && pair<6
            && read(activation.data(),(const uint16_t*)act+(size_t)row*k,k*2)
            && read(scalars,rs+row,4) && read(scalars+1,mg+row,4)
            && read(scalars+2,mu+row,4) && read(scalars+3,rw+row,4)
            && read(scalars+4,md+row,4);
        uint32_t meta[3]={(uint32_t)g,(uint32_t)eid,(uint32_t)pair};
        write(meta,sizeof(meta));write(scalars,sizeof(scalars));write(activation.data(),k*2);
        for(int plane=0;plane<6 && ok;++plane) {
            const size_t bytes=(size_t)k*n/(plane%2 ? 16 : 2);
            std::vector<uint8_t> buf(bytes);
            ok=read(buf.data(),(const void*)pointers[plane*ne+eid],bytes);
            write(buf.data(),bytes);
        }
    }
    if(fclose(file)) ok=false;
    if(!ok) {remove(path);return 40080;}
    fprintf(stdout,"GU_N32_CAPTURE rank=%d call=%d real_slots=%d path=%s\n",dev,seen[dev],live,path);
    fflush(stdout);++seen[dev];
    return 0;
}

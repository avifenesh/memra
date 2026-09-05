#include "../crates/memra-engine/cu/dsv4_gpu.cu"
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void check(cudaError_t rc) { if (rc != cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc)); }
template<typename T> struct Dev {
    T* p;
    explicit Dev(size_t n) { check(cudaMalloc(&p, n * sizeof(T))); }
    explicit Dev(const std::vector<T>& v) : Dev(v.size()) { check(cudaMemcpy(p, v.data(), v.size()*sizeof(T), cudaMemcpyHostToDevice)); }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
};
static uint32_t bits(float x) { uint32_t b; std::memcpy(&b, &x, 4); return b; }
static float fp8(uint8_t c) {
    int mag = c & 127, exp = mag >> 3, man = mag & 7;
    float v = exp ? std::ldexp(1.0f + man / 8.0f, exp - 7) : man / 512.0f;
    return c & 128 ? -v : v;
}
static void cell(int cols, bool underflow, bool nan) {
    constexpr int sources=3, rows=5, guard=11;
    std::vector<uint8_t> codes(sources * cols);
    std::vector<float> scales(sources * cols / 128);
    std::vector<int> ids{2,0,2,1,0};
    for (size_t i=0; i<codes.size(); ++i) { codes[i]=(i*97+43)%256; if ((codes[i]&127)==127) codes[i]=0; }
    for (size_t i=0; i<scales.size(); ++i) scales[i]=std::ldexp(1.0f, int(i%13)-8);
    if (underflow) { codes[cols*2]=1; scales[2*cols/128]=std::ldexp(1.0f,-70); }
    if (nan) codes[cols*2]=127;
    Dev<uint8_t> dc(codes);
    Dev<float> ds(scales), rs(rows);
    Dev<int> di(ids), status(rows);
    Dev<uint16_t> out(rows*cols+guard);
    check(cudaMemset(out.p,0xa5,(rows*cols+guard)*2));
    int rc=memra_dsv4_fp8_gather_half(dc.p,ds.p,di.p,out.p,rs.p,status.p,rows,cols,nullptr);
    if(rc)throw std::runtime_error("launcher failed");
    std::vector<uint16_t> got(rows*cols+guard);
    std::vector<float> rscale(rows);
    std::vector<int> flags(rows);
    check(cudaMemcpy(got.data(),out.p,got.size()*2,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(rscale.data(),rs.p,rows*4,cudaMemcpyDeviceToHost));
    check(cudaMemcpy(flags.data(),status.p,rows*4,cudaMemcpyDeviceToHost));
    for(int r=0;r<rows;++r) {
        bool invalid=(underflow||nan)&&ids[r]==2;
        if(bool(flags[r])!=invalid) throw std::runtime_error("row refusal missing or spurious");
        if(invalid)continue;
        for(int x=0;x<cols;++x) {
            uint16_t b=got[r*cols+x]; __half h; std::memcpy(&h,&b,2);
            float value=__half2float(h)*rscale[r];
            float expected=fp8(codes[ids[r]*cols+x])*scales[ids[r]*cols/128+x/128];
            if(bits(value)!=bits(expected)) throw std::runtime_error("mirror differs from FP8 QAT");
        }
    }
    for(int i=rows*cols;i<rows*cols+guard;++i) if(got[i]!=0xa5a5)throw std::runtime_error("guard overwrite");
    printf("PASS cols=%d underflow=%d nan=%d rows=%d\n",cols,underflow,nan,rows);
}
int main() {
    try {
        for(int cols:{128,256,2048,4096})cell(cols,false,false);
        cell(4096,true,false); cell(4096,false,true);
        puts("PASS exact FP8 half mirrors and nonrepresentable/NaN refusals");
    }catch(const std::exception& e){fprintf(stderr,"FAIL %s\n",e.what());return 1;}
}

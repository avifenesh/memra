from pathlib import Path
p=Path('crates/memra-engine/cu/mla_attn.cu')
s=p.read_text()
h=r'''
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
static void mla_verify_capture(const float* q,const float* qp,const float* cache,const int* idx,
 int h,int r,int dr,int t,int slots,float scale,cudaStream_t stream) {
 const char* root=getenv("GLM5_MLA_CAPTURE_DIR");
 if(!root || (t!=2 && t!=4 && t!=7)) return;
 static int seen[8]={};
 int ordinal=seen[t]++;
 if(ordinal>=11) return;
 cudaError_t rc=cudaStreamSynchronize(stream);
 if(rc!=cudaSuccess) {fprintf(stderr,"capture sync failed %d\n",rc);abort();}
 std::vector<int> ids(t*slots);
 if(cudaMemcpy(ids.data(),idx,ids.size()*4,cudaMemcpyDeviceToHost)!=cudaSuccess) abort();
 int rows=0;for(int x:ids) if(x>=rows) rows=x+1;
 std::string base=std::string(root)+"/t"+std::to_string(t)+"-mla"+std::to_string(ordinal);
 auto dump=[&](const char* suffix,const void* ptr,size_t n) {
   std::vector<char> data(n);
   if(n && cudaMemcpy(data.data(),ptr,n,cudaMemcpyDeviceToHost)!=cudaSuccess) abort();
   FILE* f=fopen((base+suffix).c_str(),"wb");if(!f)abort();
   if(fwrite(data.data(),1,n,f)!=n)abort();fclose(f);
 };
 dump(".q",q,(size_t)t*h*r*4);dump(".qp",qp,(size_t)t*h*dr*4);
 dump(".idx",idx,(size_t)t*slots*4);dump(".cache",cache,(size_t)rows*(r+dr)*4);
 FILE* f=fopen((base+".meta").c_str(),"w");if(!f)abort();
 fprintf(f,"%d %d %d %d %d %d %.9g\n",h,r,dr,t,slots,rows,scale);fclose(f);
 fprintf(stderr,"[mla-verify-capture] ordinal=%d t=%d heads=%d rank=%d rope=%d slots=%d rows=%d\n",ordinal,t,h,r,dr,slots,rows);
}
'''
s=s.replace('#define MLA_ERR()',h+'\n#define MLA_ERR()',1)
for name in ['memra_mla_attn_gathered_f32','memra_mla_attn_gathered_dsa_f32']:
 pos=s.index('extern "C" int '+name+'(')
 start=s.index('cudaStream_t stream = (cudaStream_t)stream_v;',pos)+len('cudaStream_t stream = (cudaStream_t)stream_v;')
 s=s[:start]+'\n    mla_verify_capture(q_lat,q_pe,cache,idx,n_head,kv_rank,d_rope,t_q,n_slots,scale,stream);'+s[start:]
p.write_text(s)

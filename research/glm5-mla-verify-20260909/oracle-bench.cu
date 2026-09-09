// Standalone B200 real-input oracle first, then rotating-layer and hot-layer ABBA.
#include "../../crates/memra-engine/cu/mla_attn.cu"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#define CHECK(x) do { int rc=(int)(x); if(rc) {fprintf(stderr,"FAIL %s:%d rc=%d %s\n",__FILE__,__LINE__,rc,#x);exit(2);} } while(0)
struct Input {
 std::string base, prompt;
 int h,r,dr,t,slots,rows,ordinal; float scale;
 float *q,*qp,*cache,*a,*b,*m,*d,*acc;int* idx;
};
static std::vector<char> read(const std::string& path) {
 std::ifstream f(path,std::ios::binary|std::ios::ate);
 if(!f) {fprintf(stderr,"missing %s\n",path.c_str());exit(2);}
 size_t n=f.tellg();f.seekg(0);std::vector<char> v(n);if(n&&!f.read(v.data(),n))exit(2);return v;
}
template<class T> static void alloc(T** p,size_t count) { CHECK(cudaMalloc(p,std::max(count,(size_t)1)*sizeof(T))); }
template<class T> static void upload(T** p,const std::string& path,size_t count) {
 auto data=read(path);if(data.size()!=count*sizeof(T))exit(2);alloc(p,count);
 if(count)CHECK(cudaMemcpy(*p,data.data(),data.size(),cudaMemcpyHostToDevice));
}
static Input load(const std::string& root,const std::string& prompt,int t,int layer) {
 Input x{};x.prompt=prompt;x.ordinal=layer;x.base=root+"/"+prompt+"/t"+std::to_string(t)+"-mla"+std::to_string(layer);
 std::ifstream f(x.base+".meta");if(!(f>>x.h>>x.r>>x.dr>>x.t>>x.slots>>x.rows>>x.scale))exit(2);
 if(x.t!=t||x.h!=64||x.r!=512||x.dr!=0||x.slots<2048||x.slots>2051) {fprintf(stderr,"geometry %s %d %d %d %d\n",x.base.c_str(),x.h,x.r,x.dr,x.slots);exit(2);}
 upload(&x.q,x.base+".q",t*x.h*x.r);upload(&x.qp,x.base+".qp",t*x.h*x.dr);
 upload(&x.idx,x.base+".idx",t*x.slots);upload(&x.cache,x.base+".cache",(size_t)x.rows*(x.r+x.dr));
 alloc(&x.a,t*x.h*x.r);alloc(&x.b,t*x.h*x.r);alloc(&x.m,t*x.h*8);alloc(&x.d,t*x.h*8);alloc(&x.acc,t*x.h*8*x.r);return x;
}
static void release(Input& x) {for(void* p:{(void*)x.q,(void*)x.qp,(void*)x.idx,(void*)x.cache,(void*)x.a,(void*)x.b,(void*)x.m,(void*)x.d,(void*)x.acc})CHECK(cudaFree(p));}
static void run(Input& x,bool candidate) {
 if(candidate)CHECK(memra_mla_verify_splitkv_f32(x.q,x.qp,x.cache,x.idx,x.b,x.m,x.d,x.acc,x.h,x.r,x.dr,x.t,x.slots,x.scale,nullptr));
 else if(x.t==4)CHECK(memra_mla_attn_gathered_dsa_f32(x.q,x.qp,x.cache,x.idx,x.a,x.h,x.r,x.dr,x.t,x.slots,x.scale,nullptr));
 else CHECK(memra_mla_attn_gathered_f32(x.q,x.qp,x.cache,x.idx,x.a,x.h,x.r,x.dr,x.t,x.slots,x.scale,nullptr));
}
__global__ void capture_gather(const float* cache,const int* idx,float* gathered,int count,int rank) {
 long k=(long)blockIdx.x*blockDim.x+threadIdx.x;if(k>=(long)count*rank)return;
 int row=idx[k/rank];gathered[k]=row<0?0.0f:cache[(long)row*rank+k%rank];
}
static void exact_gather(Input& x,FILE* out) {
 auto rawidx=read(x.base+".idx"),rawcache=read(x.base+".cache");
 auto ids=(const int*)rawidx.data();auto cache=(const float*)rawcache.data();
 size_t n=(size_t)x.t*x.slots*x.r;float* device;alloc(&device,n);
 capture_gather<<<(n+255)/256,256>>>(x.cache,x.idx,device,x.t*x.slots,x.r);CHECK(cudaGetLastError());
 std::vector<float> got(n);CHECK(cudaMemcpy(got.data(),device,n*4,cudaMemcpyDeviceToHost));CHECK(cudaFree(device));
 size_t bad=0;float zero=0;
 for(size_t k=0;k<n;k++){int id=ids[k/x.r];if(id>=x.rows||id< -1)exit(2);const float* want=id<0?&zero:cache+(size_t)id*x.r+k%x.r;if(memcmp(&got[k],want,4))bad++;}
 fprintf(out,"%s\t%d\t%d\t%zu\t%zu\n",x.prompt.c_str(),x.t,x.ordinal,n,bad);fflush(out);if(bad)exit(3);
}
static int oracle(Input& x,FILE* out) {
 run(x,false);run(x,true);CHECK(cudaDeviceSynchronize());
 int rows=x.t*x.h;size_t n=rows*x.r;
 std::vector<float>a(n),b(n);CHECK(cudaMemcpy(a.data(),x.a,n*4,cudaMemcpyDeviceToHost));CHECK(cudaMemcpy(b.data(),x.b,n*4,cudaMemcpyDeviceToHost));
 int failures=0;
 for(int row=0;row<rows;row++) {
  int aa=0,ab=0,bits=0;double maxa=0,err=0,mae=0;bool finite=true;
  for(int k=0;k<x.r;k++) {float v=a[row*x.r+k],w=b[row*x.r+k];finite=finite&&std::isfinite(v)&&std::isfinite(w);if(v>a[row*x.r+aa])aa=k;if(w>b[row*x.r+ab])ab=k;maxa=std::max(maxa,(double)fabs(v));err=std::max(err,(double)fabs(v-w));mae+=fabs(v-w);if(memcmp(&v,&w,4))bits++;}
  double band=1e-5+1e-4*maxa;bool pass=finite&&aa==ab&&err<=band;if(!pass)failures++;
  fprintf(out,"%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%.12g\t%.12g\t%.12g\t%d\n",x.prompt.c_str(),x.t,x.ordinal,row,aa,ab,bits,finite,err,mae/x.r,band,pass);
 }
 fflush(out);printf("ORACLE %s t=%d mla=%d failures=%d\n",x.prompt.c_str(),x.t,x.ordinal,failures);fflush(stdout);return failures;
}
static double timed(std::vector<Input>& inputs,int layer,bool arm,int reps) {
 cudaEvent_t a,b;CHECK(cudaEventCreate(&a));CHECK(cudaEventCreate(&b));CHECK(cudaEventRecord(a));
 for(int rep=0;rep<reps;rep++) {if(layer<0)for(auto&x:inputs)run(x,arm);else run(inputs[layer],arm);}
 CHECK(cudaEventRecord(b));CHECK(cudaEventSynchronize(b));float ms;CHECK(cudaEventElapsedTime(&ms,a,b));CHECK(cudaEventDestroy(a));CHECK(cudaEventDestroy(b));return ms/reps;
}
static void bench(std::vector<Input>& inputs,FILE* out,int layer) {
 // Same warmup for both arms; event positions synchronize outside each repeated chain.
 for(int i=0;i<20;i++){if(layer<0){for(auto&x:inputs){run(x,false);run(x,true);}}else{run(inputs[layer],false);run(inputs[layer],true);}}
 CHECK(cudaDeviceSynchronize());
 for(int block=0;block<5;block++)for(int pos=0;pos<4;pos++){
  bool arm=pos==1||pos==2;int reps=layer<0?20:50;double ms=timed(inputs,layer,arm,reps);
  fprintf(out,"%s\t%d\t%d\t%d\t%d\t%c\t%d\t%.12g\n",inputs[0].prompt.c_str(),inputs[0].t,layer,block,pos,arm?'B':'A',reps,ms);fflush(out);
 }
 printf("BENCH %s t=%d layer=%d complete\n",inputs[0].prompt.c_str(),inputs[0].t,layer);fflush(stdout);
}
int main(int argc,char**argv) {
 if(argc!=3){fprintf(stderr,"usage: oracle-bench captures output\n");return 2;}
 std::string root=argv[1],out=argv[2];
 // Contract refusals must happen before any pointer dereference or launch.
 for(auto shape:std::vector<std::vector<int>>{{32,512,0,4,2051},{64,512,0,1,2051},{64,512,0,8,2051},{64,1024,0,4,2051},{64,512,64,4,2051},{64,512,0,4,2044},{64,512,0,4,2052}}) {
  int rc=memra_mla_verify_splitkv_f32(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,shape[0],shape[1],shape[2],shape[3],shape[4],1,nullptr);if(rc!=40023)return 2;
 }
 FILE* f=fopen((out+"/oracle.tsv").c_str(),"w"),*g=fopen((out+"/gather.tsv").c_str(),"w");if(!f||!g)return 2;
 fprintf(f,"prompt\tt\tmla\trow\targmax_a\targmax_b\tbits_changed\tfinite\tmax_error\tmean_error\tband\tpass\n");fprintf(g,"prompt\tt\tmla\telements\tbits_changed\n");
 int failures=0;
 for(std::string p:{"p32k","p128k"})for(int t:{2,4,7})for(int layer=0;layer<11;layer++){auto x=load(root,p,t,layer);exact_gather(x,g);failures+=oracle(x,f);release(x);}
 fclose(f);fclose(g);if(failures){fprintf(stderr,"NEGATIVE oracle failures=%d; no timing admitted\n",failures);return 3;}
 std::ofstream(out+"/ORACLE_PASS")<<"All 66 layer-width-context captures pass gather, finite, argmax and fixed band.\n";
 f=fopen((out+"/bench.tsv").c_str(),"w");if(!f)return 2;fprintf(f,"prompt\tt\tlayer\tblock\tposition\tarm\treps\tms\n");
 for(std::string p:{"p32k","p128k"})for(int t:{2,4,7}) {
  std::vector<Input>xs;for(int layer=0;layer<11;layer++)xs.push_back(load(root,p,t,layer));
  bench(xs,f,-1);for(int layer=0;layer<11;layer++)bench(xs,f,layer);for(auto&x:xs)release(x);
 }
 fclose(f);return 0;
}

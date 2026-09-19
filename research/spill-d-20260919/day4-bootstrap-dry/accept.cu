
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { std::fprintf(stderr,"ERROR: %s: %s\n",#x,cudaGetErrorString(e)); return 2; } } while(0)
int main() {
  int count=0; CHECK(cudaGetDeviceCount(&count));
  if(count!=1) { std::fprintf(stderr,"ERROR: expected exactly one CUDA device, got %d\n",count); return 3; }
  CHECK(cudaSetDevice(0));
  const size_t bytes=size_t(8)*1024*1024*1024, chunk=4*1024*1024;
  unsigned char *d=nullptr; CHECK(cudaMalloc((void**)&d,bytes));
  CHECK(cudaMemset(d,0xa5,bytes)); CHECK(cudaDeviceSynchronize());
  std::vector<unsigned char> host(chunk);
  for(size_t offset=0;offset<bytes;offset+=chunk) {
    CHECK(cudaMemcpy(host.data(),d+offset,chunk,cudaMemcpyDeviceToHost));
    for(size_t i=0;i<chunk;i++) if(host[i]!=0xa5) {
      std::fprintf(stderr,"ERROR: readback mismatch at %zu\n",offset+i); return 4;
    }
  }
  CHECK(cudaFree(d));
  std::puts("ACCEPT: 8589934592 bytes cudaMalloc+memset+full-readback MATCH");
  return 0;
}

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  void* p = nullptr;
  auto err = cudaMalloc(&p, std::atoll(argv[1]) * 1048576ull);
  if (err != cudaSuccess) { printf("HOLD-FAILED %s\n", cudaGetErrorString(err)); return 1; }
  printf("HELD %s MiB\n", argv[1]); fflush(stdout);
  for (;;) { struct timespec t{1, 0}; nanosleep(&t, nullptr); }
}

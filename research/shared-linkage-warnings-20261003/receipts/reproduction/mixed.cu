#include <cuda_runtime.h>
extern "C" __global__ void kernel0(float *out) {
 extern __shared__ float shared_data[];
 shared_data[threadIdx.x]=float(threadIdx.x);
 __syncthreads();
 out[threadIdx.x]=shared_data[(threadIdx.x+1)%blockDim.x];
}
__global__ void kernel1(float *out) {
 extern __shared__ float shared_data[];
 shared_data[threadIdx.x]=float(threadIdx.x);
 __syncthreads();
 out[threadIdx.x]=shared_data[(threadIdx.x+1)%blockDim.x];
}

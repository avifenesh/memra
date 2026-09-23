// Initialization-only diagnostic. Build outside the GPU lease; run with the provided wrapper.
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <string>

__global__ void initialize_scratch(float *output) { output[0] = 1.0f; }
void mappings(const char *stage) {
    std::ifstream file("/proc/self/maps");
    std::string line;
    std::cout << "stage=" << stage << '\n';
    while (std::getline(file, line)) {
        if (line.find("libnvidia") != std::string::npos || line.find("libcuda") != std::string::npos)
            std::cout << line << '\n';
    }
}
void checked(cudaError_t result) {
    if (result != cudaSuccess) { std::cerr << cudaGetErrorString(result) << '\n'; std::exit(1); }
}
int main() {
    mappings("before");
    checked(cudaSetDevice(0));
    checked(cudaFree(nullptr));
    mappings("runtime-init");
    float *scratch = nullptr;
    checked(cudaMalloc(&scratch, sizeof(float)));
    initialize_scratch<<<1,1>>>(scratch);
    checked(cudaDeviceSynchronize());
    mappings("after-kernel");
    float result = 0;
    checked(cudaMemcpy(&result, scratch, sizeof(result), cudaMemcpyDeviceToHost));
    checked(cudaFree(scratch));
    if (result != 1.0f) return 2;
    std::cout << "PASS scratch-only runtime initializer; no model loaded\n";
}

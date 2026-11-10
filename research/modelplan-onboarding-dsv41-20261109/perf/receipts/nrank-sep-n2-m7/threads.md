wall 516.9 us/layer

thread 2917780: 196 calls/layer, in calls 231.5 us/layer (blocking 40.6), between calls 285.4 us/layer
  gaps: 0-1us n=174.0/layer 42.6us/layer; 1-3us n=12.3/layer 27.3us/layer; 3-10us n=4.0/layer 16.2us/layer; 10-30us n=3.9/layer 62.5us/layer; 30-100us n=0.7/layer 48.4us/layer; 100-1e+06us n=0.7/layer 88.3us/layer
    >=10us after cuMemFreeAsync before cuMemAllocAsync: 1.94/layer, 125.7 us/layer, 65 us each
    >=10us after cudaLaunchKernel before cuMemAllocAsync: 1.00/layer, 14.3 us/layer, 14 us each
    >=10us after cudaLaunchKernel before cuCtxSetCurrent: 0.50/layer, 11.2 us/layer, 23 us each
    >=10us after cudaLaunchKernel before cuKernelGetName: 0.50/layer, 10.8 us/layer, 22 us each
    >=10us after cuEventSynchronize before cuEventSynchronize: 0.10/layer, 10.3 us/layer, 103 us each
    >=10us after cuStreamWaitEvent before cuCtxSetCurrent: 0.55/layer, 8.6 us/layer, 16 us each
    >=10us after cuStreamWaitEvent before cuKernelGetName: 0.45/layer, 7.3 us/layer, 16 us each
    >=10us after cuEventSynchronize before cuStreamWaitEvent: 0.10/layer, 6.6 us/layer, 68 us each

thread 2917779: 195 calls/layer, in calls 235.0 us/layer (blocking 38.9), between calls 281.9 us/layer
  gaps: 0-1us n=173.7/layer 43.6us/layer; 1-3us n=13.2/layer 28.7us/layer; 3-10us n=3.5/layer 15.4us/layer; 10-30us n=3.7/layer 58.4us/layer; 30-100us n=0.7/layer 48.5us/layer; 100-1e+06us n=0.7/layer 87.2us/layer
    >=10us after cuMemFreeAsync before cuMemAllocAsync: 1.74/layer, 122.6 us/layer, 70 us each
    >=10us after cudaLaunchKernel before cuMemAllocAsync: 1.00/layer, 14.2 us/layer, 14 us each
    >=10us after cudaLaunchKernel before cuKernelGetName: 0.51/layer, 10.6 us/layer, 21 us each
    >=10us after cudaLaunchKernel before cuCtxSetCurrent: 0.50/layer, 10.6 us/layer, 21 us each
    >=10us after cuEventSynchronize before cuEventSynchronize: 0.10/layer, 10.1 us/layer, 101 us each
    >=10us after cuStreamWaitEvent before cuCtxSetCurrent: 0.55/layer, 8.0 us/layer, 14 us each
    >=10us after cuStreamWaitEvent before cuKernelGetName: 0.45/layer, 6.9 us/layer, 15 us each
    >=10us after cuEventSynchronize before cuStreamWaitEvent: 0.10/layer, 6.7 us/layer, 67 us each

thread 2917733: 49 calls/layer, in calls 131.2 us/layer (blocking 98.9), between calls 385.7 us/layer
  gaps: 0-1us n=44.1/layer 8.4us/layer; 1-3us n=0.1/layer 0.1us/layer; 3-10us n=1.0/layer 7.9us/layer; 10-30us n=0.9/layer 24.6us/layer; 30-100us n=2.1/layer 120.2us/layer; 100-1e+06us n=1.1/layer 224.1us/layer
    >=10us after cuMemFreeAsync before cuMemFreeAsync: 2.08/layer, 250.9 us/layer, 121 us each
    >=10us after cuMemcpyDtoHAsync before cuMemFreeAsync: 1.00/layer, 81.5 us/layer, 82 us each
    >=10us after cuMemFreeAsync before cuMemcpyDtoHAsync: 1.00/layer, 36.6 us/layer, 37 us each

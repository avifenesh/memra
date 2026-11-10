wall 340.6 us/layer

thread 3591043: 112 calls/layer, in calls 267.7 us/layer (blocking 138.9), between calls 72.9 us/layer
  gaps: 0-1us n=99.9/layer 29.0us/layer; 1-3us n=10.6/layer 19.9us/layer; 3-10us n=1.3/layer 5.3us/layer; 10-30us n=0.0/layer 0.6us/layer; 30-100us n=0.1/layer 6.0us/layer; 100-1e+06us n=0.0/layer 8.4us/layer
    >=10us after cudaLaunchKernel before cuKernelGetName: 0.02/layer, 6.7 us/layer, 440 us each
    >=10us after cuEventSynchronize before cuMemcpyHtoDAsync: 0.10/layer, 5.1 us/layer, 51 us each
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.02/layer, 2.7 us/layer, 112 us each
    >=10us after cuMemcpyDtoHAsync before cuMemcpyDtoHAsync: 0.01/layer, 0.2 us/layer, 14 us each
    >=10us after cuStreamSynchronize before cuKernelGetName: 0.01/layer, 0.1 us/layer, 14 us each
    >=10us after cuStreamWaitEvent before cuKernelGetName: 0.00/layer, 0.1 us/layer, 21 us each
    >=10us after cuMemFreeAsync before cuKernelGetName: 0.00/layer, 0.0 us/layer, 10 us each
    >=10us after cudaLaunchKernel before cuEventRecord: 0.00/layer, 0.0 us/layer, 16 us each

thread 3591111: 112 calls/layer, in calls 272.3 us/layer (blocking 138.8), between calls 68.3 us/layer
  gaps: 0-1us n=99.0/layer 29.4us/layer; 1-3us n=11.7/layer 20.9us/layer; 3-10us n=0.9/layer 3.7us/layer; 10-30us n=0.0/layer 0.8us/layer; 30-100us n=0.1/layer 6.5us/layer; 100-1e+06us n=0.0/layer 3.3us/layer
    >=10us after cuEventSynchronize before cuMemcpyHtoDAsync: 0.10/layer, 5.1 us/layer, 51 us each
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.02/layer, 2.5 us/layer, 103 us each
    >=10us after cudaLaunchKernel before cuKernelGetName: 0.01/layer, 2.2 us/layer, 156 us each
    >=10us after cuMemcpyDtoHAsync before cuMemcpyDtoHAsync: 0.02/layer, 0.6 us/layer, 25 us each
    >=10us after cuStreamWaitEvent before cuKernelGetName: 0.00/layer, 0.1 us/layer, 19 us each
    >=10us after cuMemFreeAsync before cuKernelGetName: 0.00/layer, 0.1 us/layer, 14 us each
    >=10us after cudaLaunchKernel before cuEventRecord: 0.00/layer, 0.0 us/layer, 12 us each
    >=10us after cuStreamSynchronize before cuKernelGetName: 0.00/layer, 0.0 us/layer, 18 us each

thread 3591110: 112 calls/layer, in calls 271.7 us/layer (blocking 138.8), between calls 68.9 us/layer
  gaps: 0-1us n=99.2/layer 29.3us/layer; 1-3us n=11.8/layer 20.1us/layer; 3-10us n=0.7/layer 3.0us/layer; 10-30us n=0.0/layer 0.7us/layer; 30-100us n=0.1/layer 6.9us/layer; 100-1e+06us n=0.0/layer 5.1us/layer
    >=10us after cuEventSynchronize before cuMemcpyHtoDAsync: 0.10/layer, 5.1 us/layer, 51 us each
    >=10us after cudaLaunchKernel before cuKernelGetName: 0.02/layer, 4.4 us/layer, 270 us each
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.02/layer, 2.3 us/layer, 98 us each
    >=10us after cuMemcpyDtoHAsync before cuMemcpyDtoHAsync: 0.02/layer, 0.6 us/layer, 27 us each
    >=10us after cuStreamSynchronize before cuKernelGetName: 0.01/layer, 0.1 us/layer, 16 us each
    >=10us after cuStreamWaitEvent before cuKernelGetName: 0.00/layer, 0.1 us/layer, 21 us each
    >=10us after cuMemFreeAsync before cuKernelGetName: 0.00/layer, 0.1 us/layer, 13 us each
    >=10us after cudaLaunchKernel before cuEventRecord: 0.00/layer, 0.0 us/layer, 15 us each

thread 3591109: 112 calls/layer, in calls 266.1 us/layer (blocking 139.0), between calls 74.5 us/layer
  gaps: 0-1us n=99.2/layer 28.7us/layer; 1-3us n=11.0/layer 20.7us/layer; 3-10us n=1.6/layer 6.4us/layer; 10-30us n=0.0/layer 0.7us/layer; 30-100us n=0.1/layer 6.9us/layer; 100-1e+06us n=0.0/layer 7.3us/layer
    >=10us after cudaLaunchKernel before cuKernelGetName: 0.02/layer, 6.6 us/layer, 359 us each
    >=10us after cuEventSynchronize before cuMemcpyHtoDAsync: 0.10/layer, 5.1 us/layer, 51 us each
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.02/layer, 2.3 us/layer, 97 us each
    >=10us after cuMemcpyDtoHAsync before cuMemcpyDtoHAsync: 0.02/layer, 0.6 us/layer, 27 us each
    >=10us after cuStreamWaitEvent before cuKernelGetName: 0.00/layer, 0.1 us/layer, 22 us each
    >=10us after cuMemFreeAsync before cuKernelGetName: 0.01/layer, 0.1 us/layer, 12 us each
    >=10us after cuStreamSynchronize before cuKernelGetName: 0.01/layer, 0.1 us/layer, 14 us each
    >=10us after cudaLaunchKernel before cuEventRecord: 0.00/layer, 0.0 us/layer, 12 us each

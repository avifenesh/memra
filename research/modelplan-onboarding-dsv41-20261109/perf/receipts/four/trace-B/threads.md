wall 403.9 us/layer

thread 3491697: 148 calls/layer, in calls 275.8 us/layer (blocking 62.8), between calls 128.1 us/layer
  gaps: 0-1us n=127.2/layer 39.0us/layer; 1-3us n=10.4/layer 18.9us/layer; 3-10us n=10.0/layer 51.6us/layer; 10-30us n=0.1/layer 0.8us/layer; 30-100us n=0.1/layer 6.1us/layer; 100-1e+06us n=0.0/layer 10.4us/layer
    >=10us after cuEventRecord before cuStreamWaitEvent: 0.01/layer, 7.1 us/layer, 504 us each
    >=10us after cuEventSynchronize before cuMemcpyHtoDAsync: 0.10/layer, 5.8 us/layer, 59 us each
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.02/layer, 3.4 us/layer, 143 us each
    >=10us after cuMemcpyDtoHAsync before cuMemcpyDtoHAsync: 0.02/layer, 0.3 us/layer, 16 us each
    >=10us after cudaLaunchKernel before cuEventRecord: 0.01/layer, 0.3 us/layer, 26 us each
    >=10us after cuMemcpyDtoDAsync before cuKernelGetName: 0.01/layer, 0.2 us/layer, 22 us each
    >=10us after cudaLaunchKernel before cuKernelGetName: 0.01/layer, 0.1 us/layer, 17 us each
    >=10us after cuStreamSynchronize before cuKernelGetName: 0.01/layer, 0.1 us/layer, 15 us each

thread 3491765: 148 calls/layer, in calls 291.4 us/layer (blocking 62.4), between calls 112.6 us/layer
  gaps: 0-1us n=129.6/layer 41.4us/layer; 1-3us n=11.5/layer 21.6us/layer; 3-10us n=6.4/layer 30.5us/layer; 10-30us n=0.0/layer 0.7us/layer; 30-100us n=0.1/layer 6.7us/layer; 100-1e+06us n=0.0/layer 10.3us/layer
    >=10us after cuEventRecord before cuStreamWaitEvent: 0.01/layer, 7.2 us/layer, 1109 us each
    >=10us after cuEventSynchronize before cuMemcpyHtoDAsync: 0.10/layer, 5.9 us/layer, 59 us each
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.02/layer, 3.1 us/layer, 131 us each
    >=10us after cuMemcpyDtoHAsync before cuMemcpyDtoHAsync: 0.02/layer, 0.9 us/layer, 37 us each
    >=10us after cuMemcpyDtoDAsync before cuKernelGetName: 0.01/layer, 0.2 us/layer, 18 us each
    >=10us after cudaLaunchKernel before cuMemcpyDtoHAsync: 0.01/layer, 0.1 us/layer, 12 us each
    >=10us after cudaLaunchKernel before cuEventRecord: 0.00/layer, 0.1 us/layer, 15 us each
    >=10us after cuStreamSynchronize before cuKernelGetName: 0.00/layer, 0.1 us/layer, 14 us each

thread 3491764: 148 calls/layer, in calls 308.8 us/layer (blocking 62.6), between calls 95.2 us/layer
  gaps: 0-1us n=131.0/layer 42.5us/layer; 1-3us n=13.0/layer 23.6us/layer; 3-10us n=3.5/layer 14.4us/layer; 10-30us n=0.1/layer 1.1us/layer; 30-100us n=0.1/layer 6.7us/layer; 100-1e+06us n=0.0/layer 5.4us/layer
    >=10us after cuEventSynchronize before cuMemcpyHtoDAsync: 0.10/layer, 5.8 us/layer, 59 us each
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.02/layer, 3.0 us/layer, 127 us each
    >=10us after cuEventRecord before cuStreamWaitEvent: 0.01/layer, 2.5 us/layer, 178 us each
    >=10us after cuMemcpyDtoHAsync before cuMemcpyDtoHAsync: 0.02/layer, 0.8 us/layer, 33 us each
    >=10us after cudaLaunchKernel before cuEventRecord: 0.02/layer, 0.4 us/layer, 24 us each
    >=10us after cuMemcpyDtoDAsync before cuKernelGetName: 0.01/layer, 0.2 us/layer, 25 us each
    >=10us after cudaLaunchKernel before cuKernelGetName: 0.01/layer, 0.1 us/layer, 16 us each
    >=10us after cudaLaunchKernel before cuMemcpyDtoHAsync: 0.01/layer, 0.1 us/layer, 13 us each

thread 3491763: 148 calls/layer, in calls 299.0 us/layer (blocking 62.6), between calls 104.9 us/layer
  gaps: 0-1us n=129.9/layer 41.6us/layer; 1-3us n=11.4/layer 21.7us/layer; 3-10us n=6.3/layer 25.1us/layer; 10-30us n=0.1/layer 0.8us/layer; 30-100us n=0.1/layer 6.5us/layer; 100-1e+06us n=0.0/layer 7.8us/layer
    >=10us after cuEventSynchronize before cuMemcpyHtoDAsync: 0.10/layer, 5.9 us/layer, 59 us each
    >=10us after cuEventRecord before cuStreamWaitEvent: 0.00/layer, 4.8 us/layer, 1105 us each
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.02/layer, 3.0 us/layer, 126 us each
    >=10us after cuMemcpyDtoHAsync before cuMemcpyDtoHAsync: 0.02/layer, 0.8 us/layer, 32 us each
    >=10us after cuMemcpyDtoDAsync before cuKernelGetName: 0.01/layer, 0.2 us/layer, 16 us each
    >=10us after cudaLaunchKernel before cuEventRecord: 0.01/layer, 0.2 us/layer, 21 us each
    >=10us after cudaLaunchKernel before cuMemcpyDtoHAsync: 0.01/layer, 0.1 us/layer, 13 us each
    >=10us after cudaLaunchKernel before cuKernelGetName: 0.00/layer, 0.1 us/layer, 27 us each

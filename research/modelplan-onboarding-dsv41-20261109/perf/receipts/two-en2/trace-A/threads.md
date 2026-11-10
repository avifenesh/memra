wall 365.4 us/layer

thread 3738842: 120 calls/layer, in calls 304.6 us/layer (blocking 173.6), between calls 60.8 us/layer
  gaps: 0-1us n=113.5/layer 33.8us/layer; 1-3us n=6.1/layer 7.7us/layer; 3-10us n=0.1/layer 0.4us/layer; 10-30us n=0.0/layer 0.5us/layer; 30-100us n=0.2/layer 14.4us/layer; 100-1e+06us n=0.0/layer 2.3us/layer
    >=10us after cuEventSynchronize before cuMemcpyHtoDAsync: 0.20/layer, 11.7 us/layer, 59 us each
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.05/layer, 5.0 us/layer, 104 us each
    >=10us after cuMemcpyDtoHAsync before cuMemcpyDtoHAsync: 0.03/layer, 0.3 us/layer, 12 us each
    >=10us after cuMemcpyDtoDAsync before cuKernelGetName: 0.00/layer, 0.1 us/layer, 21 us each
    >=10us after cudaLaunchKernel before cuKernelGetName: 0.00/layer, 0.1 us/layer, 27 us each
    >=10us after cuStreamSynchronize before cuEventRecord: 0.00/layer, 0.0 us/layer, 11 us each

thread 3738902: 120 calls/layer, in calls 301.4 us/layer (blocking 173.5), between calls 64.0 us/layer
  gaps: 0-1us n=110.4/layer 31.6us/layer; 1-3us n=9.0/layer 12.5us/layer; 3-10us n=0.2/layer 0.6us/layer; 10-30us n=0.0/layer 0.8us/layer; 30-100us n=0.2/layer 16.2us/layer; 100-1e+06us n=0.0/layer 0.7us/layer
    >=10us after cuEventSynchronize before cuMemcpyHtoDAsync: 0.20/layer, 11.7 us/layer, 58 us each
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.05/layer, 4.7 us/layer, 97 us each
    >=10us after cuMemcpyDtoHAsync before cuMemcpyDtoHAsync: 0.05/layer, 1.0 us/layer, 22 us each
    >=10us after cuMemcpyDtoDAsync before cuKernelGetName: 0.00/layer, 0.1 us/layer, 27 us each
    >=10us after cuStreamWaitEvent before cuKernelGetName: 0.00/layer, 0.1 us/layer, 29 us each
    >=10us after cudaLaunchKernel before cuKernelGetName: 0.00/layer, 0.1 us/layer, 25 us each

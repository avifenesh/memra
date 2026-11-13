wall 366.5 us/layer

thread 3767524: 120 calls/layer, in calls 313.5 us/layer (blocking 185.0), between calls 53.0 us/layer
  gaps: 0-1us n=112.7/layer 33.6us/layer; 1-3us n=6.8/layer 8.6us/layer; 3-10us n=0.1/layer 0.4us/layer; 10-30us n=0.2/layer 3.1us/layer; 30-100us n=0.0/layer 0.9us/layer; 100-1e+06us n=0.0/layer 4.7us/layer
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.05/layer, 5.5 us/layer, 116 us each
    >=10us after cuEventSynchronize before cuMemcpyHtoDAsync: 0.20/layer, 2.7 us/layer, 13 us each
    >=10us after cuMemcpyDtoHAsync before cuMemcpyDtoHAsync: 0.03/layer, 0.4 us/layer, 12 us each
    >=10us after cuMemcpyDtoDAsync before cuKernelGetName: 0.00/layer, 0.1 us/layer, 15 us each

thread 3767546: 120 calls/layer, in calls 311.9 us/layer (blocking 184.9), between calls 54.6 us/layer
  gaps: 0-1us n=111.2/layer 32.6us/layer; 1-3us n=8.2/layer 10.8us/layer; 3-10us n=0.2/layer 0.6us/layer; 10-30us n=0.2/layer 3.2us/layer; 30-100us n=0.0/layer 1.4us/layer; 100-1e+06us n=0.0/layer 4.3us/layer
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.05/layer, 5.2 us/layer, 108 us each
    >=10us after cuEventSynchronize before cuMemcpyHtoDAsync: 0.20/layer, 2.7 us/layer, 13 us each
    >=10us after cuMemcpyDtoHAsync before cuMemcpyDtoHAsync: 0.04/layer, 1.0 us/layer, 24 us each
    >=10us after cudaLaunchKernel before cuKernelGetName: 0.00/layer, 0.0 us/layer, 17 us each

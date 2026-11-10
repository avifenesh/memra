wall 383.1 us/layer

thread 3767650: 114 calls/layer, in calls 319.6 us/layer (blocking 181.0), between calls 63.4 us/layer
  gaps: 0-1us n=105.9/layer 33.7us/layer; 1-3us n=7.5/layer 10.3us/layer; 3-10us n=0.2/layer 0.9us/layer; 10-30us n=0.1/layer 1.0us/layer; 30-100us n=0.2/layer 13.4us/layer; 100-1e+06us n=0.0/layer 2.3us/layer
    >=10us after cuEventSynchronize before cuMemcpyHtoDAsync: 0.20/layer, 10.5 us/layer, 52 us each
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.05/layer, 4.9 us/layer, 102 us each
    >=10us after cudaLaunchKernel before cuKernelGetName: 0.02/layer, 0.5 us/layer, 26 us each
    >=10us after cuMemcpyDtoHAsync before cuMemcpyDtoHAsync: 0.02/layer, 0.3 us/layer, 12 us each
    >=10us after cuMemcpyDtoDAsync before cuKernelGetName: 0.01/layer, 0.2 us/layer, 27 us each
    >=10us after cuStreamWaitEvent before cuKernelGetName: 0.01/layer, 0.2 us/layer, 19 us each
    >=10us after cuStreamSynchronize before cuKernelGetName: 0.00/layer, 0.1 us/layer, 20 us each
    >=10us after cudaLaunchKernel before cuEventRecord: 0.00/layer, 0.0 us/layer, 10 us each

thread 3767714: 114 calls/layer, in calls 317.9 us/layer (blocking 180.6), between calls 65.1 us/layer
  gaps: 0-1us n=103.2/layer 31.6us/layer; 1-3us n=10.1/layer 14.6us/layer; 3-10us n=0.2/layer 0.7us/layer; 10-30us n=0.1/layer 0.8us/layer; 30-100us n=0.2/layer 14.9us/layer; 100-1e+06us n=0.0/layer 0.7us/layer
    >=10us after cuEventSynchronize before cuMemcpyHtoDAsync: 0.20/layer, 10.5 us/layer, 52 us each
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.05/layer, 4.6 us/layer, 96 us each
    >=10us after cuMemcpyDtoHAsync before cuMemcpyDtoHAsync: 0.05/layer, 1.0 us/layer, 23 us each
    >=10us after cuStreamSynchronize before cuKernelGetName: 0.01/layer, 0.1 us/layer, 13 us each
    >=10us after cuStreamWaitEvent before cuKernelGetName: 0.00/layer, 0.1 us/layer, 26 us each
    >=10us after cudaLaunchKernel before cuEventSynchronize: 0.00/layer, 0.0 us/layer, 21 us each
    >=10us after cudaLaunchKernel before cuKernelGetName: 0.00/layer, 0.0 us/layer, 17 us each

wall 944.2 us/layer

thread 2844630: 139 calls/layer, in calls 728.5 us/layer (blocking 461.2), between calls 215.7 us/layer
  gaps: 0-1us n=113.0/layer 41.7us/layer; 1-3us n=14.1/layer 23.4us/layer; 3-10us n=11.1/layer 55.1us/layer; 10-30us n=0.4/layer 6.6us/layer; 30-100us n=0.2/layer 7.0us/layer; 100-1e+06us n=0.2/layer 80.3us/layer
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.12/layer, 68.7 us/layer, 574 us each
    >=10us after cuEventSynchronize before cuEventSynchronize: 0.07/layer, 8.2 us/layer, 116 us each
    >=10us after cudaLaunchKernel before cuEventRecord: 0.22/layer, 6.2 us/layer, 28 us each
    >=10us after cuEventSynchronize before cuStreamWaitEvent: 0.07/layer, 6.1 us/layer, 86 us each
    >=10us after cuEventRecord before cuStreamWaitEvent: 0.15/layer, 1.8 us/layer, 12 us each
    >=10us after cudaLaunchKernel before cuKernelGetName: 0.07/layer, 1.8 us/layer, 27 us each
    >=10us after cuMemFreeAsync before cuKernelGetName: 0.06/layer, 0.9 us/layer, 16 us each
    >=10us after cuMemcpyDtoDAsync before cuKernelGetName: 0.02/layer, 0.2 us/layer, 14 us each

thread 2844734: 138 calls/layer, in calls 454.8 us/layer (blocking 180.7), between calls 489.3 us/layer
  gaps: 0-1us n=113.8/layer 42.1us/layer; 1-3us n=14.4/layer 23.4us/layer; 3-10us n=8.6/layer 41.0us/layer; 10-30us n=0.4/layer 6.7us/layer; 30-100us n=0.1/layer 4.6us/layer; 100-1e+06us n=1.2/layer 369.9us/layer
    >=10us after cudaLaunchKernel before cuKernelGetName: 1.06/layer, 311.6 us/layer, 294 us each
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.12/layer, 48.0 us/layer, 401 us each
    >=10us after cuEventSynchronize before cuEventSynchronize: 0.07/layer, 8.3 us/layer, 117 us each
    >=10us after cuEventSynchronize before cuStreamWaitEvent: 0.07/layer, 6.0 us/layer, 85 us each
    >=10us after cudaLaunchKernel before cuEventRecord: 0.11/layer, 3.0 us/layer, 28 us each
    >=10us after cudaLaunchKernel before cuEventSynchronize: 0.11/layer, 1.5 us/layer, 14 us each
    >=10us after cuMemFreeAsync before cuKernelGetName: 0.07/layer, 1.1 us/layer, 15 us each
    >=10us after cuEventRecord before cuStreamWaitEvent: 0.09/layer, 1.0 us/layer, 11 us each

thread 2844735: 138 calls/layer, in calls 471.1 us/layer (blocking 195.5), between calls 473.0 us/layer
  gaps: 0-1us n=113.9/layer 43.3us/layer; 1-3us n=14.8/layer 24.9us/layer; 3-10us n=7.5/layer 35.8us/layer; 10-30us n=0.4/layer 6.2us/layer; 30-100us n=0.1/layer 5.4us/layer; 100-1e+06us n=1.2/layer 355.9us/layer
    >=10us after cudaLaunchKernel before cuKernelGetName: 1.09/layer, 312.1 us/layer, 287 us each
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.12/layer, 33.8 us/layer, 283 us each
    >=10us after cuEventSynchronize before cuEventSynchronize: 0.07/layer, 8.2 us/layer, 117 us each
    >=10us after cuEventSynchronize before cuStreamWaitEvent: 0.07/layer, 6.0 us/layer, 85 us each
    >=10us after cudaLaunchKernel before cuEventRecord: 0.15/layer, 4.4 us/layer, 29 us each
    >=10us after cudaLaunchKernel before cuEventSynchronize: 0.12/layer, 1.6 us/layer, 14 us each
    >=10us after cuMemFreeAsync before cuKernelGetName: 0.04/layer, 0.8 us/layer, 17 us each
    >=10us after cuMemcpyDtoDAsync before cuKernelGetName: 0.02/layer, 0.2 us/layer, 12 us each

thread 2844736: 137 calls/layer, in calls 471.4 us/layer (blocking 194.5), between calls 472.8 us/layer
  gaps: 0-1us n=113.1/layer 43.8us/layer; 1-3us n=15.5/layer 25.9us/layer; 3-10us n=6.8/layer 32.0us/layer; 10-30us n=0.4/layer 8.3us/layer; 30-100us n=0.1/layer 4.8us/layer; 100-1e+06us n=1.2/layer 356.5us/layer
    >=10us after cudaLaunchKernel before cuKernelGetName: 1.07/layer, 311.7 us/layer, 293 us each
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.12/layer, 34.7 us/layer, 290 us each
    >=10us after cuEventSynchronize before cuEventSynchronize: 0.07/layer, 8.2 us/layer, 116 us each
    >=10us after cudaLaunchKernel before cuEventRecord: 0.24/layer, 6.7 us/layer, 28 us each
    >=10us after cuEventSynchronize before cuStreamWaitEvent: 0.07/layer, 6.0 us/layer, 85 us each
    >=10us after cudaLaunchKernel before cuEventSynchronize: 0.12/layer, 1.6 us/layer, 13 us each
    >=10us after cuMemFreeAsync before cuKernelGetName: 0.04/layer, 0.6 us/layer, 13 us each
    >=10us after cuStreamSynchronize before cuKernelGetName: 0.01/layer, 0.1 us/layer, 11 us each

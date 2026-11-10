wall 490.7 us/layer

thread 2916975: 124 calls/layer, in calls 410.9 us/layer (blocking 258.3), between calls 79.8 us/layer
  gaps: 0-1us n=114.1/layer 35.1us/layer; 1-3us n=9.2/layer 13.9us/layer; 3-10us n=0.5/layer 2.0us/layer; 10-30us n=0.0/layer 0.7us/layer; 30-100us n=0.2/layer 13.2us/layer; 100-1e+06us n=0.1/layer 12.4us/layer
    >=10us after cuEventSynchronize before cuEventSynchronize: 0.10/layer, 9.9 us/layer, 99 us each
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.05/layer, 7.9 us/layer, 166 us each
    >=10us after cuEventSynchronize before cuStreamWaitEvent: 0.10/layer, 6.4 us/layer, 64 us each
    >=10us after cuMemcpyDtoHAsync before cuMemcpyDtoHAsync: 0.05/layer, 1.5 us/layer, 31 us each
    >=10us after cuStreamSynchronize before cuKernelGetName: 0.02/layer, 0.5 us/layer, 32 us each
    >=10us after cudaLaunchKernel before cuEventRecord: 0.00/layer, 0.1 us/layer, 35 us each

thread 2917045: 124 calls/layer, in calls 407.7 us/layer (blocking 257.6), between calls 83.1 us/layer
  gaps: 0-1us n=113.3/layer 35.4us/layer; 1-3us n=9.8/layer 15.3us/layer; 3-10us n=0.6/layer 2.6us/layer; 10-30us n=0.1/layer 1.0us/layer; 30-100us n=0.2/layer 14.2us/layer; 100-1e+06us n=0.1/layer 12.0us/layer
    >=10us after cuEventSynchronize before cuEventSynchronize: 0.10/layer, 10.1 us/layer, 101 us each
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.05/layer, 7.7 us/layer, 161 us each
    >=10us after cuEventSynchronize before cuStreamWaitEvent: 0.10/layer, 6.5 us/layer, 66 us each
    >=10us after cuMemcpyDtoHAsync before cuMemcpyDtoHAsync: 0.05/layer, 2.3 us/layer, 47 us each
    >=10us after cuStreamSynchronize before cuKernelGetName: 0.03/layer, 0.6 us/layer, 17 us each
    >=10us after cudaLaunchKernel before cuKernelGetName: 0.00/layer, 0.0 us/layer, 10 us each

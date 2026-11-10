wall 446.9 us/layer

thread 3491117: 139 calls/layer, in calls 328.2 us/layer (blocking 138.0), between calls 118.7 us/layer
  gaps: 0-1us n=120.1/layer 36.3us/layer; 1-3us n=10.4/layer 19.5us/layer; 3-10us n=8.1/layer 40.1us/layer; 10-30us n=0.1/layer 1.9us/layer; 30-100us n=0.1/layer 5.9us/layer; 100-1e+06us n=0.0/layer 11.2us/layer
    >=10us after cudaLaunchKernel before cuEventRecord: 0.03/layer, 8.0 us/layer, 273 us each
    >=10us after cuEventSynchronize before cuMemcpyHtoDAsync: 0.10/layer, 5.5 us/layer, 55 us each
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.02/layer, 3.7 us/layer, 155 us each
    >=10us after cuEventRecord before cuStreamWaitEvent: 0.04/layer, 0.6 us/layer, 17 us each
    >=10us after cuMemcpyDtoDAsync before cuKernelGetName: 0.02/layer, 0.4 us/layer, 21 us each
    >=10us after cudaLaunchKernel before cuKernelGetName: 0.02/layer, 0.4 us/layer, 16 us each
    >=10us after cuStreamWaitEvent before cuKernelGetName: 0.01/layer, 0.2 us/layer, 16 us each
    >=10us after cuStreamSynchronize before cuKernelGetName: 0.01/layer, 0.2 us/layer, 19 us each

thread 3491143: 139 calls/layer, in calls 350.7 us/layer (blocking 138.2), between calls 96.2 us/layer
  gaps: 0-1us n=123.4/layer 39.0us/layer; 1-3us n=11.2/layer 21.8us/layer; 3-10us n=3.8/layer 15.2us/layer; 10-30us n=0.1/layer 2.1us/layer; 30-100us n=0.1/layer 6.6us/layer; 100-1e+06us n=0.0/layer 7.9us/layer
    >=10us after cuEventSynchronize before cuMemcpyHtoDAsync: 0.10/layer, 5.5 us/layer, 55 us each
    >=10us after cudaLaunchKernel before cuEventRecord: 0.06/layer, 3.4 us/layer, 61 us each
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.02/layer, 3.2 us/layer, 133 us each
    >=10us after cuEventRecord before cuStreamWaitEvent: 0.02/layer, 2.5 us/layer, 114 us each
    >=10us after cuMemcpyDtoHAsync before cuMemcpyDtoHAsync: 0.02/layer, 0.8 us/layer, 33 us each
    >=10us after cuMemcpyDtoDAsync before cuKernelGetName: 0.04/layer, 0.8 us/layer, 18 us each
    >=10us after cudaLaunchKernel before cuKernelGetName: 0.02/layer, 0.3 us/layer, 15 us each
    >=10us after cuStreamWaitEvent before cuKernelGetName: 0.00/layer, 0.1 us/layer, 25 us each

thread 3491144: 139 calls/layer, in calls 347.4 us/layer (blocking 137.9), between calls 99.5 us/layer
  gaps: 0-1us n=123.1/layer 38.9us/layer; 1-3us n=11.7/layer 22.0us/layer; 3-10us n=3.6/layer 15.5us/layer; 10-30us n=0.1/layer 1.9us/layer; 30-100us n=0.1/layer 6.7us/layer; 100-1e+06us n=0.0/layer 10.8us/layer
    >=10us after cudaLaunchKernel before cuEventRecord: 0.05/layer, 6.0 us/layer, 122 us each
    >=10us after cuEventSynchronize before cuMemcpyHtoDAsync: 0.10/layer, 5.5 us/layer, 55 us each
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.02/layer, 3.2 us/layer, 134 us each
    >=10us after cuEventRecord before cuStreamWaitEvent: 0.02/layer, 2.4 us/layer, 150 us each
    >=10us after cuMemcpyDtoHAsync before cuMemcpyDtoHAsync: 0.02/layer, 0.8 us/layer, 35 us each
    >=10us after cuMemcpyDtoDAsync before cuKernelGetName: 0.04/layer, 0.7 us/layer, 18 us each
    >=10us after cudaLaunchKernel before cuKernelGetName: 0.02/layer, 0.5 us/layer, 25 us each
    >=10us after cuStreamSynchronize before cuKernelGetName: 0.01/layer, 0.2 us/layer, 18 us each

thread 3491145: 139 calls/layer, in calls 345.4 us/layer (blocking 137.8), between calls 101.5 us/layer
  gaps: 0-1us n=122.3/layer 38.3us/layer; 1-3us n=11.2/layer 21.4us/layer; 3-10us n=4.9/layer 23.3us/layer; 10-30us n=0.2/layer 2.6us/layer; 30-100us n=0.1/layer 6.3us/layer; 100-1e+06us n=0.0/layer 5.9us/layer
    >=10us after cuEventSynchronize before cuMemcpyHtoDAsync: 0.10/layer, 5.5 us/layer, 55 us each
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.02/layer, 3.5 us/layer, 147 us each
    >=10us after cuEventRecord before cuStreamWaitEvent: 0.03/layer, 2.6 us/layer, 96 us each
    >=10us after cudaLaunchKernel before cuEventRecord: 0.06/layer, 1.1 us/layer, 18 us each
    >=10us after cuMemcpyDtoDAsync before cuKernelGetName: 0.04/layer, 0.8 us/layer, 18 us each
    >=10us after cuMemcpyDtoHAsync before cuMemcpyDtoHAsync: 0.02/layer, 0.7 us/layer, 30 us each
    >=10us after cudaLaunchKernel before cuKernelGetName: 0.03/layer, 0.4 us/layer, 14 us each
    >=10us after cuStreamSynchronize before cuKernelGetName: 0.01/layer, 0.2 us/layer, 16 us each

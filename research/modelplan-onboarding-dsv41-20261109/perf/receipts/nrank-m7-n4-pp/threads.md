wall 965.2 us/layer

thread 1067001: 202 calls/layer, in calls 473.1 us/layer (blocking 169.6), between calls 492.1 us/layer
  gaps: 0-1us n=179.5/layer 46.5us/layer; 1-3us n=5.6/layer 10.7us/layer; 3-10us n=8.8/layer 45.3us/layer; 10-30us n=5.7/layer 107.1us/layer; 30-100us n=1.8/layer 79.7us/layer; 100-1e+06us n=1.0/layer 202.5us/layer
    >=10us after cuMemFreeAsync before cuMemAllocAsync: 2.94/layer, 245.0 us/layer, 83 us each
    >=10us after cudaLaunchKernel before cuKernelGetName: 0.99/layer, 35.6 us/layer, 36 us each
    >=10us after cuMemFreeAsync before cuEventRecord: 1.06/layer, 29.3 us/layer, 28 us each
    >=10us after cuStreamWaitEvent before cuKernelGetName: 1.00/layer, 24.7 us/layer, 25 us each
    >=10us after cudaLaunchKernel before cuMemAllocAsync: 0.99/layer, 22.0 us/layer, 22 us each
    >=10us after cuMemcpyDtoDAsync before cuMemAllocAsync: 0.42/layer, 8.0 us/layer, 19 us each
    >=10us after cuEventRecord before cuStreamWaitEvent: 0.60/layer, 7.2 us/layer, 12 us each
    >=10us after cuEventSynchronize before cuEventSynchronize: 0.07/layer, 6.4 us/layer, 90 us each

thread 1067000: 202 calls/layer, in calls 496.8 us/layer (blocking 180.7), between calls 468.4 us/layer
  gaps: 0-1us n=178.8/layer 50.8us/layer; 1-3us n=7.1/layer 14.4us/layer; 3-10us n=8.9/layer 44.6us/layer; 10-30us n=4.8/layer 91.8us/layer; 30-100us n=1.7/layer 74.3us/layer; 100-1e+06us n=0.9/layer 192.1us/layer
    >=10us after cuMemFreeAsync before cuMemAllocAsync: 2.68/layer, 229.7 us/layer, 86 us each
    >=10us after cudaLaunchKernel before cuKernelGetName: 0.99/layer, 35.2 us/layer, 35 us each
    >=10us after cuStreamWaitEvent before cuKernelGetName: 1.00/layer, 24.4 us/layer, 24 us each
    >=10us after cudaLaunchKernel before cuMemAllocAsync: 0.99/layer, 21.6 us/layer, 22 us each
    >=10us after cuMemFreeAsync before cuEventRecord: 0.68/layer, 16.5 us/layer, 24 us each
    >=10us after cuEventSynchronize before cuEventSynchronize: 0.07/layer, 6.3 us/layer, 89 us each
    >=10us after cuMemFreeAsync before cuStreamWaitEvent: 0.06/layer, 6.3 us/layer, 105 us each
    >=10us after cuMemcpyDtoDAsync before cuMemAllocAsync: 0.26/layer, 5.2 us/layer, 20 us each

thread 1067002: 201 calls/layer, in calls 476.7 us/layer (blocking 173.7), between calls 488.5 us/layer
  gaps: 0-1us n=178.0/layer 47.8us/layer; 1-3us n=6.1/layer 11.8us/layer; 3-10us n=9.0/layer 46.2us/layer; 10-30us n=5.2/layer 98.7us/layer; 30-100us n=1.8/layer 81.1us/layer; 100-1e+06us n=1.0/layer 202.6us/layer
    >=10us after cuMemFreeAsync before cuMemAllocAsync: 2.84/layer, 240.0 us/layer, 85 us each
    >=10us after cudaLaunchKernel before cuKernelGetName: 0.99/layer, 36.0 us/layer, 36 us each
    >=10us after cuStreamWaitEvent before cuKernelGetName: 0.99/layer, 25.1 us/layer, 25 us each
    >=10us after cudaLaunchKernel before cuMemAllocAsync: 1.00/layer, 22.0 us/layer, 22 us each
    >=10us after cuMemFreeAsync before cuEventRecord: 0.78/layer, 20.1 us/layer, 26 us each
    >=10us after cuMemFreeAsync before cuStreamWaitEvent: 0.14/layer, 8.8 us/layer, 65 us each
    >=10us after cuEventRecord before cuStreamWaitEvent: 0.50/layer, 6.4 us/layer, 13 us each
    >=10us after cuMemcpyDtoDAsync before cuMemAllocAsync: 0.34/layer, 6.3 us/layer, 19 us each

thread 1066999: 201 calls/layer, in calls 492.0 us/layer (blocking 170.6), between calls 473.1 us/layer
  gaps: 0-1us n=177.2/layer 51.7us/layer; 1-3us n=7.0/layer 14.0us/layer; 3-10us n=9.4/layer 45.5us/layer; 10-30us n=4.5/layer 87.3us/layer; 30-100us n=1.8/layer 77.7us/layer; 100-1e+06us n=0.9/layer 196.8us/layer
    >=10us after cuMemFreeAsync before cuMemAllocAsync: 2.64/layer, 237.8 us/layer, 90 us each
    >=10us after cudaLaunchKernel before cuKernelGetName: 0.99/layer, 34.7 us/layer, 35 us each
    >=10us after cuStreamWaitEvent before cuKernelGetName: 1.00/layer, 24.2 us/layer, 24 us each
    >=10us after cudaLaunchKernel before cuMemAllocAsync: 0.99/layer, 21.8 us/layer, 22 us each
    >=10us after cuMemFreeAsync before cuEventRecord: 0.72/layer, 19.0 us/layer, 26 us each
    >=10us after cuEventSynchronize before cuEventSynchronize: 0.07/layer, 6.4 us/layer, 90 us each
    >=10us after cuMemcpyDtoDAsync before cuMemAllocAsync: 0.24/layer, 4.5 us/layer, 18 us each
    >=10us after cuEventSynchronize before cuStreamWaitEvent: 0.07/layer, 4.4 us/layer, 63 us each

thread 1066945: 98 calls/layer, in calls 207.4 us/layer (blocking 150.7), between calls 757.8 us/layer
  gaps: 0-1us n=92.8/layer 14.0us/layer; 1-3us n=0.1/layer 0.2us/layer; 3-10us n=0.5/layer 4.5us/layer; 10-30us n=0.5/layer 5.9us/layer; 30-100us n=2.0/layer 113.1us/layer; 100-1e+06us n=2.1/layer 618.4us/layer
    >=10us after cuMemFreeAsync before cuMemFreeAsync: 2.63/layer, 518.7 us/layer, 197 us each
    >=10us after cuMemcpyDtoHAsync before cuMemFreeAsync: 0.99/layer, 143.7 us/layer, 144 us each
    >=10us after cuMemFreeAsync before cuMemcpyDtoHAsync: 0.99/layer, 74.9 us/layer, 75 us each

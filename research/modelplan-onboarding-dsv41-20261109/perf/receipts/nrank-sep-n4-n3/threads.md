wall 655.2 us/layer

thread 2921998: 142 calls/layer, in calls 392.5 us/layer (blocking 193.1), between calls 262.7 us/layer
  gaps: 0-1us n=120.1/layer 34.1us/layer; 1-3us n=7.5/layer 11.2us/layer; 3-10us n=10.0/layer 59.9us/layer; 10-30us n=4.1/layer 57.1us/layer; 30-100us n=0.3/layer 18.3us/layer; 100-1e+06us n=0.2/layer 81.5us/layer
    >=10us after cudaLaunchKernel before cuEventRecord: 1.51/layer, 23.4 us/layer, 16 us each
    >=10us after cuStreamWaitEvent before cuKernelGetName: 0.07/layer, 21.8 us/layer, 334 us each
    >=10us after cuEventRecord before cuStreamWaitEvent: 0.86/layer, 20.5 us/layer, 24 us each
    >=10us after cuMemcpyDtoDAsync before cuKernelGetName: 0.86/layer, 18.7 us/layer, 22 us each
    >=10us after cuEventSynchronize before cuEventSynchronize: 0.10/layer, 16.5 us/layer, 165 us each
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.05/layer, 15.1 us/layer, 315 us each
    >=10us after cuEventSynchronize before cuStreamWaitEvent: 0.10/layer, 14.4 us/layer, 147 us each
    >=10us after cudaLaunchKernel before cuKernelGetName: 0.94/layer, 14.3 us/layer, 15 us each

thread 2922104: 142 calls/layer, in calls 465.6 us/layer (blocking 198.0), between calls 189.6 us/layer
  gaps: 0-1us n=119.0/layer 42.6us/layer; 1-3us n=15.0/layer 24.2us/layer; 3-10us n=6.9/layer 34.2us/layer; 10-30us n=0.8/layer 11.3us/layer; 30-100us n=0.2/layer 14.4us/layer; 100-1e+06us n=0.2/layer 62.4us/layer
    >=10us after cuStreamSynchronize before cuKernelGetName: 0.04/layer, 19.4 us/layer, 445 us each
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.05/layer, 12.0 us/layer, 251 us each
    >=10us after cuEventRecord before cuStreamWaitEvent: 0.17/layer, 11.7 us/layer, 71 us each
    >=10us after cuEventSynchronize before cuEventSynchronize: 0.10/layer, 10.2 us/layer, 102 us each
    >=10us after cuMemcpyDtoDAsync before cuKernelGetName: 0.17/layer, 8.2 us/layer, 49 us each
    >=10us after cuStreamWaitEvent before cuKernelGetName: 0.02/layer, 7.9 us/layer, 330 us each
    >=10us after cuEventSynchronize before cuStreamWaitEvent: 0.10/layer, 6.9 us/layer, 69 us each
    >=10us after cudaLaunchKernel before cuEventRecord: 0.27/layer, 4.6 us/layer, 17 us each

thread 2922103: 142 calls/layer, in calls 459.6 us/layer (blocking 202.8), between calls 195.5 us/layer
  gaps: 0-1us n=118.9/layer 41.6us/layer; 1-3us n=14.0/layer 23.5us/layer; 3-10us n=7.7/layer 38.2us/layer; 10-30us n=1.0/layer 14.5us/layer; 30-100us n=0.2/layer 15.5us/layer; 100-1e+06us n=0.1/layer 61.6us/layer
    >=10us after cuStreamWaitEvent before cuKernelGetName: 0.02/layer, 16.4 us/layer, 944 us each
    >=10us after cuStreamSynchronize before cuKernelGetName: 0.04/layer, 15.1 us/layer, 408 us each
    >=10us after cuEventSynchronize before cuEventSynchronize: 0.10/layer, 14.6 us/layer, 146 us each
    >=10us after cuMemcpyDtoDAsync before cuKernelGetName: 0.30/layer, 10.7 us/layer, 35 us each
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.05/layer, 9.3 us/layer, 195 us each
    >=10us after cuEventSynchronize before cuStreamWaitEvent: 0.10/layer, 6.7 us/layer, 68 us each
    >=10us after cudaLaunchKernel before cuEventRecord: 0.39/layer, 6.1 us/layer, 16 us each
    >=10us after cudaLaunchKernel before cuKernelGetName: 0.20/layer, 5.1 us/layer, 25 us each

thread 2922102: 142 calls/layer, in calls 469.2 us/layer (blocking 195.8), between calls 186.0 us/layer
  gaps: 0-1us n=119.0/layer 42.7us/layer; 1-3us n=15.1/layer 24.7us/layer; 3-10us n=6.5/layer 32.7us/layer; 10-30us n=1.0/layer 14.1us/layer; 30-100us n=0.2/layer 14.6us/layer; 100-1e+06us n=0.1/layer 56.5us/layer
    >=10us after cuEventSynchronize before cuEventSynchronize: 0.10/layer, 25.9 us/layer, 259 us each
    >=10us after cuEventSynchronize before cuStreamWaitEvent: 0.09/layer, 15.0 us/layer, 161 us each
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.05/layer, 10.4 us/layer, 218 us each
    >=10us after cuMemcpyDtoDAsync before cuKernelGetName: 0.23/layer, 8.7 us/layer, 37 us each
    >=10us after cudaLaunchKernel before cuEventRecord: 0.37/layer, 7.8 us/layer, 21 us each
    >=10us after cuStreamWaitEvent before cuKernelGetName: 0.03/layer, 5.5 us/layer, 196 us each
    >=10us after cudaLaunchKernel before cuKernelGetName: 0.21/layer, 4.8 us/layer, 23 us each
    >=10us after cuEventRecord before cuStreamWaitEvent: 0.17/layer, 4.2 us/layer, 25 us each

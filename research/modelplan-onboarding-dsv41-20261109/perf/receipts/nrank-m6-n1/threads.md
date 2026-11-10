wall 644.3 us/layer

thread 848143: 199 calls/layer, in calls 567.6 us/layer (blocking 409.9), between calls 76.7 us/layer
  gaps: 0-1us n=197.1/layer 43.1us/layer; 1-3us n=1.1/layer 1.5us/layer; 3-10us n=0.1/layer 0.6us/layer; 10-30us n=0.1/layer 3.0us/layer; 30-100us n=0.3/layer 18.5us/layer; 100-1e+06us n=0.0/layer 0.0us/layer
    >=10us after cuMemFreeAsync before cuMemFreeAsync: 0.12/layer, 9.8 us/layer, 82 us each
    >=10us after cuEventSynchronize before cuStreamWaitEvent: 0.07/layer, 4.6 us/layer, 65 us each
    >=10us after cuEventSynchronize before cuEventSynchronize: 0.07/layer, 4.1 us/layer, 58 us each
    >=10us after cuMemcpyDtoHAsync before cuMemFreeAsync: 0.12/layer, 2.9 us/layer, 24 us each
    >=10us after cuMemFreeAsync before cuMemAllocAsync: 0.01/layer, 0.1 us/layer, 10 us each

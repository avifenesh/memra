wall 836.0 us/layer

thread 3281799: 114 calls/layer, in calls 774.1 us/layer (blocking 649.5), between calls 61.9 us/layer
  gaps: 0-1us n=112.6/layer 30.7us/layer; 1-3us n=0.9/layer 1.0us/layer; 3-10us n=0.0/layer 0.0us/layer; 10-30us n=0.2/layer 3.2us/layer; 30-100us n=0.1/layer 10.8us/layer; 100-1e+06us n=0.0/layer 0.0us/layer
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.11/layer, 9.6 us/layer, 90 us each
    >=10us after cuEventSynchronize before cuEventSynchronize: 0.11/layer, 2.9 us/layer, 27 us each
    >=10us after cuMemcpyDtoHAsync before cuMemcpyDtoHAsync: 0.11/layer, 1.6 us/layer, 15 us each

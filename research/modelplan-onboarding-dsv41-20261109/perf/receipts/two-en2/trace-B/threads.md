wall 366.6 us/layer

thread 3739118: 120 calls/layer, in calls 308.8 us/layer (blocking 169.6), between calls 57.8 us/layer
  gaps: 0-1us n=111.7/layer 36.0us/layer; 1-3us n=7.8/layer 10.6us/layer; 3-10us n=0.1/layer 0.5us/layer; 10-30us n=0.2/layer 3.5us/layer; 30-100us n=0.0/layer 0.0us/layer; 100-1e+06us n=0.0/layer 5.7us/layer
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.05/layer, 5.7 us/layer, 120 us each
    >=10us after cuEventSynchronize before cuMemcpyHtoDAsync: 0.20/layer, 3.0 us/layer, 15 us each
    >=10us after cuMemcpyDtoHAsync before cuMemcpyDtoHAsync: 0.04/layer, 0.5 us/layer, 12 us each

thread 3739141: 120 calls/layer, in calls 305.7 us/layer (blocking 169.7), between calls 60.9 us/layer
  gaps: 0-1us n=108.7/layer 33.8us/layer; 1-3us n=10.5/layer 15.2us/layer; 3-10us n=0.3/layer 1.1us/layer; 10-30us n=0.2/layer 3.5us/layer; 30-100us n=0.0/layer 0.4us/layer; 100-1e+06us n=0.0/layer 5.4us/layer
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.05/layer, 5.4 us/layer, 113 us each
    >=10us after cuEventSynchronize before cuMemcpyHtoDAsync: 0.20/layer, 3.0 us/layer, 15 us each
    >=10us after cuMemcpyDtoHAsync before cuMemcpyDtoHAsync: 0.04/layer, 0.9 us/layer, 22 us each

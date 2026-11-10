wall 524.7 us/layer

thread 3714246: 113 calls/layer, in calls 459.3 us/layer (blocking 331.1), between calls 65.4 us/layer
  gaps: 0-1us n=112.5/layer 35.4us/layer; 1-3us n=0.5/layer 0.5us/layer; 3-10us n=0.0/layer 0.1us/layer; 10-30us n=0.1/layer 2.6us/layer; 30-100us n=0.1/layer 6.6us/layer; 100-1e+06us n=0.1/layer 11.8us/layer
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.11/layer, 11.2 us/layer, 105 us each
    >=10us after cuEventSynchronize before cuMemcpyHtoDAsync: 0.15/layer, 8.1 us/layer, 52 us each
    >=10us after cuMemcpyDtoHAsync before cuMemcpyDtoHAsync: 0.11/layer, 1.7 us/layer, 16 us each

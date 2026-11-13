wall 525.5 us/layer

thread 3714427: 113 calls/layer, in calls 464.5 us/layer (blocking 336.3), between calls 60.9 us/layer
  gaps: 0-1us n=112.5/layer 35.4us/layer; 1-3us n=0.4/layer 0.5us/layer; 3-10us n=0.0/layer 0.0us/layer; 10-30us n=0.3/layer 4.8us/layer; 30-100us n=0.0/layer 0.2us/layer; 100-1e+06us n=0.1/layer 11.7us/layer
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.11/layer, 11.7 us/layer, 110 us each
    >=10us after cuEventSynchronize before cuMemcpyHtoDAsync: 0.22/layer, 3.2 us/layer, 14 us each
    >=10us after cuMemcpyDtoHAsync before cuMemcpyDtoHAsync: 0.11/layer, 1.7 us/layer, 16 us each

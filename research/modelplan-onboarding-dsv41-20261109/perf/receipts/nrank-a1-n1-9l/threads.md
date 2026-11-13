wall 754.1 us/layer

thread 3635804: 114 calls/layer, in calls 676.3 us/layer (blocking 550.7), between calls 77.9 us/layer
  gaps: 0-1us n=112.8/layer 31.8us/layer; 1-3us n=0.5/layer 0.6us/layer; 3-10us n=0.0/layer 0.1us/layer; 10-30us n=0.1/layer 2.0us/layer; 30-100us n=0.1/layer 10.0us/layer; 100-1e+06us n=0.1/layer 16.7us/layer
    >=10us after cuEventSynchronize before cuMemcpyHtoDAsync: 0.10/layer, 16.4 us/layer, 161 us each
    >=10us after cuMemcpyDtoHAsync before cuMemAllocAsync: 0.11/layer, 10.7 us/layer, 101 us each
    >=10us after cuMemcpyDtoHAsync before cuMemcpyDtoHAsync: 0.11/layer, 1.7 us/layer, 16 us each

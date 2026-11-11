window 4097.0 ms (t0 = first 4096-row mHC pass); last mHC pass of the prompt at +3921.9 ms; next prompt's upload at +4034.8 ms

stream 0/16: host-late 88.2 ms (2.21 a layer): in the prompt's layers 9.4 ms; after its last mHC pass, before the next upload 0.0 ms; the next prompt's start 78.7 ms
  gap at +2.3 ms: 3.0 ms before v41_act_quant_kernel; the issuing thread was in host code between calls
  gap at +86.8 ms: 4.5 ms before copy kind 1 24 MB; the issuing thread was in host code between calls
  gap at +99.4 ms: 1.3 ms before v41_hc_mixes_part_kernel; the issuing thread was in host code between calls
  gap at +3974.3 ms: 60.6 ms before copy kind 1 320 MB; the issuing thread was in host code between calls
  gap at +4070.1 ms: 18.2 ms before copy kind 8 320 MB; the issuing thread was in cuMemAllocAsync

stream 1/28: host-late 88.4 ms (2.21 a layer): in the prompt's layers 9.8 ms; after its last mHC pass, before the next upload 0.1 ms; the next prompt's start 78.4 ms
  gap at +2.3 ms: 3.0 ms before v41_act_quant_kernel; the issuing thread was in host code between calls
  gap at +86.8 ms: 4.1 ms before copy kind 1 24 MB; the issuing thread was in host code between calls
  gap at +99.5 ms: 1.3 ms before v41_hc_mixes_part_kernel; the issuing thread was in host code between calls
  gap at +3974.3 ms: 60.6 ms before copy kind 1 320 MB; the issuing thread was in host code between calls
  gap at +4070.1 ms: 17.8 ms before copy kind 8 320 MB; the issuing thread was in cuMemAllocAsync

stream 2/40: host-late 89.5 ms (2.24 a layer): in the prompt's layers 10.5 ms; after its last mHC pass, before the next upload 0.1 ms; the next prompt's start 78.9 ms
  gap at +2.3 ms: 3.0 ms before v41_act_quant_kernel; the issuing thread was in host code between calls
  gap at +86.8 ms: 6.4 ms before copy kind 1 24 MB; the issuing thread was in cuMemHostAlloc
  gap at +3974.3 ms: 60.6 ms before copy kind 1 320 MB; the issuing thread was in host code between calls
  gap at +4070.1 ms: 18.3 ms before copy kind 8 320 MB; the issuing thread was in cuMemAllocAsync

stream 3/52: host-late 89.9 ms (2.25 a layer): in the prompt's layers 11.0 ms; after its last mHC pass, before the next upload 0.1 ms; the next prompt's start 78.8 ms
  gap at +2.3 ms: 3.0 ms before v41_act_quant_kernel; the issuing thread was in host code between calls
  gap at +86.8 ms: 3.8 ms before copy kind 1 24 MB; the issuing thread was in host code between calls
  gap at +91.0 ms: 1.2 ms before copy kind 1 0 MB; the issuing thread was in host code between calls
  gap at +99.4 ms: 1.3 ms before v41_hc_mixes_part_kernel; the issuing thread was in host code between calls
  gap at +3974.3 ms: 60.6 ms before copy kind 1 320 MB; the issuing thread was in host code between calls
  gap at +4070.1 ms: 18.3 ms before copy kind 8 320 MB; the issuing thread was in cuMemAllocAsync

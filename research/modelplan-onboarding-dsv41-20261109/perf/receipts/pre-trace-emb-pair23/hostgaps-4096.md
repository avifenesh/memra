window 480.6 ms (t0 = first 4096-row mHC pass); last mHC pass of the prompt at +457.8 ms; next prompt's upload at +473.4 ms

stream 0/11: host-late 16.9 ms (0.85 a layer): in the prompt's layers 16.9 ms
  gap at +19.4 ms: 7.6 ms before copy kind 10 1 MB; the issuing thread was in cuStreamSynchronize
  gap at +320.7 ms: 9.3 ms before copy kind 10 1 MB; the issuing thread was in cuStreamSynchronize

stream 0/14: host-late 6.6 ms (0.33 a layer): in the prompt's layers 0.1 ms; after its last mHC pass, before the next upload 0.1 ms; the next prompt's start 6.4 ms
  gap at +473.1 ms: 6.4 ms before copy kind 1 0 MB; the issuing thread was in host code between calls

stream 1/20: host-late 17.0 ms (0.85 a layer): in the prompt's layers 17.0 ms
  gap at +19.4 ms: 7.6 ms before copy kind 10 1 MB; the issuing thread was in cuStreamSynchronize
  gap at +320.7 ms: 9.3 ms before copy kind 10 1 MB; the issuing thread was in cuStreamSynchronize

stream 1/24: host-late 6.9 ms (0.35 a layer): in the prompt's layers 0.4 ms; after its last mHC pass, before the next upload 0.1 ms; the next prompt's start 6.4 ms
  gap at +473.2 ms: 6.4 ms before copy kind 1 0 MB; the issuing thread was in host code between calls

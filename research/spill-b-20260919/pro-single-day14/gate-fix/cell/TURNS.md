| turn | prompt_tokens | cached_tokens | prev prompt_tokens | route cold/restored | hit line | insert | evict (segment) | refused/skipped | effective free consumed | prefix bytes grew | V3 error | elapsed s | text sha256[:16] |
| ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | 9200 | 0 | - | none | none | 9200 tok 430.1MB (seed) | 2800 tok 240.0MB (Protected) | 0 | 1301720404 | 190054400 | 1111666004 | 2.639 | 254a65a01730e58b |
| 2 | 9500 | 9200 | 9200 | none | 9200 of 9500 | 9500 tok 439.0MB (seed) | 3000 tok 246.0MB (Probation), 3200 tok 251.9MB (Protected) | 0 | 551462996 | -58896384 | 610359380 | 0.25 | 24a97a2867768b2d |
| 3 | 9800 | 9500 | 9500 | none | 9500 of 9800 | 9800 tok 447.9MB (seed) | 9200 tok 430.1MB (Probation) | 0 | 37977600 | 17817600 | 20160000 | 0.25 | 65eeb1ef8f716af1 |
| 4 | 10100 | 9800 | 9800 | none | 9800 of 10100 | 10100 tok 456.8MB (seed) | 9500 tok 439.0MB (Probation) | 0 | 37977600 | 17817600 | 20160000 | 0.251 | 64a99158cb668b0c |
| 5 | 10400 | 10100 | 10100 | none | 10100 of 10400 | 10400 tok 465.7MB (seed) | 9800 tok 447.9MB (Probation) | 0 | 37977600 | 17817600 | 20160000 | 0.251 | c6b9d167a76a942e |
| 6 | 10700 | 10400 | 10400 | none | 10400 of 10700 | 10700 tok 474.6MB (seed) | 10100 tok 456.8MB (Probation) | 0 | 37977600 | 17817600 | 20160000 | 0.252 | 8e4798b352770d9a |
| 7 | 11000 | 10700 | 10700 | none | 10700 of 11000 | 11000 tok 483.5MB (seed) | 10400 tok 465.7MB (Probation) | 0 | 37977600 | 17817600 | 20160000 | 0.252 | f34ee12b3db5cb9c |
| 8 | 11300 | 11000 | 11000 | none | 11000 of 11300 | 11300 tok 492.5MB (seed) | 10700 tok 474.6MB (Probation) | 0 | 46695936 | 17817600 | 28878336 | 0.254 | ee53848835a29d90 |

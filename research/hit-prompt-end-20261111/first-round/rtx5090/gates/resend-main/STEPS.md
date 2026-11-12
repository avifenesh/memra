| step | prompt | cached_tokens | hit line | prime calls | off-grid calls | tails inserted | elapsed s | sha[:16] | cold sha[:16] | == cold |
| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | --- | --- | --- |
| p1-cold | 11000 | 0 | [] | 5 | 0 | [] | 16.346 | 9b5aec538d30a8f5 | 9b5aec538d30a8f5 | yes |
| p1-resend | 11000 | 10976 | [(10976, 11000)] | 1 | 0 | [] | 0.76 | 9b5aec538d30a8f5 | 9b5aec538d30a8f5 | yes |
| p2-growth | 11150 | 10976 | [(10976, 11150)] | 2 | 0 | [] | 0.986 | 472b5686de481749 | 472b5686de481749 | yes |
| p1-resend2 | 11000 | 10976 | [(10976, 11000)] | 1 | 0 | [] | 0.76 | 9b5aec538d30a8f5 | 9b5aec538d30a8f5 | yes |
| p2-resend | 11150 | 11104 | [(11104, 11150)] | 1 | 0 | [] | 0.764 | 472b5686de481749 | 472b5686de481749 | yes |

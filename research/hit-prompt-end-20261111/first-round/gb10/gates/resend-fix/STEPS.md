| step | prompt | cached_tokens | hit line | prime calls | off-grid calls | tails inserted | elapsed s | sha[:16] | cold sha[:16] | == cold |
| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | --- | --- | --- |
| p1-cold | 11000 | 0 | [] | 5 | 0 | [(24, 10976)] | 8.907 | 9b5aec538d30a8f5 | 9b5aec538d30a8f5 | yes |
| p1-resend | 11000 | 11000 | [(11000, 11000)] | 0 | 0 | [] | 1.204 | 9b5aec538d30a8f5 | 9b5aec538d30a8f5 | yes |
| p2-growth | 11150 | 10976 | [(10976, 11150)] | 2 | 0 | [(46, 11104)] | 1.534 | f8a235218b399416 | f8a235218b399416 | yes |
| p1-resend2 | 11000 | 11000 | [(11000, 11000)] | 0 | 0 | [] | 1.204 | 9b5aec538d30a8f5 | 9b5aec538d30a8f5 | yes |
| p2-resend | 11150 | 11150 | [(11150, 11150)] | 0 | 0 | [] | 1.205 | f8a235218b399416 | f8a235218b399416 | yes |

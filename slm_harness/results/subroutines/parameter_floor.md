| Subroutine | Rules | 135M FT | 362M FT | 494M FT | 1544M FT | Floor | Verdict |
|---|---|---|---|---|---|---|---|
| action_router | 0.277 | 0.612 | 0.740 | 0.936 | 0.996 | 494M | works at 494M |
| evidence_judge | 0.983 | 0.540 | 0.484 | 0.844 | 0.932 | — | rules suffice |
| json_repair | 0.147 | 0.924 | 1.000 | 0.996 | 1.000 | 135M | works at 135M |
| path_normalizer | 0.790 | 0.500 | 0.704 | 0.956 | 0.980 | 494M | works at 494M |
| read_span_selector | 0.817 | 0.000 | 0.024 | 0.224 | 0.816 | — | unsolved (best 0.82 @ qwen2.5-1.5b) |
| search_hit_ranker | 0.993 | 0.160 | 0.176 | 0.948 | 0.996 | — | rules suffice |
| search_query_gen | 0.237 | 0.636 | 0.632 | 0.864 | 0.876 | — | unsolved (best 0.88 @ qwen2.5-1.5b) |
| trace_localizer | 1.000 | 0.812 | 0.936 | 1.000 | 1.000 | — | rules suffice |

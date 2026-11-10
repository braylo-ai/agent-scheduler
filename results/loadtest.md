Load: high_21rps | SIMULATED MockLLM | seed 7 | slots 8 | SLO 1.5s | 10.0s arrivals

| config | on-time | miss (admitted) | miss+reject | p50 ok (s) | p95 ok (s) | Jain (norm.) | wasted calls |
|---|---|---|---|---|---|---|---|
| fifo_naive | 0.014 | 0.986 | 0.986 | 1.389 | 1.393 | 0.250 | 0.985 |
| fair_naive | 0.041 | 0.959 | 0.959 | 0.886 | 1.403 | 0.653 | 0.964 |
| fair_deadline | 0.256 | 0.744 | 0.744 | 0.959 | 1.443 | 0.936 | 0.447 |
| fifo_full | 0.343 | 0.265 | 0.657 | 1.222 | 1.441 | 0.548 | 0.314 |
| full | 0.352 | 0.374 | 0.648 | 1.045 | 1.420 | 0.918 | 0.203 |

Load: low_8.4rps | SIMULATED MockLLM | seed 7 | slots 8 | SLO 1.5s | 10.0s arrivals

| config | on-time | miss (admitted) | miss+reject | p50 ok (s) | p95 ok (s) | Jain (norm.) | wasted calls |
|---|---|---|---|---|---|---|---|
| fifo_naive | 0.862 | 0.138 | 0.138 | 0.770 | 1.435 | 0.971 | 0.168 |
| fair_naive | 0.830 | 0.170 | 0.170 | 0.754 | 1.277 | 0.975 | 0.199 |
| fair_deadline | 0.915 | 0.085 | 0.085 | 0.752 | 1.262 | 0.987 | 0.074 |
| fifo_full | 0.872 | 0.079 | 0.128 | 0.782 | 1.443 | 0.967 | 0.105 |
| full | 0.925 | 0.044 | 0.074 | 0.727 | 1.247 | 0.987 | 0.036 |

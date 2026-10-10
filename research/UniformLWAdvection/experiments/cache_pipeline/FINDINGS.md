# Full-step tile pipeline scaling study

The final comparison has 180 accepted fresh-process samples (five repeats per
size, worker count, and method) and a minimum timed duration of 0.372 s.
Each size uses one fixed trajectory length for both methods: 2048 steps at
1024², 128 at 2048², and 32 at 4096². Every final field matched its serial
toy oracle bitwise, the public `q` object remained the same array, and each
requested worker participated in both measured phases. The worker counts are
1, 2, 4, 8, 16, and 32 pinned Julia threads.

One 2048² toy p32 Julia process crashed in compiler/GC before its timer. The
exact command was retried once and passed; the failed launch, stderr, retry,
and remaining job order are recorded in the ignored local cohort's command
log and recovery receipt. The failed launch contributes no timing sample.

The pipeline uses tile `(128,32)` with its halo work and parallel public-`q` copyback charged on every step. The frozen toy uses tile `(512,16)`. Each method uses the same input, timestep, and fixed step count at a given mesh size. Ratios are medians of independent process samples; 95% intervals resample the numerator and denominator groups independently.

The frozen numerical source SHA-256 values are
`7c9b20156f76100691b130701c1910ef7d50870f8907a38499c495b83d98f245`
for `CachePipeline.jl` and
`cfaec4a0fcabc64b53e932b8f9347c486a5a59cd883a4ff22cf009e965fd0de7`
for `timing_driver.jl`. The runner was
`7bd2321714f4426849162d481b6fe364053d91a721c7e70b0d8bf7e7c56de34e`.
Numerical arrays, run order, launch receipts, source snapshots, full-field
hashes, and the local SVG/PNG plots are retained under ignored
`benchmark/scaling/results/cache_pipeline-final-{1024,2048,4096}-20260930`
and `benchmark/scaling/results/cache_pipeline-final-analysis-20260930`.

| N | Workers | Toy ns [IQR] | Pipeline ns [IQR] | Same-p toy/pipeline [95%] | Toy Eₚ [95%] | Pipeline Eₚ [95%] | Matched toy serial / pipeline Sₚ [95%] | Matched toy serial / pipeline Eₚ [95%] |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 1 | 2.051 [1.984–2.291] | 2.655 [2.624–2.692] | 0.773 [0.737–0.919] | 1.000 [1.000–1.000] | 1.000 [1.000–1.000] | 0.773 [0.737–0.919] | 0.773 [0.737–0.919] |
| 1024 | 2 | 1.041 [0.993–1.066] | 1.102 [1.082–1.129] | 0.945 [0.876–1.146] | 0.985 [0.812–1.184] | 1.204 [1.157–1.252] | 1.861 [1.757–2.213] | 0.931 [0.878–1.106] |
| 1024 | 4 | 0.480 [0.459–0.565] | 0.578 [0.576–0.584] | 0.830 [0.770–1.006] | 1.068 [0.877–1.286] | 1.147 [1.114–1.190] | 3.546 [3.394–4.216] | 0.886 [0.848–1.054] |
| 1024 | 8 | 0.277 [0.273–0.282] | 0.330 [0.321–0.330] | 0.839 [0.760–0.961] | 0.926 [0.809–1.116] | 1.006 [0.976–1.069] | 6.216 [6.008–7.390] | 0.777 [0.751–0.924] |
| 1024 | 16 | 0.192 [0.177–0.210] | 0.230 [0.227–0.241] | 0.835 [0.727–0.936] | 0.668 [0.591–0.828] | 0.721 [0.675–0.752] | 8.918 [8.223–10.603] | 0.557 [0.514–0.663] |
| 1024 | 32 | 0.246 [0.232–0.400] | 0.295 [0.274–0.338] | 0.831 [0.180–1.829] | 0.261 [0.119–0.325] | 0.281 [0.061–0.324] | 6.943 [1.508–8.893] | 0.217 [0.047–0.278] |
| 2048 | 1 | 2.970 [2.875–3.053] | 3.484 [3.473–3.547] | 0.852 [0.791–0.912] | 1.000 [1.000–1.000] | 1.000 [1.000–1.000] | 0.852 [0.791–0.912] | 0.852 [0.791–0.912] |
| 2048 | 2 | 2.348 [2.223–2.357] | 1.792 [1.789–1.835] | 1.310 [1.167–1.331] | 0.632 [0.597–0.713] | 0.972 [0.897–1.019] | 1.658 [1.529–1.773] | 0.829 [0.764–0.887] |
| 2048 | 4 | 1.766 [1.761–1.807] | 1.284 [1.242–1.296] | 1.376 [1.316–1.496] | 0.420 [0.387–0.450] | 0.679 [0.648–0.727] | 2.314 [2.165–2.504] | 0.578 [0.541–0.626] |
| 2048 | 8 | 1.924 [1.889–1.990] | 1.136 [1.111–1.180] | 1.693 [1.481–1.811] | 0.193 [0.179–0.207] | 0.383 [0.335–0.406] | 2.614 [2.286–2.796] | 0.327 [0.286–0.349] |
| 2048 | 16 | 1.914 [1.899–2.007] | 1.102 [1.101–1.135] | 1.736 [1.584–1.847] | 0.097 [0.088–0.104] | 0.198 [0.180–0.207] | 2.694 [2.458–2.883] | 0.168 [0.154–0.180] |
| 2048 | 32 | 2.700 [2.562–2.734] | 1.724 [1.699–1.748] | 1.566 [1.287–1.617] | 0.034 [0.032–0.042] | 0.063 [0.055–0.066] | 1.723 [1.511–1.843] | 0.054 [0.047–0.058] |
| 4096 | 1 | 3.242 [3.191–3.251] | 3.587 [3.539–3.615] | 0.904 [0.476–0.941] | 1.000 [1.000–1.000] | 1.000 [1.000–1.000] | 0.904 [0.476–0.941] | 0.904 [0.476–0.941] |
| 4096 | 2 | 2.570 [2.547–2.582] | 2.176 [2.167–2.191] | 1.181 [1.160–1.230] | 0.631 [0.606–0.647] | 0.824 [0.792–1.566] | 1.490 [1.450–1.529] | 0.745 [0.725–0.764] |
| 4096 | 4 | 2.446 [2.403–2.467] | 1.782 [1.767–1.846] | 1.373 [1.258–1.409] | 0.331 [0.321–0.341] | 0.503 [0.461–0.956] | 1.819 [1.666–1.868] | 0.455 [0.417–0.467] |
| 4096 | 8 | 2.487 [2.465–2.503] | 1.677 [1.672–1.685] | 1.483 [1.457–1.506] | 0.163 [0.158–0.167] | 0.267 [0.257–0.508] | 1.933 [1.884–1.983] | 0.242 [0.236–0.248] |
| 4096 | 16 | 2.638 [2.635–2.697] | 1.787 [1.721–1.826] | 1.476 [1.412–1.575] | 0.077 [0.073–0.079] | 0.126 [0.118–0.238] | 1.814 [1.735–1.898] | 0.113 [0.108–0.119] |
| 4096 | 32 | 2.888 [2.875–2.889] | 2.059 [2.054–2.130] | 1.402 [1.232–1.515] | 0.035 [0.034–0.037] | 0.054 [0.048–0.103] | 1.574 [1.383–1.701] | 0.049 [0.043–0.053] |

Efficiency Eₚ is speedup Sₚ divided by p. Own-method Eₚ uses that method's one-worker median. A large pipeline Eₚ can reflect a slower pipeline serial baseline, so the same-worker absolute time and toy/pipeline ratio are the direct gain measures. The matched toy-serial columns use the frozen toy's one-worker median for numerator and bootstrap interval; this is a within-study serial reference, not the strongest compatible implementation established elsewhere. The fastest observed toy serial sample is also retained in summary.json as a conservative within-study check; one fastest sample is not a stable serial median. The earlier Stage 2 observable halo path reached 2.446 ns at 2048² and 2.674 ns at 4096² in different frozen-step cohorts. Those values do not enter these confidence intervals and still matter to the 5% serial-regression guard. These timings support or reject a wall-time benefit, without measuring DRAM bytes or proving a particular traffic reduction.

One of the five 4096² pipeline serial samples measured 6.814 ns, versus
3.444–3.615 ns for the other four. It is retained, and explains the broad
upper limits of the pipeline's own-serial efficiency intervals. Five-process
bootstrap intervals describe this cohort's variability; they do not identify
the cause of that slow sample.

The pipeline's 1024² median is slower at every worker count; at p2, p4, and
p32 the same-worker ratio intervals include parity. At 2048² and 4096², its p4 and
p8 same-worker gains have intervals entirely above one. At p16 the toy's
normalized 1024²→2048² time rises 9.97×, while the pipeline's rises 4.79×.
The pipeline therefore reduces the measured large-grid cliff for this
research toy, but its own serial cost is higher, and the historical Stage 2
halo serial reference prevents declaring the broader production acceptance
target met. No production integration was made.

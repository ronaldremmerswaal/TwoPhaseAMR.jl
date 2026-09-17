# Two-velocity passive-transport benchmark

Reference run: 2026-09-17, Julia 1.12.7, four Julia threads, AMD Ryzen 9
5950X, 10 samples per operation. Times are medians. The benchmark uses the
same interface, phase scalar fields, time step, and sweep configuration for
the one-velocity and two-velocity paths.

## Uniform mesh

The uniform case has 1,024 cells and 2,048 faces.

| Operation | Time | Allocated bytes | Allocations |
|---|---:|---:|---:|
| One-velocity workspace | 0.036 ms | 219,848 | 77 |
| Projected workspace | 2.742 ms | 9,754,184 | 35,850 |
| Frozen-subface workspace | 0.033 ms | 219,848 | 77 |
| Liquid auxiliary projection | 1.690 ms | 1,709,576 | 42,779 |
| One-velocity step | 2.064 ms | 1,426,272 | 56,452 |
| Projected two-velocity step | 4.038 ms | 3,400,584 | 113,603 |
| Frozen-subface two-velocity step | 3.735 ms | 14,694,144 | 144,600 |

The complete projected step costs 1.96 times the one-velocity time and 2.38
times its allocated bytes. The frozen-subface step costs 1.81 times the time
and 10.30 times the allocated bytes.

## Fixed mixed-level mesh

The mixed-level case has 1,008 cells and 2,040 faces.

| Operation | Time | Allocated bytes | Allocations |
|---|---:|---:|---:|
| One-velocity workspace | 0.034 ms | 238,808 | 165 |
| Projected workspace | 7.521 ms | 16,487,424 | 44,054 |
| Frozen-subface workspace | 0.032 ms | 238,808 | 165 |
| Liquid auxiliary projection | 2.864 ms | 1,735,736 | 42,484 |
| One-velocity step | 7.266 ms | 1,391,280 | 55,554 |
| Projected two-velocity step | 10.667 ms | 3,375,096 | 112,182 |
| Frozen-subface two-velocity step | 9.561 ms | 6,983,120 | 132,643 |

The complete projected step costs 1.47 times the one-velocity time and 2.43
times its allocated bytes. The frozen-subface step costs 1.32 times the time
and 5.02 times the allocated bytes.

## Interpretation

The projected method pays most of its setup cost while building the auxiliary
pressure operator and its geometric solve state. Its per-step overhead is
then dominated by the liquid-velocity projection. The frozen-subface
workspace is as cheap to construct as the one-velocity workspace, but its
directional geometric clipping creates substantially more temporary
allocation, especially on the uniform mesh. This identifies frozen-subface
flux construction as the first allocation target when optimizing this
milestone. These measurements include interface and two centred phase-scalar
updates; staggered momentum transport belongs to the next benchmark stage.

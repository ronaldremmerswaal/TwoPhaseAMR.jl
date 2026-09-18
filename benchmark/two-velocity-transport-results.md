# Two-velocity passive-transport benchmark

Reference run: 2026-09-17, Julia 1.12.7, four Julia threads, AMD Ryzen 9
5950X, 10 samples per operation. Times are medians. The benchmark uses the
same interface, phase scalar fields, time step, and sweep configuration for
the one-velocity and two-velocity paths.

## Uniform mesh

The uniform case has 1,024 cells and 2,048 faces.

| Operation | Time | Allocated bytes | Allocations |
|---|---:|---:|---:|
| One-velocity workspace | 0.032 ms | 219,880 | 77 |
| Projected workspace | 2.704 ms | 9,754,216 | 35,850 |
| Frozen-subface workspace | 0.030 ms | 219,880 | 77 |
| Two-velocity momentum workspace | 0.453 ms | 2,060,000 | 4,360 |
| Liquid auxiliary projection | 1.642 ms | 1,709,576 | 42,779 |
| Phase dual-flux interpolation | 0.268 ms | 1,171,520 | 17,923 |
| One-velocity interface/centred step | 2.059 ms | 1,426,272 | 56,452 |
| Projected interface/centred step | 4.215 ms | 3,400,904 | 113,603 |
| Frozen interface/centred step | 4.099 ms | 15,107,184 | 163,390 |
| One-velocity complete step | 2.549 ms | 1,525,600 | 56,602 |
| Projected complete step | 4.794 ms | 3,538,056 | 113,910 |
| Frozen complete step | 4.772 ms | 15,210,352 | 163,657 |

The complete projected step costs 1.88 times the one-velocity time and 2.32
times its allocated bytes. The frozen-subface step costs 1.87 times the time
and 9.97 times the allocated bytes.

## Fixed mixed-level mesh

The mixed-level case has 1,008 cells and 2,040 faces.

| Operation | Time | Allocated bytes | Allocations |
|---|---:|---:|---:|
| One-velocity workspace | 0.030 ms | 238,840 | 165 |
| Projected workspace | 5.922 ms | 16,487,456 | 44,054 |
| Frozen-subface workspace | 0.034 ms | 238,840 | 165 |
| Two-velocity momentum workspace | 1.837 ms | 5,117,952 | 8,474 |
| Liquid auxiliary projection | 2.814 ms | 1,735,736 | 42,484 |
| Phase dual-flux interpolation | 0.283 ms | 1,213,808 | 19,129 |
| One-velocity interface/centred step | 7.135 ms | 1,391,280 | 55,554 |
| Projected interface/centred step | 10.520 ms | 3,375,416 | 112,182 |
| Frozen interface/centred step | 10.944 ms | 7,401,728 | 151,210 |
| One-velocity complete step | 17.854 ms | 18,849,200 | 465,648 |
| Projected complete step | 30.268 ms | 37,187,704 | 862,601 |
| Frozen complete step | 40.231 ms | 42,324,096 | 971,421 |

The complete projected step costs 1.70 times the one-velocity time and 1.97
times its allocated bytes. The frozen-subface step costs 2.25 times the time
and 2.25 times the allocated bytes.

## Interpretation

The projected method pays most of its setup cost while building the auxiliary
pressure operator and its geometric solve state. Its complete-step overhead
is 70--88%, and its allocation ratio is approximately two. The
frozen-subface workspace itself remains as cheap as the one-velocity
workspace, but directional geometric clipping allocates heavily on the
uniform mesh. On the mixed mesh, the compatible dual schedules dominate all
complete variants: adding staggered scalars and momentum raises the
one-velocity step from 7.1 to 17.9 ms and the frozen two-velocity step from
10.9 to 40.2 ms.

The separate dual-flux checksum takes only 0.27--0.28 ms, but allocates
1.17--1.21 MB. Together with the 15.2 MB uniform frozen complete step and
42.3 MB mixed frozen complete step, this identifies schedule traversal and
frozen-subface flux construction as the next allocation targets. The retained
phase-face scratch prevents an additional array allocation proportional to
the number of transported momentum components; the allocation-scaling tests
confirm linear rather than superlinear growth with active staggered degrees
of freedom.

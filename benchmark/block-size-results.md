# Block size as a type parameter

`RESEARCH.md` asks whether block dimensions should be encoded in the mesh type.
The reproducible toy benchmark is `block_size_parameter.jl`. It compares the
same allocation-free five-point stencil with dimensions passed as runtime
integers and as `StaticLayout{NX,NY,H}` value parameters.

Recorded 2026-09-13 with Julia 1.12.7, one Julia thread, on an AMD Ryzen 9
5950X. Values are warm medians from `BenchmarkTools`; both functions are marked
`@noinline` so the dynamic dimensions are not constant-propagated from the
benchmark harness.

| Interior size | Runtime dimensions | Type parameters | Dynamic/static |
|---:|---:|---:|---:|
| 8×8 | 36.31 ns | 45.36 ns | 0.800 |
| 16×16 | 86.80 ns | 69.18 ns | 1.255 |
| 32×32 | 301.86 ns | 325.40 ns | 0.928 |
| 64×64 | 1.172 μs | 1.140 μs | 1.028 |

The result is mixed and negligible at the representative 32×32 and 64×64
sizes. `BlockTreeMesh` therefore keeps block dimensions as runtime fields for
now, avoiding a compiled mesh/field specialization for every chosen size. This
decision should be repeated with the actual pressure and staggered-transport
kernels before their interfaces are frozen; those kernels may expose static
boundary structure that this simple stencil does not.

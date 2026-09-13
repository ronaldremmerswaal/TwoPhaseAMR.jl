# TwoPhaseAMR.jl

This package is being developed as a sequence of tested vertical slices. The
public API covers the centred, single-phase foundation described in milestone
1 of `DEVELOPMENT.md` and a uniform-periodic staggered-advection slice from
milestone 2.

## Data model

Each `RegionTrees.Cell` leaf represents one fixed-size Cartesian block. A
`CellField` owns one dense Julia `Matrix` per leaf, including a configurable
halo. Mesh construction turns the mutable tree into immutable leaf, face, and
halo schedules. Numerical kernels traverse those schedules and arrays only.

The face schedule subdivides a coarse face where necessary and gives each
resulting physical segment exactly one owner. Conservative updates add one
integrated flux to one cell and subtract the same value from its neighbor.

The [compatible pressure gradient](pressure-gradient.md) leaves this primal
Gauss divergence unchanged. A precomputed inverse face Riesz map is diagonal
on conforming regions and adds sparse affine-exact couplings at coarse--fine
vertices, making the composite pressure operator symmetric and stable.

`FaceField{Axis}` stores compact directional face degrees of freedom. The
staggered-advection workspace builds dual-control-volume connectivity from the
immutable centred face schedule. Its upwind, centred, and Lax--Wendroff kernels
are allocation-free after setup. See [Staggered advection](staggered-advection.md)
for the discrete construction and current boundary of support.

## Scope

The package is 2-D, CPU-only, serial, refinement-ratio-two, and globally
time-stepped. The compatible gradient supports periodic uniform meshes and
isotropic mixed-level meshes. Staggered advection currently supports uniform-resolution,
fully periodic meshes, including equivalent multi-block partitions. Staggered
AMR transfer and physical-boundary dual volumes remain explicit gates before
milestone 2 can be called complete. The literature and architecture documents
state the compatibility conditions for projection, VOF, and distributed
execution.

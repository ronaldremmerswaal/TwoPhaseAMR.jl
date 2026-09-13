# TwoPhaseAMR.jl

This package is being developed as a sequence of tested vertical slices. The
public API covers the centred, single-phase foundation described in milestone
1 of `DEVELOPMENT.md` and the periodic uniform/mixed-level staggered operators
from milestone 2.

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
are allocation-free after setup. A separate `amr_interface_scheme` choice
defaults to upwind on non-regular connections, while the bulk scheme defaults
to Lax--Wendroff. See
[Staggered advection](staggered-advection.md) for the discrete construction,
the divergence-on-the-diagonal identity, and the interface stability results.

## Scope

The package is 2-D, CPU-only, serial, refinement-ratio-two, and globally
time-stepped. The compatible gradient supports periodic uniform meshes and
isotropic mixed-level meshes. Staggered advection supports 2:1-balanced
mixed-level, fully periodic meshes as well as equivalent uniform multi-block
partitions. Coarse--fine transfer is encoded directly by the single-owner
segment fractions in its immutable dual schedule; no duplicate coarse face
unknown is synchronized afterward. Physical-boundary dual volumes remain an
explicit gate. The literature and architecture documents state the
compatibility conditions for projection, VOF, and distributed execution.

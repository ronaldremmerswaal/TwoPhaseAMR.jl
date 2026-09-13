# TwoPhaseAMR.jl

This package is being developed as a sequence of tested vertical slices. The
public API currently covers the centred, single-phase foundation described in
milestone 1 of `DEVELOPMENT.md`.

## Data model

Each `RegionTrees.Cell` leaf represents one fixed-size Cartesian block. A
`CellField` owns one dense Julia `Matrix` per leaf, including a configurable
halo. Mesh construction turns the mutable tree into immutable leaf, face, and
halo schedules. Numerical kernels traverse those schedules and arrays only.

The face schedule subdivides a coarse face where necessary and gives each
resulting physical segment exactly one owner. Conservative updates add one
integrated flux to one cell and subtract the same value from its neighbor.

## Scope

The package is 2-D, CPU-only, serial, refinement-ratio-two, and globally
time-stepped. This is deliberate. The literature and architecture documents
state the compatibility conditions that must be met before staggered fields,
projection, VOF, or distributed execution can be added.

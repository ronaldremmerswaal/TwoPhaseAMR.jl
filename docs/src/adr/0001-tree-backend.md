# ADR 0001: use RegionTrees for the serial foundation

- Status: accepted for milestones 1--2; mandatory review before distributed
  execution
- Date: 2026-09-13

## Context

The numerical data model needs a refinement-ratio-two quadtree whose leaves
represent fixed-size Cartesian blocks. Tree traversal is allowed while a mesh
is built or regridded, but not inside stencil kernels. The initial target is a
serial 2-D CPU solver; 3-D and distributed execution come later.

## Decision

Use `RegionTrees.jl` as the hierarchy backend for the serial foundation. Keep
all solver-specific state out of its traversal API: after refinement, copy the
leaf geometry into compact `LeafBlock` records and construct face and halo
schedules. Add full 2:1 balancing as a refinement policy over the existing
tree, rather than creating another tree representation.

`RegionTrees.jl` and this package are MIT licensed. The backend is pure Julia,
small, generic in dimension, and permits arbitrary leaf payloads. It supplies
splitting, parent/child traversal, leaf lookup, and adaptive sampling. It does
not supply neighbor discovery, 2:1 balancing, ghost exchange, partitioning, or
distributed execution. Consequently:

- neighbors are resolved into schedules after refinement;
- full (face and corner) 2:1 balance is enforced locally at setup;
- ghost data currently means block-array halos within one process; and
- an MPI implementation must revisit this decision rather than growing an
  ad-hoc communication layer around `RegionTrees.jl`.

## Alternatives evaluated

### P4est.jl

`P4est.jl` is an MIT-licensed low-level wrapper around the GPL-2.0 `p4est` C
library. `p4est` directly supports parallel forests of quadtrees/octrees,
refinement/coarsening, 2:1 balancing, partitioning, neighbor iteration, and
ghost layers. It is the leading candidate when distributed execution or 3-D
becomes active. Its current Julia wrapper requires an MPI-enabled native
library even for workstation use, and using a system MPI requires matching
`p4est` and MPI builds. That native/MPI and GPL runtime dependency is excessive
for the initial serial API but may be justified later.

Sources: [P4est.jl](https://github.com/trixi-framework/P4est.jl),
[p4est](https://github.com/cburstedde/p4est).

### T8code.jl

`T8code.jl` is also an MIT-licensed low-level wrapper around a GPL-2.0 native
library. `t8code` supplies 2:1 balance, MPI partitioning, ghost elements, and a
broader set of element classes, including triangles, tetrahedra, prisms, and
pyramids. Those hybrid-element features are not needed by a Cartesian block
solver. It has the same early native/MPI cost and a larger scope than required;
it becomes attractive only if non-Cartesian macro elements become a goal.

Sources: [T8code.jl](https://github.com/DLR-AMR/T8code.jl),
[t8code](https://github.com/DLR-AMR/t8code).

## Consequences and migration gate

The solver API must depend on `BlockTreeMesh`, leaf geometry, and schedules,
not on concrete `RegionTrees.Cell` types. Before adding distributed execution,
prototype the same schedule builder with P4est and compare setup time, memory,
partition stability, ghost exchange, licensing implications for distribution,
and the cost of mandatory MPI initialization. If that prototype succeeds,
replace the backend without changing field or numerical-kernel APIs.

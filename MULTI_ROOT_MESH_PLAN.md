# Multi-root block-tree mesh implementation plan

## Goal

Decouple the physical base-grid resolution from the number of cells stored in
one block. A user specifies the physical domain extent, the global base mesh,
and the block size. The base mesh must be an integer multiple of the block size
in each direction.

The implementation uses a rectangular collection of permanent level-zero
quadtree roots. These base roots may be refined and coarsened back to level
zero, but they are never removed. Boundaries between base roots are ordinary
internal block boundaries.

## Public API

```julia
mesh = BlockTreeMesh(
    origin = (0.0, 0.0),
    extent = (1.0, 3.0),
    base_mesh = (32, 96),
    block_size = (8, 8),
    periodic = (true, false),
    max_level = 2,
    refine = refine_callback,
)
```

The inputs have the following meaning:

- `extent` is the physical domain size.
- `base_mesh` is the global number of level-zero cells.
- `block_size` is the number of cells stored in every leaf block.
- `max_level` is the maximum refinement level relative to `base_mesh`.

The constructor derives

```julia
base_blocks = base_mesh .÷ block_size
h0 = extent ./ base_mesh
```

For the example above, the base layout contains `4 x 12` permanent roots. A
cell at refinement level `level` has spacing `h0 ./ 2^level`.

Use `base_mesh=block_size` by default. Existing constructors then retain their
current one-root behavior.

Validate that:

- all entries of `base_mesh` are positive integers;
- `base_mesh[d] >= block_size[d]`;
- `base_mesh[d] % block_size[d] == 0` in both directions;
- the existing block-size, halo, extent, and refinement-level requirements
  hold.

Do not silently round a requested base mesh.

## Mesh invariants

The implementation must maintain the following invariants:

1. The base roots tile the complete domain without gaps or overlaps.
2. Base roots are permanent.
3. A base root may be refined and subsequently coarsened back to level zero.
4. A base root cannot be coarsened below level zero.
5. Every active block contains exactly `block_size` cells.
6. Every active block has a globally unique `BlockKey`.
7. Base-root boundaries are numerically indistinguishable from other internal
   block boundaries.
8. Periodicity applies to the complete physical domain, not to individual
   roots.
9. A numerical result is independent of the partition of a fixed base mesh
   into blocks, up to floating-point ordering effects.

## Root representation

Introduce a mesh-level container for the base roots, for example:

```julia
struct RootGrid{Tree,T}
    roots::Vector{Tree}
    layout::NTuple{2,Int}
    root_extent::NTuple{2,T}
end
```

Store roots in deterministic `j`-major and then `i`-major order. A base root at
logical position `(i, j)` receives

```julia
BlockKey(0, i, j)
```

The existing descendant addressing remains valid:

```julia
BlockKey(level + 1, 2i + child_i, 2j + child_j)
```

`RootGrid` is a mesh container and must not pretend to be a
`RegionTrees.Cell`. In particular, it does not have a numerical level or four
dyadic children. Add package-local operations for it:

```julia
_collect_leaves!(leaves, roots::RootGrid)
_findleaf(roots::RootGrid, point)
_collect_leaf_blocks_in_window!(..., roots::RootGrid, ...)
_balance!(roots::RootGrid, ...)
```

Each contained root remains an ordinary independent `RegionTrees.Cell`.

### Root lookup

Select the base root in constant time from the physical coordinate and
`root_extent`, after applying domain-periodic wrapping. Search only that
quadtree with `RegionTrees.findleaf`.

Window queries should compute the intersecting base-root index range and visit
only those roots. Do not introduce a global scan over all roots or leaves.

## Centralized lattice geometry

Replace local assumptions that the global lattice is
`block_size .* 2^level`. Introduce shared helpers such as:

```julia
cells_at_level(mesh, level) = mesh.base_mesh .* (1 << level)
block_scale(mesh, level) = 1 << (mesh.max_level - level)
```

Use these helpers in:

- face and boundary schedules;
- halo schedules;
- gradient schedules;
- AMR cell and face transfer;
- staggered diffusion schedules;
- multigrid cell and face transfer;
- periodic line searches;
- topology and schedule cache keys.

No operator should independently reproduce the global lattice-size formula.

## Milestone 1: construction and uniform topology

1. Add `base_mesh` to `BlockTreeMesh` and construct the `RootGrid`.
2. Create each root with its physical origin, physical extent, and global
   level-zero key.
3. Generalize leaf collection, point lookup, and window lookup.
4. Build global face, boundary, and halo schedules across all roots.
5. Change global lattice dimensions to `base_mesh .* 2^max_level`.

### Gate

The following meshes must represent the same uniform Cartesian grid:

```julia
BlockTreeMesh(extent=(1.0, 3.0), block_size=(32, 96))

BlockTreeMesh(
    extent=(1.0, 3.0),
    base_mesh=(32, 96),
    block_size=(8, 8),
)
```

After ordering by physical coordinates, compare:

- cell centers and volumes;
- interior and boundary faces;
- face centers and areas;
- halo donors and weights;
- periodic connectivity.

Test base-root layouts `(1,1)`, `(1,3)`, `(3,1)`, and `(2,3)`, with periodic
and physical boundaries in each direction.

## Milestone 2: AMR and balancing

1. Apply the existing refinement callback independently to every root.
2. Collect all active nodes from the root grid.
3. Generalize adaptation rebuilds to preserve every base root.
4. Prevent parent lookup and coarsening below level zero.
5. Balance across base-root boundaries, including periodic boundaries.

Balancing should use neighbor queries through the root grid. Starting from the
blocks changed by refinement or coarsening, query adjacent leaves across faces
and corners and refine a coarse neighbor when the level difference exceeds
one. Avoid the existing all-pairs leaf comparison in the new implementation.

### Gate

Refine and coarsen immediately beside:

- a horizontal base-root boundary;
- a vertical base-root boundary;
- a four-root corner;
- a periodic domain boundary.

Verify 2:1 balance, deterministic block keys, and that distant roots retain
their topology.

## Milestone 3: restriction and prolongation

The synthetic root never participates in numerical transfer. Restriction and
prolongation continue to operate on actual four-child adaptive families.

Update:

- old and new block maps;
- point-to-block lookup;
- face-transfer lattice coordinates;
- topology matching and reuse checks;
- adaptation workspace reconstruction.

### Gate

Across base-root and coarse-fine boundaries, verify:

- exact preservation of constants;
- conservative cell restriction;
- conservative staggered-face transfer;
- velocity-field accuracy;
- geometric interface restriction and prolongation;
- geometry fractions and face apertures.

## Milestone 4: geometric multigrid

Preserve `base_mesh`, `block_size`, and the root layout in capped and pruned
multigrid meshes. The adaptive hierarchy terminates at the complete base mesh,
whose roots remain connected through the global face schedules.

Update:

- ancestor discovery;
- capped and all-leaf-pruned mesh construction;
- hierarchy topology matching;
- cell prolongation;
- face-coefficient restriction;
- reusable hierarchy matching;
- pressure-component and gauge detection.

The number of cells in the coarsest system remains the number of cells in
`base_mesh`; changing the storage partition must not change the mathematical
coarse operator.

### Gate

For one-block and multi-block partitions of the same base mesh, compare:

- Poisson operator action;
- gradient and divergence;
- GMG residual histories and iteration counts;
- converged pressure modulo its null-space constant;
- constant- and variable-coefficient diffusion results;
- hierarchy and workspace reuse after adaptation.

## Milestone 5: operator partition invariance

Compare, for example,

```julia
base_mesh=(24, 24), block_size=(24, 24)
```

with

```julia
base_mesh=(24, 24), block_size=(8, 8)
```

Map cells and faces by physical location rather than comparing raw vector
ordering. Cover:

- centered scalar advection;
- staggered advection;
- WY interface advection;
- gradient and divergence;
- single- and two-phase diffusion;
- pressure projection;
- mimetic projection;
- CMDGFM projection;
- surface tension and gravity;
- cut-cell geometry;
- height functions and curvature.

Algebraically equivalent operator results should agree to roundoff. Iterative
solver cycle counts should remain equal or differ only marginally.

## Milestone 6: interface and geometry seams

Test a phase interface and embedded geometry that:

- cross one base-root boundary;
- cross a four-root corner;
- lie directly on a root boundary;
- cross a coarse-fine boundary between roots;
- cross the periodic seam.

Verify:

- level-set volume-fraction initialization;
- PLIC reconstruction consistency;
- normal and curvature accuracy;
- volume conservation;
- phase-face fractions;
- geometry fractions and apertures;
- absence of force or reconstruction artifacts at root seams.

## Milestone 7: threading and performance

Benchmark the same global mesh with one large block and with smaller blocks:

- mesh construction;
- face and halo schedule construction;
- field kernels;
- interface reconstruction and height functions;
- centered and staggered advection;
- diffusion assembly and solve;
- Poisson assembly and solve;
- mesh adaptation;
- restriction and prolongation.

Acceptance criteria:

- no material serial regression for the same global cell count;
- improved useful threading for formerly single-block uniform meshes;
- topology construction scales with blocks and local neighbors;
- no new global leaf searches in performance-critical schedule builders.

## Milestone 8: examples and integration tests

After the partition-invariance gates pass, migrate the capillary-wave and
two-velocity-instability examples to `8 x 8` blocks.

Uniform example:

```julia
BlockTreeMesh(
    extent=(1.0, 3.0),
    base_mesh=(32, 96),
    block_size=(8, 8),
    periodic=(true, false),
)
```

Adaptive example:

```julia
BlockTreeMesh(
    extent=(1.0, 3.0),
    base_mesh=(8, 24),
    block_size=(8, 8),
    max_level=2,
    refine=refine_callback,
)
```

Inputs such as `cells_per_wavelength` should determine `base_mesh` explicitly.
Reject resolutions that are not divisible by the selected block size.

Run the wave integration tests on uniform, fixed mixed-level, and adaptive
meshes.

## Compatibility

- `base_mesh=block_size` preserves the existing constructor behavior.
- `refine(key, origin, extent)` retains its current signature.
- `max_level` and `MeshAdaptationOptions.minimum_level` remain relative to the
  base mesh.
- Field storage continues to use one array per active block.
- Existing numerical kernels continue to consume materialized cell, face,
  halo, and transfer schedules rather than traversing the root grid.
- No cut-cell geometry is introduced merely to shape the computational domain.

## Completion criteria

The work is complete when:

1. Existing one-root constructors and tests continue to pass.
2. Arbitrary rectangular base-root layouts work.
3. AMR maintains 2:1 balance across base-root and periodic boundaries.
4. Transfer remains conservative across root boundaries.
5. Operators are partition invariant.
6. GMG convergence and reuse are unaffected by block partitioning.
7. Interfaces and embedded geometries cross root seams without artifacts.
8. Serial and threaded test suites pass.
9. Wave examples use `8 x 8` blocks without coupling domain dimensions to the
   block size.

Estimated implementation and validation effort: four to seven focused days.

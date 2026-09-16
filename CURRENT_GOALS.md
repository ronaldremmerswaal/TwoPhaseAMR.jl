Recommended direction
1. Remove the assembled matrix only on the finest multigrid level
Keep explicit matrices on coarse levels, where they are small and useful.
For the finest level:
- smooth using the factorized pressure or diffusion action;
- compute the Jacobi diagonal directly:
  - pressure diagonal from the incidence/metric contributions;
  - diffusion diagonal from \(\sum_s w_s E_{si}^2\);
- replace the matrix-based residual-floor calculation with an operator-derived bound;
- construct pressure’s smoothed transfer from local stencil contributions rather than an assembled fine stiffness matrix.
This would eliminate the full fine pressure matrix and both the fine diffusion stiffness and shifted matrix from the normal solver path. It would not eliminate construction of \(E\), pressure metric connectivity, or advection schedules.
2. Introduce persistent block-local topology objects
The larger improvement is to exploit the locality of tree adaptation:
- Assign stable storage slots using BlockKey.
- Cache the standard 8×8 block-interior schedules.
- Cache AMR interface patches by their local 2:1 neighbour configuration.
- Cache variable-diffusion vertex patches by their local tree-neighbour signature.
- After adaptation, compare old and new block keys and neighbour signatures.
- Rebuild only added, removed, and adjacent patches.
- Retain every unchanged block’s pressure, diffusion, VOF, and momentum connectivity.
This avoids constructing global sparse graphs from scratch. Numerical coefficients can still be refreshed every step without repeating geometric searches.
3. Make the multigrid hierarchy incrementally local
The current hierarchy reuses a suffix only after an entire coarse mesh matches. A better tree-native representation would retain unchanged branches within every level and rebuild only ancestors of changed leaves.
That is the multigrid equivalent of the block-local fine operator and should produce much better reuse than caching whole topologies. Whole-topology caching performed poorly because 66 of 72 visited Stokes-3 topologies were unique.
4. Reuse Krylov storage by capacity
Krylov vectors can use grow-only capacity with an active range, subject to how much adaptation Krylov.jl allows. This should reduce allocation but is unlikely to remove most rebuild time. Replacing Krylov with standalone multigrid would not eliminate pressure, diffusion, or AMR connectivity construction and would sacrifice the robustness Krylov currently provides at high coefficient ratios.
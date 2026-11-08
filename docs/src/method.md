# Method reference

The [technical reference PDF](assets/techref/TwoPhaseAMR-techref.pdf) is a
chapter-by-chapter map of the numerical methods and their differences from the
thesis. It is a skeleton: detailed derivations and verification results are not
yet supplied. For the background and thesis notation, see Ronald A.
Remmerswaal, *Numerical Modelling of Variability in Liquid Impacts* (2023),
[doi:10.33612/diss.562133319](https://doi.org/10.33612/diss.562133319).

For a local documentation build, first build the PDF in `docs/techref/` with
`latexmk -pdf -interaction=nonstopmode -halt-on-error techref.tex`.
`docs/make.jl` copies the resulting PDF into the site when it exists. In CI the
PDF is built and downloaded before the site is generated.

| Topic | PDF chapter | Main API |
| --- | --- | --- |
| One- and two-velocity formulations | [1. Overview](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:overview) | [`TwoPhase`](@ref), [`OneVelocityModel`](@ref), [`TwoVelocityModel`](@ref) |
| Time integration and stage ordering | [1. Overview](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:overview) | [`NavierStokesOptions`](@ref), [`step!`](@ref), [`simulate!`](@ref) |
| Block tree, 2:1 grading, schedules and halos | [2. Mesh and AMR](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:mesh-amr) | [`BlockTreeMesh`](@ref), [`MeshAdaptationOptions`](@ref), [`fill_halo!`](@ref) |
| Discrete spaces, face and cell operators | [3. Discrete setting](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:discrete-setting) | [`CellField`](@ref), [`FaceField`](@ref), [`FaceVelocity`](@ref), [`divergence!`](@ref), [`GradientOperator`](@ref) |
| Interface initialization and reconstruction | [4. Interface](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:interface) | [`InterfaceField`](@ref), [`PolygonIndicator`](@ref), [`reconstruct!`](@ref) |
| Apertures, curvature and contact angles | [4. Interface](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:interface) | [`InterfaceField`](@ref), [`GeometryField`](@ref), [`reconstruct!`](@ref) |
| Geometric VOF advection, CFL and snapping | [5. VOF transport](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:vof-transport) | [`WYAdvectionWorkspace`](@ref), [`advect!`](@ref), [`stable_timestep`](@ref) |
| Two-fluid mass transport and interface carriers | [5. VOF transport](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:vof-transport) | [`FrozenInterfaceFluxes`](@ref), [`ProjectedLiquidVelocity`](@ref), [`PhaseScalar`](@ref) |
| Staggered and phase momentum transport, limiters | [6. Momentum transport](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:momentum-transport) | [`StaggeredVelocityAdvectionWorkspace`](@ref), [`MomentumAdvectionWorkspace`](@ref), [`MonotonizedCentral`](@ref), [`convection!`](@ref) |
| Embedded geometry and cut cells | [7. Cut cells](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:cutcells) | [`GeometryField`](@ref), [`PLICSolidGeometry`](@ref), [`enforce_geometry_no_slip!`](@ref) |
| Surface tension and balanced force | [8. Forces](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:forces) | [`SurfaceTension`](@ref), [`CSF`](@ref), [`CGFM`](@ref), [`surface_tension!`](@ref) |
| Gravity | [8. Forces](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:forces) | [`TwoPhase`](@ref), [`gravity!`](@ref) |
| Single- and two-phase viscosity | [9. Viscosity](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:viscosity) | [`StaggeredDiffusionOperator`](@ref), [`VariableStaggeredDiffusionOperator`](@ref), [`implicit_midpoint_diffusion!`](@ref) |
| Pressure projection and velocity boundaries | [10. Projection](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:projection) | [`ProjectionWorkspace`](@ref), [`project!`](@ref), [`VelocityBoundaryConditions`](@ref), [`Inflow`](@ref), [`Outflow`](@ref) |
| CMDGFM projection and jump interpolant | [10. Projection](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:projection) | [`CMDGFMProjection`](@ref), [`cmdgfm_jump_interpolant`](@ref), [`cmdgfm_gradient!`](@ref) |
| Mimetic projection | [10. Projection](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:projection) | [`MimeticProjection`](@ref), [`mimetic_gradient!`](@ref) |
| Pressure gradient and elliptic operators | [11. Solvers](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:solvers) | [`GradientOperator`](@ref), [`EllipticOperator`](@ref), [`PoissonOperator`](@ref), [`weighted_gradient!`](@ref) |
| Tree multigrid, Krylov solvers and tolerances | [11. Solvers](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:solvers) | [`EllipticSolver`](@ref), [`StaggeredEllipticSolver`](@ref), [`LinearSolverOptions`](@ref), [`MultigridOptions`](@ref) |
| AMR refinement criteria | [12. AMR criteria and transfer](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:amr-criteria-transfer) | [`InterfaceRefinementIndicator`](@ref), [`CurvatureRefinementIndicator`](@ref), [`VorticityRefinementIndicator`](@ref), [`GeometryRefinementIndicator`](@ref), [`InterpolationErrorIndicator`](@ref) |
| Conservative transfer and divergence preservation | [12. AMR criteria and transfer](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:amr-criteria-transfer) | [`adapt!`](@ref), [`prolong!`](@ref), [`restrict!`](@ref) |
| Time-step limits, stability and accuracy | [13. Stability and accuracy](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:stability-accuracy) | [`stable_timestep`](@ref), [`AdvectionOptions`](@ref) |
| Verification and conservation diagnostics | [14. Verification](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:verification) | [`DivergenceNorm`](@ref), [`PhaseMass`](@ref), [`KineticEnergy`](@ref) |

Implementation topics remain on the site: see [shared-memory
parallelism](shared-memory-parallelism.md), [architecture](architecture.md),
and [testing and validation](testing.md).

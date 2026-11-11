# Method reference

The [technical reference PDF](assets/techref/TwoPhaseAMR-techref.pdf) states the
implemented equations chapter by chapter, with boxed differences from the
thesis. All fourteen chapters are written, each audited against the commit
stated on its first page: Chapter 1 gives the governing equations, the stages
of a time step, the method options and the known limitations, Chapters 2 to 13
the discretisation, and Chapter 14 the verification. For the
background and thesis notation, see Ronald A. Remmerswaal, *Numerical Modelling of
Variability in Liquid Impacts* (2023),
[doi:10.33612/diss.562133319](https://doi.org/10.33612/diss.562133319).

For a local documentation build, first build the PDF in `docs/techref/` with
`latexmk -pdf -interaction=nonstopmode -halt-on-error techref.tex`.
`docs/make.jl` copies the resulting PDF into the site when it exists. In CI the
PDF is built and downloaded before the site is generated.

| Topic | PDF chapter | Main API |
| --- | --- | --- |
| One- and two-velocity formulations | [1.2 Governing equations](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=sec:overview:models) | [`TwoPhase`](@ref), [`OneVelocityModel`](@ref), [`TwoVelocityModel`](@ref) |
| Time integration and stage ordering | [1.4 One time step](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=sec:overview:step) | [`NavierStokesOptions`](@ref), [`step!`](@ref), [`simulate!`](@ref) |
| Block forest, levels and 2:1 grading | [2.2 Forest, blocks and levels](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=sec:mesh:forest) | [`BlockTreeMesh`](@ref), [`BlockKey`](@ref), [`isbalanced`](@ref) |
| Faces, hanging segments and stored unknowns | [2.3 Faces](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=sec:mesh:faces) | [`FaceField`](@ref), [`FaceVelocity`](@ref), [`CellField`](@ref) |
| Halo slots and boundary rules | [2.4 Halo slots](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=sec:mesh:halo) | [`fill_halo!`](@ref), [`ZeroGradient`](@ref), [`Dirichlet`](@ref) |
| Regrid and schedule rebuild | [2.5 Regrid](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=sec:mesh:regrid) | [`adapt!`](@ref), [`MeshAdaptationOptions`](@ref) |
| Dual volumes and open (cut-cell) measures | [3.2–3.3 Measures](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=sec:setting:measures) | [`GeometryField`](@ref), [`refresh_geometry!`](@ref) |
| Divergence and the hanging-face sample | [3.5 Divergence](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=sec:setting:divergence) | [`divergence!`](@ref) |
| Phase fractions and PLIC reconstruction | [4.2–4.3 Fractions and PLIC](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=sec:interface:fractions) | [`InterfaceField`](@ref), [`reconstruct!`](@ref), [`plic_segments`](@ref) |
| Normals, height functions and curvature | [4.4–4.6 Heights, normals, curvature](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=sec:interface:normals) | [`InterfaceField`](@ref), [`interface_height`](@ref) |
| Interface initialization | [4.9 Initialisation](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=sec:interface:init) | [`initialize!`](@ref), [`PolygonIndicator`](@ref) |
| Contact angles and phase apertures | [4.7–4.8 Contact angles, apertures](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=sec:interface:contact) | [`InterfaceField`](@ref), [`GeometryField`](@ref) |
| Directional WY sweeps, donating regions, bounds and CFL | [5.3–5.6 Sweeps, donating regions, bounds, time step](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=sec:vof:sweeps) | [`WYAdvectionWorkspace`](@ref), [`advect!`](@ref), [`stable_timestep`](@ref) |
| Phase scalars, two-velocity carriers and conservation | [5.7–5.9 Phase scalars, carriers, balances](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=sec:vof:scalars) | [`PhaseScalar`](@ref), [`phase_flux_diagnostics`](@ref) |
| Dual control volumes, primal--dual fluxes and face states (limiters, AMR interface) | [6.2–6.3 Dual fluxes and face states](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=sec:momentum:dual) | [`StaggeredVelocityAdvectionWorkspace`](@ref), [`MonotonizedCentral`](@ref), [`default_amr_interface_scheme`](@ref) |
| Staggered convection, conservation, kinetic energy and time step | [6.4–6.6 Convection, conservation, CFL](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=sec:momentum:single) | [`convection!`](@ref), [`advect!`](@ref), [`staggered_inner_product`](@ref), [`stable_timestep`](@ref) |
| One- and two-velocity phase momentum and staggered phase scalars | [6.7–6.8 Phase momentum](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=sec:momentum:one-velocity) | [`MomentumAdvectionWorkspace`](@ref), [`PhaseFaceScalar`](@ref), [`mixture_density`](@ref) |
| Embedded solid, open polygons and apertures | [7.2–7.4 Cut-cell geometry](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=sec:cutcells:solid) | [`GeometryField`](@ref), [`refresh_geometry!`](@ref), [`PLICSolidGeometry`](@ref) |
| Cut-cell velocity closure and regrid | [7.5–7.6 Closure and regrid](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=sec:cutcells:operators) | [`enforce_geometry_no_slip!`](@ref), [`adapt!`](@ref) |
| Surface tension and balanced force | [8. Forces](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:forces) | [`SurfaceTension`](@ref), [`CSF`](@ref), [`CGFM`](@ref), [`surface_tension!`](@ref) |
| Gravity | [8. Forces](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:forces) | [`TwoPhase`](@ref), [`gravity!`](@ref) |
| Single- and two-phase viscosity | [9. Viscosity](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:viscosity) | [`StaggeredDiffusionOperator`](@ref), [`VariableStaggeredDiffusionOperator`](@ref), [`implicit_midpoint_diffusion!`](@ref) |
| Pressure projection and velocity boundaries | [10. Projection](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:projection) | [`ProjectionWorkspace`](@ref), [`project!`](@ref), [`VelocityBoundaryConditions`](@ref), [`Inflow`](@ref), [`Outflow`](@ref) |
| CMDGFM projection and jump interpolant | [10. Projection](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:projection) | [`CMDGFMProjection`](@ref), [`cmdgfm_jump_interpolant`](@ref), [`cmdgfm_gradient!`](@ref) |
| Pressure gradient and elliptic operators | [3.6–3.7 Gradient and elliptic operator](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=sec:setting:gradient) | [`GradientOperator`](@ref), [`EllipticOperator`](@ref), [`PoissonOperator`](@ref), [`weighted_gradient!`](@ref) |
| Tree multigrid, Krylov solvers and tolerances | [11. Solvers](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:solvers) | [`EllipticSolver`](@ref), [`StaggeredEllipticSolver`](@ref), [`LinearSolverOptions`](@ref), [`MultigridOptions`](@ref) |
| AMR refinement criteria | [12. AMR criteria and transfer](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:amr-criteria-transfer) | [`InterfaceRefinementIndicator`](@ref), [`CurvatureRefinementIndicator`](@ref), [`VorticityRefinementIndicator`](@ref), [`GeometryRefinementIndicator`](@ref), [`InterpolationErrorIndicator`](@ref) |
| Conservative transfer and divergence preservation | [12. AMR criteria and transfer](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:amr-criteria-transfer) | [`adapt!`](@ref), [`prolong!`](@ref), [`restrict!`](@ref) |
| Time-step limits, stability and accuracy | [13. Stability and accuracy](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:stability-accuracy) | [`stable_timestep`](@ref), [`AdvectionOptions`](@ref) |
| Verification and conservation diagnostics | [14. Verification](assets/techref/TwoPhaseAMR-techref.pdf#nameddest=ch:verification) | [`DivergenceNorm`](@ref), [`PhaseMass`](@ref), [`KineticEnergy`](@ref) |

Implementation topics remain on the site: see [shared-memory
parallelism](shared-memory-parallelism.md), [architecture](architecture.md),
and [testing and validation](testing.md).

"""
    NavierStokesSystem(mesh, physics; options=NavierStokesOptions(),
        adaptivity=Adaptivity(), boundary=NoSlip(), geometry=nothing,
        two_velocity_projection=CMDGFMProjection())
    NavierStokesSystem(mesh; density, dynamic_viscosity,
        boundary=NoSlip(), scheme=LaxWendroff(),
        amr_interface_scheme=default_amr_interface_scheme(scheme),
        multigrid=MultigridOptions(),
        pressure_krylov=nothing,
        diffusion_krylov=nothing,
        pressure_krylov_options=(;), diffusion_krylov_options=(;),
        pressure_preconditioner=:auto,
        diffusion_preconditioner=:auto,
        two_velocity_projection=CMDGFMProjection(),
        refinement_indicators=(), adaptive_fields=())

Incompressible Navier--Stokes discretization. The typed constructor retains
the supplied [`SinglePhase`](@ref) or [`TwoPhase`](@ref) physics and
[`NavierStokesOptions`](@ref) as `system.physics` and `system.options`.
The keyword constructor accepts the corresponding material and numerical
choices directly; supplying `interface` selects two-phase coefficients and
compatible geometric transport. Both configurations run the same staged
`step!` procedure. One time step is split into:

1. a conservative staggered convection update (Lax--Wendroff by default and
   optionally using a three-point limited AMR-interface flux), applied as
   sequential coordinate sweeps whose order alternates between time steps,
2. the optional user body force followed by a Crank--Nicolson/implicit-midpoint
   staggered viscous solve,
3. optional phase forces (surface tension and gravity), and
4. one compatible pressure projection.

`density` and `dynamic_viscosity` are material properties in both modes. For
single-phase flow they are scalars; the diffusion operator uses the derived
kinematic viscosity `dynamic_viscosity / density`, while the pressure
projection uses the inverse density so that the stored pressure has physical
units. For two phases they are `(liquid, gas)` pairs (a scalar dynamic
viscosity is also accepted when both phases match). The coupled diffusion
operator evaluates the volume-fraction-weighted geometric mean of dynamic
viscosity at each normal- and shear-stress location.

The tree may contain static or dynamically adapted local refinement. Optional
`refinement_indicators` are evaluated after completed steps according to the
mesh's `MeshAdaptationOptions`; `velocity`, `pressure`, indicator fields, and
the additional `adaptive_fields` are transferred in place before
topology-dependent operators are rebuilt. The stored pressure is the latest
projection pressure. Automatic Krylov selection uses PCG for symmetric
positive (semi-)definite systems and BiCGSTAB otherwise. Pressure and both
the single-phase and coupled two-phase viscous systems use PCG, including
on composite meshes. Explicit choices take precedence. The two preconditioners
are independent. Both configurations use
rediscretized tree-geometric hierarchies by default; CG requires equal pre-
and post-sweep counts for these hierarchies.

For a liquid/gas velocity tuple, `two_velocity_projection` selects the
physical pressure-gradient model. `CMDGFMProjection()` is the nonsymmetric
conservative multidimensional ghost-fluid system and BiCGSTAB solve; it is the
only supported projection.
"""
mutable struct NavierStokesSystem{T, M <: BlockTreeMesh, O <: NavierStokesOptions, A <: Adaptivity, D, B, G, TP}
    mesh::M
    # Physics is retained for inspection and independent reconstruction; the
    # normalized single-phase density is cached for numerical kernels.
    physics::Union{SinglePhase, TwoPhase}
    single_phase_density::T
    options::O
    adaptivity::A
    kinematic_viscosity::Union{Nothing, T}
    diffusion::D
    boundary::B
    # The compatible face metric can change concrete type after adaptation.
    pressure_solver::EllipticSolver{T}
    # The phase model remains concrete as its interface evolves.
    two_phase::TP
    geometry::G
end

@inline _pressure_krylov(system::NavierStokesSystem) =
    isnothing(system.options.pressure.krylov) ?
    Krylov.CgWorkspace : system.options.pressure.krylov
@inline _diffusion_krylov(system::NavierStokesSystem) =
    isnothing(system.options.diffusion.krylov) && !isnothing(system.two_phase) ?
    Krylov.CgWorkspace : system.options.diffusion.krylov
@inline _configured_preconditioner(specification) =
    specification === :auto ? :tree_geometric : specification
@inline _pressure_preconditioner(system::NavierStokesSystem) =
    _configured_preconditioner(system.options.pressure.preconditioner)
@inline _diffusion_preconditioner(system::NavierStokesSystem) =
    _configured_preconditioner(system.options.diffusion.preconditioner)
@inline function _pressure_preconditioner_options(system::NavierStokesSystem)
    options = system.options.pressure.preconditioner_options
    return !isnothing(system.geometry) &&
        _pressure_preconditioner(system) in (:geometric, :tree_geometric) ?
        merge(options, (; geometry = system.geometry)) : options
end
@inline _single_phase_density(system::NavierStokesSystem) =
    system.single_phase_density
@inline _multigrid_options(system::NavierStokesSystem) = system.options.multigrid


function NavierStokesSystem(
        mesh::M; density, dynamic_viscosity,
        boundary = NoSlip(), scheme = LaxWendroff(),
        amr_interface_scheme = default_amr_interface_scheme(scheme),
        multigrid::MultigridOptions = MultigridOptions(),
        coarse_size = multigrid.coarse_size,
        pre_sweeps = multigrid.pre_sweeps, post_sweeps = multigrid.post_sweeps,
        max_levels = multigrid.max_levels,
        pressure_krylov = nothing,
        diffusion_krylov = nothing,
        pressure_krylov_options = (;),
        diffusion_krylov_options = (;),
        pressure_preconditioner = :auto,
        diffusion_preconditioner = :auto,
        pressure_preconditioner_options = (;),
        diffusion_preconditioner_options = (;),
        refinement_indicators = (), adaptive_fields = (),
        geometry = nothing,
        interface = nothing, surface_tension = 0.0,
        surface_tension_formulation = CGFM(), gravity = (0.0, 0.0),
        phase_scalars = (),
        two_velocity_projection = CMDGFMProjection(),
        T = eltype(mesh.origin)
    ) where {M <: BlockTreeMesh}
    options = NavierStokesOptions(
        advection = AdvectionOptions(; scheme, amr_interface_scheme),
        pressure = LinearSolverOptions(
            krylov = pressure_krylov, krylov_options = pressure_krylov_options,
            preconditioner = pressure_preconditioner,
            preconditioner_options = pressure_preconditioner_options,
        ),
        diffusion = LinearSolverOptions(
            krylov = diffusion_krylov, krylov_options = diffusion_krylov_options,
            preconditioner = diffusion_preconditioner,
            preconditioner_options = diffusion_preconditioner_options,
        ),
        multigrid = MultigridOptions(; coarse_size, pre_sweeps, post_sweeps, max_levels),
    )
    physics = isnothing(interface) ?
        SinglePhase(; density, dynamic_viscosity) :
        TwoPhase(
            interface; density, dynamic_viscosity,
            surface_tension = SurfaceTension(
                surface_tension; method = surface_tension_formulation,
            ),
            gravity, scalars = phase_scalars,
        )
    adaptivity = Adaptivity(; criteria = refinement_indicators, fields = adaptive_fields)
    return NavierStokesSystem(
        mesh, physics; options, adaptivity, boundary, geometry,
        two_velocity_projection, T,
    )
end

function NavierStokesSystem(
        mesh::M, physics::Union{SinglePhase, TwoPhase};
        options::NavierStokesOptions = NavierStokesOptions(),
        adaptivity::Adaptivity = Adaptivity(),
        boundary = NoSlip(), geometry = nothing,
        two_velocity_projection = CMDGFMProjection(),
        T = eltype(mesh.origin),
    ) where {M <: BlockTreeMesh}
    _validate_linear_solver_symbols(
        options.pressure.krylov, options.pressure.preconditioner,
    )
    _validate_linear_solver_symbols(
        options.diffusion.krylov, options.diffusion.preconditioner,
    )
    isnothing(geometry) || (
        geometry isa GeometryField{T, M} && geometry.mesh === mesh
    ) || throw(
        ArgumentError("geometry must share solver mesh and scalar type"),
    )
    model, configured_density, configured_dynamic_viscosity = if physics isa SinglePhase
        rho = T(physics.density)
        mu = T(physics.dynamic_viscosity)
        isfinite(rho) && rho > zero(T) || throw(
            ArgumentError("density must be finite and strictly positive"),
        )
        isfinite(mu) && mu > zero(T) || throw(
            ArgumentError("dynamic_viscosity must be finite and strictly positive"),
        )
        nothing, rho, mu
    else
        interface = physics.interface
        interface isa InterfaceField{T, M} && interface.mesh === mesh ||
            throw(ArgumentError("interface must share solver mesh and scalar type"))
        phase_model = TwoPhaseModel(
            physics; pressure_projection = two_velocity_projection,
        )
        phase_model, phase_model.density, phase_model.dynamic_viscosity
    end
    pressure_krylov = options.pressure.krylov
    pressure_krylov_options = options.pressure.krylov_options
    pressure_preconditioner = options.pressure.preconditioner
    pressure_preconditioner_options = options.pressure.preconditioner_options
    multigrid = options.multigrid
    selected_pressure_krylov = isnothing(pressure_krylov) ?
        Krylov.CgWorkspace : pressure_krylov
    selected_pressure_preconditioner = pressure_preconditioner === :auto ?
        :tree_geometric :
        pressure_preconditioner
    configured_pressure_preconditioner_options =
        !isnothing(geometry) && selected_pressure_preconditioner in
        (:geometric, :tree_geometric) ?
        merge(pressure_preconditioner_options, (; geometry)) :
        pressure_preconditioner_options
    interface = isnothing(model) ? nothing : model.interface
    kinematic_viscosity, diffusion = if isnothing(model)
        kinematic_viscosity = configured_dynamic_viscosity / configured_density
        operator = isnothing(geometry) ? StaggeredDiffusionOperator(
                mesh; viscosity = kinematic_viscosity, boundary, T,
            ) : StaggeredDiffusionOperator(
                geometry; viscosity = kinematic_viscosity, boundary,
            )
        kinematic_viscosity, operator
    else
        # The coupled two-phase operator depends on the advected interface.
        # TwoPhaseFlowWorkspace owns it and refreshes its stress-location
        # dynamic viscosities before every viscous substep.
        nothing, nothing
    end
    pressure_solver = EllipticSolver(
        isnothing(model) ? (
                isnothing(geometry) ? PoissonOperator(
                    mesh; face_coefficient = inv(configured_density),
                    velocity_boundary = boundary, T,
                ) : PoissonOperator(
                    geometry; face_coefficient = inv(configured_density),
                    velocity_boundary = boundary,
                )
            ) : (
                isnothing(geometry) ? PoissonOperator(
                    interface; density = model.density,
                ) : PoissonOperator(
                    interface, geometry; density = model.density,
                )
            ); krylov = selected_pressure_krylov,
        krylov_options = pressure_krylov_options,
        preconditioner = selected_pressure_preconditioner,
        preconditioner_options = configured_pressure_preconditioner_options,
        multigrid,
    )
    all(indicator -> _indicator_field(indicator).mesh === mesh, adaptivity.criteria) ||
        throw(ArgumentError("refinement indicator fields must use solver mesh"))
    all(field -> field.mesh === mesh, adaptivity.fields) ||
        throw(ArgumentError("adaptive fields must use solver mesh"))
    return NavierStokesSystem{
        T, M, typeof(options), typeof(adaptivity), typeof(diffusion),
        typeof(boundary), typeof(geometry), typeof(model),
    }(
        mesh, physics, isnothing(model) ? configured_density : zero(T),
        options, adaptivity, kinematic_viscosity, diffusion,
        boundary, pressure_solver, model, geometry,
    )
end

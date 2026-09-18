"""
    IncompressibleNavierStokesSolver(mesh; density, dynamic_viscosity,
        boundary=NoSlip(), scheme=LW(),
        amr_interface_scheme=default_amr_interface_scheme(scheme),
        pressure_krylov=nothing,
        diffusion_krylov=nothing,
        pressure_krylov_options=(;), diffusion_krylov_options=(;),
        pressure_preconditioner=:auto,
        diffusion_preconditioner=:auto,
        refinement_indicators=(), adaptive_fields=())

Incompressible Navier--Stokes discretization. Supplying `interface` selects
the one-velocity two-phase coefficients and compatible geometric transport;
without it the original single-phase operators are used. Both configurations
run the same staged `step!` procedure. One time step is split into:

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
positive (semi-)definite systems and BiCGSTAB otherwise. Thus pressure and
coupled two-phase stress diffusion use PCG, while a corrected nonsymmetric
single-phase AMR diffusion operator uses BiCGSTAB. Explicit choices take
precedence. The two preconditioners are independent. Both configurations use
rediscretized tree-geometric hierarchies by default.
"""
mutable struct IncompressibleNavierStokesSolver{
        T, M <: BlockTreeMesh, R, MU, V, D, B, S, A, PK, DK, PKO, DKO,
        PP, DP, PO, QO, RI, AF, G,
    }
    mesh::M
    density::R
    dynamic_viscosity::MU
    kinematic_viscosity::V
    diffusion::D
    boundary::B
    # The compatible face metric is diagonal on a uniform mesh and sparse on
    # a mixed-level mesh. Dynamic adaptation may therefore change the
    # concrete EllipticSolver type while preserving its scalar type.
    pressure_solver::EllipticSolver{T}
    scheme::S
    amr_interface_scheme::A
    coarse_size::Int
    pre_sweeps::Int
    post_sweeps::Int
    max_levels::Int
    pressure_krylov::PK
    diffusion_krylov::DK
    pressure_krylov_options::PKO
    diffusion_krylov_options::DKO
    pressure_preconditioner::PP
    diffusion_preconditioner::DP
    pressure_preconditioner_options::PO
    diffusion_preconditioner_options::QO
    refinement_indicators::RI
    adaptive_fields::AF
    two_phase::Union{Nothing, TwoPhaseModel{T, M}}
    geometry::G
end

"""Concise alias for [`IncompressibleNavierStokesSolver`](@ref)."""
const NavierStokesSolver = IncompressibleNavierStokesSolver

function IncompressibleNavierStokesSolver(
        mesh::M; density, dynamic_viscosity,
        boundary = NoSlip(), scheme = LW(),
        amr_interface_scheme = default_amr_interface_scheme(scheme), coarse_size = 64,
        pre_sweeps = 2, post_sweeps = 2, max_levels = 32,
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
        surface_tension_formulation = CSF(), gravity = (0.0, 0.0),
        phase_scalars = (),
        interface_method = FrozenInterfaceFluxes(),
        T = eltype(mesh.origin)
    ) where {M <: BlockTreeMesh}
    isnothing(geometry) || (
        geometry isa GeometryField{T, M} && geometry.mesh === mesh
    ) || throw(
        ArgumentError("geometry must share solver mesh and scalar type"),
    )
    model, configured_density, configured_dynamic_viscosity = if isnothing(interface)
        density isa Number || throw(
            ArgumentError("single-phase density must be a scalar"),
        )
        dynamic_viscosity isa Number || throw(
            ArgumentError("single-phase dynamic_viscosity must be a scalar"),
        )
        rho = T(density)
        mu = T(dynamic_viscosity)
        isfinite(rho) && rho > zero(T) || throw(
            ArgumentError("density must be finite and strictly positive"),
        )
        isfinite(mu) && mu > zero(T) || throw(
            ArgumentError(
                "dynamic_viscosity must be finite and strictly positive",
            ),
        )
        nothing, rho, mu
    else
        interface isa InterfaceField{T, M} && interface.mesh === mesh ||
            throw(ArgumentError("interface must share solver mesh and scalar type"))
        phase_model = TwoPhaseModel(
            interface; density, dynamic_viscosity, surface_tension, gravity,
            formulation = surface_tension_formulation,
            scalars = phase_scalars, interface_method,
        )
        phase_model, phase_model.density, phase_model.dynamic_viscosity
    end
    selected_pressure_krylov = isnothing(pressure_krylov) ?
        Krylov.CgWorkspace : pressure_krylov
    # Keep `nothing` as the automatic-selection sentinel for single-phase
    # diffusion: AMR can change that operator from symmetric to nonsymmetric
    # at a later topology rebuild. Coupled two-phase diffusion always has the
    # SPD factorization M_rho + dt/2 E'W_mu E.
    selected_diffusion_krylov = isnothing(diffusion_krylov) &&
        !isnothing(model) ? Krylov.CgWorkspace : diffusion_krylov
    selected_pressure_preconditioner = pressure_preconditioner === :auto ?
        :tree_geometric :
        pressure_preconditioner
    selected_diffusion_preconditioner = diffusion_preconditioner === :auto ?
        :tree_geometric :
        diffusion_preconditioner
    configured_pressure_preconditioner_options =
        !isnothing(geometry) && selected_pressure_preconditioner in
        (:geometric, :tree_geometric) ?
        merge(pressure_preconditioner_options, (; geometry)) :
        pressure_preconditioner_options
    _validate_amr_interface_scheme(scheme, amr_interface_scheme)
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
                    mesh; face_coefficient = inv(configured_density), T,
                ) : PoissonOperator(
                    geometry; face_coefficient = inv(configured_density),
                )
            ) : (
                isnothing(geometry) ? PoissonOperator(
                    interface; density = model.density,
                    formulation = model.formulation,
                ) : PoissonOperator(
                    interface, geometry; density = model.density,
                    formulation = model.formulation,
                )
            ); krylov = selected_pressure_krylov,
        krylov_options = pressure_krylov_options,
        preconditioner = selected_pressure_preconditioner,
        preconditioner_options = configured_pressure_preconditioner_options,
        coarse_size, pre_sweeps, post_sweeps, max_levels
    )
    indicators = _as_configuration_tuple(refinement_indicators)
    all(indicator -> indicator isa AbstractRefinementIndicator, indicators) ||
        throw(
        ArgumentError(
            "refinement_indicators must contain refinement indicators"
        )
    )
    all(indicator -> _indicator_field(indicator).mesh === mesh, indicators) ||
        throw(ArgumentError("refinement indicator fields must use solver mesh"))
    fields = _as_configuration_tuple(adaptive_fields)
    all(field -> field isa AbstractMeshField, fields) ||
        throw(
        ArgumentError(
            "adaptive_fields must contain cell or face fields"
        )
    )
    all(field -> field.mesh === mesh, fields) ||
        throw(ArgumentError("adaptive fields must use solver mesh"))
    return IncompressibleNavierStokesSolver{
        T, M, typeof(configured_density), typeof(configured_dynamic_viscosity),
        typeof(kinematic_viscosity), typeof(diffusion), typeof(boundary),
        typeof(scheme), typeof(amr_interface_scheme),
        typeof(selected_pressure_krylov), typeof(selected_diffusion_krylov),
        typeof(pressure_krylov_options), typeof(diffusion_krylov_options),
        typeof(selected_pressure_preconditioner),
        typeof(selected_diffusion_preconditioner),
        typeof(configured_pressure_preconditioner_options),
        typeof(diffusion_preconditioner_options),
        typeof(indicators), typeof(fields), typeof(geometry),
    }(
        mesh, configured_density, configured_dynamic_viscosity,
        kinematic_viscosity, diffusion, boundary, pressure_solver,
        scheme, amr_interface_scheme, coarse_size, pre_sweeps,
        post_sweeps, max_levels, selected_pressure_krylov,
        selected_diffusion_krylov,
        pressure_krylov_options, diffusion_krylov_options,
        selected_pressure_preconditioner, selected_diffusion_preconditioner,
        configured_pressure_preconditioner_options,
        diffusion_preconditioner_options,
        indicators, fields, model, geometry
    )
end

@inline _as_configuration_tuple(::Nothing) = ()
@inline _as_configuration_tuple(value::Tuple) = value
@inline _as_configuration_tuple(value::AbstractVector) = Tuple(value)
@inline _as_configuration_tuple(value) = (value,)

mutable struct NavierStokesWorkspace{T, M <: BlockTreeMesh}
    old_velocity::FaceVelocity{T, M}
    convected::FaceVelocity{T, M}
    diffusion_work::FaceVelocity{T, M}
    # Uniform and mixed-level meshes use different schedule/metric concrete
    # types. These partial types let a single workspace survive a regrid.
    advection::Union{Nothing, StaggeredVelocityAdvectionWorkspace{T, M}}
    projection::ProjectionWorkspace{T, M}
    elliptic::Union{Nothing, StaggeredEllipticSolver{T}}
    two_phase::Union{Nothing, TwoPhaseFlowWorkspace{T, M}}
end

function _replace_workspace!(
        destination::NavierStokesWorkspace,
        source::NavierStokesWorkspace
    )
    destination.old_velocity = source.old_velocity
    destination.convected = source.convected
    destination.diffusion_work = source.diffusion_work
    destination.advection = source.advection
    destination.projection = source.projection
    destination.elliptic = source.elliptic
    destination.two_phase = source.two_phase
    return destination
end

function _navier_stokes_configuration(
        solver::IncompressibleNavierStokesSolver{T};
        refinement_indicators = solver.refinement_indicators,
        adaptive_fields = solver.adaptive_fields
    ) where {T}
    return (;
        density = solver.density,
        dynamic_viscosity = solver.dynamic_viscosity,
        geometry = solver.geometry,
        interface = isnothing(solver.two_phase) ? nothing : solver.two_phase.interface,
        surface_tension = isnothing(solver.two_phase) ? 0.0 : solver.two_phase.sigma,
        surface_tension_formulation = isnothing(solver.two_phase) ? CSF() : solver.two_phase.formulation,
        gravity = isnothing(solver.two_phase) ? (0.0, 0.0) : solver.two_phase.gravity,
        phase_scalars = isnothing(solver.two_phase) ? () : solver.two_phase.scalars,
        interface_method = isnothing(solver.two_phase) ?
            FrozenInterfaceFluxes() : solver.two_phase.interface_method,
        boundary = solver.boundary,
        scheme = solver.scheme,
        amr_interface_scheme = solver.amr_interface_scheme,
        coarse_size = solver.coarse_size,
        pre_sweeps = solver.pre_sweeps,
        post_sweeps = solver.post_sweeps,
        max_levels = solver.max_levels,
        pressure_krylov = solver.pressure_krylov,
        diffusion_krylov = solver.diffusion_krylov,
        pressure_krylov_options = solver.pressure_krylov_options,
        diffusion_krylov_options = solver.diffusion_krylov_options,
        pressure_preconditioner = solver.pressure_preconditioner,
        diffusion_preconditioner = solver.diffusion_preconditioner,
        pressure_preconditioner_options =
            solver.pressure_preconditioner_options,
        diffusion_preconditioner_options =
            solver.diffusion_preconditioner_options,
        refinement_indicators,
        adaptive_fields,
        T,
    )
end

function _rebuild_topology_dependent_state!(
        solver::IncompressibleNavierStokesSolver{T},
        workspace::NavierStokesWorkspace; dt = nothing
    ) where {T}
    if !isnothing(solver.two_phase)
        return _rebuild_two_phase_topology_dependent_state!(
            solver, workspace; dt,
        )
    end
    old_pressure_solver = solver.pressure_solver
    geometry = solver.geometry
    !isnothing(geometry) && refresh_geometry!(geometry)
    pressure_operator = isnothing(geometry) ? PoissonOperator(
            solver.mesh; T, face_coefficient = inv(solver.density),
            construction_cache =
            old_pressure_solver.operator.gradient_operator.construction_cache,
        ) : PoissonOperator(
            geometry; face_coefficient = inv(solver.density),
            construction_cache =
            old_pressure_solver.operator.gradient_operator.construction_cache,
        )
    old_pressure_preconditioner = old_pressure_solver.preconditioner
    pressure_preconditioner = solver.pressure_preconditioner
    pressure_preconditioner_options = solver.pressure_preconditioner_options
    if pressure_preconditioner in (:geometric, :tree_geometric) &&
            old_pressure_preconditioner isa
            TreeGeometricMultigridPreconditioner
        settings = merge(
            (;
                coarse_size = solver.coarse_size,
                pre_sweeps = solver.pre_sweeps,
                post_sweeps = solver.post_sweeps,
                max_levels = solver.max_levels,
            ),
            pressure_preconditioner_options,
        )
        pressure_preconditioner = TreeGeometricMultigridPreconditioner(
            pressure_operator; settings...,
            reuse = old_pressure_preconditioner,
        )
        pressure_preconditioner_options = (;)
    end
    reconfigured_pressure = pressure_preconditioner isa
        TreeGeometricMultigridPreconditioner ?
        _reconfigure_elliptic_solver!(
            old_pressure_solver, pressure_operator,
            solver.pressure_krylov, pressure_preconditioner,
        ) : nothing
    solver.pressure_solver = isnothing(reconfigured_pressure) ?
        EllipticSolver(
            pressure_operator;
            krylov = solver.pressure_krylov,
            krylov_options = solver.pressure_krylov_options,
            preconditioner = pressure_preconditioner,
            preconditioner_options = pressure_preconditioner_options,
            coarse_size = solver.coarse_size,
            pre_sweeps = solver.pre_sweeps,
            post_sweeps = solver.post_sweeps,
            max_levels = solver.max_levels,
        ) : reconfigured_pressure

    old_elliptic = workspace.elliptic
    isnothing(old_elliptic) && throw(
        ArgumentError("single-phase diffusion workspace is required"),
    )
    solver.diffusion = isnothing(geometry) ? StaggeredDiffusionOperator(
            solver.mesh; viscosity = solver.kinematic_viscosity,
            boundary = solver.boundary, T,
        ) : StaggeredDiffusionOperator(
            geometry; viscosity = solver.kinematic_viscosity,
            boundary = solver.boundary,
        )
    _resize_zero_velocity!(workspace.old_velocity, solver.mesh)
    _resize_zero_velocity!(workspace.convected, solver.mesh)
    _resize_zero_velocity!(workspace.diffusion_work, solver.mesh)
    workspace.advection = isnothing(geometry) ?
        StaggeredVelocityAdvectionWorkspace(workspace.old_velocity) :
        StaggeredVelocityAdvectionWorkspace(workspace.old_velocity, geometry)
    workspace.projection = ProjectionWorkspace(solver.pressure_solver)
    reuse_diffusion = if solver.diffusion_preconditioner in
            (:geometric, :tree_geometric) &&
            old_elliptic.x.preconditioner isa
            TreeStaggeredMultigridComponent &&
            old_elliptic.y.preconditioner isa
            TreeStaggeredMultigridComponent
        TreeStaggeredGeometricMultigridPreconditioner(
            old_elliptic.operator,
            old_elliptic.x.preconditioner,
            old_elliptic.y.preconditioner,
        )
    else
        nothing
    end
    diffusion = isnothing(dt) ? zero(T) :
        _positive_finite_step(
            T, dt, "workspace dt",
        ) * solver.kinematic_viscosity / 2
    workspace.elliptic = _navier_stokes_elliptic_solver(
        solver, diffusion; reuse_preconditioner = reuse_diffusion,
    )
    workspace.two_phase = nothing
    return workspace
end

function _resize_zero_velocity!(velocity::FaceVelocity, mesh)
    resize!(velocity.interior, length(mesh.faces))
    resize!(velocity.boundary, length(mesh.boundary_faces))
    fill!(velocity.interior, zero(eltype(velocity)))
    fill!(velocity.boundary, zero(eltype(velocity)))
    return velocity
end

function _rebuild_two_phase_topology_dependent_state!(
        solver::IncompressibleNavierStokesSolver{T},
        workspace::NavierStokesWorkspace{T}; dt = nothing
    ) where {T}
    model = solver.two_phase
    isnothing(model) && throw(ArgumentError("two-phase model is required"))
    old_phase_workspace = workspace.two_phase
    isnothing(old_phase_workspace) && throw(
        ArgumentError("two-phase workspace is required"),
    )
    old_pressure_solver = solver.pressure_solver
    geometry = solver.geometry
    !isnothing(geometry) && refresh_geometry!(geometry)
    pressure_operator = isnothing(geometry) ? PoissonOperator(
            model.interface; density = model.density,
            formulation = model.formulation,
            construction_cache =
            old_pressure_solver.operator.gradient_operator.construction_cache,
        ) : PoissonOperator(
            model.interface, geometry; density = model.density,
            formulation = model.formulation,
            construction_cache =
            old_pressure_solver.operator.gradient_operator.construction_cache,
        )
    pressure_preconditioner = solver.pressure_preconditioner
    pressure_preconditioner_options = solver.pressure_preconditioner_options
    old_pressure_preconditioner = old_pressure_solver.preconditioner
    if pressure_preconditioner in (:geometric, :tree_geometric) &&
            old_pressure_preconditioner isa
            TreeGeometricMultigridPreconditioner
        settings = merge(
            (;
                coarse_size = solver.coarse_size,
                pre_sweeps = solver.pre_sweeps,
                post_sweeps = solver.post_sweeps,
                max_levels = solver.max_levels,
            ),
            pressure_preconditioner_options,
        )
        pressure_preconditioner = TreeGeometricMultigridPreconditioner(
            pressure_operator; settings...,
            reuse = old_pressure_preconditioner,
        )
        pressure_preconditioner_options = (;)
    end
    reconfigured_pressure = pressure_preconditioner isa
        TreeGeometricMultigridPreconditioner ?
        _reconfigure_elliptic_solver!(
            old_pressure_solver, pressure_operator,
            solver.pressure_krylov, pressure_preconditioner,
        ) : nothing
    solver.pressure_solver = isnothing(reconfigured_pressure) ?
        EllipticSolver(
            pressure_operator;
            krylov = solver.pressure_krylov,
            krylov_options = solver.pressure_krylov_options,
            preconditioner = pressure_preconditioner,
            preconditioner_options = pressure_preconditioner_options,
            coarse_size = solver.coarse_size,
            pre_sweeps = solver.pre_sweeps,
            post_sweeps = solver.post_sweeps,
            max_levels = solver.max_levels,
        ) : reconfigured_pressure
    _resize_zero_velocity!(workspace.old_velocity, solver.mesh)
    _resize_zero_velocity!(workspace.convected, solver.mesh)
    _resize_zero_velocity!(workspace.diffusion_work, solver.mesh)
    workspace.projection = ProjectionWorkspace(solver.pressure_solver)
    workspace.advection = nothing
    workspace.elliptic = nothing
    old_diffusion_preconditioner =
        isnothing(old_phase_workspace.diffusion_solver) ? nothing :
        old_phase_workspace.diffusion_solver.preconditioner
    reusable_diffusion_preconditioner =
        solver.diffusion_preconditioner in (:geometric, :tree_geometric) &&
        old_diffusion_preconditioner isa
        TreeVariableDiffusionPreconditioner ?
        old_diffusion_preconditioner : nothing
    workspace.two_phase = TwoPhaseFlowWorkspace(
        model; boundary = solver.boundary, dt,
        diffusion_krylov = solver.diffusion_krylov,
        diffusion_krylov_options = solver.diffusion_krylov_options,
        diffusion_preconditioner = solver.diffusion_preconditioner,
        diffusion_preconditioner_options =
            solver.diffusion_preconditioner_options,
        coarse_size = solver.coarse_size,
        pre_sweeps = solver.pre_sweeps,
        post_sweeps = solver.post_sweeps,
        max_levels = solver.max_levels,
        diffusion_cache = old_phase_workspace.diffusion_cache,
        reuse_diffusion = old_phase_workspace.diffusion,
        reuse_diffusion_preconditioner =
            reusable_diffusion_preconditioner,
        reuse_diffusion_solver =
            reusable_diffusion_preconditioner === nothing ? nothing :
            old_phase_workspace.diffusion_solver,
        reuse_density = old_phase_workspace.density,
        reuse_workspace = old_phase_workspace,
    )
    return workspace
end

function _navier_stokes_amr_workspace(
        velocity, pressure, solver::IncompressibleNavierStokesSolver
    )
    workspace = velocity isa TwoPhaseVelocity ?
        AMRWorkspace(solver.mesh, velocity[1], velocity[2], pressure) :
        AMRWorkspace(solver.mesh, velocity, pressure)
    if !isnothing(solver.two_phase)
        register_adaptive_field!(workspace, solver.two_phase.interface)
        for scalar in solver.two_phase.scalars
            register_adaptive_field!(workspace, scalar)
        end
    end
    if !isnothing(solver.geometry)
        register_adaptive_field!(workspace, solver.geometry)
    end
    for field in solver.adaptive_fields
        register_adaptive_field!(workspace, field)
    end
    for indicator in solver.refinement_indicators
        _register_indicator_field!(workspace, indicator)
    end
    return workspace
end

function _zero_velocity(mesh::M, ::Type{T}) where {M <: BlockTreeMesh, T}
    return FaceVelocity(mesh, T)
end

function _positive_finite_step(::Type{T}, value, label = "dt") where {T}
    converted = try
        T(value)
    catch exception
        exception isa InterruptException && rethrow()
        throw(ArgumentError("$label must convert to $T"))
    end
    isfinite(converted) && converted > zero(T) || throw(
        ArgumentError(
            "$label must be finite and strictly positive"
        )
    )
    return converted
end

function _finite_time(::Type{T}, value, label = "time") where {T}
    converted = try
        T(value)
    catch exception
        exception isa InterruptException && rethrow()
        throw(ArgumentError("$label must convert to $T"))
    end
    isfinite(converted) || throw(ArgumentError("$label must be finite"))
    return converted
end

function _navier_stokes_elliptic_solver(
        solver::IncompressibleNavierStokesSolver{T}, diffusion;
        reuse_preconditioner = nothing,
    ) where {T}
    return StaggeredEllipticSolver(
        StaggeredEllipticOperator(solver.diffusion; diffusion = T(diffusion));
        krylov = solver.diffusion_krylov,
        krylov_options = solver.diffusion_krylov_options,
        preconditioner = solver.diffusion_preconditioner,
        preconditioner_options = solver.diffusion_preconditioner_options,
        coarse_size = solver.coarse_size, pre_sweeps = solver.pre_sweeps,
        post_sweeps = solver.post_sweeps, max_levels = solver.max_levels,
        reuse_preconditioner,
    )
end

"""
    NavierStokesWorkspace(solver; dt=nothing)

Allocate all reusable arrays, staggered schedules, Krylov workspaces, and
preconditioners required by `step!` and `simulate!`. When the fixed time step
is known, pass `dt` to build the shifted diffusion hierarchy at its final
coefficient and avoid rebuilding it on the first step. Construct the workspace
once for advanced low-level use; `FlowSimulation` owns one automatically.
"""
function NavierStokesWorkspace(
        solver::IncompressibleNavierStokesSolver{T, M};
        dt = nothing
    ) where {T, M}
    old_velocity = _zero_velocity(solver.mesh, T)
    convected = _zero_velocity(solver.mesh, T)
    diffusion_work = _zero_velocity(solver.mesh, T)
    two_phase = solver.two_phase
    advection = isnothing(two_phase) ?
        (
            isnothing(solver.geometry) ?
            StaggeredVelocityAdvectionWorkspace(old_velocity) :
            StaggeredVelocityAdvectionWorkspace(
                old_velocity, solver.geometry,
            )
        ) : nothing
    projection = ProjectionWorkspace(solver.pressure_solver)
    if isnothing(dt)
        diffusion = zero(T)
    else
        time_step = _positive_finite_step(T, dt, "workspace dt")
        diffusion = isnothing(solver.two_phase) ?
            time_step * solver.kinematic_viscosity / 2 : zero(T)
    end
    elliptic = isnothing(two_phase) ?
        _navier_stokes_elliptic_solver(solver, diffusion) : nothing
    return NavierStokesWorkspace(
        old_velocity, convected, diffusion_work,
        advection, projection, elliptic,
        isnothing(two_phase) ? nothing : TwoPhaseFlowWorkspace(
                two_phase; boundary = solver.boundary, dt,
                diffusion_krylov = solver.diffusion_krylov,
                diffusion_krylov_options = solver.diffusion_krylov_options,
                diffusion_preconditioner = solver.diffusion_preconditioner,
                diffusion_preconditioner_options =
                solver.diffusion_preconditioner_options,
                coarse_size = solver.coarse_size,
                pre_sweeps = solver.pre_sweeps,
                post_sweeps = solver.post_sweeps,
                max_levels = solver.max_levels,
            )
    )
end

function _copy_velocity!(destination::FaceVelocity, source::FaceVelocity)
    destination.mesh === source.mesh ||
        throw(ArgumentError("velocities must share a mesh"))
    copyto!(destination.interior, source.interior)
    copyto!(destination.boundary, source.boundary)
    return destination
end

function _copy_velocity!(
        destination::TwoPhaseVelocity, source::TwoPhaseVelocity,
    )
    for phase in 1:2
        _copy_velocity!(destination[phase], source[phase])
    end
    return destination
end

function _enforce_phase_velocity_boundary!(
        velocity::TwoPhaseVelocity, solver, time,
    )
    for phase in 1:2
        enforce_velocity_boundary!(
            velocity[phase], solver.boundary;
            time, geometry = solver.geometry,
        )
    end
    isnothing(solver.geometry) ||
        enforce_geometry_no_slip!(velocity, solver.geometry)
    return velocity
end

function _enforce_current_velocity_boundary!(
        velocity::FaceVelocity, solver, time,
    )
    enforce_velocity_boundary!(
        velocity, solver.boundary; time, geometry = solver.geometry,
    )
    isnothing(solver.geometry) ||
        enforce_geometry_no_slip!(velocity, solver.geometry)
    return velocity
end

@inline _enforce_current_velocity_boundary!(
    velocity::TwoPhaseVelocity, solver, time,
) = _enforce_phase_velocity_boundary!(velocity, solver, time)

@inline _body_force(::Nothing, face, time) = zero(typeof(face.area))
@inline _body_force(force, face, time) =
    force(face.center[1], face.center[2], time, face.axis)

function _transport_substep!(
        velocity, solver, workspace, dt, time, ::Nothing
    )
    _copy_velocity!(workspace.convected, workspace.old_velocity)
    advect!(
        workspace.convected, workspace.old_velocity, dt,
        workspace.advection;
        scheme = solver.scheme,
        amr_interface_scheme = solver.amr_interface_scheme,
        boundary = solver.boundary, time,
    )
    return workspace.convected
end

function _diffusion_substep!(
        velocity, solver, workspace, dt, time, ::Nothing;
        tolerance = sqrt(eps(eltype(velocity))),
        max_cycles = 100
    )
    return implicit_midpoint_diffusion!(
        velocity, workspace.convected, workspace.old_velocity,
        solver.diffusion, dt, workspace.elliptic,
        workspace.diffusion_work; time, tolerance, max_cycles
    )
end

@inline _refresh_pressure_substep!(solver, workspace, ::Nothing) =
    workspace.projection

@inline _phase_force_substep!(velocity, solver, workspace, dt, ::Nothing) =
    velocity

@inline _timestep_bound(velocity, solver, workspace, ::Nothing; cfl) =
    stable_timestep(velocity, workspace.advection; cfl)

function _navier_stokes_cycle_limit(solver, max_cycles)
    isnothing(max_cycles) && return isnothing(solver.two_phase) ? 100 : 500
    max_cycles isa Integer || throw(
        ArgumentError("max_cycles must be an integer or nothing"),
    )
    max_cycles >= 0 || throw(
        ArgumentError("max_cycles must be nonnegative"),
    )
    return Int(max_cycles)
end

"""
    step!(velocity, pressure, solver, dt, workspace;
          time=0, force=nothing, tolerance=sqrt(eps(T)),
          max_cycles=nothing)

Advance one split momentum step followed by exactly one pressure projection.
Both single- and two-phase configurations execute the same staged procedure.
`force`, when present, is sampled as `force(x, y, time, axis)` at the time
midpoint and added to the transported state before diffusion, preserving the
single-phase solver's established splitting. Thesis phase forces are added
after diffusion so hydrostatic equilibria are unchanged by the viscous solve.
The transporting flux is
frozen at the old-time velocity while each coordinate sweep updates both
transported velocity components. No-permeation is automatic in non-periodic
directions.
"""
function step!(
        velocity::FaceVelocity{T}, pressure::CellField{T},
        solver::IncompressibleNavierStokesSolver{T}, dt,
        workspace::NavierStokesWorkspace{T};
        time = zero(T), force = nothing,
        tolerance = sqrt(eps(T)), max_cycles = nothing
    ) where {T}
    velocity.mesh === pressure.mesh === solver.mesh ===
        workspace.old_velocity.mesh ||
        throw(ArgumentError("Navier--Stokes arguments must share a mesh"))
    time_step = _positive_finite_step(T, dt)
    current_time = _finite_time(T, time)
    dimensionless_tolerance = _dimensionless_tolerance(T, tolerance)
    cycle_limit = _navier_stokes_cycle_limit(solver, max_cycles)
    _copy_velocity!(workspace.old_velocity, velocity)
    enforce_velocity_boundary!(
        workspace.old_velocity, solver.boundary;
        time = current_time, geometry = solver.geometry,
    )
    isnothing(solver.geometry) || enforce_geometry_no_slip!(
        workspace.old_velocity, solver.geometry,
    )
    model = solver.two_phase
    _transport_substep!(
        velocity, solver, workspace, time_step, current_time, model
    )
    midpoint = current_time + time_step / 2
    for (index, face) in pairs(solver.mesh.faces)
        @inbounds workspace.convected.interior[index] +=
            time_step * _body_force(force, face, midpoint)
    end
    enforce_velocity_boundary!(
        workspace.convected, solver.boundary;
        time = midpoint, geometry = solver.geometry,
    )
    _diffusion_substep!(
        velocity, solver, workspace, time_step, current_time, model;
        tolerance = dimensionless_tolerance,
        max_cycles = cycle_limit,
    )
    enforce_velocity_boundary!(
        velocity, solver.boundary;
        time = current_time + time_step, geometry = solver.geometry,
    )
    _refresh_pressure_substep!(solver, workspace, model)
    _phase_force_substep!(velocity, solver, workspace, time_step, model)

    if isnothing(solver.geometry)
        return project!(
            velocity, pressure, solver.pressure_solver, time_step,
            workspace.projection;
            velocity_boundary = solver.boundary,
            boundary_time = current_time + time_step,
            tolerance = dimensionless_tolerance,
            max_cycles = cycle_limit,
        )
    end
    return project!(
        velocity, pressure, solver.pressure_solver, time_step,
        workspace.projection, solver.geometry;
        velocity_boundary = solver.boundary,
        boundary_time = current_time + time_step,
        tolerance = dimensionless_tolerance,
        max_cycles = cycle_limit,
    )
end

"""Advance one two-velocity Navier--Stokes step.

The geometric interface and both intrinsic phase velocities use the same
transport stage as the passive two-velocity API.  Body force, surface tension,
and gravity modify the predictor, so their contribution is present in the two
right-hand sides of the subsequent three-block mimetic projection. Viscosity
acts implicitly on the phase-mass weighted one-velocity state; its common
increment is then applied to both phases, conserving combined momentum and
preserving the intrinsic velocity jump.
"""
function step!(
        velocity::TwoPhaseVelocity, pressure::CellField{T},
        solver::IncompressibleNavierStokesSolver{T}, dt,
        workspace::NavierStokesWorkspace{T};
        time = zero(T), force = nothing,
        tolerance = sqrt(eps(T)), max_cycles = nothing,
    ) where {T}
    model = solver.two_phase
    isnothing(model) && throw(
        ArgumentError(
            "a two-velocity step requires a solver configured with an interface",
        )
    )
    phase = workspace.two_phase
    isnothing(phase) && throw(
        ArgumentError(
            "a two-velocity step requires a two-phase workspace",
        )
    )
    _validate_two_phase_velocity(velocity, solver.mesh, T)
    pressure.mesh === solver.mesh || throw(
        ArgumentError("Navier--Stokes arguments must share a mesh"),
    )
    time_step = _positive_finite_step(T, dt)
    current_time = _finite_time(T, time)
    dimensionless_tolerance = _dimensionless_tolerance(T, tolerance)
    cycle_limit = _navier_stokes_cycle_limit(solver, max_cycles)

    state = _two_velocity_flow_workspace!(phase, model)
    _copy_velocity!(state.old_velocity, velocity)
    _enforce_phase_velocity_boundary!(
        state.old_velocity, solver, current_time,
    )
    _transport_substep!(
        velocity, solver, workspace, time_step, current_time, model,
    )
    midpoint = current_time + time_step / 2
    for phase_index in 1:2
        for (index, face) in pairs(solver.mesh.faces)
            @inbounds state.convected[phase_index].interior[index] +=
                time_step * _body_force(force, face, midpoint)
        end
    end
    _enforce_phase_velocity_boundary!(
        state.convected, solver, midpoint,
    )
    _diffusion_substep!(
        velocity, solver, workspace, time_step, current_time, model;
        tolerance = dimensionless_tolerance, max_cycles = cycle_limit,
    )
    _enforce_phase_velocity_boundary!(
        velocity, solver, current_time + time_step,
    )
    if model.sigma > zero(T) || any(!iszero, model.gravity)
        # The established balanced-force kernels use the one-velocity
        # pressure Riesz map. Refresh it only when a force actually needs it;
        # the mimetic projection assembles its own three-block pressure block.
        _refresh_pressure_substep!(solver, workspace, model)
        _phase_force_substep!(
            velocity, solver, workspace, time_step, model,
        )
    end
    return project!(
        velocity, pressure, state.mimetic_solver, time_step,
        state.mimetic_projection;
        tolerance = dimensionless_tolerance,
        max_iterations = cycle_limit,
    )
end

"""Return the convective CFL bound for a Navier--Stokes step."""
function stable_timestep(
        velocity::FaceVelocity,
        solver::IncompressibleNavierStokesSolver,
        workspace::NavierStokesWorkspace;
        cfl = 0.45
    )
    return _timestep_bound(
        velocity, solver, workspace, solver.two_phase; cfl
    )
end

function stable_timestep(
        velocity::TwoPhaseVelocity,
        solver::IncompressibleNavierStokesSolver,
        workspace::NavierStokesWorkspace; cfl = 0.45,
    )
    isnothing(solver.two_phase) && throw(
        ArgumentError(
            "a two-velocity CFL bound requires a two-phase solver",
        )
    )
    return _timestep_bound(
        velocity, solver, workspace, solver.two_phase; cfl,
    )
end

function _copy_cell_field!(destination::CellField, source::CellField)
    destination.mesh === source.mesh ||
        throw(ArgumentError("copied fields must share one mesh"))
    length(destination.blocks) == length(source.blocks) ||
        throw(DimensionMismatch("copied fields must share one topology"))
    for block in eachindex(source.blocks)
        copyto!(destination.blocks[block], source.blocks[block])
    end
    return destination
end

function _copy_interface!(destination::InterfaceField, source::InterfaceField)
    destination.mesh === source.mesh ||
        throw(ArgumentError("copied interfaces must share one mesh"))
    _copy_cell_field!(destination.fraction, source.fraction)
    return reconstruct!(destination)
end

function _independent_solver(
        solver::IncompressibleNavierStokesSolver{T}
    ) where {T}
    configuration = _navier_stokes_configuration(
        solver; refinement_indicators = (), adaptive_fields = ()
    )
    if !isnothing(solver.two_phase)
        interface = copy(solver.two_phase.interface)
        scalars = map(solver.two_phase.scalars) do scalar
            PhaseScalar(
                interface, copy(scalar.liquid), copy(scalar.gas)
            )
        end
        configuration = merge(
            configuration, (; interface, phase_scalars = scalars)
        )
    end
    return IncompressibleNavierStokesSolver(
        solver.mesh; configuration...
    )
end

mutable struct IndependentNavierStokesStep{SS, S, W, F, O, I, Q} <: Function
    source_solver::SS
    solver::S
    workspace::W
    force::F
    options::O
    interface_snapshot::I
    scalar_snapshots::Q
    advection_forward::Bool
    phase_forward::Bool
end

function _independent_step_procedure(
        solver::IncompressibleNavierStokesSolver, force, options,
        source_workspace::NavierStokesWorkspace
    )
    independent_solver = _independent_solver(solver)
    independent_workspace = NavierStokesWorkspace(independent_solver)
    interface_snapshot = isnothing(solver.two_phase) ? nothing :
        copy(solver.two_phase.interface)
    scalar_snapshots = isnothing(solver.two_phase) ? () :
        map(solver.two_phase.scalars) do scalar
            (copy(scalar.liquid), copy(scalar.gas))
    end
    phase_forward = isnothing(source_workspace.two_phase) ? true :
        source_workspace.two_phase.forward
    return IndependentNavierStokesStep(
        solver, independent_solver, independent_workspace, force, options,
        interface_snapshot, scalar_snapshots,
        isnothing(source_workspace.advection) ? true :
            source_workspace.advection.sweep_forward[],
        phase_forward
    )
end

function _independent_field_replacements(
        procedure::IndependentNavierStokesStep
    )
    replacements = IdDict{Any, Any}()
    source = procedure.source_solver.two_phase
    independent = procedure.solver.two_phase
    isnothing(source) && return replacements
    for name in (
            :fraction, :conditional_fraction, :normal_x, :normal_y, :intercept,
            :curvature, :reconstruction,
        )
        replacements[getproperty(source.interface, name)] =
            getproperty(independent.interface, name)
    end
    for (source_scalar, independent_scalar) in
        zip(source.scalars, independent.scalars)
        replacements[source_scalar.liquid] = independent_scalar.liquid
        replacements[source_scalar.gas] = independent_scalar.gas
    end
    return replacements
end

function _prepare_independent_step!(
        procedure::IndependentNavierStokesStep,
        solver::IncompressibleNavierStokesSolver,
        source_workspace::NavierStokesWorkspace
    )
    if !isnothing(source_workspace.advection)
        procedure.advection_forward =
            source_workspace.advection.sweep_forward[]
    end
    if !isnothing(solver.two_phase)
        _copy_interface!(
            procedure.interface_snapshot, solver.two_phase.interface
        )
        for (snapshot, scalar) in
            zip(procedure.scalar_snapshots, solver.two_phase.scalars)
            _copy_cell_field!(snapshot[1], scalar.liquid)
            _copy_cell_field!(snapshot[2], scalar.gas)
        end
        procedure.phase_forward = source_workspace.two_phase.forward
    end
    return procedure
end

_prepare_independent_step!(::Nothing, solver, workspace) = nothing

function (procedure::IndependentNavierStokesStep)(
        velocity, pressure, dt; time
    )
    if !isnothing(procedure.workspace.advection)
        procedure.workspace.advection.sweep_forward[] =
            procedure.advection_forward
    end
    if !isnothing(procedure.solver.two_phase)
        model = procedure.solver.two_phase
        _copy_interface!(model.interface, procedure.interface_snapshot)
        for (scalar, snapshot) in
            zip(model.scalars, procedure.scalar_snapshots)
            _copy_cell_field!(scalar.liquid, snapshot[1])
            _copy_cell_field!(scalar.gas, snapshot[2])
        end
        procedure.workspace.two_phase.forward = procedure.phase_forward
    end
    stats = step!(
        velocity, pressure, procedure.solver, dt,
        procedure.workspace; time, force = procedure.force,
        procedure.options...
    )
    stats.converged ||
        throw(
        ErrorException(
            _solver_failure_message(
                "pressure projection", stats,
                procedure.options.tolerance;
                context = "independent visualization step",
                details = "dt=$(dt), max_cycles=$(procedure.options.max_cycles)",
            )
        )
    )
    return stats
end

_visualization_step_procedure(
    ::NoVisualization, solver, force, options, workspace
) = nothing

function _visualization_step_procedure(
        session::VisualizationSession, solver, force, options, workspace
    )
    return session.exact_checkpoints ?
        _independent_step_procedure(
            solver, force, options, workspace
        ) : nothing
end

_visualization_step_procedure(
    ::AbstractVisualization, solver, force, options, workspace
) =
    _independent_step_procedure(solver, force, options, workspace)

"""
    simulate!(velocity, pressure, solver, final_time, workspace;
              time=0, dt=nothing, cfl=0.45,
              visualization=NoVisualization(),
              tolerance=sqrt(eps(T)), max_cycles=nothing)

Advance to `final_time`. If `dt` is omitted, a stable step is
chosen initially and retained while it remains below the current bound, which
also reuses the shifted viscous hierarchy and bottom factorization. Returns
the final time, number of steps, latest projection statistics, and the number
of topology changes. When the solver has refinement indicators, they are
evaluated after a successful physical step; field transfer does not add a
time-integration or pressure-correction step. A supplied
visualization receives the initial state, every completed step, and the final
state. During initialization it is also given a lazy independent-step
procedure. A visualization may use that procedure to advance copied state
without replacing or subdividing a step on the actual simulation path.
"""
function simulate!(
        velocity::Union{FaceVelocity, TwoPhaseVelocity},
        pressure::CellField{T},
        solver::IncompressibleNavierStokesSolver{T}, final_time,
        workspace::NavierStokesWorkspace{T};
        time = zero(T), dt = nothing, cfl = 0.45,
        force = nothing,
        visualization::AbstractVisualization = NoVisualization(),
        tolerance = sqrt(eps(T)), max_cycles = nothing
    ) where {T}
    target_time = _finite_time(T, final_time, "final_time")
    current_time = _finite_time(T, time)
    target_time >= current_time ||
        throw(ArgumentError("final_time must not precede the initial time"))
    dimensionless_tolerance = _dimensionless_tolerance(T, tolerance)
    cycle_limit = _navier_stokes_cycle_limit(solver, max_cycles)
    options = (;
        tolerance = dimensionless_tolerance,
        max_cycles = cycle_limit,
    )
    step_count = 0
    mesh_updates = 0
    stats = solver.pressure_solver.stats
    selected_step = isnothing(dt) ? T(Inf) : _positive_finite_step(T, dt)
    time_tolerance = T(16) * eps(T) *
        max(one(T), abs(target_time), abs(current_time))
    independent_step = _visualization_step_procedure(
        visualization, solver, force, options, workspace
    )
    amr_workspace = isempty(solver.refinement_indicators) ? nothing :
        _navier_stokes_amr_workspace(
            velocity, pressure, solver
        )
    isnothing(amr_workspace) || _supports_dynamic_topology(visualization) ||
        throw(
        ArgumentError(
            "exact visualization checkpoints are not supported while dynamic mesh adaptation is enabled"
        )
    )
    initialize_visualization!(
        visualization; time = current_time, step = step_count,
        final_time = target_time, velocity, pressure, solver,
        step_procedure = independent_step
    )
    try
        while target_time - current_time > time_tolerance
            if isnothing(dt)
                # Initializers and AMR transfer operate on intrinsic fields;
                # impose the configured boundary values before querying the
                # transport CFL, just as `step!` does before advection.
                _enforce_current_velocity_boundary!(
                    velocity, solver, current_time,
                )
                bound = stable_timestep(
                    velocity, solver, workspace; cfl
                )
                selected_step = min(selected_step, bound)
            end
            proposed = isfinite(selected_step) ? selected_step :
                target_time - current_time
            step_size = min(proposed, target_time - current_time)
            step_size > zero(T) ||
                throw(
                ArgumentError(
                    "time-step selection produced a non-positive step"
                )
            )

            _prepare_independent_step!(
                independent_step, solver, workspace
            )
            stats = step!(
                velocity, pressure, solver, step_size, workspace;
                time = current_time, force, options...
            )
            stats.converged ||
                throw(
                ErrorException(
                    _solver_failure_message(
                        "pressure projection", stats,
                        dimensionless_tolerance;
                        details = "dt=$(step_size), time=$(current_time), " *
                            "max_cycles=$(cycle_limit)",
                    )
                )
            )
            current_time = min(target_time, current_time + step_size)
            step_count += 1
            solver.mesh.adaptation.completed_steps += 1
            if !isnothing(amr_workspace)
                amr_stats = adapt!(
                    amr_workspace, solver.refinement_indicators;
                    step = solver.mesh.adaptation.completed_steps,
                    time = current_time
                )
                if amr_stats.changed
                    mesh_updates += 1
                    # Seed the replacement viscous hierarchy with the current
                    # shift. Fixed-step runs therefore do not build it twice
                    # after every topology change.
                    _rebuild_topology_dependent_state!(
                        solver, workspace; dt = step_size
                    )
                end
            end
            update_visualization!(
                visualization; event = :step, time = current_time,
                step = step_count, final_time = target_time,
                velocity, pressure, solver
            )
        end
        current_time = target_time
        update_visualization!(
            visualization; event = :completed, time = current_time,
            step = step_count, final_time = current_time,
            velocity, pressure, solver
        )
    catch exception
        try
            update_visualization!(
                visualization; event = :failed, time = current_time,
                step = step_count, final_time = target_time,
                velocity, pressure, solver, exception
            )
        catch
            # Preserve the simulation or visualization error that stopped the run.
        end
        rethrow()
    end
    return (
        time = current_time, steps = step_count, pressure = stats,
        mesh_updates,
    )
end

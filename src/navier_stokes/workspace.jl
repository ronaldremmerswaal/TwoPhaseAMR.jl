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
        surface_tension_formulation = isnothing(solver.two_phase) ? CGFM() : solver.two_phase.formulation,
        gravity = isnothing(solver.two_phase) ? (0.0, 0.0) : solver.two_phase.gravity,
        phase_scalars = isnothing(solver.two_phase) ? () : solver.two_phase.scalars,
        interface_transport = isnothing(solver.two_phase) ?
            FrozenInterfaceFluxes() : solver.two_phase.interface_transport,
        two_velocity_projection = isnothing(solver.two_phase) ?
            CMDGFMProjection() : solver.two_phase.pressure_projection,
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
            construction_cache =
            old_pressure_solver.operator.gradient_operator.construction_cache,
        ) : PoissonOperator(
            model.interface, geometry; density = model.density,
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
        AMRWorkspace(solver.mesh, pressure) : AMRWorkspace(solver.mesh, velocity, pressure)
    if velocity isa TwoPhaseVelocity
        register_adaptive_field!(workspace, velocity, solver.two_phase.interface)
    end
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

mutable struct SimulationStepSelection{T, M <: BlockTreeMesh}
    time::T
    # A Float32 flow with a Float64 CFL can select a Float64 proposal.
    step::Union{T, Float64, BigFloat}
    # The actual last step and the time scale of the defect can differ after
    # an AMR transfer/projection. Zero denotes the absence of a prior step.
    last_dt::T
    defect_dt::T
    topology_generation::UInt
    velocity::Union{Nothing, FaceVelocity{T, M}, NTuple{2, FaceVelocity{T, M}}}
    pressure::Union{Nothing, CellField{T, M}}
    solver::Union{Nothing, NavierStokesSystem{T, M}}
end

SimulationStepSelection(::Type{T}, ::Type{M}) where {T, M <: BlockTreeMesh} =
    SimulationStepSelection{T, M}(
    zero(T), T(Inf), zero(T), zero(T), UInt(0),
    nothing, nothing, nothing,
)

function _invalidate_simulation_step!(workspace)
    selection = workspace.simulation_step
    selection.velocity = nothing
    selection.step = typeof(selection.time)(Inf)
    selection.last_dt = zero(selection.last_dt)
    selection.defect_dt = zero(selection.defect_dt)
    return nothing
end

mutable struct NavierStokesWorkspace{T, M <: BlockTreeMesh}
    old_velocity::FaceVelocity{T, M}
    convected::FaceVelocity{T, M}
    diffusion_work::FaceVelocity{T, M}
    # Uniform and mixed-level meshes use different schedule/metric concrete
    # types. These partial types let a single workspace survive a regrid.
    advection::Union{Nothing, StaggeredVelocityAdvectionWorkspace{T, M}}
    projection::ProjectionWorkspace{T, M}
    # Open shared-velocity phase projection must not alter the Neumann
    # pressure solver used by the coupled two-velocity formulation.
    one_velocity_pressure_solver::Union{Nothing, EllipticSolver{T}}
    elliptic::Union{Nothing, StaggeredEllipticSolver{T}}
    two_phase::Union{Nothing, TwoPhaseFlowWorkspace{T, M}}
    unresolved_curvature_faces::Base.RefValue{Int}
    simulation_step::SimulationStepSelection{T, M}
end

function _replace_workspace!(
        destination::NavierStokesWorkspace,
        source::NavierStokesWorkspace
    )
    destination.old_velocity = source.old_velocity
    destination.convected = source.convected
    destination.diffusion_work = source.diffusion_work
    destination.advection = source.advection
    source.projection.warned_incompatible[] |=
        destination.projection.warned_incompatible[]
    destination.projection = source.projection
    destination.one_velocity_pressure_solver = source.one_velocity_pressure_solver
    destination.elliptic = source.elliptic
    destination.two_phase = source.two_phase
    _invalidate_simulation_step!(destination)
    return destination
end


function _rebuild_pressure_solver!(
        solver, old_pressure_solver, pressure_operator,
        pressure_preconditioner, pressure_preconditioner_options,
        old_pressure_preconditioner,
    )
    if pressure_preconditioner in (:geometric, :tree_geometric) &&
            old_pressure_preconditioner isa
            TreeGeometricMultigridPreconditioner
        pressure_preconditioner = TreeGeometricMultigridPreconditioner(
            pressure_operator; multigrid = _multigrid_options(solver),
            pressure_preconditioner_options...,
            reuse = old_pressure_preconditioner,
        )
        pressure_preconditioner_options = (;)
    end
    reconfigured_pressure = pressure_preconditioner isa
        TreeGeometricMultigridPreconditioner ?
        _reconfigure_elliptic_solver!(
            old_pressure_solver, pressure_operator,
            _pressure_krylov(solver), pressure_preconditioner,
        ) : nothing
    return isnothing(reconfigured_pressure) ?
        EllipticSolver(
            pressure_operator;
            krylov = _pressure_krylov(solver),
            krylov_options = solver.options.pressure.krylov_options,
            preconditioner = pressure_preconditioner,
            preconditioner_options = pressure_preconditioner_options,
            multigrid = _multigrid_options(solver),
        ) : reconfigured_pressure
end

function _rebuild_topology_dependent_state!(
        solver::NavierStokesSystem{T},
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
            solver.mesh; T, face_coefficient = inv(_single_phase_density(solver)),
            velocity_boundary = solver.boundary,
            construction_cache =
            old_pressure_solver.operator.gradient_operator.construction_cache,
        ) : PoissonOperator(
            geometry; face_coefficient = inv(_single_phase_density(solver)),
            velocity_boundary = solver.boundary,
            construction_cache =
            old_pressure_solver.operator.gradient_operator.construction_cache,
        )
    old_pressure_preconditioner = old_pressure_solver.preconditioner
    pressure_preconditioner = _pressure_preconditioner(solver)
    pressure_preconditioner_options = _pressure_preconditioner_options(solver)
    solver.pressure_solver = _rebuild_pressure_solver!(
        solver, old_pressure_solver, pressure_operator,
        pressure_preconditioner, pressure_preconditioner_options,
        old_pressure_preconditioner,
    )

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
    workspace.projection = ProjectionWorkspace(
        solver.pressure_solver, workspace.projection.warned_incompatible,
    )
    workspace.one_velocity_pressure_solver = nothing
    reuse_diffusion = if _diffusion_preconditioner(solver) in
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

function _two_phase_workspace_solver_options(solver)
    return (;
        diffusion_krylov = _diffusion_krylov(solver),
        diffusion_krylov_options = solver.options.diffusion.krylov_options,
        diffusion_preconditioner = _diffusion_preconditioner(solver),
        diffusion_preconditioner_options =
            solver.options.diffusion.preconditioner_options,
        multigrid = _multigrid_options(solver),
    )
end

function _rebuild_two_phase_topology_dependent_state!(
        solver::NavierStokesSystem{T},
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
    pressure_preconditioner = _pressure_preconditioner(solver)
    pressure_preconditioner_options = _pressure_preconditioner_options(solver)
    old_pressure_preconditioner = old_pressure_solver.preconditioner
    solver.pressure_solver = _rebuild_pressure_solver!(
        solver, old_pressure_solver, pressure_operator,
        pressure_preconditioner, pressure_preconditioner_options,
        old_pressure_preconditioner,
    )
    _resize_zero_velocity!(workspace.old_velocity, solver.mesh)
    _resize_zero_velocity!(workspace.convected, solver.mesh)
    _resize_zero_velocity!(workspace.diffusion_work, solver.mesh)
    workspace.projection = ProjectionWorkspace(
        solver.pressure_solver, workspace.projection.warned_incompatible,
    )
    workspace.one_velocity_pressure_solver = nothing
    workspace.advection = nothing
    workspace.elliptic = nothing
    old_diffusion_preconditioner =
        isnothing(old_phase_workspace.diffusion_solver) ? nothing :
        old_phase_workspace.diffusion_solver.preconditioner
    reusable_diffusion_preconditioner =
        _diffusion_preconditioner(solver) in (:geometric, :tree_geometric) &&
        old_diffusion_preconditioner isa
        TreeVariableDiffusionPreconditioner ?
        old_diffusion_preconditioner : nothing
    workspace.two_phase = TwoPhaseFlowWorkspace(
        model; boundary = solver.boundary, dt,
        _two_phase_workspace_solver_options(solver)...,
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
        velocity, pressure, solver::NavierStokesSystem
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
    for field in solver.adaptivity.fields
        register_adaptive_field!(workspace, field)
    end
    for indicator in solver.adaptivity.criteria
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
        solver::NavierStokesSystem{T}, diffusion;
        reuse_preconditioner = nothing,
    ) where {T}
    return StaggeredEllipticSolver(
        StaggeredEllipticOperator(solver.diffusion; diffusion = T(diffusion));
        krylov = _diffusion_krylov(solver),
        krylov_options = solver.options.diffusion.krylov_options,
        preconditioner = _diffusion_preconditioner(solver),
        preconditioner_options = solver.options.diffusion.preconditioner_options,
        multigrid = _multigrid_options(solver),
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
        solver::NavierStokesSystem{T, M};
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
        advection, projection, nothing, elliptic,
        isnothing(two_phase) ? nothing : TwoPhaseFlowWorkspace(
                two_phase; boundary = solver.boundary, dt,
                _two_phase_workspace_solver_options(solver)...,
            ), Ref(0), SimulationStepSelection(T, M)
    )
end

function _one_velocity_pressure_solver!(solver, workspace; refresh = false)
    model = solver.two_phase
    if isnothing(model) || !any(_pressure_boundary_sides(solver.boundary))
        return solver.pressure_solver
    end
    pressure_solver = workspace.one_velocity_pressure_solver
    if isnothing(pressure_solver)
        cache = solver.pressure_solver.operator.gradient_operator.construction_cache
        operator = isnothing(solver.geometry) ? PoissonOperator(
                model.interface; density = model.density,
                velocity_boundary = solver.boundary, construction_cache = cache,
            ) : PoissonOperator(
                model.interface, solver.geometry; density = model.density,
                velocity_boundary = solver.boundary, construction_cache = cache,
            )
        pressure_solver = EllipticSolver(
            operator; krylov = _pressure_krylov(solver),
            krylov_options = solver.options.pressure.krylov_options,
            preconditioner = _pressure_preconditioner(solver),
            preconditioner_options = _pressure_preconditioner_options(solver),
            multigrid = _multigrid_options(solver),
        )
        workspace.one_velocity_pressure_solver = pressure_solver
    elseif refresh
        pressure_solver.operator = isnothing(solver.geometry) ? PoissonOperator(
                model.interface; density = model.density,
                template = pressure_solver.operator,
            ) : PoissonOperator(
                model.interface, solver.geometry; density = model.density,
                template = pressure_solver.operator,
            )
        specification = _pressure_preconditioner(solver)
        if model.density[1] != model.density[2] && (
                specification isa Symbol ||
                    applicable(specification, pressure_solver.operator.matrix)
            )
            if specification in (:geometric, :tree_geometric) &&
                    pressure_solver.preconditioner isa TreeGeometricMultigridPreconditioner
                _refresh_geometric_multigrid!(
                    pressure_solver.preconditioner, pressure_solver.operator,
                )
            else
                pressure_solver.preconditioner = _construct_pressure_preconditioner(
                    specification, pressure_solver.operator;
                    multigrid = _multigrid_options(solver),
                    options = _pressure_preconditioner_options(solver),
                )
            end
            pressure_solver.stats.preconditioner =
                _preconditioner_name(pressure_solver.preconditioner)
        end
    end
    return pressure_solver
end

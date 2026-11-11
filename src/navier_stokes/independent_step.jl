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

function _copy_phase_scalar(interface, scalar::PhaseScalar)
    return PhaseScalar(interface, copy(scalar.liquid), copy(scalar.gas))
end

function _independent_solver(
        solver::NavierStokesSystem{T}
    ) where {T}
    model = solver.two_phase
    physics = if isnothing(model)
        solver.physics
    else
        interface = copy(model.interface)
        scalars = map(Base.Fix1(_copy_phase_scalar, interface), model.scalars)
        TwoPhase(
            interface; density = model.density,
            dynamic_viscosity = model.dynamic_viscosity,
            surface_tension = SurfaceTension(model.sigma; method = model.formulation),
            gravity = model.gravity, scalars,
        )
    end
    return NavierStokesSystem(
        solver.mesh, physics; options = solver.options,
        geometry = solver.geometry, boundary = solver.boundary,
        two_velocity_projection = isnothing(model) ?
            CMDGFMProjection() : model.pressure_projection,
        T,
    )
end

mutable struct IndependentNavierStokesStep{SS, S, W, F, O, I, Q, R} <: Function
    source_solver::SS
    solver::S
    workspace::W
    force::F
    options::O
    interface_snapshot::I
    scalar_snapshots::Q
    advection_forward::Bool
    phase_forward::Bool
    reference_dt::R
end

function _independent_step_procedure(
        solver::NavierStokesSystem{T}, force, options,
        source_workspace::NavierStokesWorkspace
    ) where {T}
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
    # Mesh coordinates may be narrower than the solver or its CFL proposal.
    return IndependentNavierStokesStep(
        solver, independent_solver, independent_workspace, force, options,
        interface_snapshot, scalar_snapshots,
        isnothing(source_workspace.advection) ? true :
            source_workspace.advection.sweep_forward[],
        phase_forward, one(promote_type(T, eltype(solver.mesh.origin), Float64))
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
        solver::NavierStokesSystem,
        source_workspace::NavierStokesWorkspace, reference_dt,
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
    procedure.reference_dt = reference_dt
    return procedure
end

_prepare_independent_step!(::Nothing, solver, workspace, reference_dt) = nothing

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
        _reference_dt = procedure.reference_dt,
        procedure.options...
    )
    stats.converged ||
        throw(
        ErrorException(
            _solver_failure_message(
                "pressure projection", stats,
                procedure.options.tolerance;
                context = "independent keyframe step",
                details = "dt=$(dt), max_cycles=$(procedure.options.max_cycles)",
            )
        )
    )
    return stats
end

_observer_step_procedure(
    ::NoObserver, solver, force, options, workspace
) = nothing

_observer_step_procedure(
    recorder::Recorder, solver, force, options, workspace
) = recorder.exact_keyframes ?
    _independent_step_procedure(solver, force, options, workspace) : nothing

# Custom observers receive the procedure, as they may advance copied state.
_observer_step_procedure(
    ::AbstractObserver, solver, force, options, workspace
) =
    _independent_step_procedure(solver, force, options, workspace)

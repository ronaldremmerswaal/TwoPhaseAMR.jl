module KHSingleStepDiagnostic

using LinearAlgebra
using Printf
using SparseArrays
using TwoPhaseAMR

# Reuse the integration problem so this diagnostic exercises exactly the KH
# initial data, interface transport, balanced force, and projection used by
# the production wave tests.  The diagnostic itself remains a development
# script and is not included in the test suite.
include(
    joinpath(
        @__DIR__, "..", "..", "test", "integration",
        "interfacial_instability_problem.jl",
    ),
)

const DEFAULT_TIMESTEP_FACTORS = (1.0,)
const REFINEMENT_TIMESTEP_FACTORS = (1.0, 0.5, 0.25, 0.125)

"""Named projection configurations compared by the diagnostic."""
function projection_configurations()
    return (
        cmdgfm = CMDGFMProjection(),
        divergence = MimeticProjection(jump_interpolant = :divergence),
        face = MimeticProjection(jump_interpolant = :face),
        cell = MimeticProjection(jump_interpolant = :cell),
        commuting = MimeticProjection(jump_interpolant = :commuting),
        commuting_unreduced = MimeticProjection(
            jump_interpolant = :commuting,
            commuting_sliver_epsilon = 0.0,
        ),
    )
end

@inline function velocity_vector(velocity)
    return vcat((collect(phase.interior) for phase in velocity)...)
end

@inline function relative_change(after, before, reference_norm)
    return norm(after - before) / max(reference_norm, eps(reference_norm))
end

function sheet_diagnostics(case)
    return _instability_vortex_sheet_diagnostics(
        case.interface, case.flow.velocity;
        interpolation_weights = :c1mc,
    )
end

function amplitude(case)
    return _instability_wave_amplitude(
        case.interface;
        wavelength = case.dispersion.wavelength,
        liquid_above = case.dispersion.liquid_above,
    )
end

"""Rank and conditioning of the KKT normal constraint at one geometry."""
function constraint_geometry(
        solver::MimeticProjectionSolver, workspace,
        velocity, dt,
    )
    assembled = TwoPhaseAMR._assemble_mimetic_system!(
        solver, workspace, dt, velocity,
    )
    jump_dofs = length(assembled.active_faces)
    constraints = size(assembled.constraint, 1)
    kernel_dimension = jump_dofs - constraints
    schur_condition = if iszero(constraints)
        1.0
    else
        inverse_kinetic = spdiagm(0 => inv.(assembled.relative_diagonal))
        schur = assembled.constraint * inverse_kinetic *
            transpose(assembled.constraint)
        cond(Matrix(schur))
    end

    # This is the actual retained constraint defect used on the last KKT
    # block row.  It includes the affine boundary term where applicable and
    # is divided by the same aperture-normal row scale as the solver defect.
    np = length(assembled.pressure_indices)
    nx = jump_dofs
    defect = dt .* view(
        assembled.right_hand_side,
        (np + nx + 1):(np + nx + constraints),
    )
    scaled_defect_linf = maximum(
        abs.(defect ./ assembled.constraint_scales); init = 0.0,
    )
    return (;
        kind = :kkt,
        jump_dofs,
        constraints,
        kernel_dimension,
        candidates = assembled.candidate_count,
        dependent = assembled.dependent_count,
        underresolved = assembled.underresolved,
        scaled_defect_linf,
        schur_condition,
        inferred_tangent = 0,
        minimum_beta = 1.0,
    )
end

"""Stencil information playing the role of a jump-space diagnostic in CMDGFM."""
function constraint_geometry(
        solver::CMDGFMProjectionSolver, workspace,
        velocity, dt,
    )
    assembled = TwoPhaseAMR._assemble_cmdgfm_system!(
        solver, workspace, dt, velocity,
    )
    inferred_tangent = count(
        stencil -> stencil.gas_inference_face > 0,
        assembled.stencils,
    )
    minimum_beta = minimum(
        (stencil.beta for stencil in assembled.stencils); init = 1.0,
    )
    return (;
        kind = :cmdgfm,
        jump_dofs = length(assembled.mixed_faces),
        constraints = missing,
        kernel_dimension = missing,
        candidates = length(assembled.stencils) + assembled.underresolved,
        dependent = missing,
        underresolved = assembled.underresolved,
        scaled_defect_linf = missing,
        schur_condition = missing,
        inferred_tangent,
        minimum_beta,
    )
end

function projection_identity(stats::MimeticProjectionStats)
    return (;
        energy = stats.energy_identity_error,
        momentum = stats.momentum_identity_error,
        normal = stats.normal_velocity_defect,
        commuting = stats.commuting_constraint_rows,
        fallback = stats.fallback_constraint_rows,
        sliver_extrapolated = stats.sliver_extrapolated_faces,
        sliver_zeroed = stats.sliver_zeroed_faces,
    )
end

function projection_identity(stats::CMDGFMProjectionStats)
    return (;
        energy = missing,
        momentum = stats.momentum_identity_error,
        normal = stats.normal_velocity_defect,
        commuting = missing,
        fallback = missing,
        sliver_extrapolated = missing,
        sliver_zeroed = missing,
    )
end

"""
    single_step_diagnostic(name, projection; kwargs...)

Run exactly one production KH step, but expose the state after transport,
balanced forcing, and pressure projection.  The geometry diagnostics before
and after transport show whether the normal constraint changes rank or loses
its tangential-sheet kernel.
"""
function single_step_diagnostic(
        name::Symbol, projection;
        cells_per_wavelength = 8,
        density_ratio = 1.0,
        weber = 0.5,
        timestep_factor = 1.0,
        cfl = 0.35,
        tolerance = 1.0e-9,
        max_cycles = 5000,
    )
    case = two_velocity_instability_setup(;
        cells_per_wavelength,
        density_ratio,
        weber,
        bond = 0.0,
        liquid_above = false,
        interface_transport = FrozenInterfaceFluxes(),
        pressure_projection = projection,
    )
    flow = case.flow
    model = flow.solver.two_phase
    state = TwoPhaseAMR._two_velocity_flow_workspace!(
        flow.workspace.two_phase, model,
    )
    TwoPhaseAMR._enforce_phase_velocity_boundary!(
        flow.velocity, flow.solver, flow.time,
    )
    dt = timestep_factor * stable_timestep(flow; cfl)

    initial_velocity = velocity_vector(flow.velocity)
    reference_norm = norm(initial_velocity)
    initial_sheet = sheet_diagnostics(case)
    initial_amplitude = amplitude(case)
    geometry_before = constraint_geometry(
        state.projection_solver, state.projection_workspace,
        flow.velocity, dt,
    )

    # This is the production step split into its constituent stages.  Keeping
    # the ordering identical is important: the observed KH impulse occurs
    # when the normal constraint is rebuilt on the transported interface.
    TwoPhaseAMR._copy_velocity!(state.old_velocity, flow.velocity)
    TwoPhaseAMR._enforce_phase_velocity_boundary!(
        state.old_velocity, flow.solver, flow.time,
    )
    TwoPhaseAMR._transport_substep!(
        flow.velocity, flow.solver, flow.workspace,
        dt, flow.time, model,
    )
    TwoPhaseAMR._enforce_phase_velocity_boundary!(
        state.convected, flow.solver, flow.time + dt / 2,
    )
    TwoPhaseAMR._diffusion_substep!(
        flow.velocity, flow.solver, flow.workspace,
        dt, flow.time, model;
        tolerance, max_cycles,
    )
    TwoPhaseAMR._enforce_phase_velocity_boundary!(
        flow.velocity, flow.solver, flow.time + dt,
    )
    transported_velocity = velocity_vector(flow.velocity)
    transported_sheet = sheet_diagnostics(case)
    transported_amplitude = amplitude(case)
    geometry_after = constraint_geometry(
        state.projection_solver, state.projection_workspace,
        flow.velocity, dt,
    )

    if model.sigma > 0 || any(!iszero, model.gravity)
        TwoPhaseAMR._refresh_pressure_substep!(
            flow.solver, flow.workspace, model,
        )
        TwoPhaseAMR._phase_force_substep!(
            flow.velocity, flow.solver, flow.workspace, dt, model,
        )
    end
    forced_velocity = velocity_vector(flow.velocity)
    forced_sheet = sheet_diagnostics(case)

    stats = project!(
        flow.velocity, flow.pressure,
        state.projection_solver, dt, state.projection_workspace;
        tolerance, max_iterations = max_cycles,
    )
    projected_velocity = velocity_vector(flow.velocity)
    projected_sheet = sheet_diagnostics(case)
    projected_amplitude = amplitude(case)
    analytical_amplitude = instability_linear_amplitude(
        dt, case.dispersion;
        initial_amplitude = case.initial_amplitude,
    )

    changes = (;
        transport = relative_change(
            transported_velocity, initial_velocity, reference_norm,
        ),
        force = relative_change(
            forced_velocity, transported_velocity, reference_norm,
        ),
        projection = relative_change(
            projected_velocity, forced_velocity, reference_norm,
        ),
        total = relative_change(
            projected_velocity, initial_velocity, reference_norm,
        ),
    )
    circulation_changes = (;
        transport = transported_sheet.circulation -
            initial_sheet.circulation,
        force = forced_sheet.circulation - transported_sheet.circulation,
        projection = projected_sheet.circulation -
            forced_sheet.circulation,
        total = projected_sheet.circulation - initial_sheet.circulation,
    )
    return (;
        name,
        timestep_factor,
        dt,
        converged = stats.converged,
        iterations = stats.iterations,
        changes,
        circulation_changes,
        initial_sheet,
        transported_sheet,
        forced_sheet,
        projected_sheet,
        initial_amplitude,
        transported_amplitude,
        projected_amplitude,
        analytical_amplitude,
        amplitude_error = abs(projected_amplitude - analytical_amplitude),
        geometry_before,
        geometry_after,
        identity = projection_identity(stats),
        stats,
    )
end

function display_value(value::Missing)
    return "--"
end

function display_value(value::Integer)
    return string(value)
end

function display_value(value::Real)
    return @sprintf("%.3e", value)
end

function print_geometry_table(results)
    println("\nJump-space geometry before -> after interface transport")
    @printf(
        "%-22s %10s %10s %10s   %10s %10s %10s %10s %10s\n",
        "method", "dofs", "rows", "kernel", "dofs", "rows", "kernel",
        "N defect", "cond(S_N)",
    )
    for result in results
        before = result.geometry_before
        after = result.geometry_after
        @printf(
            "%-22s %10s %10s %10s   %10s %10s %10s %10s %10s\n",
            string(result.name),
            display_value(before.jump_dofs),
            display_value(before.constraints),
            display_value(before.kernel_dimension),
            display_value(after.jump_dofs),
            display_value(after.constraints),
            display_value(after.kernel_dimension),
            display_value(after.scaled_defect_linf),
            display_value(after.schur_condition),
        )
        if after.kind === :cmdgfm
            @printf(
                "  CMDGFM inferred-tangent rows %d -> %d; ",
                before.inferred_tangent, after.inferred_tangent,
            )
            @printf(
                "underresolved rows %d -> %d; ",
                before.underresolved, after.underresolved,
            )
            @printf(
                "minimum beta %.3e\n",
                after.minimum_beta,
            )
        end
    end
end

function print_step_table(results)
    println("\nOne-step changes, normalized by the initial velocity norm")
    @printf(
        "%-22s %10s %10s %10s %10s %12s %12s %10s\n",
        "method", "transport", "force", "projection", "total",
        "Delta Gamma P", "Delta Z P", "amp error",
    )
    for result in results
        @printf(
            "%-22s %10.3e %10.3e %10.3e %10.3e %12.3e %12.3e %10.3e\n",
            string(result.name),
            result.changes.transport,
            result.changes.force,
            result.changes.projection,
            result.changes.total,
            result.circulation_changes.projection,
            result.projected_sheet.enstrophy - result.forced_sheet.enstrophy,
            result.amplitude_error,
        )
    end
end

function print_solver_table(results)
    println("\nProjection and fallback diagnostics")
    @printf(
        "%-22s %9s %8s %11s %11s %10s %10s %10s\n",
        "method", "converged", "iters", "normal", "energy",
        "commuting", "fallback", "zeroed",
    )
    for result in results
        identity = result.identity
        @printf(
            "%-22s %9s %8d %11.3e %11s %10s %10s %10s\n",
            string(result.name),
            string(result.converged),
            result.iterations,
            identity.normal,
            display_value(identity.energy),
            display_value(identity.commuting),
            display_value(identity.fallback),
            display_value(identity.sliver_zeroed),
        )
    end
end

"""
    run_diagnostic(; timestep_factors=(1.0,), methods=nothing, kwargs...)

Run and print the one-step KH diagnostic.  Passing several time-step factors
repeats the same single-step experiment from identical initial data; it does
not integrate for multiple time steps.  This makes a non-vanishing projection
impulse under `dt -> 0` directly visible.
"""
function run_diagnostic(;
        timestep_factors = DEFAULT_TIMESTEP_FACTORS,
        methods = nothing,
        kwargs...,
    )
    configurations = projection_configurations()
    selected = isnothing(methods) ? collect(pairs(configurations)) : [
        name => getproperty(configurations, name) for name in methods
    ]
    all_results = NamedTuple[]
    for factor in timestep_factors
        println("\nKH single-step diagnostic: timestep factor = $factor")
        results = NamedTuple[]
        for (name, projection) in selected
            println("  running $name ...")
            push!(
                results,
                single_step_diagnostic(
                    name, projection;
                    timestep_factor = factor, kwargs...,
                ),
            )
        end
        print_geometry_table(results)
        print_step_table(results)
        print_solver_table(results)
        append!(all_results, results)
    end
    return all_results
end

function main(args = ARGS)
    timestep_factors = "--refine-dt" in args ?
        REFINEMENT_TIMESTEP_FACTORS : DEFAULT_TIMESTEP_FACTORS
    return run_diagnostic(; timestep_factors)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module KHSingleStepDiagnostic

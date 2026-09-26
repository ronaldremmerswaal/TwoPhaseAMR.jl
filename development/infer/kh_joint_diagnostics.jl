module JointKHDiagnostics

using Printf
using TwoPhaseAMR

include(joinpath(@__DIR__, "..", "..", "test", "integration",
    "interfacial_instability_problem.jl"))

"""Inspect why a refined KH run changes even when each KKT solve converges."""
function study(; density_ratio = 1000.0, weber = 5.0,
        resolution = 16, timestep_factor = 0.25,
        final_time = nothing, pressure_weight = 1000.0,
        commutator_weight = 1000.0, correction_cap = 1.0,
        sweeps = 3, active_mass_fraction = 1.0e-6,
        patch_radius = 2, variant = :joint_smoothed)
    projection = MimeticProjection(;
        variant, active_mass_fraction, pressure_weight,
        commutator_weight, correction_cap, fit_sweeps = sweeps,
        patch_radius,
    )
    case = two_velocity_instability_setup(
        ; cells_per_wavelength = resolution, density_ratio, weber,
        bond = 0.0, liquid_above = false,
        pressure_projection = projection,
    )
    add_diagnostic!(case.flow, KineticEnergy())
    add_diagnostic!(case.flow, SurfaceEnergy())
    add_diagnostic!(case.flow, TotalEnergy())
    target = _instability_target(case, final_time)
    trajectory = _run_instability!(
        case, target; cfl = 0.35, tolerance = 1.0e-7,
        max_cycles = 3000, timestep_factor,
    )
    analytical = [instability_linear_amplitude(
        time, case.dispersion;
        initial_amplitude = case.initial_amplitude,
    ) for time in trajectory.times]
    run = (; case..., trajectory..., analytical, final_time = target)
    error = normalized_amplitude_error(
        run.times, run.amplitudes, run.analytical,
    )
    diagnostics = run.projection_diagnostics
    maxima(key) = maximum(getproperty.(diagnostics, key))
    count_changes(key) = count(!iszero, diff(getproperty.(diagnostics, key)))
    println("density_ratio=$density_ratio We=$weber n=$resolution ",
        "factor=$timestep_factor steps=$(run.flow.steps) error=$error")
    for key in (
            :candidate_rows, :fallback_rows, :commuting_residual,
            :pressure_fallback_rows, :pressure_commuting_residual,
            :energy_identity_error, :normal_velocity_defect,
        )
        values = getproperty.(diagnostics, key)
        println("$key: min=$(minimum(values)) max=$(maximum(values)) ",
            "changes=$(count_changes(key)) final=$(last(values))")
    end
    for name in (:kinetic_energy, :surface_energy, :total_energy)
        series = diagnostic_history(run.flow, name)
        fractional = (last(series.values) - first(series.values)) /
            max(abs(first(series.values)), eps())
        println("$name: initial=$(first(series.values)) ",
            "final=$(last(series.values)) fractional_change=$fractional")
    end
    println("final amplitude=$(last(run.amplitudes)), ",
        "analytical=$(last(run.analytical))")
    for key in (:commuting_residual, :pressure_commuting_residual)
        selected = sortperm(getproperty.(diagnostics, key); rev = true)[
            1:min(5, length(diagnostics))]
        println("largest $key events:")
        for index in selected
            item = diagnostics[index]
            @printf("  step=%d t=%.6g value=%.6g rows=%d fallback=%d pressure_fallback=%d\n",
                index, item.time, getproperty(item, key),
                item.candidate_rows, item.fallback_rows,
                item.pressure_fallback_rows)
        end
    end
    return run
end

if abspath(PROGRAM_FILE) == @__FILE__
    study()
end

end # module

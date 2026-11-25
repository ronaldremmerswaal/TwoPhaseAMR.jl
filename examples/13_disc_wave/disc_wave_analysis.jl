# Measurements of the disc wave against its exact solution, shared by
# make_figures.jl and test/integration/disc_wave.jl: error norms, the phase and
# frequency of each mode, and sampled initial data. Include after disc_modes.jl;
# needs TwoPhaseAMR (`value_points`, `stable_timestep`), no plotting.

"""
    disc_wave_errors(operator, displacement, rate, modes, time; radius=1, speed=1)
    disc_wave_errors(run)

Volume-weighted RMS (`l2`) and maximum (`max`) errors of the displacement and
its rate at the `value_points` of `operator`, over the fluid unknowns (positive
open volume). `norms` holds the same norms of the exact solution.
"""
function disc_wave_errors(operator, displacement, rate, modes, time; radius = 1.0, speed = 1.0)
    squares = zeros(4)
    maxima = zeros(4)
    volume = 0.0
    for axis in 1:2
        nodes = value_points(operator, axis)
        for (index, face) in pairs(nodes.faces)
            weight = nodes.volumes[index]
            iszero(weight) && continue
            x, y = nodes.points[index]
            exact = disc_wave_exact(modes, x, y, time, axis; radius, speed)
            values = (
                displacement.interior[face] - exact[1], rate.interior[face] - exact[2],
                exact[1], exact[2],
            )
            for (slot, value) in pairs(values)
                squares[slot] += weight * value^2
                maxima[slot] = max(maxima[slot], abs(value))
            end
            volume += weight
        end
    end
    l2 = sqrt.(squares ./ volume)
    return (;
        displacement_l2 = l2[1], displacement_max = maxima[1],
        rate_l2 = l2[2], rate_max = maxima[2],
        norms = (;
            displacement_l2 = l2[3], displacement_max = maxima[3],
            rate_l2 = l2[4], rate_max = maxima[4],
        ),
    )
end

disc_wave_errors(run) = disc_wave_errors(
    run.operator, run.displacement, run.rate, run.modes, run.time; run.radius, run.speed,
)

"""Error `name` of `errors` relative to the same norm of the exact solution."""
relative_error(errors, name::Symbol) = getproperty(errors, name) / getproperty(errors.norms, name)

"""
    sample_initial_displacement!(displacement, operator, geometry, modes; radius, tolerance)

The exact initial displacement sampled at the `value_points`, for
`run_example(; initial_displacement! = sample_initial_displacement!)`: it excites
the stiff modes of the smallest cut cells (compare the Helmholtz projection).
"""
function sample_initial_displacement!(displacement, operator, geometry, modes; radius, tolerance)
    for axis in 1:2
        nodes = value_points(operator, axis)
        for (index, face) in pairs(nodes.faces)
            iszero(nodes.volumes[index]) && continue
            displacement.interior[face] = disc_wave_exact(modes, nodes.points[index]..., 0.0, axis; radius)[1]
        end
    end
    return displacement
end

"""
    PhaseTracker()

`run_example(; on_step = tracker)` records the unwrapped phase of every mode:
the projections `a`, `b` of the displacement and its rate on the exact shape `φ`
give `atan(-b/ω, a) = ωt` for the exact solution. See [`mode_frequencies`](@ref).
"""
mutable struct PhaseTracker
    shapes::Vector{Any}
    phases::Vector{Float64}
    history::Vector{Vector{Float64}}
end

PhaseTracker() = PhaseTracker(Any[], Float64[], Vector{Float64}[])

function (tracker::PhaseTracker)(state)
    if state.step == 0
        nodes = (value_points(state.operator, 1), value_points(state.operator, 2))
        tracker.shapes = [
            (nodes[mode.axis], [mode_shape(mode, point..., state.radius) for point in nodes[mode.axis].points])
                for mode in state.modes
        ]
        tracker.phases = zeros(length(state.modes))
        empty!(tracker.history)
    end
    for (index, mode) in pairs(state.modes)
        nodes, shape = tracker.shapes[index]
        projected_displacement = 0.0
        projected_rate = 0.0
        for (local_index, face) in pairs(nodes.faces)
            weight = nodes.volumes[local_index] * shape[local_index]
            projected_displacement += weight * state.displacement.interior[face]
            projected_rate += weight * state.rate.interior[face]
        end
        omega = mode_frequency(mode; state.radius, state.speed)
        phase = atan(-projected_rate / omega, projected_displacement)
        # Unwrap: steps advance each phase by less than π.
        tracker.phases[index] += rem2pi(phase - tracker.phases[index], RoundNearest)
    end
    push!(tracker.history, copy(tracker.phases))
    return nothing
end

"""
    mode_frequencies(tracker, run)

For each mode the exact `ω`, the measured `discrete` frequency (least-squares
slope of its unwrapped phase over the run) and its `spatial` part
`(2/dt) tan(Ω dt/2)`, without the implicit-midpoint lag.
"""
function mode_frequencies(tracker::PhaseTracker, run)
    times = collect(range(0.0, run.time; length = length(tracker.history)))
    centred = times .- sum(times) / length(times)
    return ntuple(length(run.modes)) do index
        phases = [entry[index] for entry in tracker.history]
        discrete = sum(centred .* phases) / sum(abs2, centred)
        exact = mode_frequency(run.modes[index]; run.radius, run.speed)
        (; mode = run.modes[index], exact, discrete, spatial = 2 / run.dt * tan(discrete * run.dt / 2))
    end
end

"""Leapfrog's stability limit `2/(c√ρ)`, `ρ` the largest absolute row sum of `Δ_h`."""
explicit_limit(operator; speed = 1.0) = 2sqrt(stable_timestep(operator; cfl = 1)) / speed

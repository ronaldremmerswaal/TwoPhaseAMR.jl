using TwoPhaseAMR, CairoMakie, Printf

Base.include(@__MODULE__, joinpath(@__DIR__, "sloshing.jl"))

# Bottom-mounted baffle of three quarters of the depth (Liu & Lin, case 5 in
# Yu, Xue & Zhu 2020). The strongly forced case shakes the baffled tank with
# A = 1 cm at its linear first-mode frequency.
const BAFFLE_HEIGHT = 0.375
const HERO = (; forcing_amplitude = 0.01, forcing_frequency = 4.314, final_time = 10.0)

"""Times at which a sampled signal changes sign, by linear interpolation."""
function sign_changes(times, values)
    crossings = Float64[]
    for i in 2:length(values)
        a, b = values[i - 1], values[i]
        if (a < 0 <= b) || (a > 0 >= b)
            push!(crossings, times[i - 1] - a * (times[i] - times[i - 1]) / (b - a))
        end
    end
    return crossings
end

_slope(x, y) = sum((x .- sum(x) / length(x)) .* (y .- sum(y) / length(y))) /
    sum(abs2, x .- sum(x) / length(x))

"""
Frequency and damping ratio of the antisymmetric wall elevation
`(η_left - η_right)/2` of free sloshing, which removes the even modes: a
least-squares line through its sign changes gives the half period, and one
through the logarithm of the largest `|η|` between consecutive sign changes
the decay rate. The first half cycle is skipped.
"""
function free_sloshing_fit(times, η)
    crossings = sign_changes(times, η)[2:end]
    frequency = pi / _slope(collect(eachindex(crossings)), crossings)
    peaks = map(2:length(crossings)) do i
        window = findall(t -> crossings[i - 1] < t < crossings[i], times)
        k = window[argmax(abs.(η[window]))]
        (times[k], abs(η[k]))
    end
    rate = -_slope(first.(peaks), log.(last.(peaks)))
    return (; frequency, damping_ratio = rate / frequency, cycles = length(peaks) / 2)
end

function antisymmetric_elevation(flow)
    left = diagnostic_history(flow, :left_elevation)
    right = diagnostic_history(flow, :right_elevation)
    return left.times, (left.values .- right.values) ./ 2
end

function sloshing_case(label; options...)
    elapsed = @elapsed run = run_example(; options...)
    println(
        "sloshing ", label, ": ", run.result.steps, " steps, ",
        run.result.mesh_updates, " mesh updates, ",
        TwoPhaseAMR.ncells(run.flow.system.mesh), " final cells, ",
        @sprintf("%.1f s", elapsed)
    )
    return run
end

function free_sloshing_figure(outdir, plain_modes, baffled_modes; level, amplitude = 0.01)
    figure = Figure(size = (900, 500))
    for (row, (label, modes, baffle_height)) in enumerate(
            (("plain tank", plain_modes, 0.0), ("baffled tank", baffled_modes, BAFFLE_HEIGHT))
        )
        run = sloshing_case(
            "free, $label, level $level"; max_level = level, forcing_amplitude = 0.0,
            initial_amplitude = amplitude, baffle_height, final_time = 12.0,
        )
        times, η = antisymmetric_elevation(run.flow)
        fit = free_sloshing_fit(times, η)
        linear = [modal_elevation(t, modes; initial_amplitude = amplitude) for t in times]
        reference = free_sloshing_fit(times, linear)
        @printf(
            "  frequency %.5f rad/s (same fit of linear theory %.5f, %+.3f%%; first linear mode %.5f), damping ratio %.5f over %.1f cycles\n",
            fit.frequency, reference.frequency,
            100 * (fit.frequency / reference.frequency - 1), first(modes).frequency,
            fit.damping_ratio, fit.cycles,
        )
        axis = Axis(
            figure[row, 1]; ylabel = "η (m)", xlabel = row == 2 ? "t (s)" : "",
            title = @sprintf(
                "%s: ω = %.4f rad/s (linear %.4f), ζ = %.4f",
                label, fit.frequency, reference.frequency, fit.damping_ratio
            ),
        )
        lines!(axis, times, linear; color = :gray60, linewidth = 1, label = "linear, undamped")
        lines!(axis, times, η; color = :dodgerblue3, linewidth = 1.2, label = "(η_left − η_right)/2")
        row == 1 && axislegend(axis; position = :rb, labelsize = 10)
    end
    # Long oscillating series: one pixel per unit keeps the PNG under 150 kB.
    save(joinpath(outdir, "free_sloshing.png"), figure; px_per_unit = 1)
    return nothing
end

# Extrema (t, η) of the nonlinear numerical benchmark of Liu & Lin (2008) at
# the left wall, read by eye from the plots reproduced by Yu, Xue & Zhu (2020),
# Figs. 3 (plain tank) and 6 (baffled tank); reading accuracy about ±1 mm.
const LIU_LIN = (
    plain = [
        (4.2, 0.026), (4.75, -0.03), (5.35, 0.0334), (5.95, -0.034), (6.55, 0.0434),
        (7.15, -0.04), (7.7, 0.0529), (8.3, -0.0447), (8.9, 0.0618), (9.5, -0.0492),
    ],
    baffled = [
        (0.57, 0.0021), (1.4, -0.0041), (1.95, 0.0056), (2.65, -0.0054),
        (3.25, 0.0059), (3.95, -0.0045), (4.5, 0.0045), (5.05, -0.0024),
        (5.7, 0.002), (6.15, -0.0022), (6.75, 0.002), (7.3, -0.0034),
        (7.9, 0.0038), (8.55, -0.0041), (9.15, 0.0046), (9.8, -0.0038),
    ],
)

"""Largest value of `sign*values` within `half` of `time`, times `sign`."""
function extremum_near(times, values, time, sign; half = 0.3)
    indices = findall(t -> abs(t - time) < half, times)
    return values[indices[argmax(sign .* values[indices])]]
end

function forced_response_figure(outdir, plain_modes, baffled_modes; level)
    figure = Figure(size = (900, 540))
    cases = (
        ("plain tank", plain_modes, 0.0, LIU_LIN.plain),
        ("baffled tank", baffled_modes, BAFFLE_HEIGHT, LIU_LIN.baffled),
    )
    for (row, (label, modes, baffle_height, benchmark)) in enumerate(cases)
        run = sloshing_case("forced, $label, level $level"; max_level = level, baffle_height)
        left = diagnostic_history(run.flow, :left_elevation)
        linear = [
            modal_elevation(t, modes; forcing_amplitude = 0.002, forcing_frequency = 5.29)
                for t in left.times
        ]
        for window in ((0, 2), (2, 5), (5, 10))
            indices = findall(t -> window[1] <= t <= window[2], left.times)
            @printf(
                "  t in [%g, %g]: max |η - linear| = %.5f m, η range [%.5f, %.5f], linear [%.5f, %.5f]\n",
                window..., maximum(abs.(left.values[indices] .- linear[indices])),
                extrema(left.values[indices])..., extrema(linear[indices])...,
            )
        end
        for (name, values) in (("simulation", left.values), ("linear theory", linear))
            deviations = [
                extremum_near(left.times, values, t, sign(η)) - η for (t, η) in benchmark
            ]
            @printf(
                "  %s extrema minus Liu & Lin: mean |.| %.5f m, mean %.5f m, max |.| %.5f m\n",
                name, sum(abs, deviations) / length(deviations),
                sum(deviations) / length(deviations), maximum(abs, deviations),
            )
        end
        axis = Axis(
            figure[row, 1]; ylabel = "η at left wall (m)", xlabel = row == 2 ? "t (s)" : "",
            title = "$label, A = 2 mm, ω = 5.29 rad/s",
        )
        lines!(axis, left.times, linear; color = :gray60, linewidth = 1, label = "linear theory")
        lines!(
            axis, left.times, left.values; color = :dodgerblue3, linewidth = 1.2,
            label = "simulation, level $level",
        )
        scatter!(
            axis, first.(benchmark), last.(benchmark); color = :darkorange2,
            markersize = 7, label = "Liu & Lin (2008), extrema read from plot",
        )
        row == 1 && axislegend(axis; position = :lt, labelsize = 10)
    end
    save(joinpath(outdir, "forced_response.png"), figure; px_per_unit = 1)
    return nothing
end

"""Cumulative trapezoidal integral of a sampled history."""
cumulative(times, values) = [0.0; cumsum((values[2:end] .+ values[1:(end - 1)]) .* diff(times) ./ 2)]

function hero_figures(outdir; level = 3)
    run = sloshing_case(
        "strongly forced baffled, level $level"; max_level = level,
        baffle_height = BAFFLE_HEIGHT, HERO...,
        # A level-3 frame needs about 2 MiB, so 512 MiB keeps every keyframe.
        recorder_options = (;
            keyframes = EquispacedTimes(81), retention = Retention(memory = 2^29),
        ),
    )
    print_simulation_summary(run.flow, run.result)
    left = diagnostic_history(run.flow, :left_elevation)
    right = diagnostic_history(run.flow, :right_elevation)
    strongest = argmax(max.(abs.(left.values), abs.(right.values)))
    @printf(
        "  largest wall elevation %.4f m (left) / %.4f m (right) at t = %.3f s\n",
        left.values[strongest], right.values[strongest], left.times[strongest],
    )
    energy = diagnostic_history(run.flow, :total_energy)
    power = diagnostic_history(run.flow, :forcing_power)
    work = cumulative(power.times, power.values)
    change = energy.values .- first(energy.values)
    @printf(
        "  input work %.3f J/m, energy change %.3f J/m, dissipated %.3f J/m at t = %.2f s\n",
        last(work), last(change), last(work) - last(change), last(energy.times),
    )
    figure = Figure(size = (900, 540))
    axis = Axis(figure[1, 1]; ylabel = "wall elevation (m)")
    lines!(axis, left.times, left.values; linewidth = 1.2, label = "left wall")
    lines!(axis, right.times, right.values; linewidth = 1.2, label = "right wall")
    axislegend(axis; position = :lt, labelsize = 10)
    axis = Axis(figure[2, 1]; xlabel = "t (s)", ylabel = "energy per unit depth (J/m)")
    lines!(axis, power.times, work; linewidth = 1.2, label = "work of the forcing")
    lines!(
        axis, energy.times, change; linewidth = 1.2,
        label = "change of kinetic + potential + surface energy",
    )
    lines!(axis, energy.times, work .- change; linewidth = 1.2, label = "dissipated (difference)")
    axislegend(axis; position = :lt, labelsize = 10)
    save(joinpath(outdir, "response.png"), figure; px_per_unit = 1)

    # Two build-up instants, the largest wall elevation and a quarter period
    # later, when the water runs over the baffle.
    store = snapshot_store(run.recorder)
    quarter = pi / (2 * HERO.forcing_frequency)
    times = [1.5, 3.5, left.times[strongest], left.times[strongest] + quarter]
    hero_frames(outdir, store, min.(times, HERO.final_time))
    return run
end

# Vorticity on a 31-level diverging scale with the PLIC surface and the solid,
# over the part of the tank that the water reaches.
const HERO_VIEW = (;
    field = :velocity_vorticity, camera = (0.0, 1.0, 0.15, 0.8),
    colorrange = (-20.0, 20.0), colormap = cgrad(:balance, 31; categorical = true),
)

"""Four stored frames in a 2 × 2 grid with the PLIC surface, baffle and AMR blocks."""
function snapshot_figure(path, store, times; field, camera, label, options...)
    figure = Figure(size = (1000, 720))
    plot = nothing
    for (k, time) in enumerate(times)
        frame = frame_at(store, time)
        axis = fieldaxis(
            figure[fld1(k, 2), mod1(k, 2)]; camera,
            title = @sprintf("t = %.2f s", frame.time),
        )
        plot = amrfield!(axis, frame, field; blocks = true, options...)
    end
    Colorbar(figure[1:2, 3], plot; label)
    resize_to_layout!(figure)
    save(path, figure)
    return path
end

function hero_frames(outdir, store, times)
    snapshot_figure(
        joinpath(outdir, "snapshots.png"), store, times; HERO_VIEW...,
        label = "vorticity (1/s)",
    )
    export_animation(
        joinpath(outdir, "sloshing.gif"), store; HERO_VIEW..., fps = 12,
        nframes = length(keyframes(store)), size = (800, 560),
    )
    return nothing
end

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    level = 2
    plain_modes = plain_tank_modes(sloshing_tank())
    seconds = @elapsed baffled_modes = baffled_tank_modes(
        sloshing_tank(; baffle_height = BAFFLE_HEIGHT),
    )
    @printf(
        "linear first-mode frequencies: plain %.5f rad/s, baffled %.5f rad/s (%.1f s)\n",
        first(plain_modes).frequency, first(baffled_modes).frequency, seconds,
    )
    free_sloshing_figure(outdir, plain_modes, baffled_modes; level)
    forced_response_figure(outdir, plain_modes, baffled_modes; level)
    hero_figures(outdir)
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end

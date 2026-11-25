using TwoPhaseAMR, CairoMakie, Printf

Base.include(@__MODULE__, joinpath(@__DIR__, "rayleigh_taylor.jl"))

"""Least-squares slope and intercept of `y` against `x`."""
function linear_fit(x, y)
    mx, my = sum(x) / length(x), sum(y) / length(y)
    slope = sum((x .- mx) .* (y .- my)) / sum(abs2, x .- mx)
    return slope, my - slope * mx
end

"""Central differences of `values` resampled at the spacing `step`."""
function resampled_rate(times, values, step)
    grid = collect(first(times):step:last(times))
    sampled = map(grid) do t
        i = clamp(searchsortedlast(times, t), 1, length(times) - 1)
        w = (t - times[i]) / (times[i + 1] - times[i])
        (1 - w) * values[i] + w * values[i + 1]
    end
    middle = (grid[1:(end - 2)] .+ grid[3:end]) ./ 2
    return middle, (sampled[3:end] .- sampled[1:(end - 2)]) ./ (2step)
end

# Linear stage: a fixed step of 0.005 (about 120 steps per e-folding) keeps
# the time error out of the measured rate; the automatic step is limited by
# the gravity-wave bound `√(h/(2Ag))` there, which leaves only 5–11 steps per
# e-folding at these levels. The rate is the least-squares slope of
# log((h_b + h_s)/2) while the amplitude grows from 6 to 20 times its initial
# value (k a ≈ 0.075–0.25), after the start from rest has relaxed to the
# growing mode and before the fundamental saturates.
const LINEAR_STEP = 0.005
const LINEAR_TIME = 2.3
const GROWTH_WINDOW = (6, 20)

function linear_case(level)
    local run
    seconds = @elapsed run = run_example(;
        case = :single_mode, max_level = level, final_time = LINEAR_TIME,
        dt = LINEAR_STEP,
    )
    bubble = diagnostic_history(run.flow, :bubble_height)
    spike = diagnostic_history(run.flow, :spike_height)
    times = bubble.times
    amplitude = (bubble.values .+ spike.values) ./ 2
    a0 = first(amplitude)
    linear = findall(a -> GROWTH_WINDOW[1] * a0 <= a <= GROWTH_WINDOW[2] * a0, amplitude)
    rate, _ = linear_fit(times[linear], log.(amplitude[linear]))
    p = run.parameters
    k = 2pi
    theory = linear_growth_rate(
        k; p.density, p.dynamic_viscosity, p.gravity, p.surface_tension,
    )
    inviscid = sqrt(p.atwood * p.gravity * k)
    ν = p.kinematic_viscosity
    println(
        @sprintf(
            "linear stage, level %d: %d steps, %.1f s; rate %.5f over t = %.3f–%.3f (viscous theory %.5f, %+.2f%%; inviscid %.5f; explicit √(Agk + ν²k⁴) - νk² %.5f)",
            level, run.result.steps, seconds, rate, times[first(linear)],
            times[last(linear)], theory, 100 * (rate / theory - 1), inviscid,
            sqrt(inviscid^2 + ν^2 * k^4) - ν * k^2,
        )
    )
    return (; level, seconds, times, amplitude, rate, theory, linear, inviscid)
end

# Nonlinear stage with automatic steps: measure the least-squares slope of
# the bubble height over `VELOCITY_WINDOW`. Later behaviour depends on
# resolution; see the README.
const VELOCITY_WINDOW = (4.0, 5.0)

function terminal_case(level)
    local run
    seconds = @elapsed run = run_example(;
        case = :single_mode, max_level = level,
        recorder_options = (; keyframes = EquispacedTimes(27)),
    )
    bubble = diagnostic_history(run.flow, :bubble_height)
    spike = diagnostic_history(run.flow, :spike_height)
    times = bubble.times
    plateau = findall(t -> VELOCITY_WINDOW[1] <= t <= VELOCITY_WINDOW[2], times)
    velocity, _ = linear_fit(times[plateau], bubble.values[plateau])
    spike_velocity, _ = linear_fit(times[plateau], spike.values[plateau])
    late = findall(t -> t >= last(times) - 0.5, times)
    late_velocity, _ = linear_fit(times[late], bubble.values[late])
    p = run.parameters
    reference = bubble_velocity(; p.atwood, p.gravity, wavelength = 1.0)
    froude = sqrt(p.atwood * p.gravity / (1 + p.atwood))
    cells = diagnostic_history(run.flow, :cells)
    println(
        @sprintf(
            "terminal stage, level %d: %d steps, %d mesh updates, %d final / %d peak cells, %.1f s; bubble velocity %.5f (Fr %.4f; Goncharov %.5f, Fr %.4f, %+.2f%%), over the last 0.5 %.5f (Fr %.4f); spike velocity %.5f (Fr %.4f); h_b = %.4f, h_s = %.4f at t = %.2f",
            level, run.result.steps, run.result.mesh_updates, last(cells.values),
            maximum(cells.values), seconds, velocity, velocity / froude, reference,
            reference / froude, 100 * (velocity / reference - 1), late_velocity,
            late_velocity / froude, spike_velocity, spike_velocity / froude,
            last(bubble.values), last(spike.values), last(times),
        )
    )
    return (;
        run, level, seconds, times, bubble = bubble.values, spike = spike.values,
        velocity, late_velocity, spike_velocity, reference, froude,
    )
end

function single_mode_figure(path, linear, terminal)
    figure = Figure(size = (1000, 380))
    rate_axis = Axis(
        figure[1, 1]; xlabel = "t", ylabel = "d ln a / dt",
        title = "Linear stage of a = (h_b + h_s)/2",
    )
    velocity_axis = Axis(
        figure[1, 2]; xlabel = "t", ylabel = "dh_b/dt",
        title = "Bubble velocity",
    )
    colors = Makie.wong_colors()
    for case in linear
        label = "λ/Δx = $(8 * 2^case.level)"
        t, rate = resampled_rate(case.times, log.(case.amplitude), 0.05)
        lines!(rate_axis, t, rate; color = colors[case.level - 1], label)
    end
    finest = last(linear)
    hlines!(rate_axis, [finest.theory]; color = :black, label = "viscous theory")
    hlines!(
        rate_axis, [finest.inviscid]; color = :black, linestyle = :dash,
        label = "inviscid",
    )
    window = finest.times[[first(finest.linear), last(finest.linear)]]
    vspan!(rate_axis, window...; color = (:gray, 0.15))
    ylims!(rate_axis, 1.2, 1.85)
    for case in terminal
        t, velocity = resampled_rate(case.times, case.bubble, 0.1)
        lines!(
            velocity_axis, t, velocity; color = colors[case.level - 1],
            label = "λ/Δx = $(8 * 2^case.level)",
        )
    end
    hlines!(
        velocity_axis, [last(terminal).reference]; color = :black,
        linestyle = :dash, label = "Goncharov (2-D)",
    )
    vspan!(velocity_axis, VELOCITY_WINDOW...; color = (:gray, 0.15))
    axislegend(rate_axis; position = :rb, labelsize = 11)
    axislegend(velocity_axis; position = :rb, labelsize = 11)
    save(path, figure; px_per_unit = 1)
    return nothing
end

# Multimode: α_b and α_s are the squared slopes of √h against t divided by
# A g over `MIXING_WINDOW`, after the initial modes have saturated and before
# the box-wide bubbles feel the walls.
const MIXING_WINDOW = (2.0, 4.0)

function multimode_case(level; final_time = 4.0, keyframes = 81)
    started = time_ns()
    run = run_example(; max_level = level, final_time = 0.0)
    initial_dye = (integrate(run.dye, :liquid), integrate(run.dye, :gas))
    fields = rayleigh_taylor_fields(run.flow, run.dye)
    recorder = Recorder(
        fields; diagnostics = run.flow, keyframes = EquispacedTimes(keyframes),
        preview = nothing,
    )
    result = simulate!(run.flow, final_time; observer = recorder)
    seconds = (time_ns() - started) / 1.0e9
    print_simulation_summary(run.flow, result)
    bubble = diagnostic_history(run.flow, :bubble_height)
    spike = diagnostic_history(run.flow, :spike_height)
    cells = diagnostic_history(run.flow, :cells)
    p = run.parameters
    function alpha_fit(history, window)
        selected = findall(t -> window[1] <= t <= window[2], history.times)
        slope, intercept = linear_fit(history.times[selected], sqrt.(history.values[selected]))
        return slope, intercept, slope^2 / (p.atwood * p.gravity)
    end
    slope_b, origin_b, alpha_b = alpha_fit(bubble, MIXING_WINDOW)
    slope_s, origin_s, alpha_s = alpha_fit(spike, MIXING_WINDOW)
    # Sensitivity to the window.
    for window in ((1.5, 3.5), (2.5, 4.0))
        println(
            @sprintf(
                "  window t = %.1f–%.1f: α_b = %.4f, α_s = %.4f", window...,
                alpha_fit(bubble, window)[3], alpha_fit(spike, window)[3],
            )
        )
    end
    final_dye = (integrate(run.dye, :liquid), integrate(run.dye, :gas))
    origin = diagnostic_history(run.flow, :spike_origin)
    light_origin = diagnostic_history(run.flow, :bubble_origin)
    for t in 1.0:1.0:final_time
        i = argmin(abs.(origin.times .- t))
        println(
            @sprintf(
                "  t = %.2f: h_b = %.4f, h_s = %.4f, heavy origin / h_b = %.3f, light origin / h_s = %.3f",
                origin.times[i], bubble.values[i], spike.values[i],
                origin.values[i] / bubble.values[i],
                light_origin.values[i] / spike.values[i],
            )
        )
    end
    println(
        @sprintf(
            "multimode, level %d: %d steps, %d mesh updates, %d final / %d peak cells, %.1f s; α_b = %.4f, α_s = %.4f (t = %.1f–%.1f, virtual origins %.3f, %.3f); h_b = %.4f, h_s = %.4f at t = %.3f; dye integrals %.9f / %.9f, change %.3e / %.3e",
            level, result.steps, result.mesh_updates, last(cells.values),
            maximum(cells.values), seconds, alpha_b, alpha_s, MIXING_WINDOW...,
            -origin_b / slope_b, -origin_s / slope_s, last(bubble.values),
            last(spike.values), last(bubble.times), initial_dye...,
            (final_dye .- initial_dye)...,
        )
    )
    return (;
        run, recorder, seconds, alpha_b, alpha_s, slope_b, origin_b, slope_s,
        origin_s, bubble, spike, origin, light_origin, initial_dye, final_dye,
    )
end

function mixing_figure(path, case)
    p = case.run.parameters
    figure = Figure(size = (960, 320))
    scale = p.atwood * p.gravity
    width_axis = Axis(
        figure[1, 1]; xlabel = "A g t²", ylabel = "h",
        title = "Penetration (1 % thresholds)",
    )
    root_axis = Axis(figure[1, 2]; xlabel = "t", ylabel = "√h", title = "√h and the fitted α")
    origin_axis = Axis(
        figure[1, 3]; xlabel = "t", ylabel = "mean initial distance / h",
        title = "Origin of the fluid that crossed y = 0",
    )
    for (history, slope, intercept, alpha, label, color) in (
            (case.bubble, case.slope_b, case.origin_b, case.alpha_b, "bubbles", :royalblue),
            (case.spike, case.slope_s, case.origin_s, case.alpha_s, "spikes", :firebrick),
        )
        lines!(width_axis, scale .* history.times .^ 2, history.values; color, label)
        lines!(root_axis, history.times, sqrt.(history.values); color, label)
        fit = range(MIXING_WINDOW...; length = 20)
        lines!(
            root_axis, fit, slope .* fit .+ intercept; color = :black,
            linestyle = :dash, label = @sprintf("α = %.3f", alpha),
        )
    end
    vspan!(root_axis, MIXING_WINDOW...; color = (:gray, 0.15))
    # Heavy fluid below y = 0 against the bubble penetration it was drawn
    # from, light fluid above y = 0 against the spike penetration.
    late = findall(>=(0.5), case.origin.times)
    lines!(
        origin_axis, case.origin.times[late],
        case.origin.values[late] ./ case.bubble.values[late];
        color = :firebrick, label = "heavy below y = 0, / h_b",
    )
    lines!(
        origin_axis, case.light_origin.times[late],
        case.light_origin.values[late] ./ case.spike.values[late];
        color = :royalblue, label = "light above y = 0, / h_s",
    )
    ylims!(origin_axis, 0, 0.4)
    axislegend(width_axis; position = :lt, labelsize = 11)
    axislegend(root_axis; position = :lt, labelsize = 11)
    axislegend(origin_axis; position = :lb, labelsize = 11)
    save(path, figure)
    return nothing
end

"""Dye panels of `store` at the keyframes nearest `times`, saved at one pixel per unit."""
function dye_panels(path, store, times; camera, colormap, aspect, size)
    frames = keyframes(store)
    figure = Figure(; size)
    plot = nothing
    for (i, time) in enumerate(times)
        frame = store[frames[argmin(abs.([info.time for info in frames] .- time))]]
        axis = fieldaxis(figure[1, i]; camera, title = @sprintf("t = %.2f", frame.time))
        plot = amrfield!(
            axis, frame, :dye; phase_rendering = :plic, colormap, interfacewidth = 0.7,
        )
        colsize!(figure.layout, i, Aspect(1, aspect))
    end
    Colorbar(figure[1, length(times) + 1], plot; label = "Dye: initial height")
    resize_to_layout!(figure)
    save(path, figure; px_per_unit = 1)
    return nothing
end

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    # Both sizes run the gallery configuration: the hero needs the level-5
    # multimode run and the validation the three single-mode resolutions.
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    started = time_ns()
    linear = [linear_case(level) for level in 2:5]
    terminal = [terminal_case(level) for level in 2:4]
    multimode = multimode_case(5)
    single_mode_figure(joinpath(outdir, "single_mode.png"), linear, terminal)
    # Twelve bands of the initial height make the dye read as stacked layers.
    banded = cgrad(:balance, 12; categorical = true)
    dye_panels(
        joinpath(outdir, "single_mode_dye.png"),
        snapshot_store(last(terminal).run.recorder), [3.5, 4.5, 5.5, 6.5];
        camera = (0.0, 1.0, -2.0, 1.0), aspect = 1 / 3, size = (760, 520),
        colormap = banded,
    )
    mixing_figure(joinpath(outdir, "mixing.png"), multimode)
    store = snapshot_store(multimode.recorder)
    window = (0.0, 1.0, -0.75, 0.75)
    dye_panels(
        joinpath(outdir, "dye.png"), store, [1.5, 2.75, 4.0];
        camera = window, aspect = 2 / 3, size = (800, 420), colormap = banded,
    )
    # One output frame per keyframe: a uniform time step and no held frames.
    export_animation(
        joinpath(outdir, "multimode.gif"), store; field = :dye, camera = window,
        phase_rendering = :plic, colormap = banded,
        diagnostics = [:bubble_height, :spike_height], fps = 12,
        nframes = length(keyframes(store)), size = (800, 520),
        title = frame -> @sprintf("t = %.2f", frame.time),
    )
    println(
        "make_figures($size): ", (time_ns() - started) / 1.0e9,
        " s including rendering (package loading excluded)",
    )
    return (; linear, terminal, multimode)
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end

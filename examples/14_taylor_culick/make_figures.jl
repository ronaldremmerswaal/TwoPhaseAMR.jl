using TwoPhaseAMR, CairoMakie, Printf

Base.include(@__MODULE__, joinpath(@__DIR__, "taylor_culick.jl"))

# In the example's units V_TC = 1, e = 1 and σ = 1/2.
const SURFACE_TENSION = 0.5

"""Least-squares slope of the finite `values` against `times` on `[t0, t1]`."""
function window_slope(times, values, t0, t1)
    keep = findall(i -> t0 <= times[i] <= t1 && isfinite(values[i]), eachindex(times))
    t, v = times[keep], values[keep]
    tm, vm = sum(t) / length(t), sum(v) / length(v)
    return sum((t .- tm) .* (v .- vm)) / sum(abs2, t .- tm)
end

"""Column-integrated liquid thickness on bins of width `width` along `x`."""
function thickness_profile(interface, width)
    mesh = interface.mesh
    h = zeros(round(Int, mesh.extent[1] / width))
    nx, ny = mesh.block_size
    for (b, block) in pairs(mesh.blocks)
        fraction = interior(interface, b)
        dx, dy = block.spacing
        bins = round(Int, dx / width)
        for j in 1:ny, i in 1:nx
            first = round(Int, (block.origin[1] + (i - 1) * dx) / width) + 1
            h[first:(first + bins - 1)] .+= fraction[i, j] * dy
        end
    end
    return (; x = ((1:length(h)) .- 0.5) .* width, h)
end

"""
Neck (thinnest point within 1.5 rim heights ahead of the thickest point) and
the crests and troughs of the capillary waves ahead of it, nearest first.
"""
function rim_profile(profile)
    (; x, h) = profile
    top = argmax(h)
    ahead = findall(i -> x[top] - 1.5h[top] < x[i] < x[top], eachindex(x))
    neck = ahead[argmin(h[ahead])]
    window = findall(i -> x[neck] - 12 < x[i] < x[neck], eachindex(x))[2:(end - 1)]
    crest(i) = h[i] > h[i - 1] && h[i] >= h[i + 1] && h[i] > 1 + 2.0e-3
    trough(i) = h[i] < h[i - 1] && h[i] <= h[i + 1] && h[i] < 1 - 2.0e-3
    return (;
        neck = x[neck], neck_thickness = h[neck],
        crests = reverse(x[filter(crest, window)]),
        troughs = reverse(x[filter(trough, window)]),
    )
end

"""
Centroid of the liquid the edge has gathered: of the excess `h − e` of the
column thickness `h` over the sheet thickness `e = 1`, where it is positive.
Undefined (`NaN`) before any liquid has gathered.
"""
function gathered_centroid(interface, width)
    (; x, h) = thickness_profile(interface, width)
    excess = max.(h .- 1, 0)
    return sum(x .* excess) / sum(excess)
end

"""
Run one case and reduce it to its histories and summary; `keep=true` also
records `keyframes` frames for the field figures.
"""
function retraction_case(label; keep = false, keyframes = 21, final_time = 12.0, kwargs...)
    local flow, recorder, result
    elapsed = @elapsed begin
        flow = run_example(; final_time = 0.0, kwargs...).flow
        width = 1 / 2^kwargs[:max_level]
        add_diagnostic!(
            flow, :gathered_centroid, state -> gathered_centroid(state.interface, width);
            label = "Gathered-liquid centroid", dimension = PhysicalDimension(0, 1, 0),
        )
        recorder = if keep
            fields = VisualizationFieldRegistry()
            register_default_fields!(fields, flow)
            Recorder(
                fields; diagnostics = flow, keyframes = EquispacedTimes(keyframes),
                preview = nothing,
            )
        end
        result = simulate!(flow, final_time; observer = something(recorder, NoObserver()))
    end
    names = (
        :tip_position, :edge_thickness, :liquid_momentum, :liquid_kinetic_energy,
        :kinetic_energy, :surface_energy, :phase_volume, :gathered_centroid,
    )
    histories = Dict(name => diagnostic_history(flow, name) for name in names)
    times = histories[:tip_position].times
    tip = histories[:tip_position].values
    # A proxy for the centre of a round rim: the tip minus half the largest
    # thickness. For a club-shaped edge it is the centre of the rounded nose.
    centre = tip .- histories[:edge_thickness].values ./ 2
    gathered = histories[:gathered_centroid]
    windows = collect(0.5:0.5:(final_time - 0.5))
    unit_speed(t, values) = [-window_slope(t, values, w - 0.5, w + 0.5) for w in windows]
    speeds = (;
        times = windows, tip = unit_speed(times, tip), centre = unit_speed(times, centre),
        gathered = unit_speed(gathered.times, gathered.values),
    )
    first_above(speed) = (i = findfirst(>=(0.9), speed); isnothing(i) ? NaN : windows[i])
    # The surface tension of the two flat surfaces pulls the liquid with 2σ,
    # so its momentum is -2σt from rest.
    momentum = -histories[:liquid_momentum].values ./ (2SURFACE_TENSION .* times)
    released = first(histories[:surface_energy].values) .- histories[:surface_energy].values
    liquid = histories[:liquid_kinetic_energy].values
    kinetic = histories[:kinetic_energy].values
    late = (final_time - 4, final_time)
    rate(values) = window_slope(times, values, final_time / 2, final_time)
    volume = histories[:phase_volume].values
    summary = (;
        label, elapsed, steps = flow.steps,
        cells = TwoPhaseAMR.ncells(flow.system.mesh), final_time,
        tip = last(tip), thickness = last(histories[:edge_thickness].values),
        tip_speed = -window_slope(times, tip, late...),
        centre_speed = -window_slope(times, centre, late...),
        gathered_speed = -window_slope(gathered.times, gathered.values, late...),
        tip_90 = first_above(speeds.tip), centre_90 = first_above(speeds.centre),
        gathered_90 = first_above(speeds.gathered),
        momentum = last(momentum),
        momentum_error = maximum(abs.(momentum[times .>= 1] .- 1)),
        released = last(released), liquid = last(liquid),
        gas = last(kinetic) - last(liquid), dissipated = last(released) - last(kinetic),
        liquid_rate = rate(liquid), release_rate = rate(released),
        volume_change = last(volume)[1] - first(volume)[1],
        unresolved = result.total_unresolved_curvature_faces,
    )
    @printf(
        "%s: %.1f s, %d steps, %d final cells, t=%.4f, faces without curvature %d; tip \
        %.4f, largest thickness %.4f; speeds on [T-4,T]: tip %.4f, round-rim centre \
        %.4f, gathered-liquid centroid %.4f; first unit window at 0.9 V: tip %.1f, \
        round-rim centre %.1f, gathered %.1f; momentum/(2σt) %.4f, largest deviation \
        for t >= 1 %.4f\n",
        label, elapsed, summary.steps, summary.cells, final_time, summary.unresolved,
        summary.tip, summary.thickness, summary.tip_speed, summary.centre_speed,
        summary.gathered_speed, summary.tip_90, summary.centre_90, summary.gathered_90,
        summary.momentum, summary.momentum_error
    )
    @printf(
        "  energy: released %.4f, liquid kinetic %.4f (%.3f), gas kinetic %.4f (%.3f), \
        dissipated %.4f (%.3f); rates on [T/2,T]: release %.4f, liquid kinetic %.4f \
        (%.3f of the release); liquid volume change %.3e\n",
        summary.released, summary.liquid, summary.liquid / summary.released, summary.gas,
        summary.gas / summary.released, summary.dissipated,
        summary.dissipated / summary.released, summary.release_rate, summary.liquid_rate,
        summary.liquid_rate / summary.release_rate, summary.volume_change
    )
    profile = thickness_profile(flow.interface, 1 / 2^kwargs[:max_level])
    flow = nothing
    GC.gc()
    return (; summary, histories, speeds, profile, recorder)
end

# Inviscid stationary varicose waves in the rim frame have the phase speed V:
# V² = (σk/ρ) tanh(ke/2), so (ke/2) tanh(ke/2) = 1 in the example's units.
function varicose_wavelength()
    q = 1.2
    for _ in 1:50
        q -= (q * tanh(q) - 1) / (tanh(q) + q * sech(q)^2)
    end
    return pi / q
end

"""Speeds on the unit windows centred at `times`, and dissipated energy shares."""
function speed_report(case, times)
    (; speeds) = case
    for t in times
        i = findfirst(==(t), speeds.times)
        isnothing(i) && continue
        @printf(
            "  %s, unit window at t=%.1f: tip %.4f, round-rim centre %.4f, gathered \
            centroid %.4f, (1 - V_tip/V) sqrt(t) %.3f, (1 - V_gathered/V) t %.3f\n",
            case.summary.label, t, speeds.tip[i], speeds.centre[i], speeds.gathered[i],
            (1 - speeds.tip[i]) * sqrt(t), (1 - speeds.gathered[i]) * t
        )
    end
    surface = case.histories[:surface_energy]
    kinetic = case.histories[:kinetic_energy]
    for t in (2.0, 4.0, 8.0, 12.0, 16.0, 20.0)
        t <= case.summary.final_time || continue
        i = argmin(abs.(surface.times .- t))
        released = first(surface.values) - surface.values[i]
        @printf(
            "  %s, t=%.3f: dissipated share of the released surface energy %.3f\n",
            case.summary.label, surface.times[i], (released - kinetic.values[i]) / released
        )
    end
    return nothing
end

"""
Largest thickness against the high-Oh law `e(1 + t/(4τ_vis))` of Savva & Bush
(2009, eq. 2.29) for a semicircular end, `τ_vis = μ_l e/(2σ) = Oh/√2 e/V`.
"""
function thickness_report(case, ohnesorge, times)
    viscous_time = ohnesorge / sqrt(2)
    thickness = case.histories[:edge_thickness]
    for t in times
        i = argmin(abs.(thickness.times .- t))
        sample = thickness.times[i]
        law = 1 + sample / (4viscous_time)
        @printf(
            "  %s, t=%.3f (%.2f τ_vis): largest thickness %.4f, law %.4f (%+.1f%%)\n",
            case.summary.label, sample, sample / viscous_time, thickness.values[i], law,
            100 * (thickness.values[i] / law - 1)
        )
    end
    return nothing
end

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    scale = size === :docs ? 1.0 : 1.25
    figure_size(w, h) = (round(Int, scale * w), round(Int, scale * h))
    colors = Makie.wong_colors()

    # Oh = 0.1: a rim with a neck and capillary waves, at 4, 8 and 16 cells
    # per thickness; the finest run records the animation.
    low = [
        retraction_case(
            "Oh = 0.1, level $level"; max_level = level, keep = level == 4,
            keyframes = 81,
        ) for level in (2, 3, 4)
    ]
    # Oh = 3: the edge thickens on the viscous time scale. Its region of
    # influence spreads along the sheet like sqrt(16νt), so a longer sheet keeps
    # it away from the symmetry wall until t = 20.
    high = [
        retraction_case(
            "Oh = 3, level $level"; ohnesorge = 3.0, sheet_length = 56,
            final_time = 20.0, max_level = level, keep = level == 3,
        ) for level in (2, 3)
    ]
    # The theory assumes a vacuum around the sheet.
    gas = [
        retraction_case(
            "Oh = 0.1, level 3, density ratio $(round(Int, ratio))";
            density_ratio = ratio, max_level = 3,
        ) for ratio in (100.0, 10.0)
    ]
    shared = retraction_case(
        "Oh = 0.1, level 3, one velocity"; max_level = 3,
        velocity_model = OneVelocityModel(),
    )
    for case in (last(low), last(high))
        speed_report(case, 1.0:1.0:19.0)
    end
    for case in high
        thickness_report(case, 3.0, (1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0))
    end

    # Tip retraction and speeds against the Taylor--Culick speed.
    figure = Figure(size = figure_size(900, 380))
    distance = Axis(
        figure[1, 1]; xlabel = "t V/e", ylabel = "retraction / e", title = "Tip retraction",
    )
    speed = Axis(
        figure[1, 2]; xlabel = "t V/e", ylabel = "speed / V",
        title = "Tip (solid) and gathered-liquid centroid (dashed) speed",
    )
    lines!(distance, [0, 20], [0, 20]; color = :gray50, linestyle = :dot, label = "V t")
    hlines!(speed, [1.0]; color = :gray50, linestyle = :dot)
    for (cases, color) in ((low, colors[1]), (high, colors[2]))
        for (k, case) in enumerate(cases)
            tip = case.histories[:tip_position]
            lines!(
                distance, tip.times, first(tip.values) .- tip.values;
                color = (color, k / length(cases)), label = case.summary.label,
            )
        end
        finest = last(cases)
        lines!(speed, finest.speeds.times, finest.speeds.tip; color, label = finest.summary.label)
        lines!(speed, finest.speeds.times, finest.speeds.gathered; color, linestyle = :dash)
    end
    axislegend(distance; position = :lt, labelsize = 10)
    axislegend(speed; position = :rb, labelsize = 10)
    ylims!(speed, 0, 1.2)
    save(joinpath(outdir, "retraction.png"), figure)

    # Final interfaces and horizontal phase velocities of the field runs.
    figure = Figure(size = figure_size(900, 470))
    plot = nothing
    for (row, case, window) in (
            (1, last(low), (4.0, 24.0, -2.5, 2.5)),
            (2, high[2], (24.0, 44.0, -2.5, 2.5)),
        )
        store = snapshot_store(case.recorder)
        frame = store[last(keyframes(store))]
        axis = fieldaxis(
            figure[row, 1]; camera = window,
            title = @sprintf("%s, t = %.2f e/V", case.summary.label, frame.time),
        )
        plot = amrfield!(axis, frame, :velocity_x; blocks = true, colorrange = (-1.25, 1.25))
    end
    Colorbar(figure[1:2, 2], plot; label = "horizontal phase velocity / V")
    save(joinpath(outdir, "interfaces.png"), figure)

    # Capillary waves ahead of the Oh = 0.1 rim.
    finest = last(low)
    rim = rim_profile(finest.profile)
    inviscid = varicose_wavelength()
    # Gordillo et al.'s Ω = μ/sqrt(ρσe/2) = sqrt(2) Oh.
    omega = sqrt(2) * 0.1
    long_wave = pi / sqrt(1 - 4omega^2)
    @printf(
        "Oh = 0.1, level 4 profile at t=%.4f: neck at x=%.4f, thickness %.4f; crests %s; \
        troughs %s; wavelengths: inviscid varicose %.4f, Gordillo et al. %.4f\n",
        finest.summary.final_time, rim.neck, rim.neck_thickness,
        string(round.(rim.crests; digits = 4)), string(round.(rim.troughs; digits = 4)),
        inviscid, long_wave
    )
    figure = Figure(size = figure_size(900, 300))
    axis = Axis(
        figure[1, 1]; xlabel = "x/e", ylabel = "thickness / e",
        title = @sprintf(
            "%s, t = %.2f e/V: liquid thickness ahead of the rim",
            finest.summary.label, finest.summary.final_time
        ),
    )
    (; x, h) = finest.profile
    keep = findall(s -> rim.neck - 10 <= s <= rim.neck + 1.5, x)
    lines!(axis, x[keep], h[keep]; color = :black, label = "column thickness")
    vlines!(
        axis, rim.neck .- inviscid .* (1:3); color = colors[3], linestyle = :dash,
        label = @sprintf("neck - n λ, λ = %.2f e (inviscid)", inviscid),
    )
    vlines!(
        axis, rim.neck .- long_wave .* (1:2); color = colors[4], linestyle = :dot,
        label = @sprintf("neck - n λ, λ = %.2f e (long wave)", long_wave),
    )
    axislegend(axis; position = :lt, labelsize = 10)
    save(joinpath(outdir, "profile.png"), figure)

    # Energy budget of the finest runs, after the first two time units in
    # which the released energy is still small.
    figure = Figure(size = figure_size(900, 340))
    for (column, case) in enumerate((last(low), last(high)))
        surface = case.histories[:surface_energy]
        times = surface.times
        released = first(surface.values) .- surface.values
        keep = findall(>=(2.0), times)
        liquid = case.histories[:liquid_kinetic_energy].values
        kinetic = case.histories[:kinetic_energy].values
        axis = Axis(
            figure[1, column]; xlabel = "t V/e", ylabel = "share of released surface energy",
            title = case.summary.label,
        )
        lines!(axis, times[keep], (liquid ./ released)[keep]; label = "liquid kinetic")
        lines!(axis, times[keep], ((released .- kinetic) ./ released)[keep]; label = "dissipated")
        hlines!(axis, [0.5]; color = :gray50, linestyle = :dot)
        ylims!(axis, 0, 1)
        column == 1 && axislegend(axis; position = :rt, labelsize = 10)
    end
    save(joinpath(outdir, "energy.png"), figure)

    # Hero animation: horizontal phase velocity and interface of the level-4
    # Oh = 0.1 run, with the tip against the Taylor--Culick line V t.
    store = snapshot_store(last(low).recorder)
    levels = 31
    export_animation(
        joinpath(outdir, "retraction.gif"), store; field = :velocity_x,
        camera = (4.0, 25.0, -3.0, 3.0), colorrange = (-1.25, 1.25),
        colormap = cgrad(:balance, levels; categorical = true),
        diagnostics = [:tip_position], references = Dict(:tip_position => t -> 24 - t),
        arrangement = :below,
        title = frame -> @sprintf("Oh = 0.1, t = %.2f e/V", frame.time),
        fps = 12, nframes = length(keyframes(store)), size = figure_size(760, 440),
    )
    return (; low, high, gas, shared)
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end

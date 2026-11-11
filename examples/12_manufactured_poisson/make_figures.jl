using CairoMakie

Base.include(@__MODULE__, joinpath(@__DIR__, "manufactured_poisson.jl"))

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    local run
    elapsed = @elapsed run = run_example(; resolution = 32)
    println("\nCMDGFM: setup and projection time (s; includes first-call compilation) = ", elapsed)
    println("Projection statistics: ", run.stats)
    println("Pressure, phase-gradient, and jump errors: ", run.errors)
    println("Pressure gauge offset: ", run.gauge_offset)
    println("Analytical scaled-jump components: ", run.analytical_jump_components)
    # Extreme cut-face errors would otherwise wash out the rest of the map.
    pmax = max(4 * run.errors.pressure.l2, eps())
    gmax = max(4 * run.errors.gradient.l2, eps())
    registry = VisualizationFieldRegistry()
    register_field!(
        registry, :p, run.pressure_error; colorrange = (-pmax, pmax),
        colormap = :balance
    )
    register_field!(
        registry, :gx, run.pressure_gradient_error, run.interface, Face{1}();
        colorrange = (-gmax, gmax), colormap = :balance
    )
    register_field!(
        registry, :gy, run.pressure_gradient_error, run.interface, Face{2}();
        colorrange = (-gmax, gmax), colormap = :balance
    )
    figure = Figure(size = size == :docs ? (900, 340) : (1200, 440))
    for (column, (field, title)) in enumerate(
            zip(
                (:p, :gx, :gy),
                ("Pressure error", "Phase ∂p/∂x error", "Phase ∂p/∂y error")
            )
        )
        axis = fieldaxis(figure[1, 2column - 1]; title)
        plot = amrfield!(axis, registry, field; phase_rendering = :plic)
        Colorbar(figure[1, 2column], plot)
    end
    save(joinpath(outdir, "hero.png"), figure)
    return (; run)
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end

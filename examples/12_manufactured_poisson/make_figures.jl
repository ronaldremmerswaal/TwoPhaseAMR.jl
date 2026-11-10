using CairoMakie

Base.include(@__MODULE__, joinpath(@__DIR__, "manufactured_poisson.jl"))

function make_figures(; outdir = joinpath(@__DIR__, "assets"), size = :docs)
    size in (:docs, :default) || throw(ArgumentError("size must be :docs or :default"))
    mkpath(outdir)
    methods = (:mimetic, :joint_face, :cmdgfm)
    runs = map(methods) do method
        local run
        elapsed = @elapsed run = run_example(; resolution = 32, method)
        println("\n", method, ": setup and projection time (s; includes first-call compilation) = ", elapsed)
        println("Projection statistics: ", run.stats)
        println("Pressure, phase-gradient, and jump errors: ", run.errors)
        println("Pressure gauge offset: ", run.gauge_offset)
        println("Analytical scaled-jump components: ", run.analytical_jump_components)
        run
    end
    table = Figure(size = size == :docs ? (900, 220) : (1200, 270))
    headings = ("Method", "p RMS", "p max", "∇p RMS", "∇p max", "ξx RMS", "ξx max", "ξy RMS", "ξy max")
    for (column, heading) in enumerate(headings)
        Label(table[1, column], heading; fontsize = 15)
    end
    for (row, (method, run)) in enumerate(zip(methods, runs))
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
        filename = method == :mimetic ? "hero.png" : "$(method)_errors.png"
        save(joinpath(outdir, filename), figure)
        e = run.errors
        values = (
            e.pressure.l2, e.pressure.linf, e.gradient.l2, e.gradient.linf,
            e.xi.x.l2, e.xi.x.linf, e.xi.y.l2, e.xi.y.linf,
        )
        Label(table[row + 1, 1], string(method); fontsize = 14)
        for (column, value) in enumerate(values)
            Label(table[row + 1, column + 1], string(round(value; sigdigits = 3)); fontsize = 14)
        end
    end
    save(joinpath(outdir, "error_table.png"), table)
    return (; methods, runs)
end

if abspath(PROGRAM_FILE) == @__FILE__
    make_figures()
end

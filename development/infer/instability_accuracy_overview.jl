"""Reproducible instability accuracy and energy comparison.

Run from the repository root with `julia --project=.
development/infer/instability_accuracy_overview.jl`. Results are appended to
`instability_accuracy_results_fixed.tsv` after every run and rendered to
`INSTABILITY_ACCURACY_OVERVIEW.md`; rerunning skips completed cases. Optional
comma-separated environment filters are `STUDY_CASES`, `STUDY_METHODS`,
`STUDY_MESHES`, and `STUDY_FACTORS`. `STUDY_OUTPUT` overrides the TSV path;
`STUDY_IMPORT` imports completed rows from comma-separated TSV paths before
running, for merging independent case batches.
"""
module InstabilityAccuracyOverview

using Dates
using Printf
using TwoPhaseAMR

include(joinpath(@__DIR__, "..", "..", "test", "integration",
    "interfacial_instability_problem.jl"))

const CASES = [
    (id = "kh_r1_we05", title = "KH: density 1:1, Weber 0.5",
        parameters = (; density_ratio = 1.0, weber = 0.5,
            bond = 0.0, liquid_above = false)),
    (id = "kh_r1000_we05", title = "KH: density 1000:1, Weber 0.5",
        parameters = (; density_ratio = 1000.0, weber = 0.5,
            bond = 0.0, liquid_above = false)),
    (id = "kh_r1_we5", title = "KH: density 1:1, Weber 5",
        parameters = (; density_ratio = 1.0, weber = 5.0,
            bond = 0.0, liquid_above = false)),
    (id = "kh_r1000_we5", title = "KH: density 1000:1, Weber 5",
        parameters = (; density_ratio = 1000.0, weber = 5.0,
            bond = 0.0, liquid_above = false)),
    (id = "rt_r3_bo05", title = "RT: density 3:1, Bond 0.5",
        parameters = (; density_ratio = 3.0, weber = 0.0,
            bond = 0.5, liquid_above = true)),
    (id = "rt_r1000_bo2", title = "RT: density 1000:1, Bond 2",
        parameters = (; density_ratio = 1000.0, weber = 0.0,
            bond = 2.0, liquid_above = true)),
    (id = "combined_r3", title = "Combined KH/RT: density 3:1, Weber 0.5, Bond 0.75",
        parameters = (; density_ratio = 3.0, weber = 0.5,
            bond = 0.75, liquid_above = true)),
    (id = "combined_r1000", title = "Combined KH/RT: density 1000:1, Weber 0.5, Bond 0.75",
        parameters = (; density_ratio = 1000.0, weber = 0.5,
            bond = 0.75, liquid_above = true)),
]
const METHODS = (
    :cmdgfm, :infer_aperture, :infer_smoothed,
    :infer_smoothed_stencil, :infer_plic,
)
const MESHES = (8, 16, 32)
const FACTORS = (1.0, 0.5, 0.25)
const TIME_MESH = 16
const FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)
const COLUMNS = (
    :case, :method, :mesh, :factor, :status, :error, :steps,
    :energy_initial, :energy_scale, :loss_0, :loss_25, :loss_50,
    :loss_75, :loss_100, :peak_loss, :peak_gain, :peak_abs_drift,
    :projection_identity_max, :final_amplitude, :analytical_amplitude,
    :message,
)

projection(::Val{:cmdgfm}) = CMDGFMProjection()
projection(::Val{:infer_aperture}) = MimeticProjection(
    jump_interpolant = :infer, infer_face_selection = :paired,
)
projection(::Val{:infer_smoothed}) = MimeticProjection(
    jump_interpolant = :infer, infer_face_selection = :smoothed,
)
projection(::Val{:infer_smoothed_stencil}) = MimeticProjection(
    jump_interpolant = :infer, infer_face_selection = :smoothed_stencil,
)
projection(::Val{:infer_plic}) = MimeticProjection(
    jump_interpolant = :infer, infer_face_selection = :plic,
)

function selected(name, allowed)
    filter = strip(get(ENV, name, ""))
    isempty(filter) && return collect(allowed)
    wanted = Set(strip.(split(filter, ',')))
    result = [item for item in allowed if string(item) in wanted]
    length(result) == length(wanted) || throw(ArgumentError(
        "$name contains an unknown value: $filter",
    ))
    return result
end

function sample_linear(times, values, time)
    time <= first(times) && return first(values)
    time >= last(times) && return last(values)
    right = searchsortedfirst(times, time)
    left = right - 1
    weight = (time - times[left]) / (times[right] - times[left])
    return (1 - weight) * values[left] + weight * values[right]
end

function energy_data(flow)
    total = diagnostic_history(flow, :total_energy)
    kinetic = diagnostic_history(flow, :kinetic_energy)
    potential = diagnostic_history(flow, :potential_energy)
    surface = diagnostic_history(flow, :surface_energy)
    # Potential energy can be negative or cancel another component. This
    # positive initial component norm makes relative drift comparable across
    # gravity and no-gravity cases without dividing by a cancellation.
    scale = kinetic.values[1] + abs(potential.values[1]) +
        surface.values[1]
    scale > 0 || throw(ArgumentError("initial energy scale is zero"))
    initial = total.values[1]
    losses = 100 .* (initial .- total.values) ./ scale
    final_time = total.times[end]
    samples = [
        sample_linear(total.times, losses, fraction * final_time)
        for fraction in FRACTIONS
    ]
    return (;
        initial, scale, samples,
        peak_loss = maximum(losses),
        peak_gain = -minimum(losses),
        peak_abs_drift = maximum(abs, losses),
    )
end

function run_case(case, method, mesh, factor)
    setup = two_velocity_instability_setup(
        ; cells_per_wavelength = mesh, case.parameters...,
        pressure_projection = projection(Val(method)),
        interface_transport = FrozenInterfaceFluxes(),
    )
    for diagnostic in (
            KineticEnergy(), PotentialEnergy(), SurfaceEnergy(), TotalEnergy(),
        )
        add_diagnostic!(setup.flow, diagnostic)
    end
    target = setup.default_final_time
    result = _run_instability!(
        setup, target; cfl = 0.35, tolerance = 1.0e-7,
        max_cycles = 3000, timestep_factor = factor,
        progress_interval = parse(Int, get(ENV, "STUDY_PROGRESS_INTERVAL", "250")),
    )
    analytical = [
        instability_linear_amplitude(
            time, setup.dispersion;
            initial_amplitude = setup.initial_amplitude,
        ) for time in result.times
    ]
    amplitude_error = normalized_amplitude_error(
        result.times, result.amplitudes, analytical,
    )
    energy = energy_data(setup.flow)
    identity_max = isempty(result.projection_diagnostics) ? NaN : maximum(
        entry.energy_identity_error for entry in result.projection_diagnostics
    )
    return (;
        case = case.id, method = string(method), mesh, factor,
        status = "ok", error = amplitude_error, steps = setup.flow.steps,
        energy_initial = energy.initial, energy_scale = energy.scale,
        loss_0 = energy.samples[1], loss_25 = energy.samples[2],
        loss_50 = energy.samples[3], loss_75 = energy.samples[4],
        loss_100 = energy.samples[5], peak_loss = energy.peak_loss,
        peak_gain = energy.peak_gain,
        peak_abs_drift = energy.peak_abs_drift,
        projection_identity_max = identity_max,
        final_amplitude = abs(last(result.amplitudes)),
        analytical_amplitude = abs(last(analytical)), message = "",
    )
end

function failed_run(case, method, mesh, factor, exception)
    message = replace(sprint(showerror, exception), r"[\t\r\n]+" => " ")
    return (;
        case = case.id, method = string(method), mesh, factor,
        status = "failed", error = NaN, steps = 0,
        energy_initial = NaN, energy_scale = NaN,
        loss_0 = NaN, loss_25 = NaN, loss_50 = NaN,
        loss_75 = NaN, loss_100 = NaN, peak_loss = NaN,
        peak_gain = NaN, peak_abs_drift = NaN,
        projection_identity_max = NaN, final_amplitude = NaN,
        analytical_amplitude = NaN, message,
    )
end

function load_rows(path)
    isfile(path) || return NamedTuple[]
    lines = readlines(path)
    isempty(lines) && return NamedTuple[]
    split(first(lines), '\t') == collect(string.(COLUMNS)) || throw(ArgumentError(
        "unexpected result-file header in $path",
    ))
    rows = NamedTuple[]
    float_columns = Set(COLUMNS[6:end-1])
    for line in Iterators.drop(lines, 1)
        parts = split(line, '\t'; keepempty = true)
        length(parts) == length(COLUMNS) || throw(ArgumentError(
            "incomplete result-file row in $path",
        ))
        values = map(COLUMNS, parts) do column, value
            column == :factor || column in float_columns && column != :steps ?
                parse(Float64, value) :
                column in (:mesh, :steps) ? parse(Int, value) : value
        end
        push!(rows, NamedTuple{COLUMNS}(Tuple(values)))
    end
    return rows
end

function append_row(path, row)
    mkpath(dirname(path))
    if !isfile(path)
        open(path, "w") do io
            println(io, join(string.(COLUMNS), '\t'))
        end
    end
    open(path, "a") do io
        fields = collect(string.(getproperty.(Ref(row), COLUMNS)))
        isempty(fields[end]) && (fields[end] = "-")
        println(io, join(fields, '\t'))
    end
end

key(row) = (row.case, row.method, row.mesh, Float64(row.factor))
lookup(rows) = Dict(key(row) => row for row in rows)

function number(value; digits = 3)
    isfinite(value) || return "—"
    return @sprintf("%.*g", digits, value)
end

function row_at(index, case, method, mesh, factor)
    return get(index, (case.id, string(method), mesh, factor), nothing)
end

function endpoint_rate(coarse, fine, refinement_ratio)
    (isnothing(coarse) || isnothing(fine) ||
        coarse.status != "ok" || fine.status != "ok") && return NaN
    return log(coarse.error / fine.error) / log(refinement_ratio)
end

function endpoint_cell(coarse, fine, refinement_ratio)
    isnothing(fine) && return "not run"
    fine.status == "ok" || return fine.status
    order = number(endpoint_rate(coarse, fine, refinement_ratio))
    return "$(number(fine.error)) ($order)"
end

function cell(row, field; digits = 3)
    isnothing(row) && return "not run"
    row.status == "failed" && return "failed"
    row.status == "skipped" && return "skipped"
    row.status == "ok" || return row.status
    return number(getproperty(row, field); digits)
end

function render(path, rows; cases = CASES, methods = METHODS,
        meshes = MESHES, factors = FACTORS)
    index = lookup(rows)
    open(path, "w") do io
        println(io, "# Instability accuracy: CMDGFM and inferred jumps")
        println(io)
        println(io, "Generated $(Dates.now(Dates.UTC)) UTC by `development/infer/instability_accuracy_overview.jl` with Julia $(VERSION).")
        println(io)
        println(io, "The comparison includes CMDGFM and four mimetic " *
            "inferred-jump variants. All runs use a uniform mesh, " *
            "frozen interface fluxes, " *
            "the same aperture-based divergence, CFL 0.35, pressure tolerance " *
            "1e-7 and the case's linear-theory final time. The interface " *
            "starts with amplitude 0.01 wavelength. `:infer_aperture` uses " *
            "raw cell-integrated D(a_l n), `:infer_smoothed` uses a one-ring " *
            "sum of aligned cut-cell aperture vectors, and `:infer_plic` " *
            "uses the reconstructed PLIC normal. These three use the " *
            "same paired K-orthogonal tangent/normal split. " *
            "`:infer_smoothed_stencil` uses the smoothed aperture normal " *
            "interpolated to faces but CMDGFM's beta-based facewise choice " *
            "between imposed and inferred jump components. On unmatched " *
            "paired faces, both aperture variants use the raw vector " *
            "interpolated from neighboring cells. All four share the " *
            "existing diagonal K and symmetric KKT projection. " *
            "The mimetic solver " *
            "uses separate exact Cholesky factors for its independent " *
            "preconditioner blocks. CMDGFM uses its own " *
            "pressure-dependent jump inference.")
        println(io)
        println(io, "CMDGFM rows already measured with the same Julia 1.12.7 " *
            "and unchanged CMDGFM implementation were carried into the " *
            "post-fix data file. Every inferred-method row in this report " *
            "is recomputed with the separated exact-block preconditioner; " *
            "pre-fix inferred rows are not used.")
        println(io)
        println(io, "Amplitude error is the time-integrated complex first-mode " *
            "L2 error relative to linear theory, normalized by the theory's " *
            "time-integrated norm. A mesh rate is log2(error at n/2 / error " *
            "at n); a time rate is log2(error at 2Δt / error at Δt). " *
            "These are *analytical-error* rates; a non-increasing rate is not " *
            "required for temporal boundedness. Ratios compare to CMDGFM at " *
            "the same mesh and time-step factor.")
        println(io)
        println(io, "Energy loss is `100 (E(0)-E(t))/S` in percent, where " *
            "`E=kinetic+gravity potential+surface` and " *
            "`S=kinetic(0)+|gravity potential(0)|+surface(0)`. " *
            "Positive means loss, negative means gain. The measure includes " *
            "errors from transport, body force, curvature, and projection; " *
            "it is not solely the projection energy identity. Time-history " *
            "samples are linearly interpolated at fractions of final time. " *
            "The `max |drift|` column uses every recorded step.")
        println(io)
        println(io, "| Case | ρₗ/ρg | We | Bo | Orientation | T |")
        println(io, "|---|---:|---:|---:|---|---:|")
        for case in cases
            p = case.parameters
            d = instability_dispersion(; p...)
            orientation = p.liquid_above ? "liquid above" : "liquid below"
            println(io, "| $(case.title) | $(number(p.density_ratio)) | " *
                "$(number(p.weber)) | $(number(p.bond)) | " *
                "$orientation | $(number(_default_instability_final_time(d); digits=5)) |")
        end
        println(io)
        println(io, "## Observed convergence orders from endpoints")
        println(io)
        println(io, "Each cell shows `fine-endpoint amplitude error (observed order)`. " *
            "The order is `log(error at coarse endpoint / error at fine " *
            "endpoint) / log(4)`. " *
            "A negative value means the measured error increased.")
        println(io)
        println(io, "### Mesh: n=8 to n=32 (time-step factor 1)")
        println(io)
        println(io, "| Physics case | " *
            join(["`$method`" for method in methods], " | ") * " |")
        println(io, "|---|" * join(fill("---:", length(methods)), "|") * "|")
        for case in cases
            cells = [endpoint_cell(
                row_at(index, case, method, 8, 1.0),
                row_at(index, case, method, 32, 1.0), 4.0,
            ) for method in methods]
            println(io, "| $(case.title) | " * join(cells, " | ") * " |")
        end
        println(io)
        println(io, "### Time step: factor 1 to 0.25 (n=$TIME_MESH)")
        println(io)
        println(io, "| Physics case | " *
            join(["`$method`" for method in methods], " | ") * " |")
        println(io, "|---|" * join(fill("---:", length(methods)), "|") * "|")
        for case in cases
            cells = [endpoint_cell(
                row_at(index, case, method, TIME_MESH, 1.0),
                row_at(index, case, method, TIME_MESH, 0.25), 4.0,
            ) for method in methods]
            println(io, "| $(case.title) | " * join(cells, " | ") * " |")
        end
        println(io)
        for case in cases
            println(io, "## $(case.title)")
            println(io)
            println(io, "### Mesh refinement (time-step factor 1)")
            println(io)
            println(io, "| Method | n | Amplitude error | Rate | / CMDGFM | " *
                "Final energy loss % | Max abs drift % | Steps |")
            println(io, "|---|---:|---:|---:|---:|---:|---:|---:|")
            for method in methods, (i, mesh) in pairs(meshes)
                row = row_at(index, case, method, mesh, 1.0)
                previous = i == 1 ? nothing :
                    row_at(index, case, method, meshes[i - 1], 1.0)
                reference = row_at(index, case, :cmdgfm, mesh, 1.0)
                rate = !isnothing(row) && !isnothing(previous) &&
                    row.status == "ok" && previous.status == "ok" ?
                    log(previous.error / row.error) /
                        log(mesh / meshes[i - 1]) : NaN
                ratio = !isnothing(row) && !isnothing(reference) &&
                    row.status == "ok" && reference.status == "ok" ?
                    row.error / reference.error : NaN
                println(io, "| `$method` | $mesh | $(cell(row, :error)) | " *
                    "$(number(rate)) | $(number(ratio)) | " *
                    "$(cell(row, :loss_100)) | " *
                    "$(cell(row, :peak_abs_drift)) | " *
                    "$(isnothing(row) ? "—" : row.steps) |")
            end
            println(io)
            println(io, "### Time-step refinement (n=$TIME_MESH)")
            println(io)
            println(io, "| Method | Δt factor | Amplitude error | Rate | / CMDGFM | " *
                "Final energy loss % | Max abs drift % | Steps |")
            println(io, "|---|---:|---:|---:|---:|---:|---:|---:|")
            for method in methods, (i, factor) in pairs(factors)
                row = row_at(index, case, method, TIME_MESH, factor)
                previous = i == 1 ? nothing :
                    row_at(index, case, method, TIME_MESH, factors[i - 1])
                reference = row_at(index, case, :cmdgfm, TIME_MESH, factor)
                rate = !isnothing(row) && !isnothing(previous) &&
                    row.status == "ok" && previous.status == "ok" ?
                    log(previous.error / row.error) /
                        log(factors[i - 1] / factor) : NaN
                ratio = !isnothing(row) && !isnothing(reference) &&
                    row.status == "ok" && reference.status == "ok" ?
                    row.error / reference.error : NaN
                println(io, "| `$method` | $(number(factor)) | " *
                    "$(cell(row, :error)) | $(number(rate)) | " *
                    "$(number(ratio)) | $(cell(row, :loss_100)) | " *
                    "$(cell(row, :peak_abs_drift)) | " *
                    "$(isnothing(row) ? "—" : row.steps) |")
            end
            println(io)
            println(io, "### Energy loss over time (n=$TIME_MESH, Δt factor 1)")
            println(io)
            println(io, "| Method | 0 | 0.25T | 0.5T | 0.75T | T | " *
                "Peak loss % | Peak gain % | Max projection identity residual |")
            println(io, "|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for method in methods
                row = row_at(index, case, method, TIME_MESH, 1.0)
                println(io, "| `$method` | $(cell(row, :loss_0)) | " *
                    "$(cell(row, :loss_25)) | $(cell(row, :loss_50)) | " *
                    "$(cell(row, :loss_75)) | $(cell(row, :loss_100)) | " *
                    "$(cell(row, :peak_loss)) | $(cell(row, :peak_gain)) | " *
                    "$(cell(row, :projection_identity_max)) |")
            end
            println(io)
            failures = [row for row in rows if row.case == case.id &&
                row.status != "ok"]
            for failure in failures
                println(io, "- $(uppercasefirst(failure.status)) run `$(failure.method)` n=$(failure.mesh) " *
                    "factor=$(failure.factor): $(failure.message)")
            end
            isempty(failures) || println(io)
        end
        println(io, "## Reproduction and interpretation")
        println(io)
        println(io, "Run `julia --project=. " *
            "development/infer/instability_accuracy_overview.jl`. " *
            "The script resumes from `development/infer/instability_accuracy_results_fixed.tsv` " *
            "and writes this report after every run. A failed run is recorded " *
            "and is not silently omitted. To retry failed runs, remove only " *
            "their TSV rows. The raw table preserves unrounded metrics. " *
            "The benchmark uses the exact setup in " *
            "`test/integration/interfacial_instability_problem.jl`.")
        println(io)
        println(io, "A small projection energy-identity residual does not imply " *
            "small total-energy drift, because the complete time step also " *
            "transports the interface and applies gravity and capillarity. " *
            "Likewise, agreement with linear theory can cease to be a pure " *
            "discretization-error measure after strong nonlinear growth.")
    end
end

function main()
    cases = selected("STUDY_CASES", getproperty.(CASES, :id))
    chosen_cases = [case for case in CASES if case.id in cases]
    methods = Symbol.(selected("STUDY_METHODS", METHODS))
    meshes = Int.(selected("STUDY_MESHES", MESHES))
    factors = Float64.(selected("STUDY_FACTORS", FACTORS))
    data_path = get(ENV, "STUDY_OUTPUT",
        joinpath(@__DIR__, "instability_accuracy_results_fixed.tsv"))
    report_path = joinpath(dirname(data_path), "INSTABILITY_ACCURACY_OVERVIEW.md")
    rows = load_rows(data_path)
    seen = Set(key(row) for row in rows)
    imports = strip(get(ENV, "STUDY_IMPORT", ""))
    if !isempty(imports)
        for source in strip.(split(imports, ','))
            abspath(source) == abspath(data_path) && continue
            for row in load_rows(source)
                identifier = key(row)
                identifier in seen && continue
                append_row(data_path, row)
                push!(rows, row)
                push!(seen, identifier)
            end
        end
    end
    for case in chosen_cases, method in methods
        scheduled = [(mesh, 1.0) for mesh in meshes]
        append!(scheduled, [(TIME_MESH, factor) for factor in factors
            if factor != 1.0])
        for (mesh, factor) in scheduled
            identifier = (case.id, string(method), mesh, factor)
            identifier in seen && continue
            @printf("START case=%s method=%s n=%d factor=%g\n",
                case.id, string(method), mesh, factor)
            flush(stdout)
            row = try
                run_case(case, method, mesh, factor)
            catch exception
                failed_run(case, method, mesh, factor, exception)
            end
            append_row(data_path, row)
            push!(rows, row)
            push!(seen, identifier)
            render(report_path, rows; cases = chosen_cases,
                methods, meshes, factors)
            @printf("END case=%s method=%s n=%d factor=%g status=%s error=%g energy_loss_pct=%g steps=%d\n", case.id, string(method),
                mesh, factor, row.status, row.error, row.loss_100, row.steps)
            flush(stdout)
        end
    end
    render(report_path, rows; cases = chosen_cases, methods, meshes, factors)
    println("Report: $report_path")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module

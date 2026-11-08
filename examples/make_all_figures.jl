# Generate the committed gallery assets with the examples environment.
# Usage: julia --project=examples examples/make_all_figures.jl [--only 06,07] [--size docs|default]
using CairoMakie

CairoMakie.activate!(visible = false)

function gallery_options(args)
    selected = nothing
    size = :docs
    i = 1
    while i <= length(args)
        arg = args[i]
        if arg == "--only" || arg == "--size"
            i += 1
            i <= length(args) || error("$arg requires a value")
            value = args[i]
        elseif startswith(arg, "--only=") || startswith(arg, "--size=")
            arg, value = split(arg, '='; limit = 2)
        else
            error("Unknown argument: $arg")
        end
        if arg == "--only"
            selected = split(value, ',')
            all(id -> occursin(r"^\d{2}$", id), selected) ||
                error("--only expects comma-separated two-digit example numbers")
        else
            value in ("docs", "default") || error("--size must be docs or default")
            size = Symbol(value)
        end
        i += 1
    end
    return (; selected, size)
end

function main(args)
    options = gallery_options(args)
    folders = filter(readdir(@__DIR__)) do name
        occursin(r"^\d{2}_[a-z0-9_]+$", name) &&
            isdir(joinpath(@__DIR__, name))
    end
    if !isnothing(options.selected)
        missing = setdiff(options.selected, first.(split.(folders, '_'; limit = 2)))
        isempty(missing) || error("Unknown example number(s): $(join(missing, ", "))")
        filter!(name -> first(split(name, '_'; limit = 2)) in options.selected, folders)
    end
    isempty(folders) && error("No example figure scripts found")
    for folder in folders
        isfile(joinpath(@__DIR__, folder, "make_figures.jl")) ||
            error("Missing figure script: $folder")
    end
    failures = String[]
    started = time()
    for folder in folders
        println("\n=== $folder ($(options.size)) ===")
        flush(stdout)
        t = time()
        try
            measurement = @timed begin
                example = Module(gensym(:GalleryExample))
                Base.include(example, joinpath(@__DIR__, folder, "make_figures.jl"))
                entrypoint = Base.invokelatest(getproperty, example, :make_figures)
                Base.invokelatest(entrypoint; size = options.size)
                nothing
            end
            println("Incremental compilation in generator: ",
                round(measurement.compile_time; digits = 2),
                " s (shared process, includes rendering)")
            for (root, _, files) in walkdir(joinpath(@__DIR__, folder, "assets"))
                for file in sort(files)
                    path = joinpath(root, file)
                    bytes = filesize(path)
                    println(relpath(path, @__DIR__), ": ", bytes, " bytes")
                    extension = lowercase(splitext(file)[2])
                    limit = extension == ".png" ? 150_000 :
                        extension == ".gif" ? 2_000_000 : nothing
                    options.size !== :docs || isnothing(limit) || bytes <= limit ||
                        error("$path exceeds the $(limit)-byte gallery limit")
                end
            end
        catch err
            push!(failures, folder)
            showerror(stderr, err, catch_backtrace())
            println(stderr)
        end
        println(folder, " elapsed: ", round(time() - t; digits = 2), " s")
        flush(stdout)
        GC.gc()
    end
    println("Total generation time: ", round(time() - started; digits = 2), " s")
    isempty(failures) || error("Figure generation failed: $(join(failures, ", "))")
    return nothing
end

main(ARGS)

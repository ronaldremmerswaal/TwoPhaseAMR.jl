using Documenter
using TwoPhaseAMR

# CI builds the technical reference in a separate job and places the PDF in
# docs/src/assets/techref before this script runs. Locally, build it here when
# LaTeX is available so method.md's PDF links resolve.
let techref = joinpath(@__DIR__, "techref"),
        local_pdf = joinpath(techref, "techref.pdf"),
        assets = joinpath(@__DIR__, "src", "assets", "techref"),
        published = joinpath(assets, "TwoPhaseAMR-techref.pdf")
    if !isfile(published) && !isfile(local_pdf) && !isnothing(Sys.which("latexmk"))
        cd(techref) do
            run(`latexmk -pdf -interaction=nonstopmode -halt-on-error -quiet techref.tex`)
        end
    end
    if isfile(local_pdf)
        mkpath(assets)
        cp(local_pdf, published; force = true)
    end
end

# The repository examples are the source of the gallery. Copy their images and
# adapt relative links for Documenter's generated pages; never run simulations
# during a documentation build.
function prepare_examples()
    root = normpath(joinpath(@__DIR__, "..", "examples"))
    pages_dir = joinpath(@__DIR__, "src", "examples")
    assets_dir = joinpath(@__DIR__, "src", "assets", "examples")
    rm(pages_dir; recursive = true, force = true)
    rm(assets_dir; recursive = true, force = true)
    mkpath(pages_dir)
    mkpath(assets_dir)
    folders = filter(readdir(root)) do name
        occursin(r"^\d{2}_[a-z0-9_]+$", name) && isdir(joinpath(root, name))
    end
    isempty(folders) && error("No numbered example folders found")
    pages = Pair{String, String}[]
    for folder in folders
        source = joinpath(root, folder)
        isfile(joinpath(source, "README.md")) ||
            error("Missing example README: $folder")
        isdir(joinpath(source, "assets")) ||
            error("Missing example assets: $folder")
        text = read(joinpath(source, "README.md"), String)
        for image in eachmatch(r"!\[[^\]]*\]\((assets/[^)]+)\)", text)
            isfile(joinpath(source, image.captures[1])) ||
                error("Missing image $(image.captures[1]) in $folder")
        end
        text = replace(
            text,
            "](assets/" => "](../assets/examples/$folder/",
            "](./assets/" => "](../assets/examples/$folder/",
            "](../../docs/src/" => "](../",
        )
        for file in readdir(source)
            endswith(file, ".jl") || continue
            url = "https://github.com/ronaldremmerswaal/TwoPhaseAMR.jl/blob/main/examples/$folder/$file"
            text = replace(text, "]($file)" => "]($url)", "](./$file)" => "]($url)")
        end
        for other in folders
            text = replace(
                text,
                "](../$other/)" => "]($other.md)",
                "](../$other/README.md)" => "]($other.md)",
            )
        end
        title = strip(replace(first(split(text, '\n')), r"^#\s*" => ""))
        write(joinpath(pages_dir, "$folder.md"), text)
        cp(joinpath(source, "assets"), joinpath(assets_dir, folder))
        push!(pages, title => "examples/$folder.md")
    end
    index = read(joinpath(root, "README.md"), String)
    for folder in folders
        index = replace(
            index,
            "]($folder/)" => "](examples/$folder.md)",
            "]($folder/README.md)" => "](examples/$folder.md)",
        )
    end
    write(joinpath(@__DIR__, "src", "examples.md"), index)
    return pages
end

example_pages = prepare_examples()


makedocs(
    modules = [TwoPhaseAMR],
    sitename = "TwoPhaseAMR.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        collapselevel = 1,
        edit_link = "main",
        canonical = "https://ronaldremmerswaal.github.io/TwoPhaseAMR.jl/stable/",
        size_threshold = 400 * 2^10,
        size_threshold_warn = 300 * 2^10,
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => [
            "Quick start" => "getting-started.md",
            "Choosing a model" => "choosing-a-model.md",
        ],
        "Tutorials" => [
            "Single-phase flow" => "single-phase-tutorial.md",
            "Adaptive refinement" => "adaptive-refinement.md",
            "Diagnostics" => "diagnostics.md",
            "Visualization" => "visualization.md",
        ],
        "How-to guides" => [
            "Common tasks" => "how-to.md",
            "Advanced workflows" => "advanced-workflows.md",
        ],
        "Examples" => [
            "Gallery" => "examples.md",
            example_pages...,
        ],
        "Explanation" => [
            "Method reference" => "method.md",
            "Two-phase flow" => "two-phase-flow.md",
            "Two-velocity advection" => "two-velocity-advection.md",
            "Staggered advection" => "staggered-advection.md",
            "Pressure gradient" => "pressure-gradient.md",
            "Poisson solver" => "poisson.md",
            "Navier--Stokes method" => "navier-stokes.md",
            "Shared-memory parallelism" =>
                "shared-memory-parallelism.md",
        ],
        "Reference" => [
            "Public API" => "api.md",
            "API functions" => "api-functions.md",
        ],
        "Developer guide" => [
            "Developer architecture" => "architecture.md",
            "Adding an equation system" =>
                "adding-an-equation-system.md",
            "Testing and validation" => "testing.md",
            "Literature coverage TODOs" => "literature-todos.md",
        ],
    ],
    checkdocs = :exports,
)

deploydocs(
    repo = "github.com/ronaldremmerswaal/TwoPhaseAMR.jl.git",
    devbranch = "main",
    push_preview = true,
)

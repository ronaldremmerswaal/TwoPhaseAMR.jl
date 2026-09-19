using Documenter
using TwoPhaseAMR

makedocs(
    modules = [TwoPhaseAMR],
    sitename = "TwoPhaseAMR.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        collapselevel = 1,
        edit_link = "main",
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting-started.md",
        "Tutorials" => [
            "Single-phase flow" => "single-phase-tutorial.md",
            "Adaptive refinement" => "adaptive-refinement.md",
            "Diagnostics" => "diagnostics.md",
            "Visualization" => "visualization.md",
        ],
        "Advanced workflows" => "advanced-workflows.md",
        "Numerical methods" => [
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
        "Contributing" => [
            "Developer architecture" => "architecture.md",
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

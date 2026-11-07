using Documenter
using TwoPhaseAMR

makedocs(
    modules = [TwoPhaseAMR],
    sitename = "TwoPhaseAMR.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        collapselevel = 1,
        edit_link = "main",
        canonical = "https://ronaldremmerswaal.github.io/TwoPhaseAMR.jl/stable/",
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
            "Run and extend examples" => "examples.md",
        ],
        "Explanation" => [
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
        "Migration" => "migration.md",
        "Developer guide" => [
            "Developer architecture" => "architecture.md",
            "Adding an equation system" =>
                "adding-an-equation-system.md",
            "Testing and validation" => "testing.md",
            "API decisions" => "api-decisions.md",
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

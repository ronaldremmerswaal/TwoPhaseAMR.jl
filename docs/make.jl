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
            "Visualization" => "visualization.md",
        ],
        "Advanced workflows" => "advanced-workflows.md",
        "Numerical methods" => [
            "Two-phase extension" => "two-phase-development.md",
            "Two-velocity advection" => "two-velocity-advection.md",
            "Staggered advection" => "staggered-advection.md",
            "Pressure gradient" => "pressure-gradient.md",
            "Poisson solver" => "poisson.md",
            "Navier--Stokes method" => "navier-stokes.md",
            "Implicit diffusion review" =>
                "implicit-diffusion-solver-review.md",
            "Matrix-free preconditioning" =>
                "matrix-free-geometric-preconditioning.md",
            "Shared-memory parallelism" =>
                "shared-memory-parallelism.md",
        ],
        "Reference" => [
            "Public API" => "api.md",
            "API functions" => "api-functions.md",
            "Literature review" => "literature-review.md",
            "Tree backend decision" => "adr/0001-tree-backend.md",
        ],
    ],
    checkdocs = :exports,
)

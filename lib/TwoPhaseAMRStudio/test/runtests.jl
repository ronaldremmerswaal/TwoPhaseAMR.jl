using Test

# Discover test_*.jl files adjacent to this harness, sorted by filename.
# ARGS selects filenames by case-sensitive substring, OR across filters.
# No filters runs all discovered files; an empty or unmatched filter is an error.
files = sort(
    filter(
        name -> startswith(name, "test_") && endswith(name, ".jl") &&
            isfile(joinpath(@__DIR__, name)), readdir(@__DIR__)
    )
)
isempty(files) && error("No Studio test files discovered")
any(isempty, ARGS) && error("Empty Studio test filter")
selected = isempty(ARGS) ? files : filter(name -> any(arg -> occursin(arg, name), ARGS), files)
isempty(selected) && error("No Studio test files selected by $(repr(ARGS))")

@testset verbose = true showtiming = true "TwoPhaseAMRStudio" begin
    for file in selected
        @testset "$file" begin
            include(joinpath(@__DIR__, file))
        end
    end
end

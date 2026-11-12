using Test
using TwoPhaseAMR

include("groups.jl")

function run_test_file(path)
    if get(ENV, "TWOPHASE_TEST_PROGRESS", "false") == "true"
        println("TEST_FILE ", path)
        flush(stdout)
    end
    return @testset "$path" begin
        include(joinpath(@__DIR__, path))
    end
end

function check_registry()
    registered = Set(vcat(last.(TEST_GROUPS)..., TEST_HELPERS, TEST_EXCLUSIONS))
    found = Set{String}()
    for (dir, _, files) in walkdir(@__DIR__)
        for file in files
            endswith(file, ".jl") || continue
            push!(found, relpath(joinpath(dir, file), @__DIR__))
        end
    end
    @test isempty(setdiff(found, registered))
    @test isempty(setdiff(registered, found))
    # Every default group runs in exactly one CI unit shard.
    sharded = vcat(last.(UNIT_SHARDS)...)
    @test allunique(sharded)
    @test Set(sharded) == Set(default_groups())
    return nothing
end

"""
Groups that run in their own CI job instead of the sharded default suite:
`performance` (allocation contracts, `contracts` job) and
`threading_equivalence` (fresh 1-, 2- and 4-thread processes,
`threading-equivalence` job). The registry check requires every other group
to be in exactly one unit shard.
"""
const SEPARATE_GROUPS = ("performance", "threading_equivalence")

default_groups() = [
    "registry";
    filter(
        name -> !startswith(name, "integration/") && !(name in SEPARATE_GROUPS),
        first.(TEST_GROUPS),
    )
]

function selected_groups(args)
    names = first.(TEST_GROUPS)
    if isempty(args)
        return default_groups()
    end
    selected = String[]
    for arg in args
        isempty(arg) && error("Empty test group")
        if arg == "integration"
            append!(selected, filter(name -> startswith(name, "integration/"), names))
        elseif startswith(arg, "shard:")
            shard = arg[(length("shard:") + 1):end]
            matches = filter(pair -> first(pair) == shard, UNIT_SHARDS)
            isempty(matches) && error("Unknown unit shard: $shard")
            append!(selected, last(only(matches)))
        elseif arg == "registry" || arg in names
            push!(selected, arg)
        else
            error("Unknown test group: $arg")
        end
    end
    return unique(selected)
end

args = isempty(ARGS) ? split(get(ENV, "TWOPHASE_TEST_GROUP", "")) : ARGS
selected = selected_groups(args)
@testset verbose = true showtiming = true "TwoPhaseAMR" begin
    included = Set{String}()
    for group in selected
        @testset "$group" begin
            if group == "registry"
                check_registry()
            else
                files = last(only(filter(pair -> first(pair) == group, TEST_GROUPS)))
                for file in files
                    if !(file in included)
                        run_test_file(file)
                        push!(included, file)
                    end
                end
            end
        end
    end
end

#!/usr/bin/env julia
"""Validate one copied CFL tile-max source against frozen compact and focused tests."""

using SHA
using TOML
using Test

length(ARGS) == 7 || error(
    "usage: validation_driver.jl SOURCE COMPACT_TEST FOCUSED_TEST FOCUSED_SHA SOURCE_SHA EXPECT_TILE_MAX REPORT")
source, test_file, focused_file, expected_focused_sha, expected_source_sha,
    expected_flag, report = ARGS
expected_flag in ("true", "false") || error("EXPECT_TILE_MAX must be true or false")
expected_tile_max = expected_flag == "true"
hashfile(path) = bytes2hex(SHA.sha256(read(path)))
hashfile(focused_file) == expected_focused_sha || error("focused test hash differs")
module_file = abspath(joinpath(source, "src", "TwoPhaseAMR.jl"))
advection_file = abspath(joinpath(source, "src", "operators", "advection",
                                  "cell_advection.jl"))
hashfile(advection_file) == expected_source_sha || error("copied advection source differs")

original_test = read(test_file, String)
header = "using Test\nusing TwoPhaseAMR\n"
startswith(original_test, header) || error("compact suite import changed")
adapted_test = replace(original_test, header => "using Test\nusing .TwoPhaseAMR\n";
                       count=1)

Base.include(Main, module_file)
using .TwoPhaseAMR
parentmodule(TwoPhaseAMR) === Main || error("copied module did not load in Main")
has_tile_max = isdefined(TwoPhaseAMR, :_cell_advection_rate_tile_layout)
has_tile_max == expected_tile_max || error("tile-max helper presence differs")
(:rate_tile_max in fieldnames(Base.unwrap_unionall(TwoPhaseAMR.AdvectionWorkspace))) ==
    expected_tile_max || error("tile-max workspace field presence differs")
tile_max_method_file = ""
if has_tile_max
    method = only(collect(methods(TwoPhaseAMR._cell_advection_rate_tile_layout)))
    tile_max_method_file = realpath(String(method.file))
    tile_max_method_file == realpath(advection_file) ||
        error("tile-max method resolved outside copied source")
end

Base.include_string(Main, adapted_test, abspath(test_file))
Base.include(Main, abspath(focused_file))
focused_checks = run_tile_max_focused(expected_tile_max)
focused_checks > 0 || error("focused validation contained no assertions")

open(report, "w") do io
    TOML.print(io, Dict(
        "copied_module_file" => module_file,
        "copied_module_sha256" => hashfile(module_file),
        "loaded_advection_file" => advection_file,
        "loaded_advection_sha256" => hashfile(advection_file),
        "tile_max_method_file" => tile_max_method_file,
        "tile_max_expected" => expected_tile_max,
        "original_test_file" => abspath(test_file),
        "original_test_sha256" => hashfile(test_file),
        "adapted_test_sha256" => bytes2hex(SHA.sha256(codeunits(adapted_test))),
        "focused_test_file" => abspath(focused_file),
        "focused_test_sha256" => hashfile(focused_file),
        "focused_assertions" => focused_checks,
        "julia_version" => string(VERSION),
        "threads" => Threads.nthreads(:default),
        "status" => "passed"))
end

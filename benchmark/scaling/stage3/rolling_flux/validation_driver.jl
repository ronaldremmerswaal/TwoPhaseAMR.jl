#!/usr/bin/env julia
"""Validate a copied rolling-flux source against compact and focused suites."""

using SHA
using TOML
using Test

length(ARGS) == 5 ||
    error("usage: validation_driver.jl SOURCE TEST_FILE EXPECTED_SOURCE_SHA EXPECT_ROLLING REPORT")
source, test_file, expected_sha, rolling_flag, report = ARGS
rolling_flag in ("true", "false") || error("EXPECT_ROLLING must be true or false")
expected_rolling = rolling_flag == "true"
module_file = abspath(joinpath(source, "src", "TwoPhaseAMR.jl"))
advection_file = abspath(joinpath(source, "src", "operators", "advection",
                                  "cell_advection.jl"))
focused_file = joinpath(@__DIR__, "focused_validation.jl")
hashfile(path) = bytes2hex(SHA.sha256(read(path)))
hashfile(advection_file) == expected_sha || error("copied advection source hash differs")
isfile(focused_file) || error("focused rolling validation file missing")

original_test = read(test_file, String)
header = "using Test\nusing TwoPhaseAMR\n"
startswith(original_test, header) || error("compact suite import changed")
adapted_test = replace(original_test, header => "using Test\nusing .TwoPhaseAMR\n";
                       count=1)

Base.include(Main, module_file)
using .TwoPhaseAMR
parentmodule(TwoPhaseAMR) === Main || error("copied module did not load in Main")
has_rolling = isdefined(TwoPhaseAMR, :_cell_advection_rolling_increments!)
has_rolling == expected_rolling || error("rolling-flux helper presence differs")
rolling_method_file = ""
if has_rolling
    rolling_method = only(collect(methods(TwoPhaseAMR._cell_advection_rolling_increments!)))
    rolling_method_file = realpath(String(rolling_method.file))
    rolling_method_file == realpath(advection_file) ||
        error("rolling-flux method resolved outside copied source")
end
(:rolling_flux in fieldnames(Base.unwrap_unionall(TwoPhaseAMR.AdvectionWorkspace))) ==
    expected_rolling ||
    error("rolling workspace field presence differs")

Base.include_string(Main, adapted_test, abspath(test_file))
Base.include(Main, focused_file)

open(report, "w") do io
    TOML.print(io, Dict(
        "copied_module_file" => module_file,
        "copied_module_sha256" => hashfile(module_file),
        "loaded_advection_file" => advection_file,
        "loaded_advection_sha256" => hashfile(advection_file),
        "rolling_method_file" => rolling_method_file,
        "rolling_expected" => expected_rolling,
        "original_test_file" => abspath(test_file),
        "original_test_sha256" => hashfile(test_file),
        "adapted_test_sha256" => bytes2hex(SHA.sha256(codeunits(adapted_test))),
        "focused_test_file" => abspath(focused_file),
        "focused_test_sha256" => hashfile(focused_file),
        "julia_version" => string(VERSION),
        "threads" => Threads.nthreads(:default),
        "status" => "passed"))
end

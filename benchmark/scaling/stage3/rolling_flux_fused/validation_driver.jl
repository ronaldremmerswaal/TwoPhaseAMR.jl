#!/usr/bin/env julia
"""Validate a frozen row-fused copy with compact and focused advection suites."""

using SHA
using TOML
using Test

length(ARGS) == 7 || error(
    "usage: validation_driver.jl SOURCE TEST_FILE FOCUSED_FILE FOCUSED_SHA SOURCE_SHA EXPECT_FUSED REPORT")
source, test_file, focused_file, expected_focused_sha, expected_source_sha,
    fused_flag, report = ARGS
fused_flag in ("true", "false") || error("EXPECT_FUSED must be true or false")
expected_fused = fused_flag == "true"
hashfile(path) = bytes2hex(SHA.sha256(read(path)))
hashfile(focused_file) == expected_focused_sha || error("focused test hash differs")
module_file = abspath(joinpath(source, "src", "TwoPhaseAMR.jl"))
advection_file = abspath(joinpath(source, "src", "operators", "advection",
                                  "cell_advection.jl"))
hashfile(advection_file) == expected_source_sha || error("copied advection source hash differs")

original_test = read(test_file, String)
header = "using Test\nusing TwoPhaseAMR\n"
startswith(original_test, header) || error("compact suite import changed")
adapted_test = replace(original_test, header => "using Test\nusing .TwoPhaseAMR\n";
                       count=1)

Base.include(Main, module_file)
using .TwoPhaseAMR
parentmodule(TwoPhaseAMR) === Main || error("copied module did not load in Main")
has_fused = isdefined(TwoPhaseAMR, :_cell_advection_rolling_rows!)
has_fused == expected_fused || error("row-fused helper presence differs")
isdefined(TwoPhaseAMR, :_cell_advection_rolling_increments!) &&
    error("two-pass rolling helper remains in copied source")
isdefined(TwoPhaseAMR, :_cell_advection_apply_increments!) &&
    error("separate increment update remains in copied source")
(:rolling_flux in fieldnames(Base.unwrap_unionall(TwoPhaseAMR.AdvectionWorkspace))) ==
    expected_fused || error("rolling workspace field presence differs")
fused_method_file = ""
if expected_fused
    fused_method = only(collect(methods(TwoPhaseAMR._cell_advection_rolling_rows!)))
    fused_method_file = realpath(String(fused_method.file))
    fused_method_file == realpath(advection_file) ||
        error("row-fused method resolved outside copied source")
end

Base.include_string(Main, adapted_test, abspath(test_file))
Base.include(Main, abspath(focused_file))

open(report, "w") do io
    TOML.print(io, Dict(
        "copied_module_file" => module_file,
        "copied_module_sha256" => hashfile(module_file),
        "loaded_advection_file" => advection_file,
        "loaded_advection_sha256" => hashfile(advection_file),
        "fused_method_file" => fused_method_file,
        "fused_expected" => expected_fused,
        "original_test_file" => abspath(test_file),
        "original_test_sha256" => hashfile(test_file),
        "adapted_test_sha256" => bytes2hex(SHA.sha256(codeunits(adapted_test))),
        "focused_test_file" => abspath(focused_file),
        "focused_test_sha256" => hashfile(focused_file),
        "julia_version" => string(VERSION),
        "threads" => Threads.nthreads(:default),
        "status" => "passed"))
end

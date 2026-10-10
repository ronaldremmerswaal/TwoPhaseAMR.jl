#!/usr/bin/env julia
"""Run the compact frozen-reference suite against an explicitly copied module."""

using SHA
using TOML

length(ARGS) == 4 ||
    error("usage: flux_tile_validation_driver.jl SOURCE TEST_FILE EXPECTED_SOURCE_SHA REPORT")
source, test_file, expected_sha, report = ARGS
module_file = abspath(joinpath(source, "src", "TwoPhaseAMR.jl"))
advection_file = abspath(joinpath(source, "src", "operators", "advection",
                                  "cell_advection.jl"))
hashfile(path) = bytes2hex(SHA.sha256(read(path)))
hashfile(advection_file) == expected_sha || error("copied advection source hash differs")

# The archived test begins with `using TwoPhaseAMR`. Replace only that import
# so package resolution cannot silently load the working tree or an installed
# package instead of the requested copied source.
original_test = read(test_file, String)
header = "using Test\nusing TwoPhaseAMR\n"
startswith(original_test, header) || error("frozen-reference test import changed")
adapted_test = replace(original_test, header => "using Test\nusing .TwoPhaseAMR\n";
                       count=1)

Base.include(Main, module_file)
using .TwoPhaseAMR
parentmodule(TwoPhaseAMR) === Main || error("copied module did not load in Main")
regular_method = only(collect(methods(TwoPhaseAMR._cell_advection_regular_fluxes!)))
method_file = realpath(String(regular_method.file))
method_file == realpath(advection_file) ||
    error("regular-face method resolved outside the requested copied source: $method_file")
Base.include_string(Main, adapted_test, abspath(test_file))

open(report, "w") do io
    TOML.print(io, Dict(
        "copied_module_file" => module_file,
        "copied_module_sha256" => hashfile(module_file),
        "loaded_advection_file" => advection_file,
        "loaded_advection_sha256" => hashfile(advection_file),
        "regular_flux_method_file" => method_file,
        "regular_flux_method_line" => regular_method.line,
        "original_test_file" => abspath(test_file),
        "original_test_sha256" => hashfile(test_file),
        "adapted_test_sha256" =>
            bytes2hex(SHA.sha256(codeunits(adapted_test))),
        "import_adapter" => "using TwoPhaseAMR -> using .TwoPhaseAMR",
        "julia_version" => string(VERSION),
        "threads" => Threads.nthreads(:default),
        "status" => "passed"))
end

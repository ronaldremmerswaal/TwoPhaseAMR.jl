#!/usr/bin/env julia
"""Validate every packed face code and the compact suite in a copied module."""

using SHA
using TOML
using Test

length(ARGS) == 5 ||
    error("usage: validation_driver.jl SOURCE TEST_FILE EXPECTED_SOURCE_SHA EXPECT_SELECTOR REPORT")
source, test_file, expected_sha, selector_flag, report = ARGS
selector_flag in ("true", "false") || error("EXPECT_SELECTOR must be true or false")
expected_selector = selector_flag == "true"
module_file = abspath(joinpath(source, "src", "TwoPhaseAMR.jl"))
advection_file = abspath(joinpath(source, "src", "operators", "advection",
                                  "cell_advection.jl"))
hashfile(path) = bytes2hex(SHA.sha256(read(path)))
hashfile(advection_file) == expected_sha || error("copied advection source hash differs")

original_test = read(test_file, String)
header = "using Test\nusing TwoPhaseAMR\n"
startswith(original_test, header) || error("compact suite import changed")
adapted_test = replace(original_test, header => "using Test\nusing .TwoPhaseAMR\n";
                       count=1)

Base.include(Main, module_file)
using .TwoPhaseAMR
parentmodule(TwoPhaseAMR) === Main || error("copied module did not load in Main")
sum_method = only(collect(methods(TwoPhaseAMR._cell_advection_ordered_sum)))
method_file = realpath(String(sum_method.file))
method_file == realpath(advection_file) ||
    error("ordered-sum method resolved outside copied source: $method_file")
has_selector = isdefined(TwoPhaseAMR, :_cell_advection_select_flux)
has_selector == expected_selector || error("scalar-selector helper presence differs")
selector_method_file = ""
if has_selector
    selector_method = only(collect(methods(TwoPhaseAMR._cell_advection_select_flux)))
    selector_method_file = realpath(String(selector_method.file))
    selector_method_file == realpath(advection_file) ||
        error("scalar-selector method resolved outside copied source")
end

function original_sum(values, order::UInt8, ::Type{T}) where {T}
    integrated = zero(T)
    integrated += values[Int(order & 0x03) + 1]
    integrated += values[Int((order >> 2) & 0x03) + 1]
    integrated += values[Int((order >> 4) & 0x03) + 1]
    integrated += values[Int(order >> 6) + 1]
    return integrated
end

function validate_codes()
    checks = Ref(0)
    @testset "all packed face codes preserve exact accumulation" begin
        for Flux in (Float32, Float64), Accum in unique((Flux, Float64))
            large = ldexp(one(Flux), precision(Flux))
            patterns = ((Flux(-0.0), Flux(0.0), Flux(-0.0), Flux(0.0)),
                        (large, -large, one(Flux), -one(Flux)),
                        (Flux(-0.0), one(Flux), -one(Flux), Flux(0.0)))
            for values in patterns, integer_code in 0:255
                code = UInt8(integer_code)
                observed = TwoPhaseAMR._cell_advection_ordered_sum(
                    values..., code, Accum)
                expected = original_sum(values, code, Accum)
                @test typeof(observed) === Accum && isequal(observed, expected)
                checks[] += 1
            end
        end
    end
    checks[] == 2304 || error("packed-code check count differs")
    return checks[]
end

checks = validate_codes()
Base.include_string(Main, adapted_test, abspath(test_file))

open(report, "w") do io
    TOML.print(io, Dict(
        "copied_module_file" => module_file,
        "copied_module_sha256" => hashfile(module_file),
        "loaded_advection_file" => advection_file,
        "loaded_advection_sha256" => hashfile(advection_file),
        "ordered_sum_method_file" => method_file,
        "ordered_sum_method_line" => sum_method.line,
        "scalar_selector_method_file" => selector_method_file,
        "selector_expected" => expected_selector,
        "original_test_file" => abspath(test_file),
        "original_test_sha256" => hashfile(test_file),
        "adapted_test_sha256" => bytes2hex(SHA.sha256(codeunits(adapted_test))),
        "packed_code_count" => 256,
        "order_checks" => checks,
        "julia_version" => string(VERSION),
        "threads" => Threads.nthreads(:default),
        "status" => "passed"))
end

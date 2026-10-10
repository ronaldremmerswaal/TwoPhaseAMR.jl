#!/usr/bin/env julia
"""Validate all face orders and the compact suite in an explicitly copied module."""

using SHA
using TOML
using Test

length(ARGS) == 4 ||
    error("usage: validation_driver.jl SOURCE TEST_FILE EXPECTED_SOURCE_SHA REPORT")
source, test_file, expected_sha, report = ARGS
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
fallback_file = ""
if isdefined(TwoPhaseAMR, :_cell_advection_ordered_sum_generic)
    fallback_method = only(collect(methods(TwoPhaseAMR._cell_advection_ordered_sum_generic)))
    fallback_file = realpath(String(fallback_method.file))
    fallback_file == realpath(advection_file) ||
        error("generic fallback resolved outside copied source: $fallback_file")
end

function packed_code(directions)
    code = zero(UInt8)
    for (position, direction) in enumerate(directions)
        code |= UInt8(direction) << (2(position - 1))
    end
    return code
end

function original_sum(a, b, c, d, order::UInt8, ::Type{T}) where {T}
    fluxes = (a, b, c, d)
    integrated = zero(T)
    integrated += fluxes[Int(order & 0x03) + 1]
    integrated += fluxes[Int((order >> 2) & 0x03) + 1]
    integrated += fluxes[Int((order >> 4) & 0x03) + 1]
    integrated += fluxes[Int(order >> 6) + 1]
    return integrated
end

function validate_orders()
    codes = UInt8[]
    for a in 0:3, b in 0:3, c in 0:3, d in 0:3
        length(Set((a,b,c,d))) == 4 || continue
        push!(codes, packed_code((a,b,c,d)))
    end
    sort!(codes)
    length(codes) == 24 && length(unique(codes)) == 24 ||
        error("permutation enumeration did not cover all 24 face orders")
    canonical = Set(UInt8[0xe4, 0xe1, 0xb4, 0xb1])
    all(code in codes for code in canonical) || error("canonical orders missing")

    checks = Ref(0)
    fast_checks = Ref(0)
    fallback_checks = Ref(0)
    @testset "all original-face permutations preserve exact accumulation" begin
        for Flux in (Float32, Float64), Accum in unique((Flux, Float64))
            large = ldexp(one(Flux), precision(Flux))
            patterns = ((Flux(-0.0), Flux(0.0), Flux(-0.0), Flux(0.0)),
                        (large, -large, one(Flux), -one(Flux)),
                        (Flux(-0.0), one(Flux), -one(Flux), Flux(0.0)))
            for (a,b,c,d) in patterns, order in codes
                observed = TwoPhaseAMR._cell_advection_ordered_sum(a,b,c,d,order,Accum)
                expected = original_sum(a,b,c,d,order,Accum)
                @test typeof(observed) === typeof(expected) &&
                      typeof(observed) === Accum && isequal(observed, expected)
                checks[] += 1
                if order in canonical
                    fast_checks[] += 1
                else
                    fallback_checks[] += 1
                end
            end
            # These are outside the permutation set but still have defined
            # packed tuple-index semantics and must retain the fallback.
            for order in UInt8[0x00, 0xff]
                a,b,c,d = patterns[2]
                observed = TwoPhaseAMR._cell_advection_ordered_sum(a,b,c,d,order,Accum)
                expected = original_sum(a,b,c,d,order,Accum)
                @test typeof(observed) === typeof(expected) &&
                      typeof(observed) === Accum && isequal(observed, expected)
                checks[] += 1
                fallback_checks[] += 1
            end
        end
    end
    return (; permutation_count=length(codes), order_checks=checks[],
            canonical_fast_checks=fast_checks[],
            generic_fallback_checks=fallback_checks[])
end

counts = validate_orders()

Base.include_string(Main, adapted_test, abspath(test_file))

open(report, "w") do io
    TOML.print(io, Dict(
        "copied_module_file" => module_file,
        "copied_module_sha256" => hashfile(module_file),
        "loaded_advection_file" => advection_file,
        "loaded_advection_sha256" => hashfile(advection_file),
        "ordered_sum_method_file" => method_file,
        "ordered_sum_method_line" => sum_method.line,
        "generic_fallback_method_file" => fallback_file,
        "original_test_file" => abspath(test_file),
        "original_test_sha256" => hashfile(test_file),
        "adapted_test_sha256" => bytes2hex(SHA.sha256(codeunits(adapted_test))),
        "permutation_count" => counts.permutation_count,
        "order_checks" => counts.order_checks,
        "canonical_fast_checks" => counts.canonical_fast_checks,
        "generic_fallback_checks" => counts.generic_fallback_checks,
        "julia_version" => string(VERSION),
        "threads" => Threads.nthreads(:default),
        "status" => "passed"))
end

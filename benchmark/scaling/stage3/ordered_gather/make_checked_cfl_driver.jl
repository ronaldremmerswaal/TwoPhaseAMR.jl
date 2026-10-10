#!/usr/bin/env julia
"""Combine ordinary full-mesh validation with the separate CFL workload."""
module CheckedCFLOrderedGatherDriver

include(joinpath(@__DIR__, "make_checked_driver.jl"))
include(joinpath(@__DIR__, "make_cfl_driver.jl"))
using .CheckedOrderedGatherDriver
using .OrderedGatherCFLDriver
using SHA

function make_checked_cfl_driver(output)
    source = CheckedOrderedGatherDriver.SOURCE
    OrderedGatherCFLDriver.hashfile(source) ==
        OrderedGatherCFLDriver.EXPECTED_DRIVER_SHA256 ||
        error("frozen Stage 3 source driver differs")
    CheckedOrderedGatherDriver.make_checked_driver(output)
    result = OrderedGatherCFLDriver.cfl_driver(read(output, String))
    write(output, result)
    return bytes2hex(sha256(codeunits(result)))
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("usage: make_checked_cfl_driver.jl OUTPUT")
    println(make_checked_cfl_driver(ARGS[1]))
end

end

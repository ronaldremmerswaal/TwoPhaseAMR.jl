"""Check the padded Fortran diagnostic against the frozen implementation."""
module FortranPaddedTest

include(joinpath(@__DIR__, "..", "..", "..", "..", "benchmark", "scaling", "Support.jl"))
using .ScalingSupport: run_capture
using JSON

const FROZEN = joinpath(@__DIR__, "..", "build", "lw_omp")
const PADDED = joinpath(@__DIR__, "build", "lw_omp_padded")

function execute(binary, source, target, nx, ny, workers; pad=nothing)
    command = String[binary, "--nx", string(nx), "--ny", string(ny),
                     "--steps", "7", "--threads", string(workers),
                     "--tile-x", "16", "--tile-y", "8",
                     "--input", source, "--output", target]
    pad === nothing || append!(command, ["--pad", string(pad)])
    result = run_capture(command)
    result.exitcode == 0 || error("Fortran failed: $(result.stderr)")
    record = JSON.parse(strip(result.stdout))
    field = open(target) do io
        read!(io, Vector{Float64}(undef, nx * ny))
    end
    return record, field
end

function main()
    passed = 0
    mktempdir() do directory
        for (nx, ny, workers) in ((1, 1, 1), (33, 17, 3), (257, 193, 7))
            source = joinpath(directory, "input.bin")
            initial = [1.0 + 0.2sin(0.13 * (i - 1)) + 0.1cos(0.17 * (j - 1))
                       for j in 1:ny for i in 1:nx]
            open(source, "w") do io
                write(io, initial)
            end
            _, expected = execute(FROZEN, source, joinpath(directory, "reference.bin"),
                                  nx, ny, workers)
            for pad in (0, 8, 32)
                record, actual = execute(PADDED, source,
                    joinpath(directory, "padded.bin"), nx, ny, workers; pad)
                @assert actual == expected (nx, ny, workers, pad)
                @assert record["padding"] == pad
                @assert record["leading_dimension"] == nx + pad
                passed += 1
                println("$(nx)x$(ny) p=$workers pad=$pad: exact full-field parity")
            end
        end
    end
    println("passed=$passed")
end

end # module

abspath(PROGRAM_FILE) == (@__FILE__) && FortranPaddedTest.main()

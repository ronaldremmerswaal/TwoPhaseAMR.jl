"""Independent scalar-flux oracle for the dormant Fortran constant-carrier toy."""
module FortranToyTest

include(joinpath(@__DIR__, "..", "..", "..", "benchmark", "scaling", "Support.jl"))
using .ScalingSupport: run_capture
using JSON

@inline flux(left, right, speed, ratio) =
    speed * ((left + right) * 0.5) - ((speed * speed) * ratio * 0.5) * (right - left)

function sweep(q, nx, ny, speed, ratio, axis)
    output = similar(q)
    for j in 1:ny, i in 1:nx
        index = i + nx * (j - 1)
        right = axis == 1 ? mod1(i + 1, nx) + nx * (j - 1) :
                            i + nx * (mod1(j + 1, ny) - 1)
        left = axis == 1 ? mod1(i - 1, nx) + nx * (j - 1) :
                           i + nx * (mod1(j - 1, ny) - 1)
        output[index] = q[index] - ratio *
            (flux(q[index], q[right], speed, ratio) -
             flux(q[left], q[index], speed, ratio))
    end
    return output
end

function reference(q, nx, ny, steps, u, v, dt)
    for _ in 1:steps
        q = sweep(q, nx, ny, u, dt * nx * 0.5, 1)
        q = sweep(q, nx, ny, v, dt * ny, 2)
        q = sweep(q, nx, ny, u, dt * nx * 0.5, 1)
    end
    return q
end

function check(binary, nx, ny, steps, workers, u, v, tile_x, tile_y)
    # The Fortran file uses row-major ordering (i varies fastest).
    q0 = [1.0 + 0.21sin(0.19 * (i - 1)) + 0.13cos(0.11 * (j - 1))
          for j in 1:ny for i in 1:nx]
    dt = 0.35 / max(abs(u) * nx + abs(v) * ny, 1.0)
    expected = reference(q0, nx, ny, steps, u, v, dt)
    mktempdir() do directory
        source, target = joinpath(directory, "q0.bin"), joinpath(directory, "qfinal.bin")
        open(source, "w") do io
            write(io, q0)
        end
        command = String[binary, "--nx", string(nx), "--ny", string(ny),
                         "--steps", string(steps), "--threads", string(workers),
                         "--u", string(u), "--v", string(v), "--dt", string(dt),
                         "--tile-x", string(tile_x), "--tile-y", string(tile_y),
                         "--input", source, "--output", target]
        result = run_capture(command)
        result.exitcode == 0 || error("Fortran failed: $(result.stderr)")
        record = JSON.parse(strip(result.stdout))
        actual = open(target) do io
            read!(io, Vector{Float64}(undef, nx * ny))
        end
        delta = maximum(abs.(actual .- expected))
        @assert delta <= 8e-14 (nx, ny, steps, workers, delta)
        expected_active = min(workers, cld(nx, tile_x) * cld(ny, tile_y))
        @assert all(==(expected_active), record["active_worker_counts"])
        @assert abs(record["mass"] - sum(expected) / length(expected)) <= 3e-14
        @assert record["workers"] == workers
        println("$(nx)x$(ny) steps=$steps threads=$workers: max |Fortran-oracle|=$delta")
    end
end

function main(args=ARGS)
    length(args) == 1 || error("usage: julia test.jl PATH_TO_LW_OMP")
    binary = abspath(args[1])
    for case in ((1, 1, 3, 1, 0.7, -0.3, 1, 1),
                 (1, 9, 4, 2, 0.7, -0.3, 1, 3),
                 (9, 1, 4, 2, -0.7, 0.0, 4, 1),
                 (17, 13, 7, 3, -0.7, 0.3, 5, 4),
                 (257, 193, 3, 7, 0.7, -0.3, 64, 19))
        check(binary, case...)
    end
end

end # module

abspath(PROGRAM_FILE) == (@__FILE__) && FortranToyTest.main()

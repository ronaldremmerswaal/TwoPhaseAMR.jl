#!/usr/bin/env julia
"""Tiny independent Float16 CFL oracle for copied control and tile-max source."""

using SHA
using TOML
using Test

length(ARGS) == 4 || error(
    "usage: float16_validation.jl SOURCE SOURCE_SHA EXPECT_TILE_MAX REPORT")
source, source_sha, expected_flag, report = ARGS
expected_flag in ("true", "false") || error("EXPECT_TILE_MAX must be true or false")
expected_tile_max = expected_flag == "true"
hashfile(path) = bytes2hex(SHA.sha256(read(path)))
module_file = abspath(joinpath(source, "src", "TwoPhaseAMR.jl"))
advection_file = abspath(joinpath(source, "src", "operators", "advection",
                                  "cell_advection.jl"))
hashfile(advection_file) == source_sha || error("copied advection source differs")
Base.include(Main, module_file)
using .TwoPhaseAMR
(:rate_tile_max in fieldnames(Base.unwrap_unionall(AdvectionWorkspace))) ==
    expected_tile_max || error("tile-max workspace identity differs")

function frozen_small_cfl!(rates, velocity, cfl)
    mesh = velocity.mesh
    T = promote_type(eltype(velocity), eltype(rates), typeof(cfl))
    interior_flux = [convert(eltype(rates), abs(velocity.interior[index]) * face.area)
                     for (index, face) in pairs(mesh.faces)]
    boundary_flux = [convert(eltype(rates), abs(velocity.boundary[index]) * face.area)
                     for (index, face) in pairs(mesh.boundary_faces)]
    nx, ny = mesh.block_size
    h = mesh.halo
    maxrate = zero(T)
    for b in eachindex(mesh.blocks), j in 1:ny, i in 1:nx
        integrated = zero(T)
        # Original CSR order is mesh face ID order, then boundary face order.
        for (index, face) in pairs(mesh.faces)
            if face.minus.block == b && face.minus.i == i && face.minus.j == j
                integrated += interior_flux[index]
            end
            if face.plus.block == b && face.plus.i == i && face.plus.j == j
                integrated += interior_flux[index]
            end
        end
        for (index, face) in pairs(mesh.boundary_faces)
            if face.cell.block == b && face.cell.i == i && face.cell.j == j
                integrated += boundary_flux[index]
            end
        end
        rates.blocks[b][i + h, j + h] = integrated / mesh.blocks[b].volume
        maxrate = max(maxrate, rates.blocks[b][i + h, j + h])
    end
    return iszero(maxrate) ? T(Inf) : T(cfl) / maxrate
end

checks = Ref(0)
@testset "Float16 copied CFL rate and bound" begin
    mesh = BlockTreeMesh(base_mesh=(8, 4), block_size=(8, 4),
                         periodic=(true, false))
    @test eltype(mesh.origin) === Float64
    checks[] += 1
    field = CellField(mesh, Float16)
    workspace = AdvectionWorkspace(field)
    reference_rates = CellField(mesh, Float16)
    velocity = FaceVelocity(mesh, Float16; init=Float16(0))
    for (phase, magnitude) in enumerate((Float16(0.125), Float16(0),
                                         Float16(0.03125)))
        for index in eachindex(velocity.interior)
            velocity.interior[index] = isodd(index + phase) ? -magnitude : magnitude
        end
        for index in eachindex(velocity.boundary)
            velocity.boundary[index] = isodd(index + phase) ? magnitude : -magnitude
        end
        if expected_tile_max
            fill!(workspace.rate_tile_max, Float16(NaN))
            fill!(workspace.rate_tile_exception, 0xff)
        end
        got = stable_timestep(velocity, workspace; cfl=Float16(0.5))
        expected = frozen_small_cfl!(reference_rates, velocity, Float16(0.5))
        @test typeof(got) === Float16 && isequal(got, expected)
        checks[] += 1
        @test isequal(interior(workspace.rates, 1), interior(reference_rates, 1))
        checks[] += 1
        if expected_tile_max
            @test all(isfinite, workspace.rate_tile_max)
            checks[] += 1
            @test all(iszero, workspace.rate_tile_exception)
            checks[] += 1
        end
    end
end

open(report, "w") do io
    TOML.print(io, Dict(
        "status" => "passed",
        "source" => abspath(source),
        "source_sha256" => hashfile(advection_file),
        "copied_module_file" => module_file,
        "test_sha256" => hashfile(@__FILE__),
        "tile_max_expected" => expected_tile_max,
        "assertions" => checks[],
        "threads" => Threads.nthreads(:default),
        "julia_version" => string(VERSION),
    ))
end

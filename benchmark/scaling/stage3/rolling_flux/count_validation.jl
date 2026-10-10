#!/usr/bin/env julia
"""Require exactly one ordinary helper call per regular owned face."""

using SHA
using TOML

length(ARGS) == 4 ||
    error("usage: count_validation.jl INSTRUMENTED_SOURCE EXPECTED_SHA REPORT WORKERS")
source, expected_sha, report_path, workers_arg = ARGS
parse(Int, workers_arg) == Threads.nthreads(:default) ||
    error("requested worker count differs")
advection_file = abspath(joinpath(source, "src", "operators", "advection",
                                  "cell_advection.jl"))
hashfile(path) = bytes2hex(SHA.sha256(read(path)))
hashfile(advection_file) == expected_sha || error("instrumented source hash differs")
Base.include(Main, abspath(joinpath(source, "src", "TwoPhaseAMR.jl")))
using .TwoPhaseAMR
isdefined(TwoPhaseAMR, :_rolling_flux_face_calls) ||
    error("call-count helper is absent")

cases = (
    (base_mesh=(64, 32), block_size=(64, 32), periodic=(false, false),
     max_level=0, refine=(key, origin, extent)->false),
    (base_mesh=(4, 32), block_size=(2, 16), periodic=(true, false),
     max_level=1, refine=(key, origin, extent)->key.level == 0 && key.i == 0),
    (base_mesh=(32, 4), block_size=(16, 2), periodic=(false, true),
     max_level=1, refine=(key, origin, extent)->key.level == 0 && key.j == 0),
)

case_reports = Any[]
for (index, spec) in enumerate(cases)
    mesh = BlockTreeMesh(base_mesh=spec.base_mesh, block_size=spec.block_size,
                         periodic=spec.periodic, max_level=spec.max_level,
                         refine=spec.refine, extent=(1.3, 0.9))
    scalar = CellField(mesh, (x, y)->0.8 + 0.31sinpi(2x) - 0.17cospi(2y))
    velocity = FaceVelocity(mesh, (x, y, t, axis)->
        axis == 1 ? 0.12 + 0.31sinpi(2x) : -0.15 + 0.23cospi(2y))
    workspace = AdvectionWorkspace(scalar)
    calls = [Threads.Atomic{Int}(0) for _ in eachindex(mesh.faces)]
    TwoPhaseAMR._rolling_flux_face_calls[] = calls
    advect!(scalar, velocity, 0.0037, workspace;
            boundary=Dirichlet((x, y, time)->0.4 + 0.13x - 0.27y))
    regular = 0
    residual = 0
    for (face_id, face) in pairs(mesh.faces)
        kind = TwoPhaseAMR._cell_advection_regular_kind(face)
        expected = kind in (0x01, 0x02) ? 1 : 0
        actual = calls[face_id][]
        actual == expected ||
            error("case $index face $face_id regular kind $kind called $actual times, expected $expected")
        regular += expected
        residual += iszero(expected)
    end
    push!(case_reports, Dict("case" => index, "blocks" => length(mesh.blocks),
                             "faces" => length(mesh.faces),
                             "regular_once" => regular,
                             "residual_zero" => residual))
end

open(report_path, "w") do io
    TOML.print(io, Dict("status" => "passed",
                        "instrumented_source" => abspath(source),
                        "instrumented_source_sha256" => hashfile(advection_file),
                        "threads" => Threads.nthreads(:default),
                        "julia_version" => string(VERSION),
                        "cases" => case_reports))
end

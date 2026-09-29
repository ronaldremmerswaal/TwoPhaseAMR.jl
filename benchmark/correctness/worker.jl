using LinearAlgebra: BLAS
using Serialization
using SHA
using TwoPhaseAMR

BLAS.set_num_threads(1)

const BASE_VELOCITY = (x, y, t, axis) -> axis == 1 ? 0.3 : -0.2
const INITIAL_SCALAR = (x, y) -> 1 + 0.2sinpi(2x) * cospi(2y)

function source_sha256()
    source_root = joinpath(dirname(Base.active_project()), "src")
    paths = String[]
    for (directory, _, files) in walkdir(source_root)
        append!(paths, joinpath.(directory, filter(file -> endswith(file, ".jl"), files)))
    end
    buffer = IOBuffer()
    for path in sort!(paths)
        write(buffer, relpath(path, source_root), UInt8(0))
        write(buffer, read(path))
    end
    return bytes2hex(sha256(take!(buffer)))
end

mesh_keys(mesh) = [
    "$(b.key.level):$(b.key.i):$(b.key.j)" for b in mesh.blocks
]
cell_values(field) = reduce(vcat, [
    vec(collect(interior(field, b))) for b in eachindex(field.mesh.blocks)
])

function record(mesh; quantities = Dict{String, Any}(), values = Dict{String, Any}())
    return Dict{String, Any}(
        "mesh_keys" => mesh_keys(mesh),
        "faces" => length(mesh.faces),
        "cells" => ncells(mesh),
        "quantities" => quantities,
        "values" => values,
    )
end

function scalar_fixture(kind)
    mesh = if kind == :one
        BlockTreeMesh(base_mesh = (32, 32), block_size = (32, 32))
    elseif kind == :many
        BlockTreeMesh(base_mesh = (32, 32), block_size = (8, 8))
    else
        BlockTreeMesh(
            base_mesh = (32, 32), block_size = (8, 8), max_level = 2,
            refine = (key, origin, extent) ->
                (key.level == 0 && key.i == 0 && key.j == 0) ||
                (key.level == 1 && key.i == 0 && key.j == 0),
        )
    end
    scalar = CellField(mesh, INITIAL_SCALAR)
    velocity = FaceVelocity(mesh, BASE_VELOCITY)
    workspace = AdvectionWorkspace(scalar)
    mass_before = integrate(scalar)
    dt = min(0.01, stable_timestep(velocity, workspace; cfl = 0.2))
    for _ in 1:3
        advect!(scalar, velocity, dt, workspace)
    end
    return record(
        mesh;
        quantities = Dict("dt" => dt, "steps" => 3,
                          "mass_before" => mass_before,
                          "mass_after" => integrate(scalar)),
        values = Dict("scalar" => cell_values(scalar)),
    )
end

function wy_fixture()
    mesh = BlockTreeMesh(base_mesh = (32, 32), block_size = (8, 8))
    interface = InterfaceField(
        mesh, (x, y) -> hypot(x - 0.43, y - 0.57) - 0.23;
        lipschitz = 1,
    )
    tracer = PhaseScalar(
        interface;
        liquid = (x, y) -> 1 + 0.2sinpi(2y),
        gas = (x, y) -> 2 + 0.1cospi(2x),
    )
    velocity = FaceVelocity(mesh, BASE_VELOCITY)
    workspace = WYAdvectionWorkspace(interface)
    initial = (
        integrate(interface), integrate(tracer, :liquid),
        integrate(tracer, :gas),
    )
    dt = min(0.01, stable_timestep(velocity, workspace; cfl = 0.2))
    for step in 1:2
        advect!(
            interface, velocity, dt, workspace; scalars = (tracer,),
            sweep_order = isodd(step) ? (1, 2) : (2, 1),
        )
    end
    return record(
        mesh;
        quantities = Dict(
            "dt" => dt, "steps" => 2, "phase_before" => collect(initial),
            "phase_after" => [
                integrate(interface), integrate(tracer, :liquid),
                integrate(tracer, :gas),
            ],
        ),
        values = Dict(
            "fraction" => cell_values(interface.fraction),
            "liquid" => cell_values(tracer.liquid),
            "gas" => cell_values(tracer.gas),
        ),
    )
end

function staggered_fixture()
    mesh = BlockTreeMesh(base_mesh = (32, 32), block_size = (8, 8))
    carrier = FaceVelocity(mesh, BASE_VELOCITY)
    transported = FaceVelocity(
        mesh,
        (x, y, t, axis) -> axis == 1 ?
            1 + 0.1sinpi(2x) * cospi(2y) :
            -0.5 + 0.2cospi(2x) * sinpi(2y),
    )
    workspace = StaggeredAdvectionWorkspace(transported)
    dt = min(0.01, stable_timestep(carrier, workspace; cfl = 0.2))
    for _ in 1:2
        advect!(transported, carrier, dt, workspace; scheme = LW())
    end
    return record(
        mesh;
        quantities = Dict("dt" => dt, "steps" => 2),
        values = Dict("velocity" => copy(transported.interior)),
    )
end

function adaptive_fixture()
    mesh = BlockTreeMesh(
        block_size = (8, 8), max_level = 1,
        adaptation = MeshAdaptationOptions(coarsen_delay = 0),
    )
    scalar = CellField(mesh, (x, y) -> 1 + sinpi(4x) * cospi(4y))
    workspace = AMRWorkspace(mesh, scalar)
    indicator = InterpolationErrorIndicator(
        scalar; E_refine = 0.01, E_coarsen = 0.005,
    )
    initial_mass = integrate(scalar)
    labels_before = string.(refinement_labels!(workspace, indicator))
    refined = adapt!(workspace, indicator; step = 1, force = true)
    refined.changed || error("adaptive fixture did not refine")
    keys_refined = mesh_keys(mesh)
    mass_refined = integrate(scalar)
    initialize!(scalar, (x, y) -> 1.0)
    labels_after = string.(refinement_labels!(workspace, indicator))
    coarsened = adapt!(workspace, indicator; step = 2, force = true)
    coarsened.changed || error("adaptive fixture did not coarsen")
    return record(
        mesh;
        quantities = Dict(
            "mass_before" => initial_mass, "mass_refined" => mass_refined,
            "mass_after" => integrate(scalar),
            "labels_before" => labels_before,
            "labels_after" => labels_after,
            "keys_refined" => keys_refined,
        ),
        values = Dict("scalar" => cell_values(scalar)),
    )
end

function main()
    length(ARGS) in (1, 2) || error("usage: worker.jl OUTPUT_FILE [--only=core|concurrent]")
    mode = length(ARGS) == 1 ? "all" :
        startswith(ARGS[2], "--only=") ? split(ARGS[2], "="; limit = 2)[2] : "invalid"
    mode in ("all", "core", "concurrent") || error("invalid fixture mode: $mode")
    results = Dict{String, Any}()
    if mode != "concurrent"
        results["scalar_one"] = scalar_fixture(:one)
        results["scalar_many"] = scalar_fixture(:many)
        results["scalar_mixed"] = scalar_fixture(:mixed)
        results["wy"] = wy_fixture()
        results["staggered"] = staggered_fixture()
        results["adaptive"] = adaptive_fixture()
    end
    if mode != "core" && Threads.nthreads(:default) > 1
        a = Threads.@spawn scalar_fixture(:one)
        b = Threads.@spawn scalar_fixture(:many)
        results["concurrent_one"] = fetch(a)
        results["concurrent_many"] = fetch(b)
    end
    output = Dict{String, Any}(
        "schema" => 1,
        "repository_revision" => readchomp(`git -C $(dirname(dirname(@__DIR__))) rev-parse HEAD`),
        "source_sha256" => source_sha256(),
        "julia_version" => string(VERSION),
        "default_threads" => Threads.nthreads(:default),
        "blas_threads" => BLAS.get_num_threads(),
        "results" => results,
    )
    open(ARGS[1], "w") do io
        serialize(io, output)
    end
end

main()

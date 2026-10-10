#!/usr/bin/env julia
"""Construct the unchanged one-block mesh while flushing phase evidence."""

using SHA
using TOML

length(ARGS) == 3 || error("usage: mesh_driver.jl SOURCE EVENTS_TOML fresh2048|seq1024_2048")
source, events_path, mode = ARGS
mode in ("fresh2048", "seq1024_2048") || error("unknown mesh sequence: $mode")

const EXPECTED_ADVECTION_SHA256 =
    "047457e224baa1c5b5dfa26376eff71acd61de02b225ca7e13360df6c3184541"
const EXPECTED_MESH_SHA256 =
    "d62cebb9cb5775f3b72de974ec55e2d4bf572f6daefdb28b1a89ff46eccf86fd"
hashfile(path) = bytes2hex(SHA.sha256(read(path)))
const SOURCE = abspath(source)
const MODULE_FILE = joinpath(SOURCE, "src", "TwoPhaseAMR.jl")
const MESH_FILE = joinpath(SOURCE, "src", "mesh", "topology", "mesh.jl")
const ADVECTION_FILE = joinpath(SOURCE, "src", "operators", "advection", "cell_advection.jl")
hashfile(ADVECTION_FILE) == EXPECTED_ADVECTION_SHA256 ||
    error("source is not the frozen narrow-ID control")
hashfile(MESH_FILE) == EXPECTED_MESH_SHA256 ||
    error("mesh source differs from both failed cohorts")
Base.include(Main, MODULE_FILE)
using .TwoPhaseAMR
method = only(collect(methods(TwoPhaseAMR._face_dual_volumes)))
realpath(String(method.file)) == realpath(MESH_FILE) ||
    error("loaded dual-volume method is outside the copied source")

function rss_kib()
    status = read("/proc/self/status", String)
    match_result = match(r"(?m)^VmRSS:\s+(\d+) kB$", status)
    return match_result === nothing ? -1 : parse(Int, match_result.captures[1])
end

function event!(phase; n=0, face_count=-1)
    entry = Dict{String,Any}(
        "phase" => phase, "n" => n, "face_count" => face_count,
        "time_ns" => time_ns(), "rss_kib" => rss_kib(),
        "julia_version" => string(VERSION),
        "julia_executable" => joinpath(Sys.BINDIR, Base.julia_exename()),
        "source" => SOURCE, "mesh_sha256" => hashfile(MESH_FILE),
        "advection_sha256" => hashfile(ADVECTION_FILE),
        "dual_volume_method_file" => realpath(String(method.file)),
        "threads" => Threads.nthreads(:default), "mode" => mode)
    open(events_path, "a") do io
        TOML.print(io, Dict("events" => [entry]))
        flush(io)
        ccall(:fsync, Cint, (Cint,), Base.fd(io)) == 0 || error("event fsync failed")
    end
    return nothing
end

function construct(n)
    event!("before_mesh"; n)
    # Match the Stage 3 driver's constructor and leave the production source
    # unmodified. The separate event calls perturb process allocation history.
    mesh = BlockTreeMesh(base_mesh=(n,n), block_size=(n,n),
                         periodic=(true,true))
    event!("after_mesh"; n, face_count=length(mesh.faces))
    return nothing
end

event!("module_loaded")
if mode == "fresh2048"
    construct(2048)
else
    construct(1024)
    event!("before_intercase_gc")
    GC.gc() # The original batch driver does this after each completed case.
    event!("after_intercase_gc")
    construct(2048)
end
event!("complete")

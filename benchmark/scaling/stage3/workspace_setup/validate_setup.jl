#!/usr/bin/env julia
"""Validate one frozen 1024²/8² workspace setup without advancing a field."""

using SHA
using TOML

length(ARGS) == 5 || error(
    "usage: validate_setup.jl INSTRUMENTED_SOURCE THREADING_SHA ADVECTION_SHA VARIANT REPORT")
source, expected_threading_sha, expected_advection_sha, variant, report = ARGS
variant in ("control", "rolling_flux_fused") || error("unknown variant")
hashfile(path) = bytes2hex(SHA.sha256(read(path)))
module_file = abspath(joinpath(source, "src", "TwoPhaseAMR.jl"))
threading_file = abspath(joinpath(source, "src", "threading.jl"))
advection_file = abspath(joinpath(source, "src", "operators", "advection",
                                  "cell_advection.jl"))
hashfile(threading_file) == expected_threading_sha || error("threading source differs")
hashfile(advection_file) == expected_advection_sha || error("advection source differs")
Base.include(Main, module_file)
using .TwoPhaseAMR
parentmodule(TwoPhaseAMR) === Main || error("copied module did not load in Main")
isdefined(TwoPhaseAMR, :_stage3_setup_schedule_checks) ||
    error("setup assertions absent from copied module")
schedule_method = only(collect(methods(TwoPhaseAMR._grouped_output_schedule)))
realpath(String(schedule_method.file)) == realpath(threading_file) ||
    error("schedule method resolved outside copied source")

function validate_mesh(mesh)
    nx, ny = mesh.block_size
    nblocks = length(mesh.blocks)
    nx == ny == 8 || error("block size differs")
    nblocks == 16_384 || error("block count differs")
    length(mesh.faces) == 2_097_152 || error("periodic face count differs")
    length(mesh.face_dual_volumes) == length(mesh.faces) ||
        error("face dual volume count differs")
    isempty(mesh.boundary_faces) || error("periodic mesh has boundary faces")
    valid_ref(ref) = 1 <= ref.block <= nblocks &&
        1 <= ref.i <= nx && 1 <= ref.j <= ny
    for (index, face) in pairs(mesh.faces)
        valid_ref(face.minus) && valid_ref(face.plus) ||
            error("invalid face cell reference at $index")
        face.axis in (1, 2) || error("invalid face axis at $index")
        all(isfinite, face.center) && isfinite(face.area) && face.area > 0 ||
            error("invalid face geometry at $index")
        minus_spacing = mesh.blocks[face.minus.block].spacing[face.axis]
        plus_spacing = mesh.blocks[face.plus.block].spacing[face.axis]
        expected_dual = face.area * (minus_spacing + plus_spacing) / 2
        isfinite(mesh.face_dual_volumes[index]) &&
            mesh.face_dual_volumes[index] == expected_dual ||
            error("invalid face dual volume at $index")
    end
    length(mesh.halo_donors) == length(mesh.halo_weights) ||
        error("halo donor/weight count differs")
    for (index, entry) in pairs(mesh.halo_entries)
        entry.destination.block in 1:nblocks ||
            error("invalid halo destination block at $index")
        1 <= entry.first_donor <= length(mesh.halo_donors) &&
            1 <= entry.donor_count <= length(mesh.halo_donors) - entry.first_donor + 1 ||
            error("invalid halo donor range at $index")
        all(isfinite, entry.center) || error("invalid halo center at $index")
    end
    all(valid_ref, mesh.halo_donors) || error("invalid halo donor reference")
    all(isfinite, mesh.halo_weights) || error("invalid halo weight")
    return nothing
end

mesh = BlockTreeMesh(base_mesh=(1024, 1024), block_size=(8, 8),
                     periodic=(true, true))
validate_mesh(mesh)
println("setup_check=mesh_valid blocks=$(length(mesh.blocks)) faces=$(length(mesh.faces))")
flush(stdout)
field = CellField(mesh, Float64)
velocity = FaceVelocity(mesh, (x, y, time, axis)->axis == 1 ? 0.7 : -0.3)
length(velocity.interior) == length(mesh.faces) || error("carrier face count differs")
println("setup_check=field_velocity_ready")
flush(stdout)
TwoPhaseAMR._stage3_setup_schedule_checks[] = true
workspace = AdvectionWorkspace(field)
TwoPhaseAMR._stage3_setup_schedule_checks[] = false
ncells = 1_048_576
length(workspace.cell_offsets) == ncells + 1 || error("workspace offset length differs")
length(workspace.face_indices) == 1_835_008 || error("workspace incidence count differs")
length(workspace.face_signs) == length(workspace.face_indices) ==
    length(workspace.face_kinds) || error("workspace incidence vectors differ")
workspace.cell_offsets[end] == length(workspace.face_indices) + 1 ||
    error("workspace final offset differs")
println("setup_check=workspace_ready")
flush(stdout)

open(report, "w") do io
    TOML.print(io, Dict(
        "schema" => 1,
        "status" => "passed",
        "variant" => variant,
        "source" => abspath(source),
        "threading_sha256" => hashfile(threading_file),
        "advection_sha256" => hashfile(advection_file),
        "module_sha256" => hashfile(module_file),
        "schedule_method_file" => realpath(String(schedule_method.file)),
        "julia_version" => string(VERSION),
        "threads" => Threads.nthreads(:default),
        "blocks" => length(mesh.blocks),
        "faces" => length(mesh.faces),
        "outputs" => ncells,
        "edge_incidences" => length(workspace.face_indices),
    ))
end

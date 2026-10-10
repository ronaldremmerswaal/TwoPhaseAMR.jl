#!/usr/bin/env julia
"""Make a copied timing driver with full ordinary mesh checks before timing."""
module CheckedOrderedGatherDriver

using SHA

const SOURCE = normpath(joinpath(@__DIR__, "..", "main_driver.jl"))
const ANCHOR = "function run(o)\n"
const MESH_ANCHOR = "    mesh_s = (time_ns()-t0)/1e9\n"
const CHECK = raw"""
function validate_mesh(mesh, n, block)
    nx, ny = mesh.block_size
    nblocks = length(mesh.blocks)
    nblocks == (n ÷ block)^2 || error("mesh block count differs")
    length(mesh.faces) == 2n*n || error("periodic face count differs")
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

"""

function make_checked_driver(output)
    source = read(SOURCE, String)
    length(split(source, ANCHOR)) == 2 || error("driver run anchor changed")
    length(split(source, MESH_ANCHOR)) == 2 || error("driver mesh anchor changed")
    result = replace(source, ANCHOR => CHECK * ANCHOR; count=1)
    result = replace(result, MESH_ANCHOR =>
        MESH_ANCHOR * "    validate_mesh(mesh, n, block)\n"; count=1)
    write(output, result)
    return bytes2hex(sha256(codeunits(result)))
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("usage: make_checked_driver.jl OUTPUT")
    println(make_checked_driver(ARGS[1]))
end

end

# Helpers of the audit of Chapter 12 (refinement criteria and conservative transfer): meshes, random regrids and
# measures that are evaluated independently of the transfer kernels (cell divergence per block key, flux through the
# pieces of the face lines, staggered integrals, coarse-cell lookups). Not an audit.
#
# Included by the code scripts of the audits eq-amr-*; needs `using TwoPhaseAMR`. Everything is deterministic given the RNG.

module AmrAudit

using Random
using TwoPhaseAMR
using TwoPhaseAMR: BlockKey

# ---------------------------------------------------------------- meshes

function mesh_with(; base, extent, periodic, max_level = 2, block = (8, 8), delay = 0, halo = 1)
    return BlockTreeMesh(;
        base_mesh = base, block_size = block, extent, periodic, max_level, halo,
        adaptation = MeshAdaptationOptions(update_interval = 1, coarsen_delay = delay),
    )
end

walled_mesh(; kw...) = mesh_with(; base = (32, 16), extent = (2.0, 1.0), periodic = (false, false), kw...)
periodic_mesh(; kw...) = mesh_with(; base = (32, 32), extent = (1.0, 1.0), periodic = (true, true), kw...)
channel_mesh(; kw...) = mesh_with(; base = (32, 16), extent = (2.0, 1.0), periodic = (true, false), kw...)

parent_key(key) = BlockKey(key.level - 1, key.i ÷ 2, key.j ÷ 2)
child_keys(key) = [BlockKey(key.level + 1, 2key.i + ci, 2key.j + cj) for cj in 0:1 for ci in 0:1]

# One regrid with random labels: refine with probability pr, coarsen with pc, else keep (the balance may add splits).
function random_labels(rng, mesh; pr = 0.3, pc = 0.5)
    return [(r = rand(rng); r < pr ? :refine : r < pr + pc ? :coarsen : :keep) for _ in 1:nblocks(mesh)]
end

# ---------------------------------------------------------------- independent geometry

cell_volume(mesh, b) = prod(mesh.blocks[b].spacing)

# Net integrated outflow  sum_f o_{c,f} |f| u_f  of every cell, by (block key, i, j); faces of `velocity` (FaceVelocity).
function net_flux_by_key(mesh, velocity)
    nx, ny = mesh.block_size
    flux = Dict(block.key => zeros(nx, ny) for block in mesh.blocks)
    key(cell) = mesh.blocks[cell.block].key
    for (index, face) in pairs(mesh.faces)
        f = velocity.interior[index] * face.area
        flux[key(face.minus)][face.minus.i, face.minus.j] += f
        flux[key(face.plus)][face.plus.i, face.plus.j] -= f
    end
    for (index, face) in pairs(mesh.boundary_faces)
        flux[key(face.cell)][face.cell.i, face.cell.j] += face.side * velocity.boundary[index] * face.area
    end
    return flux
end

function divergence_by_key(mesh, velocity)
    flux = net_flux_by_key(mesh, velocity)
    for (b, block) in pairs(mesh.blocks)
        flux[block.key] ./= cell_volume(mesh, b)
    end
    return flux
end

# Worst deviation of a cell quantity (dict keyed by block key) from the three regrid rules: kept blocks (unchanged),
# children of a refined cell (parent value) and merged cells (mean of the four children).
function identity_errors(mesh, old::Dict, new::Dict)
    nx, ny = mesh.block_size
    unchanged = refined = coarsened = 0.0
    for (key, values) in new
        if haskey(old, key)
            unchanged = max(unchanged, maximum(abs.(values .- old[key])))
        elseif key.level > 0 && haskey(old, parent_key(key))
            parent = old[parent_key(key)]
            ci, cj = mod(key.i, 2), mod(key.j, 2)
            for j in 1:ny, i in 1:nx
                fi, fj = ci * nx + i, cj * ny + j
                refined = max(refined, abs(values[i, j] - parent[(fi + 1) ÷ 2, (fj + 1) ÷ 2]))
            end
        else
            kids = child_keys(key)
            all(k -> haskey(old, k), kids) || error("block $key has neither an old parent nor four old children")
            for j in 1:ny, i in 1:nx
                total = 0.0
                for fj in (2j - 1):2j, fi in (2i - 1):2i
                    child = kids[(fi - 1) ÷ nx + 2((fj - 1) ÷ ny) + 1]
                    total += old[child][mod(fi - 1, nx) + 1, mod(fj - 1, ny) + 1]
                end
                coarsened = max(coarsened, abs(values[i, j] - total / 4))
            end
        end
    end
    return (; unchanged, refined, coarsened)
end

# Face segments on the lattice of the finest level: (axis, normal index, lo, hi, value) for interior and boundary faces.
# Face centres and areas are physical; the lattice step is the finest spacing in the tangential direction.
function face_segments(mesh, velocity)
    fine = mesh.extent ./ (mesh.base_mesh .* (1 << mesh.max_level))
    segs = Tuple{Int, Int, Int, Int, Float64}[]
    for (faces, values) in ((mesh.faces, velocity.interior), (mesh.boundary_faces, velocity.boundary))
        for (index, face) in pairs(faces)
            axis = face.axis
            t = 3 - axis
            n = round(Int, face.center[axis] / fine[axis])
            mesh.periodic[axis] && (n = mod(n, round(Int, mesh.extent[axis] / fine[axis])))
            lo = round(Int, (face.center[t] - face.area / 2) / fine[t])
            hi = round(Int, (face.center[t] + face.area / 2) / fine[t])
            push!(segs, (axis, n, lo, hi, values[index]))
        end
    end
    return segs
end

# Flux through the pieces of every grid line: for every line, the clusters (connected unions of old and new
# segments); where old and new segments cover the same union the integrated flux sum |f| u_f must agree.
# Returns (worst relative difference over surviving clusters, number of surviving clusters, number of vanished or
# newly created clusters).
function line_flux_check(old_segs, new_segs)
    lines = Dict{Tuple{Int, Int}, Tuple{Vector, Vector}}()
    for s in old_segs
        push!(get!(() -> (Tuple[], Tuple[]), lines, (s[1], s[2]))[1], s)
    end
    for s in new_segs
        push!(get!(() -> (Tuple[], Tuple[]), lines, (s[1], s[2]))[2], s)
    end
    worst = 0.0
    detail = nothing
    survived = vanished = 0
    for ((axis, n), (olds, news)) in lines
        events = sort!(vcat([(s[3], s[4], :old, s) for s in olds], [(s[3], s[4], :new, s) for s in news]); by = e -> (e[1], e[2]))
        # sweep clusters
        i = 1
        while i <= length(events)
            lo, hi = events[i][1], events[i][2]
            members = [events[i]]
            j = i + 1
            while j <= length(events) && events[j][1] < hi
                hi = max(hi, events[j][2])
                push!(members, events[j])
                j += 1
            end
            oldm = [e[4] for e in members if e[3] === :old]
            newm = [e[4] for e in members if e[3] === :new]
            cover(ms) = isempty(ms) ? 0 : sum(m[4] - m[3] for m in ms)
            if cover(oldm) == hi - lo && cover(newm) == hi - lo
                fo = sum((m[4] - m[3]) * m[5] for m in oldm)
                fn = sum((m[4] - m[3]) * m[5] for m in newm)
                scale = (hi - lo) * max(maximum(abs(m[5]) for m in oldm), maximum(abs(m[5]) for m in newm), eps())
                if abs(fo - fn) / scale > worst
                    worst = abs(fo - fn) / scale
                    detail = (; axis, n, lo, hi, oldm, newm, fo, fn)
                end
                survived += 1
            else
                vanished += 1
            end
            i = j
        end
    end
    return (; worst, survived, vanished, detail)
end

# Staggered integral sum_f dual_f u_f over interior faces of each axis, the sum of |u|, kinetic energy sum dual u^2/2;
# `with_boundary` adds the boundary faces with the half dual volume of their cell (|f|/h_t * V/2).
function staggered_integrals(mesh, velocity; with_boundary = false)
    M = [0.0, 0.0]
    A = [0.0, 0.0]
    E = 0.0
    for (index, face) in pairs(mesh.faces)
        w = mesh.face_dual_volumes[index]
        u = velocity.interior[index]
        M[face.axis] += w * u
        A[face.axis] += w * abs(u)
        E += w * u^2 / 2
    end
    if with_boundary
        for (index, face) in pairs(mesh.boundary_faces)
            block = mesh.blocks[face.cell.block]
            w = face.area / block.spacing[3 - face.axis] * block.volume / 2
            u = velocity.boundary[index]
            M[face.axis] += w * u
            A[face.axis] += w * abs(u)
            E += w * u^2 / 2
        end
    end
    return (; M, A, E)
end

end # module

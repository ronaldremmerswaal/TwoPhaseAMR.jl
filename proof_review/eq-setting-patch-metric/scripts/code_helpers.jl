# Helpers shared by the code_*.jl scripts (they need `TwoPhaseAMR` loaded as `AMR`).
#
# Label: eq:setting:patch-metric. Not a check by itself.

using Random: MersenneTwister
using SparseArrays: sparse, spdiagm

isdefined(Main, :PatchModel) || include(joinpath(@__DIR__, "model.jl"))
using .PatchModel

"""Random 2:1-balanced mesh; cell widths `(aspect*h0, h0)` (square cells for `aspect = 1`)."""
function random_mesh(seed; block_size, roots, max_level, periodic, h0 = 1.0, aspect = 1.0)
    rng = MersenneTwister(seed)
    nx, ny = block_size
    base = (roots[1] * nx, roots[2] * ny)
    extent = (base[1] * h0 * aspect, base[2] * h0)
    probability = (0.55, 0.5, 0.45, 0.4, 0.35)
    refine = (key, origin, ext) -> rand(rng) < probability[min(key.level + 1, 5)]
    return AMR.BlockTreeMesh(;
        base_mesh = base, block_size = block_size, extent = extent,
        periodic = periodic, max_level = max_level, refine = refine,
    )
end

"""
Cells touching the vertex `key` as lattice cells around v = (0,0), with the lattice unit
`(hx, hy)` = smallest cell widths among them. Periodic offsets use `_wrapped_delta`.
"""
function lattice_cells(mesh, key, faces)
    sx, sy = AMR._finest_lattice_size(mesh)
    vx = mesh.origin[1] + key[1] * mesh.extent[1] / sx
    vy = mesh.origin[2] + key[2] * mesh.extent[2] / sy
    refs = unique(vcat(
        [mesh.faces[f.index].minus for f in faces],
        [mesh.faces[f.index].plus for f in faces],
    ))
    info = map(refs) do ref
        block = mesh.blocks[ref.block]
        c = AMR.cellcenter(mesh, ref.block, ref.i, ref.j)
        dx = AMR._wrapped_delta(c[1] - vx, mesh.extent[1], mesh.periodic[1])
        dy = AMR._wrapped_delta(c[2] - vy, mesh.extent[2], mesh.periodic[2])
        (dx, dy, block.spacing[1], block.spacing[2])
    end
    hx = minimum(i[3] for i in info)
    hy = minimum(i[4] for i in info)
    cells = Cell[]
    for (dx, dy, wx, wy) in info
        s = round(Int, wx / hx)
        @assert abs(wx / hx - s) < 1.0e-9 && abs(wy / hy - s) < 1.0e-9 "non-dyadic cell sizes at a vertex"
        x = (dx - wx / 2) / hx
        y = (dy - wy / 2) / hy
        @assert abs(x - round(x)) < 1.0e-9 && abs(y - round(y)) < 1.0e-9 "cell corner off the vertex lattice"
        push!(cells, Cell(round(Int, x), round(Int, y), s))
    end
    return cells, hx, hy
end

"""`Y` (rows `area*d'`) and `N` (rows `(1/2) e_axis'`) of a patch, as in `_local_inverse_metric`."""
function moment_matrices(faces)
    Y = [f.area * f.displacement[k] for f in faces, k in 1:2]
    N = [f.axis == k ? 0.5 : 0.0 for f in faces, k in 1:2]
    return Y, N
end

function metric_matrix(operator)
    metric = operator.inverse_metric
    return metric isa AMR.DiagonalInverseFaceMetric ? spdiagm(metric.diagonal) : metric.matrix
end

"""Global metric assembled directly from the patches (diagonal closure at walls / <= 2 faces)."""
function reference_metric(mesh, patches)
    n = length(mesh.faces)
    rows = Int[]; cols = Int[]; vals = Float64[]
    for (key, faces) in patches
        if AMR._physical_vertex(mesh, key) || length(faces) <= 2
            for f in faces
                push!(rows, f.index); push!(cols, f.index)
                push!(vals, inv(2 * f.area * abs(f.displacement[f.axis])))
            end
        else
            R = AMR._local_inverse_metric(copy(faces))
            for (jc, fc) in pairs(faces), (jr, fr) in pairs(faces)
                push!(rows, fr.index); push!(cols, fc.index); push!(vals, R[jr, jc])
            end
        end
    end
    return sparse(rows, cols, vals, n, n)
end

"""Label of a patch from its lattice cells: D4 class of the corner pattern, or `:hanging`."""
function patch_label(cells)
    touch = touching(cells, 0, 0)
    kind = kind_of(touch, 0, 0)
    kind == :corner4 && return class_of(sigma_pattern(touch, 0, 0))
    kind == :hanging && return :hanging
    return (:other, kind)
end

"""Sorted (axis, area, d1, d2) rows, rounded for a robust ordering."""
function face_rows(faces)
    rows = [(f.axis, f.area, f.displacement[1], f.displacement[2]) for f in faces]
    return sort(rows; by = t -> (t[1], round.(t[2:4]; digits = 9)))
end

"""Same rows from the integer model for lattice unit `(hx, hy)`."""
function model_rows(ideal, hx, hy)
    rows = [(f.axis, f.len * (f.axis == 1 ? hy : hx), f.d2[1] * hx / 2, f.d2[2] * hy / 2) for f in ideal]
    return sort(rows; by = t -> (t[1], round.(t[2:4]; digits = 9)))
end

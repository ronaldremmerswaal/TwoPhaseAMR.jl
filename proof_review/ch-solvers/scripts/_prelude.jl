# Shared prelude of the scripts of the audit of ch:solvers (Chapter 11, operators and linear solvers):
# loads TwoPhaseAMR (the working tree through docs/techref/figures) and the fixtures of
# docs/techref/figures/solvers.jl (copied here: that file needs the helpers of generate.jl).
# Include with `isdefined(Main, :SOLV_DROP) || include(joinpath(@__DIR__, "_prelude.jl"))`.

using Test, Random, Printf
using LinearAlgebra
using SparseArrays
using TwoPhaseAMR
const AMR = TwoPhaseAMR
const Krylov = AMR.Krylov
const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))

const SOLV_DENSITY = (1000.0, 1.0)
const SOLV_DROP = (; centre = (0.513, 0.468), radius = 0.2371, band = 0.02)
solv_drop(x, y) = hypot(x - SOLV_DROP.centre[1], y - SOLV_DROP.centre[2]) - SOLV_DROP.radius
solv_band_solid(x, y) = abs(y - 0.46) - 0.17
solv_noise(index) = (value = sin(12.9898 * index) * 43758.5453; value - floor(value) - 0.5)

solv_uniform(n; periodic = (false, false), block = 8) = BlockTreeMesh(;
    base_mesh = (n, n), block_size = (block, block), periodic,
)

function solv_near_circle(origin, extent; radius = SOLV_DROP.radius)
    xs = (origin[1], origin[1] + extent[1])
    ys = (origin[2], origin[2] + extent[2])
    cx, cy = SOLV_DROP.centre
    nearest = hypot(clamp(cx, xs...) - cx, clamp(cy, ys...) - cy)
    farthest = maximum(hypot(x - cx, y - cy) for x in xs, y in ys)
    return nearest - SOLV_DROP.band <= radius <= farthest + SOLV_DROP.band
end

solv_amr(n; levels = 2, radius = SOLV_DROP.radius, block = 8, periodic = (false, false)) = BlockTreeMesh(;
    base_mesh = (n, n), block_size = (block, block), periodic,
    max_level = levels,
    refine = (key, origin, extent) -> solv_near_circle(origin, extent; radius),
)

solv_drop_field(mesh) = InterfaceField(mesh, solv_drop; lipschitz = 1)

# Dense matrix of a linear map given as a function (columns = images of the unit vectors).
function dense_matrix(f!, n)
    B = zeros(n, n)
    unit = zeros(n)
    column = zeros(n)
    for j in 1:n
        unit[j] = 1.0
        f!(column, unit)
        B[:, j] .= column
        unit[j] = 0.0
    end
    return B
end

# Dense matrix B of the projected V-cycle of a preconditioner.
vcycle_matrix(preconditioner, n) = dense_matrix((y, x) -> ldiv!(y, preconditioner, x), n)

# Independent component analysis of an elliptic operator (union-find over the interior faces with a
# nonzero coefficient between active cells), anchors by boundary conductance or by a nonzero
# connection to an inactive cell. Returns (labels, unanchored::Vector{Vector{Int}}).
function predicted_components(operator)
    mesh = operator.mesh
    nx, ny = mesh.block_size
    n = length(operator.volumes)
    active = operator.volumes .> 0
    parent = collect(1:n)
    find(i) = (
        while parent[i] != i
            parent[i] = parent[parent[i]]
            i = parent[i]
        end; i
    )
    anchored = falses(n)
    for (index, face) in pairs(mesh.faces)
        iszero(operator.face_coefficient[index]) && continue
        a = AMR._cell_index(face.minus, nx, ny)
        b = AMR._cell_index(face.plus, nx, ny)
        if active[a] && active[b]
            parent[find(a)] = find(b)
        elseif active[a] && !active[b]
            anchored[a] = true
        elseif active[b] && !active[a]
            anchored[b] = true
        end
    end
    for c in 1:n
        active[c] && operator.boundary_diagonal[c] > 0 && (anchored[c] = true)
    end
    groups = Dict{Int, Vector{Int}}()
    for c in 1:n
        active[c] && push!(get!(groups, find(c), Int[]), c)
    end
    unanchored = [sort(g) for g in values(groups) if !any(anchored[c] for c in g)]
    sort!(unanchored; by = first)
    return unanchored
end

# Library of the CMDGFM audit: cell normals and the independent classification of the faces following the printed rules of
# eq:projection:unresolved (transcribed from docs/techref/chapters/10_projection.tex:262-360; patches from face coordinates, own face data).
# Included by code_01_stencils.jl and code_02_system.jl.

isdefined(Main, :ProjectionAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "projection_audit.jl"))
using .ProjectionAudit
using TwoPhaseAMR
const PA = ProjectionAudit

"""Unit normal of cell `ref` (into the gas) from the PLIC data scaled by the cell widths; zero if tiny."""
function cell_normal(field, ref)
    h = field.mesh.blocks[ref.block].spacing
    nx = AMR._get(field.normal_x, ref) / h[1]; ny = AMR._get(field.normal_y, ref) / h[2]
    m = hypot(nx, ny)
    return m > sqrt(eps()) ? (nx / m, ny / m) : (0.0, 0.0)
end

"""Independent classification following the printed rules; returns Dict(face => NamedTuple)."""
function classify(field, d)
    mesh = field.mesh; nx_, ny_ = mesh.block_size
    fl = d.floor
    open_(g) = d.al[g] + d.ag[g]
    active(g) = fl < d.C[g] < 1 - fl
    incident = Dict{Int, Vector{Int}}()
    for (g, face) in pairs(mesh.faces)
        push!(get!(incident, AMR._cell_index(face.minus, nx_, ny_), Int[]), g)
        push!(get!(incident, AMR._cell_index(face.plus, nx_, ny_), Int[]), g)
    end
    out = Dict{Int, Any}()
    for f in d.mixed
        active(f) || continue
        face = mesh.faces[f]; k = face.axis; t = 3 - k
        if open_(f) <= fl
            out[f] = (; class = :rule1); continue
        end
        if (d.al[f] <= fl || d.ag[f] <= fl)
            far = [g for cell in (face.minus, face.plus) for g in incident[AMR._cell_index(cell, nx_, ny_)] if g != f && mesh.faces[g].axis == k]
            if any(g -> d.al[g] > fl && d.ag[g] > fl, far)
                out[f] = (; class = :rule2); continue
            end
        end
        # staggered normal
        cm = AMR._conditional_fraction(field, face.minus); cp = AMR._conditional_fraction(field, face.plus)
        wm, wp = cm * (1 - cm), cp * (1 - cp)
        nm, np_ = cell_normal(field, face.minus), cell_normal(field, face.plus)
        N = (wm * nm[1] + wp * np_[1], wm * nm[2] + wp * np_[2])
        if wm + wp == 0 || hypot(N...) <= sqrt(eps()) * (wm + wp)
            δ = cm - cp
            N = k == 1 ? (δ, 0.0) : (0.0, δ)
        end
        nrm = hypot(N...)
        if nrm <= sqrt(eps())
            out[f] = (; class = :rule3); continue
        end
        n = (N[1] / nrm, N[2] / nrm)
        if abs(n[k]) <= sqrt(eps())
            out[f] = (; class = :rule3); continue
        end
        R = n[t] / n[k]
        if abs(n[k]) >= 1 - 64eps()
            out[f] = (; class = :resolved, aligned = true, R, beta = 1.0, trans = Int[], w = Float64[], ig = 0, il = 0, s = 0.0, n)
            continue
        end
        trans = [g for g in PA.patch_faces(mesh, f, t) if active(g) && open_(g) > fl]
        w = [d.C[g] * (1 - d.C[g]) for g in trans]
        # physical boundary end point: lies on x in {0, 8} or y in {0, 8} (walled box)
        ends = PA.face_endpoints(face)
        onwall = any(e -> e[1] <= 1.0e-12 || e[1] >= 8 - 1.0e-12 || e[2] <= 1.0e-12 || e[2] >= 8 - 1.0e-12, ends)
        level = any(g -> mesh.faces[g].area != face.area, trans)
        need = (onwall || level) ? 1 : 2
        if length(trans) < need || sum(w) <= 0
            out[f] = (; class = :rule4); continue
        end
        w ./= sum(w)
        beta = 1.0; ig = il = 0; s = 0.0
        if abs(R) > 1
            par = PA.patch_faces(mesh, f, k)
            isnothing(field.geometry) || filter!(g -> field.geometry.face_dual_volumes[g] > 0, par)
            if length(par) < 2
                out[f] = (; class = :rule5); continue
            end
            sort!(par; by = g -> (d.ml[g] / d.m[g], g))
            ig, il = first(par), last(par)
            if d.mg[ig] < d.ml[ig] || d.ml[il] < d.mg[il]
                out[f] = (; class = :rule5); continue
            end
            s = d.ml[ig] / d.m[ig] + d.mg[il] / d.m[il]
            beta = (1 - s) / (abs(R) - s)
        end
        out[f] = (; class = :resolved, aligned = false, R, beta, trans, w, ig, il, s, n)
    end
    return out
end

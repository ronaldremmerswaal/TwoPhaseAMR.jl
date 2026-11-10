# Code check 01: transfer of liquid fractions and intrinsic phase scalars (eq:amr:vof, eq:amr:scalar).
#
# Labels: eq:amr:vof, eq:amr:scalar
# Source: docs/techref/chapters/12_amr_criteria_transfer.tex:490-545; src/interface/interface_field.jl:689-750, src/vof/scalar_transfer.jl
# Checks (periodic mesh without solid and walled mesh with a wavy solid; random drops; regrid sequences refine-refine-coarsen-coarsen
# plus random labels; every quantity recomputed independently of the package kernels):
#   V1  liquid volume: restriction conserves exactly; refinement changes the volume only by the plane-constant defect, which is measured
#       directly (sum of the four children = parent fraction? per parent, independent clipping of the stored plane);
#   V2  the child fraction equals 4|Q_r ∩ open ∩ liquid half-plane| (independent Sutherland-Hodgman clipping, snapped children counted
#       separately), lies in [0, open fraction of the child], pure parents give pure children;
#   S1  phase amounts |c| w a conserved before the snap: refinement shift (eq:amr:scalar) and weighted-mean restriction, both phases, with a solid;
#   S2  child values equal the chapter's formula (independent replica) in families without snapped children; merged values are the
#       weighted mean and lie in the range of the children (convexity).
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-amr-vof/scripts/code_01_vof_scalar.jl

using Test, Random, LinearAlgebra
include(joinpath(@__DIR__, "..", "..", "_shared", "amr_audit.jl"))
using .AmrAudit
using TwoPhaseAMR
const AA = AmrAudit

# Sutherland-Hodgman: keep { n . p <= a } (keep_less) or { n . p >= a }
function clip(poly, n, a; keep_less = true)
    out = NTuple{2, Float64}[]
    side(p) = keep_less ? a - (n[1] * p[1] + n[2] * p[2]) : (n[1] * p[1] + n[2] * p[2]) - a
    for k in eachindex(poly)
        p, q = poly[k], poly[mod1(k + 1, length(poly))]
        sp, sq = side(p), side(q)
        sp >= 0 && push!(out, p)
        if (sp >= 0) != (sq >= 0)
            t = sp / (sp - sq)
            push!(out, (p[1] + t * (q[1] - p[1]), p[2] + t * (q[2] - p[2])))
        end
    end
    return out
end
area(poly) = length(poly) < 3 ? 0.0 : abs(sum(poly[k][1] * poly[mod1(k + 1, length(poly))][2] - poly[mod1(k + 1, length(poly))][1] * poly[k][2] for k in eachindex(poly))) / 2

mc(a, b) = a * b <= 0 ? 0.0 : sign(a + b) * min(abs(a + b) / 2, 2abs(a), 2abs(b))
interior_of(mesh, M) = M[(mesh.halo + 1):(mesh.halo + mesh.block_size[1]), (mesh.halo + 1):(mesh.halo + mesh.block_size[2])]

struct Fix
    mesh; interface; geometry; scalar; workspace
end

function fixture(kind, rng)
    cx, cy, R = 0.3 + 0.4rand(rng), 0.3 + 0.4rand(rng), 0.12 + 0.12rand(rng)
    if kind === :periodic
        mesh = AA.periodic_mesh()
        geometry = nothing
        interface = InterfaceField(mesh, (x, y) -> hypot(x - cx, y - cy) - R; lipschitz = 1)
    elseif kind === :wetted      # drop resting on (overlapping) the wavy solid: triple cells (solid, liquid, gas)
        mesh = AA.walled_mesh()
        geometry = GeometryField(mesh, (x, y) -> y - 0.15 - 0.1sinpi(x); lipschitz = 1.5)
        wx, wy = 0.7 + 0.1rand(rng), 0.40 + 0.04rand(rng)
        interface = InterfaceField(mesh, (x, y) -> hypot(x - wx, y - wy) - 0.22; geometry, lipschitz = 1)
    else
        mesh = AA.walled_mesh()
        geometry = GeometryField(mesh, (x, y) -> y - 0.15 - 0.1sinpi(x); lipschitz = 1.5)
        interface = InterfaceField(mesh, (x, y) -> hypot(x - 2cx, y - 0.45 - 0.2rand(rng)) - R; geometry, lipschitz = 1)
    end
    scalar = PhaseScalar(interface; liquid = (x, y) -> 1 + x, gas = (x, y) -> 2 - y)
    for b in 1:nblocks(mesh)       # add noise so that the identities do not rely on smooth data
        scalar.liquid.blocks[b] .+= 0.3 .* randn(rng, size(scalar.liquid.blocks[b]))
        scalar.gas.blocks[b] .+= 0.3 .* randn(rng, size(scalar.gas.blocks[b]))
    end
    workspace = isnothing(geometry) ? AMRWorkspace(mesh, interface, scalar) : AMRWorkspace(mesh, interface, geometry, scalar)
    return Fix(mesh, interface, geometry, scalar, workspace)
end

openblock(f::Fix, b) = isnothing(f.geometry) ? ones(size(f.interface.fraction.blocks[b])) : copy(f.geometry.open_fraction.blocks[b])

function amounts(f::Fix)
    mesh = f.mesh
    L = G = V = 0.0
    for (b, block) in pairs(mesh.blocks)
        phi = interior_of(mesh, f.interface.fraction.blocks[b])
        op = interior_of(mesh, openblock(f, b))
        L += block.volume * sum(phi .* interior_of(mesh, f.scalar.liquid.blocks[b]))
        G += block.volume * sum((op .- phi) .* interior_of(mesh, f.scalar.gas.blocks[b]))
        V += block.volume * sum(phi)
    end
    return (; L, G, V)
end

function snapshot(f::Fix)
    mesh = f.mesh
    reconstruct!(f.interface)
    fill_halo!(f.scalar.liquid, ZeroGradient()); fill_halo!(f.scalar.gas, ZeroGradient())
    snap = Dict{BlockKey, Any}()
    for (b, block) in pairs(mesh.blocks)
        snap[block.key] = (
            phi = copy(f.interface.fraction.blocks[b]), open = openblock(f, b),
            nx = copy(f.interface.normal_x.blocks[b]), ny = copy(f.interface.normal_y.blocks[b]), a = copy(f.interface.intercept.blocks[b]),
            sx = isnothing(f.geometry) ? nothing : copy(f.geometry.solid.normal_x.blocks[b]),
            sy = isnothing(f.geometry) ? nothing : copy(f.geometry.solid.normal_y.blocks[b]),
            sa = isnothing(f.geometry) ? nothing : copy(f.geometry.solid.intercept.blocks[b]),
            ql = copy(f.scalar.liquid.blocks[b]), qg = copy(f.scalar.gas.blocks[b]),
        )
    end
    return snap
end

function quadrant(fi, fj)
    xlo, ylo = isodd(fi) ? -0.5 : 0.0, isodd(fj) ? -0.5 : 0.0
    return [(xlo, ylo), (xlo + 0.5, ylo), (xlo + 0.5, ylo + 0.5), (xlo, ylo + 0.5)]
end

# plane defect of a parent: stored fraction vs the area of its own plane, independent clipping in local coordinates
function plane_defect(P, pi, pj, has_solid)
    cell = [(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)]
    open = has_solid ? clip(cell, (P.sx[pi, pj], P.sy[pi, pj]), P.sa[pi, pj]; keep_less = false) : cell
    liquid = clip(open, (P.nx[pi, pj], P.ny[pi, pj]), P.a[pi, pj])
    return area(liquid) - P.phi[pi, pj]
end

function run_sequence(kind, seed; random_labels = false)
    rng = MersenneTwister(seed)
    f = fixture(kind, rng)
    mesh, h = f.mesh, f.mesh.halo
    nx, ny = mesh.block_size
    has_solid = !isnothing(f.geometry)
    floor_ = f.interface.fraction_floor
    stats = Dict(:refine_dv => 0.0, :merge_dv => 0.0, :plane => 0.0, :replica => 0.0, :bounds => 0.0, :pure => 0.0,
                 :amountL_refine => 0.0, :amountG_refine => 0.0, :amountL_merge => 0.0, :amountG_merge => 0.0, :s_replica => 0.0,
                 :s_merge => 0.0, :s_convex => 0.0, :s_overshoot => 0.0, :s_overshoot_lin => 0.0, :snapped_families => 0, :families => 0, :pure_children => 0, :unpure => 0)
    nsteps = random_labels ? 6 : 4
    for step in 1:nsteps
        am0 = amounts(f)
        snap = snapshot(f)
        labels = random_labels ? AA.random_labels(rng, mesh; pr = step <= 2 ? 0.6 : 0.3, pc = step <= 2 ? 0.1 : 0.6) :
            (step <= 2 ? [mesh.blocks[b].key.level < 2 && (rand(rng) < 0.7) ? :refine : :keep for b in 1:nblocks(mesh)] : fill(:coarsen, nblocks(mesh)))
        st = adapt!(f.workspace, labels; step, force = true)
        st.changed || continue
        am1 = amounts(f)
        refined, merged = st.refined_blocks > 0, st.coarsened_families > 0
        # V1: volume change, attributed to refinement-only (when nothing merged) or merging-only (nothing refined)
        dv = abs(am1.V - am0.V) / am0.V
        if refined && !merged
            stats[:refine_dv] = max(stats[:refine_dv], dv)
        elseif merged && !refined
            stats[:merge_dv] = max(stats[:merge_dv], dv)
        end
        # per parent cell measures
        for (b, block) in pairs(mesh.blocks)
            key = block.key
            phi = f.interface.fraction.blocks[b]
            op = openblock(f, b)
            if !haskey(snap, key) && key.level > 0 && haskey(snap, AA.parent_key(key))
                P = snap[AA.parent_key(key)]
                ci, cj = mod(key.i, 2), mod(key.j, 2)
                for j in 1:ny, i in 1:nx
                    fi, fj = ci * nx + i, cj * ny + j
                    pi, pj = (fi + 1) ÷ 2 + h, (fj + 1) ÷ 2 + h
                    Q = quadrant(fi, fj)
                    open = has_solid ? clip(Q, (P.sx[pi, pj], P.sy[pi, pj]), P.sa[pi, pj]; keep_less = false) : Q
                    liquid = clip(open, (P.nx[pi, pj], P.ny[pi, pj]), P.a[pi, pj])
                    alpha = clamp(4area(liquid), 0.0, 4area(open))
                    c = op[i + h, j + h] > 0 ? phi[i + h, j + h] / op[i + h, j + h] : 0.0
                    snapped = 0 < c <= floor_ || 1 - floor_ <= c < 1
                    snapped || (stats[:replica] = max(stats[:replica], abs(phi[i + h, j + h] - alpha)))
                    # bounds: 0 <= phi <= open (post regrid, post snap)
                    stats[:bounds] = max(stats[:bounds], max(-phi[i + h, j + h], phi[i + h, j + h] - op[i + h, j + h], 0.0))
                    # pure parents give pure children (conditional fraction)
                    cpar = P.open[pi, pj] > 0 ? P.phi[pi, pj] / P.open[pi, pj] : 0.0
                    if cpar == 0.0 || cpar == 1.0
                        stats[:pure_children] += 1
                        stats[:pure] = max(stats[:pure], abs(c - cpar) * (op[i + h, j + h] > 0))
                    end
                end
            end
        end
        # plane defects of parents that were refined
        newkeys = Dict(bl.key => b for (b, bl) in pairs(mesh.blocks))
        for (key, P) in snap
            haskey(newkeys, key) && continue
            all(k -> haskey(newkeys, k), AA.child_keys(key)) || continue
            for j in 1:ny, i in 1:nx
                pi, pj = i + h, j + h
                (0 < P.phi[pi, pj] < P.open[pi, pj]) || continue
                stats[:plane] = max(stats[:plane], abs(plane_defect(P, pi, pj, has_solid)))
            end
        end
        # S1/S2
        amL, amG = am1.L - am0.L, am1.G - am0.G
        (refined && !merged) && (stats[:amountL_refine] = max(stats[:amountL_refine], abs(amL) / (abs(am0.L) + eps()));
                                  stats[:amountG_refine] = max(stats[:amountG_refine], abs(amG) / (abs(am0.G) + eps())))
        (merged && !refined) && (stats[:amountL_merge] = max(stats[:amountL_merge], abs(amL) / (abs(am0.L) + eps()));
                                  stats[:amountG_merge] = max(stats[:amountG_merge], abs(amG) / (abs(am0.G) + eps())))
        # refined parents: replica of the chapter's shift formula, both phases, in families without snapped children
        for (pkey, P) in snap
            haskey(newkeys, pkey) && continue
            kids = AA.child_keys(pkey)
            if all(k -> haskey(newkeys, k), kids)
                for pj in (h + 1):(h + ny), pi in (h + 1):(h + nx)
                    stats[:families] += 1
                    # the 4 children of the parent cell (pi,pj): fine indices
                    fis = (2(pi - h) - 1, 2(pi - h)); fjs = (2(pj - h) - 1, 2(pj - h))
                    ref = [];
                    snapped_any = false
                    for fj in fjs, fi in fis
                        cb = newkeys[kids[(fi - 1) ÷ nx + 2((fj - 1) ÷ ny) + 1]]
                        ii, jj = mod(fi - 1, nx) + 1 + h, mod(fj - 1, ny) + 1 + h
                        phic = f.interface.fraction.blocks[cb][ii, jj]
                        opc = openblock(f, cb)[ii, jj]
                        Q = quadrant(fi, fj)
                        open = has_solid ? clip(Q, (P.sx[pi, pj], P.sy[pi, pj]), P.sa[pi, pj]; keep_less = false) : Q
                        liquid = clip(open, (P.nx[pi, pj], P.ny[pi, pj]), P.a[pi, pj])
                        alpha = clamp(4area(liquid), 0.0, 4area(open))
                        c = opc > 0 ? phic / opc : 0.0
                        (0 < c <= floor_ || 1 - floor_ <= c < 1 || abs(phic - alpha) > 1e-13 || abs(opc - 4area(open)) > 1e-13) && (snapped_any = true)
                        push!(ref, (; cb, ii, jj, alpha, openarea = 4area(open), fi, fj))
                    end
                    if snapped_any
                        stats[:snapped_families] += 1
                        continue
                    end
                    for (phase, Q) in ((:l, P.ql), (:g, P.qg))
                        v = Q[pi, pj]
                        sx, sy = mc(v - Q[pi - 1, pj], Q[pi + 1, pj] - v), mc(v - Q[pi, pj - 1], Q[pi, pj + 1] - v)
                        lin = [v + (isodd(r.fi) ? -0.25 : 0.25) * sx + (isodd(r.fj) ? -0.25 : 0.25) * sy for r in ref]
                        w = phase === :l ? [r.alpha for r in ref] : [clamp(r.openarea - r.alpha, 0.0, 1.0) for r in ref]
                        wpar = phase === :l ? P.phi[pi, pj] : P.open[pi, pj] - P.phi[pi, pj]
                        expect = if iszero(sx) && iszero(sy)
                            fill(v, 4)
                        elseif sum(w) > sqrt(eps())
                            lin .+ (4wpar * v - sum(w .* lin)) / sum(w)
                        else
                            lin
                        end
                        got = [(phase === :l ? f.scalar.liquid : f.scalar.gas).blocks[r.cb][r.ii, r.jj] for r in ref]
                        stats[:s_replica] = max(stats[:s_replica], maximum(abs, got .- expect))
                        # not claimed by the chapter, measured: does the shift push a child out of the range of the parent and its four neighbours (the unshifted limited values stay inside)?
                        lo, hi = extrema((v, Q[pi - 1, pj], Q[pi + 1, pj], Q[pi, pj - 1], Q[pi, pj + 1]))
                        rng_ = max(hi - lo, eps())
                        stats[:s_overshoot] = max(stats[:s_overshoot], maximum(max.(lo .- got, got .- hi, 0.0)) / rng_)
                        stats[:s_overshoot_lin] = max(stats[:s_overshoot_lin], maximum(max.(lo .- lin, lin .- hi, 0.0)) / rng_)
                    end
                end
            end
        end
        # merged parents: weighted mean of the old children
        for (b, block) in pairs(mesh.blocks)
            key = block.key
            (haskey(snap, key) || (key.level > 0 && haskey(snap, AA.parent_key(key)))) && continue
            kids = AA.child_keys(key)
            for j in 1:ny, i in 1:nx
                vals_l = Float64[]; w_l = Float64[]; vals_g = Float64[]; w_g = Float64[]
                for fj in (2j - 1):2j, fi in (2i - 1):2i
                    C = snap[kids[(fi - 1) ÷ nx + 2((fj - 1) ÷ ny) + 1]]
                    ii, jj = mod(fi - 1, nx) + 1 + h, mod(fj - 1, ny) + 1 + h
                    push!(vals_l, C.ql[ii, jj]); push!(w_l, C.phi[ii, jj])
                    push!(vals_g, C.qg[ii, jj]); push!(w_g, C.open[ii, jj] - C.phi[ii, jj])
                end
                for (vals, w, got) in ((vals_l, w_l, f.scalar.liquid.blocks[b][i + h, j + h]), (vals_g, w_g, f.scalar.gas.blocks[b][i + h, j + h]))
                    # skip merged cells whose fraction was snapped by the reconstruction (the intrinsic value is unaffected, but weights differ in the amount only)
                    expect = all(==(vals[1]), vals) ? vals[1] : sum(w) > 0 ? sum(w .* vals) / sum(w) : sum(vals) / 4
                    stats[:s_merge] = max(stats[:s_merge], abs(got - expect))
                    stats[:s_convex] = max(stats[:s_convex], max(minimum(vals) - got, got - maximum(vals), 0.0))
                end
            end
        end
    end
    return stats
end

@testset "eq:amr:vof / eq:amr:scalar" begin
    allstats = []
    for (kind, seed, rl) in ((:periodic, 1, false), (:periodic, 2, true), (:periodic, 3, true), (:walled, 4, false), (:walled, 5, true), (:walled, 6, true), (:wetted, 7, false), (:wetted, 8, true), (:wetted, 9, true))
        s = run_sequence(kind, seed; random_labels = rl)
        push!(allstats, s)
        println(rpad("$kind seed $seed" * (rl ? " random" : " refine/coarsen"), 32),
            " refine dV/V=", s[:refine_dv], " merge dV/V=", s[:merge_dv], " plane defect=", s[:plane], " child replica=", s[:replica], " child 0<=phi<=open viol=", s[:bounds],
            " pure children=", s[:pure_children], " (dev ", s[:pure], ")  amounts L/G refine=", s[:amountL_refine], "/", s[:amountG_refine], " merge=", s[:amountL_merge], "/", s[:amountG_merge],
            "  scalar replica=", s[:s_replica], " merge=", s[:s_merge], " convex=", s[:s_convex], " shift overshoot (rel. to the stencil range) ", s[:s_overshoot], " (unshifted ", s[:s_overshoot_lin], ")", " families checked=", s[:families] - s[:snapped_families], " snapped families=", s[:snapped_families])
        @test s[:merge_dv] <= 1e-14 && s[:refine_dv] <= 1e-12 && s[:bounds] <= 1e-14 && s[:replica] <= 1e-14 && s[:pure] == 0.0
        @test s[:s_replica] <= 1e-12 && s[:s_merge] <= 1e-12 && s[:s_convex] <= 1e-12
    end
    @test maximum(s[:families] - s[:snapped_families] for s in allstats) > 100
end

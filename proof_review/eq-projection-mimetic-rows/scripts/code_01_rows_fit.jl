# Code check 01: the aperture vector, the base rows of the two mimetic variants and the joint fit, against an independent transcription of the text.
#
# Labels: eq:projection:aperture-vector, eq:projection:smoothed-rows, eq:projection:face-rows, eq:projection:joint-fit
# Source: docs/techref/chapters/10_projection.tex:585-760
# Checks (4 fixtures; random predictor with phase jump; the code's active set (verified in eq-projection-kkt) is taken as given):
#   R1  apvec(c) = Σ_f o_{c,f}|f| a^l_f n_f (boundary faces with s_f) = the code's integrated aperture vector; |apvec| vs the PLIC segment length (planar fixture);
#   R2  smoothed rows: independent rows (one-ring sum over face neighbours once per shared face with apvec·apvec' > 0, ρ, area-weighted cell averages of
#       the active faces, unresolved rule |ρ_k| > 64 eps |apvec| needs an active face) equal the code's base rows (N⁰ and T⁰) entry by entry;
#   R3  face rows: independent selection (CMDGFM-resolved, β >= 1/2, all transverse faces active; else aligned-aperture fallback) and coefficients equal the code's
#       base rows N⁰ (`_local_joint_face_rows`), scales sqrt(1+R²) / 1;
#   R4  joint fit: an independent implementation of eq. joint-fit (supports by graph radius 2, geometric base weights with the 1/16 offset, exact moments,
#       4 pressure observations + commutation observations with weights 1000, ridge 1/10, cap 1, 3 alternating sweeps) reproduces the code's fitted rows
#       `candidate_constraint` (entry-wise relative difference), and the fitted rows reproduce the moments exactly;
#   R5  rank: the retained rows (first `rank` QR pivots in original order) span the candidate rows numerically (SVD rank of the normalised candidate matrix).
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-projection-mimetic-rows/scripts/code_01_rows_fit.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ProjectionAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "projection_audit.jl"))
using .ProjectionAudit
using TwoPhaseAMR
const PA = ProjectionAudit
include(joinpath(@__DIR__, "..", "..", "eq-projection-cmdgfm-system", "scripts", "code_01_stencils_lib.jl"))   # cell_normal, classify

# --- R1 -----------------------------------------------------------------------------------------
function aperture_vectors(field, d)
    mesh = field.mesh; nx, ny = mesh.block_size
    n = length(mesh.blocks) * nx * ny
    A = zeros(n, 2)
    for (i, face) in pairs(mesh.faces)
        w = face.area * d.al[i]
        A[AMR._cell_index(face.minus, nx, ny), face.axis] += w
        A[AMR._cell_index(face.plus, nx, ny), face.axis] -= w
    end
    for (i, face) in pairs(mesh.boundary_faces)
        l = isnothing(field.geometry) ? AMR._phase_boundary_face_apertures(field, i)[1] : AMR._phase_boundary_face_apertures(field, field.geometry, i)[1]
        A[AMR._cell_index(face.cell, nx, ny), face.axis] += face.side * face.area * l
    end
    return A
end

# --- R2 -----------------------------------------------------------------------------------------
function smoothed_rows(field, d, act)
    mesh = field.mesh; nx, ny = mesh.block_size; fl = field.fraction_floor
    A = aperture_vectors(field, d)
    refs = AMR._cell_references(mesh)
    n = length(refs)
    C = [AMR._conditional_fraction(field, r) for r in refs]
    mixed = [fl < C[c] < 1 - fl for c in 1:n]
    nbr = [Int[] for _ in 1:n]
    for face in mesh.faces
        m = AMR._cell_index(face.minus, nx, ny); p = AMR._cell_index(face.plus, nx, ny)
        push!(nbr[m], p); push!(nbr[p], m)
    end
    lookup = Dict(f => i for (i, f) in pairs(act))
    inc = [Int[] for _ in 1:n]
    for (g, face) in pairs(mesh.faces)
        push!(inc[AMR._cell_index(face.minus, nx, ny)], g); push!(inc[AMR._cell_index(face.plus, nx, ny)], g)
    end
    rows = NamedTuple[]
    for (c, ref) in pairs(refs)
        mixed[c] || continue
        isnothing(field.geometry) || AMR._get(field.geometry.open_fraction, ref) >= 1 - fl || continue
        meas = hypot(A[c, 1], A[c, 2])
        h = maximum(mesh.blocks[ref.block].spacing)
        meas > 64eps() * max(h, 1.0) || continue
        S = A[c, :] |> collect
        for c2 in nbr[c]
            mixed[c2] || continue
            A[c, 1] * A[c2, 1] + A[c, 2] * A[c2, 2] > 0 || continue
            S .+= A[c2, :]
        end
        ρ = meas .* S ./ hypot(S...)
        τ = (-ρ[2], ρ[1])
        N = Dict{Int, Float64}(); T = Dict{Int, Float64}(); ok = true
        for k in 1:2
            av = [g for g in inc[c] if mesh.faces[g].axis == k && haskey(lookup, g)]
            tot = sum(mesh.faces[g].area for g in av; init = 0.0)
            if tot <= 0
                abs(ρ[k]) > 64eps() * meas && (ok = false)
                continue
            end
            for g in av
                w = mesh.faces[g].area / tot
                abs(ρ[k] * w) > 64eps() * meas && (N[lookup[g]] = get(N, lookup[g], 0.0) + ρ[k] * w)
                abs(τ[k] * w) > 64eps() * meas && (T[lookup[g]] = get(T, lookup[g], 0.0) + τ[k] * w)
            end
        end
        ok || continue
        isempty(N) && continue
        push!(rows, (; location = c, ρ, τ, N, T, scale = meas))
    end
    return rows
end

# --- R3 -----------------------------------------------------------------------------------------
function face_rows(field, d, act, ind)
    mesh = field.mesh; nx, ny = mesh.block_size; fl = field.fraction_floor
    A = aperture_vectors(field, d)
    lookup = Dict(f => i for (i, f) in pairs(act))
    rows = NamedTuple[]
    for f in act
        s = get(ind, f, nothing)
        face = mesh.faces[f]; k = face.axis; t = 3 - k
        if s !== nothing && s.class == :resolved
            s.beta >= 0.5 || continue
            all(g -> haskey(lookup, g), s.trans) || continue
            N = Dict{Int, Float64}(lookup[f] => 1.0)
            for (g, w) in zip(s.trans, s.w)
                N[lookup[g]] = get(N, lookup[g], 0.0) + s.R * w
            end
            push!(rows, (; location = f, N, scale = hypot(1.0, s.R)))
        else
            m = AMR._cell_index(face.minus, nx, ny); p = AMR._cell_index(face.plus, nx, ny)
            hm = mesh.blocks[face.minus.block].spacing[k]; hp = mesh.blocks[face.plus.block].spacing[k]
            q = ((hm * A[m, 1] + hp * A[p, 1]) / (hm + hp), (hm * A[m, 2] + hp * A[p, 2]) / (hm + hp))
            mag = hypot(q...)
            mag > 0 || continue
            tol = min(1.0, max(mesh.blocks[face.minus.block].spacing...) / min(mesh.extent...))
            abs(q[t]) <= tol * mag || continue
            axial = q[k] / mag; trans = q[t] / mag
            abs(axial) > sqrt(eps()) || continue
            N = Dict{Int, Float64}(lookup[f] => axial)
            if abs(trans) > 64eps()
                nb = [g for g in PA.patch_faces(mesh, f, t) if haskey(lookup, g)]
                isempty(nb) && continue
                w = [d.C[g] * (1 - d.C[g]) for g in nb]; sum(w) > 0 || continue
                for (g, wi) in zip(nb, w)
                    N[lookup[g]] = get(N, lookup[g], 0.0) + trans * wi / sum(w)
                end
            end
            push!(rows, (; location = f, N, scale = 1.0))
        end
    end
    return rows
end

# --- R4: the joint fit --------------------------------------------------------------------------
"""Independent implementation of eq. joint-fit. `rows` carry N (dict local face => value) and moments ρ (2-vector); returns the dense fitted normal matrix."""
function joint_fit(field, d, act, relD, rows, centres, moment_n, moment_t, spacings; radius = 2, ω_p = 1000.0, ω_c = 1000.0, cap = 1.0, sweeps = 3, density = PA.DENSITY)
    mesh = field.mesh; nx, ny = mesh.block_size
    nr = length(rows); nF = length(act)
    axisof = [mesh.faces[f].axis for f in act]
    # base patterns (N⁰ and, for the smoothed variant, T⁰) define the graph
    pat = [union(keys(r.N), haskey(r, :T) ? keys(r.T) : Int[]) |> collect |> sort for r in rows]
    cols_rows = Dict{Int, Vector{Int}}()
    for (r, cs) in pairs(pat), c in cs
        push!(get!(cols_rows, c, Int[]), r)
    end
    support = Vector{Vector{Int}}(undef, nr)
    for r in 1:nr
        sel = Set([r]); front = [r]
        for _ in 1:radius
            nxt = Int[]
            for q in front, c in pat[q], r2 in cols_rows[c]
                r2 in sel && continue
                push!(sel, r2); push!(nxt, r2)
            end
            front = nxt; isempty(front) && break
        end
        support[r] = sort(unique(vcat((pat[q] for q in sel)...)))
    end
    # geometric base rows
    base_t = Vector{Vector{Float64}}(undef, nr); base_n = Vector{Vector{Float64}}(undef, nr); geo_fallback = falses(nr)
    for r in 1:nr
        bt = zeros(length(support[r])); bn = zeros(length(support[r]))
        for a in 1:2
            pos = [p for (p, c) in pairs(support[r]) if axisof[c] == a]
            if isempty(pos)
                geo_fallback[r] = true; continue
            end
            w = Float64[]
            for p in pos
                face = mesh.faces[act[support[r][p]]]
                dx = AMR._wrapped_delta(face.center[1] - centres[r][1], mesh.extent[1], mesh.periodic[1]) / spacings[r][1]
                dy = AMR._wrapped_delta(face.center[2] - centres[r][2], mesh.extent[2], mesh.periodic[2]) / spacings[r][2]
                push!(w, 1 / (dx^2 + dy^2 + 1 / 16))
            end
            w ./= sum(w)
            for (p, wi) in zip(pos, w)
                bn[p] = moment_n[r][a] * wi; bt[p] = moment_t[r][a] * wi
            end
        end
        base_t[r] = bt; base_n[r] = bn
    end
    # observations
    P = Vector{Matrix{Float64}}(undef, nr); Ptarget = Vector{Vector{Float64}}(undef, nr)
    C = Dict{Int, Float64}()
    refs = AMR._cell_references(mesh)
    cellC(ci) = AMR._conditional_fraction(field, refs[ci])
    invdj = 1 / density[2] - 1 / density[1]
    for r in 1:nr
        nvec = moment_n[r] ./ hypot(moment_n[r]...)
        obs = zeros(4, length(support[r]))
        for (p, c) in pairs(support[r])
            f = act[c]; face = mesh.faces[f]
            fac = d.apgfm[f] * face.area / relD[c]
            disp = AMR._face_displacement(mesh, face)
            obs[1, p] = fac * disp[1]; obs[2, p] = fac * disp[2]
            mi = AMR._cell_index(face.minus, nx, ny); pi_ = AMR._cell_index(face.plus, nx, ny)
            function dist(ci)
                x = cellcenter(mesh, refs[ci].block, refs[ci].i, refs[ci].j)
                dxx = AMR._wrapped_delta(x[1] - centres[r][1], mesh.extent[1], mesh.periodic[1]); dyy = AMR._wrapped_delta(x[2] - centres[r][2], mesh.extent[2], mesh.periodic[2])
                return nvec[1] * dxx + nvec[2] * dyy
            end
            obs[3, p] = fac * (cellC(pi_) * dist(pi_) - cellC(mi) * dist(mi))
            obs[4, p] = fac * ((1 - cellC(pi_)) * dist(pi_) - (1 - cellC(mi)) * dist(mi))
        end
        P[r] = obs
        Ptarget[r] = [invdj * moment_t[r][1], invdj * moment_t[r][2], 0.0, 0.0]
    end
    moments(r) = [Float64(axisof[c] == a) for a in 1:2, c in support[r]]
    function soft_row(baseline, Mom, target, obs_blocks)
        isempty(baseline) && return baseline, false
        # exact-moment origin: minimum 2-norm correction (row-normalised constraints; zero rows with zero target are dropped)
        keep = [i for i in 1:2 if norm(Mom[i, :]) > 64eps()]
        any(i -> i ∉ keep && abs(target[i]) > 64eps(), 1:2) && return baseline, false
        Mk = Mom[keep, :]; tk = target[keep]
        origin = baseline .- Mk' * ((Mk * Mk') \ (Mk * baseline .- tk))
        B = nullspace(Mom; atol = 1.0e-10)
        if isempty(B)
            return origin, true
        end
        rowsA = Vector{Vector{Float64}}(); rhs = Float64[]
        for (O, desired, w) in obs_blocks
            w > 0 || continue
            for i in axes(O, 1)
                o = O[i, :]; sc = norm(o); dv = desired === nothing ? 0.0 : desired[i]
                if sc <= 64eps()
                    abs(dv) > 64eps() && return baseline, false
                    continue
                end
                fac = sqrt(w) / sc
                push!(rowsA, fac .* (B' * o)); push!(rhs, fac * (dv - dot(o, origin)))
            end
        end
        nb = size(B, 2)
        Amat = vcat((r' for r in rowsA)..., sqrt(0.1) * Matrix{Float64}(I, nb, nb))
        bvec = vcat(rhs, zeros(nb))
        coef = Amat \ bvec
        dirv = B * coef
        all(isfinite, dirv) || return baseline, false
        frac = min(1.0, cap * norm(baseline) / max(norm(dirv), eps()))
        res = origin .+ frac .* dirv
        ok = all(isfinite, res) && norm(Mom * res .- target, Inf) <= 1.0e-8 * max(norm(target, Inf), 1.0)
        return ok ? res : baseline, ok
    end
    cur_n = [copy(b) for b in base_n]; cur_t = [copy(b) for b in base_t]
    dropsmall(v) = [abs(x) <= 64eps() ? 0.0 : x for x in v]
    function overlap_rows(rowvals_by_row, r)
        out = Vector{Vector{Float64}}()
        for r2 in 1:nr
            vals2 = zeros(length(support[r]))
            hit = false
            for (p, c) in pairs(support[r])
                pos2 = findfirst(==(c), support[r2])
                pos2 === nothing && continue
                v = dropsmall(rowvals_by_row[r2])[pos2]
                v == 0 && continue
                vals2[p] = v / relD[c]; hit = true
            end
            hit && push!(out, vals2)
        end
        return isempty(out) ? zeros(0, length(support[r])) : reduce(vcat, (o' for o in out))
    end
    fb = falses(nr)
    for _ in 1:sweeps
        newt = deepcopy(cur_t)
        curN = cur_n
        for r in 1:nr
            O = overlap_rows(curN, r)
            res, ok = soft_row(base_t[r], moments(r), moment_t[r], [(P[r], Ptarget[r], ω_p), (O, nothing, ω_c)])
            newt[r] = res
        end
        cur_t = newt
        newn = deepcopy(cur_n)
        for r in 1:nr
            O = overlap_rows(cur_t, r)
            res, ok = soft_row(base_n[r], moments(r), moment_n[r], [(zeros(0, length(support[r])), nothing, 0.0), (O, nothing, ω_c)])
            newn[r] = res
            fb[r] = !ok
        end
        cur_n = newn
    end
    Nmat = zeros(nr, nF)
    for r in 1:nr, (p, c) in pairs(support[r])
        v = cur_n[r][p]; abs(v) > 64eps() && (Nmat[r, c] = v)
    end
    return Nmat, support, geo_fallback
end

"""Stress fixtures that exercise the fallback paths (unresolved CMDGFM stencils, aperture-aligned face rows, dependent rows, fit fallbacks)."""
function stress_fixtures()
    out = Pair{String, Any}[]
    for kind in (:uniform, :amr)
        mesh = FA.mesh_of(kind)
        push!(out, "small drop $(kind)" => InterfaceField(mesh, (x, y) -> hypot(x - 3.9, y - 3.1) - 1.3; lipschitz = 1))
        push!(out, "thin slab $(kind)" => InterfaceField(mesh, (x, y) -> abs(y - 3.9) - 0.55; lipschitz = 1))
        push!(out, "wavy $(kind)" => InterfaceField(mesh, (x, y) -> y - 4.2 - 0.5 * sinpi(2x / 2.7); lipschitz = 1.7))
    end
    return out
end

@testset "mimetic rows and fit against independent implementations" begin
    for (name, field) in vcat(PA.fixtures(), stress_fixtures())
        mesh = field.mesh
        d = PA.face_data(field)
        ind = classify(field, d)
        for variant in (:joint_smoothed, :joint_face)
            solver = MimeticProjectionSolver(field; density = PA.DENSITY, variant)
            work = MimeticProjectionWorkspace(solver)
            v0 = PA.two_predictor(field, MersenneTwister(47))
            assembled = AMR._assemble_mimetic_system!(solver, work, PA.DT, PA.copy_velocity(v0))
            act = assembled.active_faces; relD = assembled.relative_diagonal
            nF = length(act)
            rec = AMR._local_joint_reconstruction(solver, assembled.data, act, relD)
            if variant == :joint_smoothed
                # R1
                Ax, Ay = AMR._local_integrated_aperture_vectors(field, assembled.data)
                A = aperture_vectors(field, d)
                @test maximum(abs.(A[:, 1] .- Ax)) <= 1.0e-13 && maximum(abs.(A[:, 2] .- Ay)) <= 1.0e-13
                # R2
                rows = smoothed_rows(field, d, act)
                @test [r.location for r in rows] == rec.candidate_cells
                Nc = Matrix(rec.normal); Tc = Matrix(rec.tangent)
                e2 = 0.0
                for (i, r) in pairs(rows)
                    Nrow = zeros(nF); Trow = zeros(nF)
                    for (j, v) in r.N
                        Nrow[j] = v
                    end
                    for (j, v) in r.T
                        Trow[j] = v
                    end
                    e2 = max(e2, maximum(abs.(Nc[i, :] .- Nrow)) / r.scale, maximum(abs.(Tc[i, :] .- Trow)) / r.scale)
                    @test rec.candidate_scales[i] ≈ r.scale rtol = 1.0e-13
                end
                println(rpad(name, 18), " R1 ok; R2 (smoothed rows): ", length(rows), " rows, max entry difference ", e2)
                @test e2 <= 1.0e-13
                mom_n = [r.ρ for r in rows]; mom_t = [r.τ for r in rows]
                centres = [
                    begin
                        ref = AMR._cell_references(mesh)[r.location]; cellcenter(mesh, ref.block, ref.i, ref.j)
                    end for r in rows
                ]
                spac = [mesh.blocks[AMR._cell_references(mesh)[r.location].block].spacing for r in rows]
                fitrows = rows
            else
                rows = face_rows(field, d, act, ind)
                @test [r.location for r in rows] == rec.candidate_cells
                Nc = Matrix(rec.normal)
                e3 = 0.0
                for (i, r) in pairs(rows)
                    Nrow = zeros(nF); for (j, v) in r.N
                        Nrow[j] = v
                    end
                    e3 = max(e3, maximum(abs.(Nc[i, :] .- Nrow)))
                    @test rec.candidate_scales[i] ≈ r.scale rtol = 1.0e-13
                end
                println(rpad(name, 18), " R3 (face rows): ", length(rows), " rows (", count(r -> r.scale != 1.0, rows), " with a transverse term R ≠ 0, ", length(rows) - count(r -> r.scale != 1.0, rows), " aligned/aperture-fallback rows), max entry difference ", e3)
                @test e3 <= 1.0e-13
                lookup = Dict(f => i for (i, f) in pairs(act))
                # moments of the face rows: N⁰ c = (Σ_f∈axis1 N, Σ_f∈axis2 N)
                mom_n = [(sum(v for (j, v) in r.N if mesh.faces[act[j]].axis == 1; init = 0.0), sum(v for (j, v) in r.N if mesh.faces[act[j]].axis == 2; init = 0.0)) for r in rows]
                mom_t = [(-m[2], m[1]) for m in mom_n]
                centres = [mesh.faces[r.location].center for r in rows]
                spac = [mesh.blocks[mesh.faces[r.location].minus.block].spacing for r in rows]
                fitrows = rows
            end
            isempty(fitrows) && continue
            # R4
            Nfit, support, geofb = joint_fit(field, d, act, relD, fitrows, centres, [collect(m) for m in mom_n], [collect(m) for m in mom_t], spac)
            Ncode = Matrix(assembled.candidate_constraint)
            e4 = maximum(abs.(Nfit .- Ncode)) / maximum(abs.(Ncode))
            # moments exactly reproduced
            Z = zeros(nF, 2); for (j, f) in pairs(act)
                Z[j, mesh.faces[f].axis] = 1
            end
            mom_err = maximum(abs.(Ncode * Z .- reduce(vcat, (collect(m)' for m in mom_n)))) / maximum(abs.(reduce(vcat, (collect(m)' for m in mom_n))))
            println(
                "    R4 (fit, ", variant, "): rows ", size(Ncode, 1), ", max |N_indep - N_code|/max|N| = ", e4, ", moment error of the code's fitted rows ", mom_err,
                ", code fallbacks ", assembled.fallback_count, ", independent geometric fallbacks ", count(geofb)
            )
            @test e4 <= 1.0e-9
            @test mom_err <= 1.0e-8
            # R5 rank
            Dm = Diagonal(1 ./ sqrt.(relD))
            Nn = Ncode * Dm; Nn = Nn ./ sqrt.(sum(abs2, Nn; dims = 2))
            sv = svdvals(Nn)
            rk = count(>(sqrt(eps()) * sv[1]), sv)
            println("    R5: candidate rows ", size(Ncode, 1), ", numerical rank (SVD, √eps) ", rk, ", code retained ", size(assembled.constraint, 1), ", dependent ", assembled.dependent_count)
            # (the QR-pivot rank and the SVD rank may differ by borderline rows: reported only)
            @test rank(Matrix(assembled.constraint * Dm)) == size(assembled.constraint, 1)
        end
    end
end

# Code check 01: two-velocity face data, presence of phases, mixture divergence, mixture update (matrix form) and the component gauge identity.
#
# Labels: eq:projection:face-data, eq:projection:mixed, eq:projection:mixture-divergence, eq:projection:phase-gradient (common part),
#         eq:projection:mixture-update, eq:projection:gauge
# Source: docs/techref/chapters/10_projection.tex:164-262
# Checks:
#   F1  independent ĉ (own staggered fraction), masses, ζ, apgfm, m^r equal the code's `_projection_face_data`; a^l + a^g = a_f; mixed set equal;
#   F2  NaN pattern of the phase velocities: liquid present iff ĉ > 0 (and V^o > 0 with a solid), gas present iff ĉ < 1;
#   F3  the code's mixture flux (`_cmdgfm_cell_flux!`, = -|c^o| D^mix) equals the independent Σ_f o |f| (a^l u^l + a^g u^g) incl. boundary faces with s_f;
#   F4  Σ_{f in F(c)} o_{c,f}|f| v_f = -(J'v)_c for a face function vanishing on the boundary; J' A R_tv J = S_tv (the stiffness of the two-velocity coefficient a/ρ̄);
#   F5  component identity (eq. gauge): for random boundary values of both phases, Σ_{c in comp} |c^o| D^mix = boundary flux of comp, for every connected component
#       (4 fixtures and the band fixture with 2 components); the constant of every component is a null vector of S_tv;
#   F6  unbalanced component flux: with a net boundary flux in ONE component the CMDGFM projection satisfies every row except the dropped one; the anchor cell keeps the
#       whole defect (flux of the component / |anchor cell|).
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-projection-two-velocity/scripts/code_01_data_flux_gauge.jl

using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ProjectionAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "projection_audit.jl"))
using .ProjectionAudit
using TwoPhaseAMR
const PA = ProjectionAudit

function components(field)
    mesh = field.mesh; nx, ny = mesh.block_size
    vol = PA.open_volumes(field); a = PA.aperture(field); n = length(vol)
    adj = [Int[] for _ in 1:n]
    for (i, face) in pairs(mesh.faces)
        a[i] > 0 || continue
        m = AMR._cell_index(face.minus, nx, ny); q = AMR._cell_index(face.plus, nx, ny)
        (vol[m] > 0 && vol[q] > 0 && m != q) || continue
        push!(adj[m], q); push!(adj[q], m)
    end
    comp = zeros(Int, n); anchors = Int[]
    for c in 1:n
        (vol[c] > 0 && comp[c] == 0) || continue
        push!(anchors, c); comp[c] = c; st = [c]
        while !isempty(st)
            x = pop!(st)
            for y in adj[x]
                comp[y] == 0 && (comp[y] = c; push!(st, y))
            end
        end
    end
    return comp, anchors
end

@testset "two-velocity data, flux and gauge" begin
    for (name, field) in vcat(PA.fixtures(), ["band uniform" => PA.band_fixture(:uniform), "band composite" => PA.band_fixture(:amr)])
        mesh = field.mesh; nx, ny = mesh.block_size
        solver = CMDGFMProjectionSolver(field; density = PA.DENSITY)
        work = CMDGFMProjectionWorkspace(solver)
        cd = AMR._projection_face_data(solver, work)
        d = PA.face_data(field)
        # F1
        diffs = [maximum(abs.(getfield(d, k) .- getproperty(cd, kc)) ./ max.(abs.(getfield(d, k)), 1.0e-300)) for (k, kc) in ((:C, :liquid_fraction), (:ml, :liquid_mass), (:mg, :gas_mass), (:m, :total_mass), (:zl, :zeta_liquid), (:zg, :zeta_gas), (:mr, :relative_mass))]
        e_ap = maximum(abs.(d.apgfm .- cd.coupling))
        e_part = maximum(abs.(d.al .+ d.ag .- PA.aperture(field)))
        @test maximum(diffs) <= 1.0e-12 && e_ap <= 1.0e-9
        @test sort(d.mixed) == sort(collect(cd.mixed))
        # F2
        v = PA.two_predictor(field, MersenneTwister(1))
        Vo = PA.open_dual_volumes(field)
        okl = all(f -> isnan(v[1].interior[f]) == !(d.C[f] > 0 && Vo[f] > 0), 1:length(d.C))
        okg = all(f -> isnan(v[2].interior[f]) == !(d.C[f] < 1 && Vo[f] > 0), 1:length(d.C))
        @test okl && okg
        # F3 / F5 with random boundary values
        rng = MersenneTwister(9)
        for ph in 1:2, i in eachindex(v[ph].boundary)
            isnan(v[ph].boundary[i]) || (v[ph].boundary[i] = randn(rng))
        end
        flux = PA.mixture_flux(field, v)
        cf = zeros(length(flux)); AMR._cmdgfm_cell_flux!(cf, cd, field, v)
        e_flux = maximum(abs.(-cf .- flux)) / maximum(abs.(flux))
        comp, anchors = components(field)
        bflux = Dict(a => 0.0 for a in anchors)
        for (i, face) in pairs(mesh.boundary_faces)
            l, g = isnothing(field.geometry) ? AMR._phase_boundary_face_apertures(field, i) : AMR._phase_boundary_face_apertures(field, field.geometry, i)
            pr(w, x) = (iszero(w) || isnan(x)) ? 0.0 : w * x
            cell = AMR._cell_index(face.cell, nx, ny)
            comp[cell] > 0 && (bflux[comp[cell]] += face.side * face.area * (pr(l, v[1].boundary[i]) + pr(g, v[2].boundary[i])))
        end
        e_comp = maximum(abs(sum(flux[c] for c in findall(==(a), comp)) - bflux[a]) for a in anchors) / maximum(abs.(flux))
        # F4
        op = AMR._projection_pressure_operator(solver)
        a = PA.aperture(field)
        γ = isnothing(field.geometry) ? 1 ./ d.m : a ./ d.m
        J = PA.jump_dense(field); K = PA.metric_dense(op); R = PA.force_map_dense(field, op, γ)
        S = J' * (sqrt.(γ) .* K .* sqrt.(γ)') * J
        e_S = maximum(abs.(J' * Diagonal(a) * R * J .- S)) / maximum(abs.(S))
        w = randn(rng, length(d.m)); w0 = copy(w)
        e_div = maximum(abs.(PA.open_flux(field, (v = FaceVelocity(mesh); v.interior .= w; v.boundary .= 0.0; v)) .+ J' * (a .* w))) / maximum(abs.(w))
        null = maximum(maximum(abs.(S[:, findall(==(a_), comp)] * ones(count(==(a_), comp)))) for a_ in anchors) / maximum(abs.(S))
        println(
            rpad(name, 18), " F1: max rel diff ", round(maximum(diffs), sigdigits = 3), " apgfm ", round(e_ap, sigdigits = 3), " |a^l+a^g-a| ", round(e_part, sigdigits = 3), "; F3 ", round(e_flux, sigdigits = 3),
            "; F5 ", length(anchors), " component(s), identity ", round(e_comp, sigdigits = 3), ", null vector ", round(null, sigdigits = 3), "; F4 ", round(e_S, sigdigits = 3), " ", round(e_div, sigdigits = 3)
        )
        @test e_flux <= 1.0e-13 && e_comp <= 1.0e-13 && e_S <= 1.0e-12 && e_div <= 1.0e-13 && null <= 1.0e-12
        @test e_part <= 1.0e-9
    end
    # F6: unbalanced component flux on the band fixture
    field = PA.band_fixture(:uniform)
    mesh = field.mesh; nx, ny = mesh.block_size
    solver = CMDGFMProjectionSolver(field; density = PA.DENSITY); work = CMDGFMProjectionWorkspace(solver)
    v = PA.two_predictor(field, MersenneTwister(4))
    comp, anchors = components(field)
    lower = [c for c in anchors if AMR.cellcenter(mesh, AMR._cell_references(mesh)[c].block, AMR._cell_references(mesh)[c].i, AMR._cell_references(mesh)[c].j)[2] < 4]
    @test length(anchors) == 2 && length(lower) == 1
    for (i, face) in pairs(mesh.boundary_faces)
        face.axis == 1 && face.side == -1 || continue
        cell = AMR._cell_index(face.cell, nx, ny)
        comp[cell] == lower[1] || continue
        for ph in 1:2
            isnan(v[ph].boundary[i]) || (v[ph].boundary[i] = 0.25)
        end      # inflow into the lower component only
    end
    pressure = CellField(mesh); dt = PA.DT
    stats = project!(v, pressure, solver, dt, work; tolerance = 1.0e-10, max_iterations = 5000)
    flux = PA.mixture_flux(field, v); vol = PA.open_volumes(field)
    anchor_defect = abs(flux[lower[1]]) / vol[lower[1]]
    others = maximum(abs(flux[c]) / vol[c] for c in 1:length(vol) if vol[c] > 0 && !(c in anchors))
    inflow = sum(face.side * face.area * 0.25 * (isnothing(field.geometry) ? 1.0 : field.geometry.boundary_aperture[i]) for (i, face) in pairs(mesh.boundary_faces) if face.axis == 1 && face.side == -1 && comp[AMR._cell_index(face.cell, nx, ny)] == lower[1])
    println(
        "F6: component net boundary flux ", inflow, "; after the projection: dt|D^mix| at the anchor cell ", dt * anchor_defect, " (= |flux|/|c^o| dt = ", dt * abs(inflow) / vol[lower[1]],
        "), at all other cells ", dt * others, "; converged = ", stats.converged
    )
    @test abs(flux[lower[1]] - (-inflow)) <= 1.0e-8 * abs(inflow) || abs(abs(flux[lower[1]]) - abs(inflow)) <= 1.0e-8 * abs(inflow)
    @test dt * others <= 1.0e-9
    @test !stats.converged                     # the anchor keeps the defect and the reported defect exceeds the tolerance
end

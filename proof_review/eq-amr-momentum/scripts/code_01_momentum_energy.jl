# Code check 01: the staggered integral and kinetic energy under a regrid (eq:amr:momentum and the quoted 0.8-1.2 % / 0.86 % changes).
#
# Labels: eq:amr:momentum, tab:amr:conservation (momentum, kinetic energy rows)
# Source: docs/techref/chapters/12_amr_criteria_transfer.tex:600-615 (eq:amr:momentum), 650-700 (check, table); src/mesh/adaptivity/field_transfer.jl:445-466, 495-532
# Checks:
#   M1  eq:amr:momentum: on a periodic axis the constant correction restores sum dual_f u_f to round-off AND leaves every cell divergence
#       unchanged (random-noise velocity, doubly periodic and x-periodic channel meshes, random regrids);
#   M2  the quoted per-regrid changes (staggered integral M_k = sum_{f interior} dual_f u_f relative to sum dual_f |u_f|, kinetic energy
#       E = sum dual_f u_f^2/2 relative to E; interior faces only) on the walled 2 x 1 fixture are reproduced for the chapter's two fields
#       (neither has zero normal velocity at the walls);
#   M3  attribution: (a) closing the integrals with the wall faces and their half dual volumes, (b) fields with zero normal velocity at the
#       walls (stream function sin^2(pi x/2) sin^2(pi y), the flow fixture of the chapter) -- the changes drop by an order of magnitude;
#   M4  order in h: base meshes (32,16), (64,32), (128,64) with the same regrid sequence: interior-only integral with wall-crossing velocity is O(h),
#       the wall-compatible / closed integral and the periodic energy are O(h^2).
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-amr-momentum/scripts/code_01_momentum_energy.jl

using Test, Random, LinearAlgebra
include(joinpath(@__DIR__, "..", "..", "_shared", "amr_audit.jl"))
using .AmrAudit
using TwoPhaseAMR
const AA = AmrAudit

# ---------------------------------------------------------------- M1
@testset "M1 periodic integral correction" begin
    for (name, mk, axes_periodic) in (("periodic", AA.periodic_mesh, (1, 2)), ("channel", AA.channel_mesh, (1,)))
        rng = MersenneTwister(5)
        mesh = mk()
        u = FaceVelocity(mesh, Float64)
        u.interior .= randn(rng, length(u.interior)); u.boundary .= randn(rng, length(u.boundary))
        workspace = AMRWorkspace(mesh, u)          # default registration: conserve_integral = true
        worst_M = worst_div = 0.0
        for step in 1:6
            old = AA.staggered_integrals(mesh, u)
            old_div = AA.divergence_by_key(mesh, u)
            old_scale = maximum(maximum(abs, d) for d in values(old_div))
            labels = AA.random_labels(rng, mesh; pr = step <= 2 ? 0.6 : 0.3, pc = step <= 2 ? 0.1 : 0.6)
            adapt!(workspace, labels; step, force = true).changed || continue
            new = AA.staggered_integrals(mesh, u)
            for k in axes_periodic
                worst_M = max(worst_M, abs(new.M[k] - old.M[k]) / old.A[k])
            end
            ie = AA.identity_errors(mesh, old_div, AA.divergence_by_key(mesh, u))
            worst_div = max(worst_div, max(ie.unchanged, ie.refined, ie.coarsened) / old_scale)
        end
        println("M1 ", name, ": |ΔM_k|/Σ|ω u| on periodic axes = ", worst_M, ";  divergence identities (rel. max|D|) = ", worst_div)
        @test worst_M <= 1.0e-13 && worst_div <= 1.0e-12
    end
end

# ---------------------------------------------------------------- M2-M4
fields = Dict(
    :general => (x, y, t, axis) -> axis == 1 ? sinpi(x) * cospi(y) : 0.5x * y,                               # chapter fixture, normal velocity 0.5 x at y = 1
    :solenoidal => nothing,                                                                                     # stream function below
    :compatible => (x, y, t, axis) -> axis == 1 ? pi * sinpi(x / 2)^2 * sinpi(2y) : -pi / 2 * sinpi(x) * sinpi(y)^2,   # u.n = 0 on all four walls
)

function streamfunction!(velocity, psi)
    mesh = velocity.mesh
    for (faces, values) in ((mesh.faces, velocity.interior), (mesh.boundary_faces, velocity.boundary))
        for (index, face) in pairs(faces)
            x, y = face.center
            a = face.area
            values[index] = face.axis == 1 ? (psi(x, y + a / 2) - psi(x, y - a / 2)) / a : -(psi(x + a / 2, y) - psi(x - a / 2, y)) / a
        end
    end
    return velocity
end

function make_velocity(name, mesh)
    name === :solenoidal && return streamfunction!(FaceVelocity(mesh, Float64), (x, y) -> 0.3y + 0.1sinpi(2x) * sinpi(y) + 0.05x)
    return FaceVelocity(mesh, fields[name])
end

# labels that follow a band around the drop of the chapter's fixture: refine blocks within 0.35 of (0.7, 0.42), then coarsen all
function band_labels(mesh, step)
    step > 2 && return fill(:coarsen, nblocks(mesh))
    return [
        (
            let c = block.origin .+ block.extent ./ 2
                hypot(c[1] - 0.7, c[2] - 0.42) < 0.35 ? :refine : :keep
            end
        ) for block in mesh.blocks
    ]
end

function changes(base, name; with_boundary)
    # blocks of the same physical size (0.5 x 0.5) at every resolution, so that the same region is refined
    mesh = AA.mesh_with(; base, extent = (2.0, 1.0), periodic = (false, false), block = (base[1] ÷ 4, base[1] ÷ 4))
    u = make_velocity(name, mesh)
    workspace = AMRWorkspace(mesh, u)
    worstM = 0.0; worstE = 0.0
    byop = Dict(:refine => (0.0, 0.0), :coarsen => (0.0, 0.0))
    for step in 1:4
        I0 = AA.staggered_integrals(mesh, u; with_boundary)
        adapt!(workspace, band_labels(mesh, step); step, force = true).changed || continue
        I1 = AA.staggered_integrals(mesh, u; with_boundary)
        dM = maximum(abs(I1.M[k] - I0.M[k]) / I0.A[k] for k in 1:2)
        dE = abs(I1.E - I0.E) / I0.E
        key = step <= 2 ? :refine : :coarsen
        byop[key] = (max(byop[key][1], dM), max(byop[key][2], dE))
        worstM = max(worstM, dM); worstE = max(worstE, dE)
    end
    return (; worstM, worstE, byop)
end

@testset "M2-M4 staggered integral and kinetic energy under a regrid (walled 2x1)" begin
    results = Dict()
    for name in (:general, :solenoidal, :compatible), wb in (false, true)
        for base in ((32, 16), (64, 32), (128, 64))
            results[(name, wb, base)] = changes(base, name; with_boundary = wb)
        end
    end
    println("columns: base mesh (h0 = 2/nx);  max over 4 regrids of  max_k |ΔM_k|/Σω|u|   and  |ΔE|/E ;  refine-only / coarsen-only in brackets")
    for name in (:general, :solenoidal, :compatible), wb in (false, true)
        println("  field=", name, wb ? "  integral CLOSED with wall faces" : "  interior faces only (chapter definition)")
        prev = nothing
        for base in ((32, 16), (64, 32), (128, 64))
            r = results[(name, wb, base)]
            ord = isnothing(prev) ? "" : "  order M: " * string(round(log2(prev.worstM / r.worstM); digits = 2)) * "  order E: " * string(round(log2(prev.worstE / r.worstE); digits = 2))
            println(
                "    ", base, "  ΔM = ", round(r.worstM; sigdigits = 3), "  ΔE = ", round(r.worstE; sigdigits = 3),
                "   [refine ΔM ", round(r.byop[:refine][1]; sigdigits = 2), " ΔE ", round(r.byop[:refine][2]; sigdigits = 2),
                " | coarsen ΔM ", round(r.byop[:coarsen][1]; sigdigits = 2), " ΔE ", round(r.byop[:coarsen][2]; sigdigits = 2), "]", ord
            )
            prev = r
        end
    end
    # chapter numbers (mesh (32,16): the chapter's wall momentum 0.83-1.2 %, energy 0.86 % are the maxima over its two fixtures)
    g = results[(:general, false, (32, 16))]
    println("chapter-like fixture, interior-only: ΔM = ", g.worstM, ", ΔE = ", g.worstE)
    # M3: closing the integral, or a wall-compatible field, reduces the wall-axis change by at least a factor 10 at the chapter's resolution
    @test results[(:general, false, (32, 16))].worstM > 3.0e-3
    @test results[(:general, true, (32, 16))].worstM < results[(:general, false, (32, 16))].worstM / 10
    @test results[(:compatible, false, (32, 16))].worstM < results[(:general, false, (32, 16))].worstM / 10
    # M4: orders
    order(name, wb, sel) = log2(sel(results[(name, wb, (32, 16))]) / sel(results[(name, wb, (128, 64))])) / 2
    @test 0.7 < order(:general, false, r -> r.worstM) < 1.5          # first order: the missing wall half dual volumes follow the wall-cell size
    @test order(:compatible, false, r -> r.worstM) > 1.5
    @test order(:general, true, r -> r.worstM) > 1.5
end

# ---------------------------------------------------------------- M5: signed energy change and order on the periodic square
function periodic_energy(base)
    mesh = AA.mesh_with(; base, extent = (1.0, 1.0), periodic = (true, true), block = (base[1] ÷ 4, base[1] ÷ 4))
    # smooth periodic solenoidal velocity from a purely periodic stream function
    u = streamfunction!(FaceVelocity(mesh, Float64), (x, y) -> 0.1sinpi(2x) * sinpi(2y) + 0.05cospi(2x) * sinpi(4y))
    workspace = AMRWorkspace(mesh, u)
    labels(step) = step > 2 ? fill(:coarsen, nblocks(mesh)) : [
            (
                let c = block.origin .+ block.extent ./ 2
                    hypot(c[1] - 0.45, c[2] - 0.5) < 0.3 ? :refine : :keep
            end
            ) for block in mesh.blocks
        ]
    out = Float64[]
    for step in 1:4
        E0 = AA.staggered_integrals(mesh, u).E
        adapt!(workspace, labels(step); step, force = true).changed || continue
        push!(out, (AA.staggered_integrals(mesh, u).E - E0) / E0)
    end
    return out
end

@testset "M5 periodic kinetic energy: sign and order" begin
    r = [periodic_energy(b) for b in ((32, 32), (64, 64), (128, 128))]
    for (b, v) in zip(((32, 32), (64, 64), (128, 128)), r)
        println("M5 periodic ", b, ": signed ΔE/E of the two refining and the two coarsening regrids = ", v)
    end
    # steps 3, 4 coarsen back: the coarsening that follows restores the energy (refine and coarsen changes cancel)
    totals = [prod(1 .+ v) - 1 for v in r]
    println("M5 energy change of the full refine-refine-coarsen-coarsen cycle: ", totals, " (relative to the refine changes: ", [abs(t) / maximum(abs, v[1:2]) for (t, v) in zip(totals, r)], ")")
    for (t, v) in zip(totals, r)
        @test abs(t) < 0.1 * maximum(abs, v[1:2])
    end
    e = [maximum(abs, v[1:2]) for v in r]
    ord = log2(e[1] / e[3]) / 2
    println("M5 observed order of |ΔE|/E: ", ord)
    @test 1.7 < ord < 2.5
end

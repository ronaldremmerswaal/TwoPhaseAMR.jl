# Sharpness experiments 01 for the uniform-mesh Courant bound of the two-phase staggered transport.
#
# Label:  sec:momentum:cfl  ("On a uniform mesh without solid this bounds the dual Courant numbers ... so nu_g <= nu^dagger <= 1/2")
# Source: docs/techref/chapters/06_momentum_transport.tex:535-575; docs/techref/chapters/05_vof_transport.tex (eq:vof:cfl, eq:vof:dtwy)
# Checks:
#   B2  (hard) a uniform translation attains nu^dagger exactly (the bound is sharp), a diagonal translation gives nu^dagger / 2;
#   A1  (hard) EXACT supremum of the dual Courant number over ALL carrier volumes admitted by the WY bound, by linear programs over
#       the sign patterns of the involved fluxes, for every distinct local configuration of dual faces of random 2:1-balanced meshes
#       WITHOUT solid (periodic, walled, mixed; one and two refinement levels): it equals nu^dagger on every mesh, never more.  The restriction
#       "uniform mesh" of the text is therefore not needed (numerical, per configuration type found);
#   A2  (excluded) the same LP with a solid: the supremum exceeds nu^dagger (reported): small open volumes, as the text says for the
#       single-phase bound; the sentence must keep "without solid".
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/sec-momentum-cfl/scripts/sharp_01_courant_extremizer.jl

using Test, Random, LinearAlgebra
isdefined(Main, :MomentumAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "momentum_audit.jl"))
using .MomentumAudit
using TwoPhaseAMR
const MA = MomentumAudit
const D = MA.Docs
const AMR = TwoPhaseAMR

"""Largest dual Courant number |Upsilon_g| / min(V) of the sweep volumes `flux` (indexed by global face) over all dual faces."""
function max_dual_courant(mesh, geometry, flux)
    worst = 0.0
    for k in (1, 2)
        s = D.phase_schedule(mesh, geometry, k)
        for kappa in (1, 2), ((i, j, axis), r) in D.phase_dual_volumes(s, flux, kappa)
            worst = max(worst, abs(r) / min(s.V[i], s.V[j]))
        end
    end
    return worst
end

function sweep_volumes(field, velocity, dt, work)
    reconstruct!(field)
    for axis in (1, 2)
        AMR._compute_directional_wy_fluxes!(field, velocity, dt, work, axis)
    end
    return copy(work.open_flux), copy(work.liquid_flux), copy(work.gas_flux)
end

@testset "sec:momentum:cfl: translations attain the bound (B2)" begin
    mesh = D.vof_uniform_mesh()
    field = InterfaceField(mesh, (x, y) -> -1.0; lipschitz = 1)
    work = WYAdvectionWorkspace(field)
    for nu in (0.5, 0.45)
        u = FaceVelocity(mesh, (x, y, t, a) -> a == 1 ? 0.7 : 0.0)
        dt = stable_timestep(u, work; cfl = nu)
        carrier, _, _ = sweep_volumes(field, u, dt, work)
        @test max_dual_courant(mesh, nothing, carrier) / nu ≈ 1 atol = 1.0e-13
        d = FaceVelocity(mesh, (x, y, t, a) -> a == 1 ? 0.7 : -0.7)
        dt = stable_timestep(d, work; cfl = nu)
        carrier, _, _ = sweep_volumes(field, d, dt, work)
        @test max_dual_courant(mesh, nothing, carrier) / nu ≈ 0.5 atol = 1.0e-13
    end
end

@testset "sec:momentum:cfl: exact worst case by linear programs (A1, A2)" begin
    @test MA.lp_max([1.0, 1.0], [1.0 0.0; 0.0 1.0; 1.0 1.0], [1.0, 2.0, 2.5]) ≈ 2.5       # solver self-test
    @test MA.lp_max([3.0, 2.0], [1.0 1.0; 1.0 3.0], [4.0, 6.0]) ≈ 12.0
    rng = MersenneTwister(7)
    table = Dict{Any, Float64}()
    history = Int[]
    worst = 0.0
    n_dual = 0
    for (periodic, levels) in (
                ((true, true), 0), ((true, true), 1), ((true, true), 1), ((false, false), 1), ((true, false), 1), ((true, true), 2),
                ((false, false), 2), ((false, true), 1), ((true, true), 2), ((false, false), 2), ((true, true), 2), ((false, false), 2),
                ((true, false), 2), ((false, true), 2),
            ), rep in 1:2
        mesh = levels == 0 ? D.vof_uniform_mesh() : MA.random_mesh(rng; periodic, levels, refine_probability = 0.5)
        for k in (1, 2)
            sched = D.phase_schedule(mesh, nothing, k)
            n_dual += length(sched.weights)
            worst = max(worst, MA.worst_courant_by_signature!(table, mesh, sched; nu = 0.5))
        end
        push!(history, length(table))
    end
    println("A1: ", n_dual, " dual faces, ", length(table), " distinct local configurations; signatures after each mesh: ", history)
    println("A1: exact worst-case ratio  nu_g / nu^dagger  over all of them: ", worst, "  (minimum over configurations ", minimum(values(table)), ")")
    @test worst <= 1 + 1.0e-9
    @test worst >= 1 - 1.0e-9                    # attained (translation)
    # nu^dagger = 0.45 gives the same ratio (the problem is homogeneous in nu)
    mesh = MA.random_mesh(rng; periodic = (true, true), levels = 2, refine_probability = 0.5)
    t45 = Dict{Any, Float64}()
    w45 = maximum(MA.worst_courant_by_signature!(t45, mesh, D.phase_schedule(mesh, nothing, k); nu = 0.45) for k in (1, 2))
    @test w45 <= 1 + 1.0e-9
    # A2 with a solid (excluded: pre-existing behaviour of the open volumes; not part of the claim)
    ratios = Float64[]
    for kind in (:disc, :wall), rep in 1:3
        mesh = MA.random_mesh(rng; periodic = kind == :disc ? (true, true) : (false, false), levels = 1, refine_probability = 0.5)
        geometry = GeometryField(mesh, MA.random_solid(rng, mesh.extent; kind); lipschitz = 2)
        aper = Float64.(geometry.aperture)
        open_volume = c -> TwoPhaseAMR._get(geometry.open_fraction, c) * mesh.blocks[c.block].volume
        t = Dict{Any, Float64}()
        for k in (1, 2)
            sched = D.phase_schedule(mesh, geometry, k)
            skip = key -> any(h -> aper[h] == 0, keys(sched.weights[key]))
            push!(ratios, MA.worst_courant_by_signature!(t, mesh, sched; aper, open_volume, nu = 0.5, skip))
        end
    end
    MA.excluded(
        "A2 with a solid: exact worst-case nu_g / nu^dagger per mesh", round.(ratios; digits = 3),
        note = "exceeds 1 in cut cells (small open volumes); the claim is for meshes without solid"
    )
    @test maximum(ratios) > 1.01
end

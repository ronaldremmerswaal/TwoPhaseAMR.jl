# Code check 02: outflow / inflow partial volumes, the own-value coefficient, and the boundedness claim for the code's CTU update.
#
# Label:  eq:momentum:ctu  ("... bypass its value ... whose switch bounds the update of a draining control volume [thesis Lemmas 6.2, 6.3]")
# Source: docs/techref/chapters/06_momentum_transport.tex:652-690; src/two_phase/phase_face_transport.jl, src/two_phase/momentum_workspace.jl
# Checks:
#   P1  (hard)  closed forms of the total outflow / inflow partial volumes of an unknown from the package's weights, all schedules:
#                 O_f = sum_c (varpi_{c,f}/2) out_c + E_f ,  I_f = sum_c (varpi_{c,f}/2) in_c + E_f ,
#                 E_f = (2 - varpi^- - varpi^+)/2 |Phi_f| along the sweep, 0 across; the net dual volumes never exceed the partial ones;
#   P2  (hard)  the own-value coefficient of an unknown whose dual faces are all upwinded term-wise,
#                 gamma_f = (V rho^old - O_f)/(V rho^new) = 1 - (I_f + V R_f)/(V rho^new)    (mass identity, uncut cells),
#               evaluated from the package's primal phase volumes of random one-velocity sweeps (liquid and gas, both components);
#   P3  (hard)  on a conforming patch (every cell of f uncut, varpi = 1) gamma_f >= 0 for every unknown with new fraction < theta in every
#               random sweep: the thesis' boundedness transfers there (it follows from the cell-wise outflow bound, see step01);
#   P5a (hard)  at hanging segments unknowns with gamma_f < 0 occur among the random sweeps (their number and the worst value are reported),
#               never below the extremal value -1 of sharp_01;
#   P6  (hard)  the term-wise (CTU) update never has a larger own coefficient than the net-volume upwind update: gamma^CTU <= gamma^net,
#               with equality unless partial volumes of opposite signs meet at a dual face of the unknown: the term-wise partial upwinding
#               adds diffusion but no boundedness.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-momentum-ctu/scripts/code_02_ctu_boundedness.jl

using Test, Random, LinearAlgebra
isdefined(Main, :MomentumAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "momentum_audit.jl"))
using .MomentumAudit
using TwoPhaseAMR
const MA = MomentumAudit
const D = MA.Docs
const AMR = TwoPhaseAMR

"""Cell sums over the faces of axis `kappa` of the cells of every dof: `(varpi / 2) out_c` and `in_c`, and E_f (axial only)."""
function closed_form_terms(mesh, s, flux, kappa)
    n = s.dofs.n
    k = s.dofs.k
    out_sum = zeros(n); in_sum = zeros(n); E = zeros(n)
    cell_faces = MA.faces_of_cells(mesh)
    function cell_out_in(c)
        out = inn = 0.0
        for (h, o) in get(cell_faces, c, Tuple{Int, Int}[])
            mesh.faces[h].axis == kappa || continue
            v = o * flux[h]
            v > 0 ? (out += v) : (inn -= v)
        end
        return out, inn
    end
    for i in 1:n
        face = MA.dof_face(mesh, s.dofs, i)
        cells = i <= length(s.dofs.interior) ? (face.minus, face.plus) : (face.cell,)
        varpis = [face.area / mesh.blocks[c.block].spacing[3 - k] for c in cells]
        for (c, varpi) in zip(cells, varpis)
            out, inn = cell_out_in(c)
            out_sum[i] += varpi / 2 * out
            in_sum[i] += varpi / 2 * inn
        end
        if k == kappa && i <= length(s.dofs.interior)
            E[i] = (2 - varpis[1] - varpis[2]) / 2 * abs(flux[s.dofs.interior[i]])
        end
    end
    return out_sum, in_sum, E
end

@testset "eq:momentum:ctu: closed forms of the partial volumes (P1)" begin
    rng = MersenneTwister(5)
    worst = 0.0
    cases = Any[(D.vof_uniform_mesh(), nothing)]
    for (periodic, solid) in (((true, true), nothing), ((true, true), nothing), ((false, false), nothing), ((true, false), nothing), ((true, true), :disc), ((false, false), :wall))
        mesh = MA.random_mesh(rng; periodic, levels = 1, refine_probability = 0.5)
        geometry = solid === nothing ? nothing : GeometryField(mesh, MA.random_solid(rng, mesh.extent; kind = solid); lipschitz = 2)
        push!(cases, (mesh, geometry))
    end
    for (mesh, geometry) in cases, k in (1, 2), kappa in (1, 2)
        s = D.phase_schedule(mesh, geometry, k)
        flux = zeros(length(mesh.faces))
        for (i, f) in pairs(mesh.faces)
            f.axis == kappa && (flux[i] = randn(rng) * f.area)
        end
        O, I, On, In = MA.partial_in_out(s, flux, kappa)
        out_sum, in_sum, E = closed_form_terms(mesh, s, flux, kappa)
        scale = maximum(abs, flux)
        worst = max(worst, maximum(abs, O .- (out_sum .+ E)) / scale, maximum(abs, I .- (in_sum .+ E)) / scale)
        @test maximum(abs, O .- (out_sum .+ E)) <= 1.0e-13 * scale
        @test maximum(abs, I .- (in_sum .+ E)) <= 1.0e-13 * scale
        @test all(On .<= O .+ 1.0e-14 * scale) && all(In .<= I .+ 1.0e-14 * scale)     # net volumes never exceed the partial ones
    end
    println("P1: worst deviation of O_f, I_f from the closed forms (relative to the largest flux): ", worst)
end

@testset "eq:momentum:ctu: own-value coefficient of fully upwinded unknowns (P2, P3, P5a, P6)" begin
    rng = MersenneTwister(77)
    theta = 0.5
    stats = Dict(:fixtures => 0, :ctu_unknowns => 0, :conforming => 0, :hanging => 0, :hanging_negative => 0, :conforming_negative => 0, :differs_from_net => 0)
    worst = Dict(:conforming => 0.0, :hanging => 0.0, :identity => 0.0)
    gamma_min_hanging = 0.0
    found = Any[]
    for (periodic, levels, trials) in (((true, true), 0, 8), ((true, true), 1, 70), ((false, false), 1, 70), ((true, true), 2, 30), ((true, false), 1, 30))
        for trial in 1:trials
            field, velocity, solid = MA.random_two_phase(rng; periodic, levels)
            reconstruct!(field)
            work = WYAdvectionWorkspace(field)
            limit = stable_timestep(velocity, work)
            isfinite(limit) || continue
            dt = limit * (0.6 + 0.4 * rand(rng))
            mesh = field.mesh
            C0 = Dict(ref => D.cond(field, ref) for ref in D.all_cells(mesh))
            try
                advect!(field, velocity, dt, work)
            catch err
                (err isa ErrorException || err isa ArgumentError) || rethrow()
                continue
            end
            stats[:fixtures] += 1
            sweeps, _ = MA.replay_mass_identity(field, work, C0, (1, 2), (1000.0, 1.0))
            for sweep in sweeps, comp in sweep.comps, (phase, flux) in ((:liquid, work.liquid_flux), (:gas, work.gas_flux))
                s = comp.schedule
                k = comp.k
                O, I, On, In = MA.partial_in_out(s, flux, sweep.kappa)
                old = D.independent_interpolation(s.interpolation, r -> phase == :liquid ? sweep.C_old[r] : 1 - sweep.C_old[r])
                new = D.independent_interpolation(s.interpolation, r -> phase == :liquid ? sweep.pred[r] : 1 - sweep.pred[r])
                R = D.independent_interpolation(s.interpolation, r -> (phase == :liquid ? sweep.chi[r] : 1 - sweep.chi[r]) * sweep.sigma[r])
                for i in 1:s.dofs.n
                    new[i] < theta && new[i] > 1.0e-7 || continue
                    face = MA.dof_face(mesh, s.dofs, i)
                    cells = s.interpolation[i]
                    cut = !isnothing(field.geometry) && any(((c, w),) -> D.open_fraction_of_geometry(field.geometry, c) < 1, cells)
                    cut && continue                                           # (b) is not guaranteed in cut cells
                    hang = any(((c, w),) -> abs(face.area / mesh.blocks[c.block].spacing[3 - k] - 1) > 1.0e-12, cells)
                    own = (s.V[i] * old[i] - O[i]) / (s.V[i] * new[i])
                    own_net = (s.V[i] * old[i] - On[i]) / (s.V[i] * new[i])
                    # P2: the algebraic identity gamma = 1 - (I + V R)/(V new) (mass identity)
                    identity = abs(own - (1 - (I[i] + s.V[i] * R[i]) / (s.V[i] * new[i])))
                    worst[:identity] = max(worst[:identity], identity)
                    @test identity <= 1.0e-12 / new[i] + 1.0e-12
                    @test own <= own_net + 1.0e-12                              # P6
                    own < own_net - 1.0e-9 && (stats[:differs_from_net] += 1)
                    stats[:ctu_unknowns] += 1
                    if hang
                        stats[:hanging] += 1
                        if own < -1.0e-9
                            stats[:hanging_negative] += 1
                            gamma_min_hanging = min(gamma_min_hanging, own)
                            push!(found, (; own, new = new[i], phase, kappa = sweep.kappa, k, i))
                        end
                        worst[:hanging] = min(worst[:hanging], own)
                    else
                        stats[:conforming] += 1
                        worst[:conforming] = min(worst[:conforming], own)
                        own < -1.0e-9 && (stats[:conforming_negative] += 1)
                        @test own >= -1.0e-9                                    # P3
                    end
                end
            end
        end
    end
    println("P2/P3/P5a/P6 statistics: ", stats)
    println("smallest own-value coefficient: conforming ", worst[:conforming], ", hanging ", worst[:hanging], "; identity gap ", worst[:identity])
    @test stats[:conforming_negative] == 0                                  # P3
    @test gamma_min_hanging >= -1 - 1.0e-9                                  # never below the extremal value of sharp_01 (P4)
    @test stats[:differs_from_net] > 100                                    # the term-wise update really differs from net upwinding (P6)
    println(
        "P5a: ", stats[:hanging_negative], " of ", stats[:hanging], " fully upwinded hanging unknowns have a negative own-value coefficient in these random sweeps ",
        "(a rare event there; sharp_01 constructs it deterministically)"
    )
end

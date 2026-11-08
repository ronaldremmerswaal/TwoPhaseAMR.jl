# Code check 01: the dilation of the frozen carrier vanishes (in the cells with chi = 1) exactly when the discrete mixture
# constraint holds on the frozen apertures.
#
# Label:  none in the source (suggested: prop:vof:frozen-dilation); text of Section 5.8.1 after (5.23) and the first bullet of
#         Section 5.9: "The dilation of a step, chi_c dt sum_f o_{c,f} |f| (a^l_f u^l_f + a^g_f u^g_f), vanishes where the phase
#         velocities satisfy the discrete mixture constraint (thesis (6.19)) on the frozen apertures"
# Source: docs/techref/chapters/05_vof_transport.tex:596-620, 797-806
# Claims: (i)   the dilation of the code's frozen carrier in cell c is chi_c dt |c| (div <a u>)_c with
#               <a u>_f = a^l_f u^l_f + a^g_f u^g_f on the frozen apertures (per cell, both sweeps summed);
#         (ii)  phase velocities with a velocity JUMP (u^l != u^g, even opposite signs) that satisfy the mixture constraint
#               exactly give a vanishing dilation and conserve the liquid volume to round-off (no snaps);
#         (iii) the constraint is needed only in the cells with chi = 1: violations in chi = 0 cells do not change the volume;
#         (iv)  a violation in a chi = 1 cell changes the volume by exactly its dilation (the independent replica predicts it);
#         (v)   the dilation of ONE sweep does not vanish for a mixture-solenoidal carrier, only the sum of the two sweeps.
# Method: mixture-solenoidal fields from a streamfunction (face fluxes = psi differences at the segment end points, exactly
#         solenoidal on every leaf cell of a composite mesh, also at hanging vertices; on cut meshes the open streamfunction of
#         docs/techref/figures/vof.jl) split into phase velocities with random jumps preserving a^l u^l + a^g u^g; frozen
#         apertures from the package (`_freeze_phase_apertures!`); the independent replica of the frozen sweeps
#         (Docs.wy_replica, carrier = frozen) predicts the dilation.
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/vof-frozen-dilation/scripts/code_01_mixture_constraint.jl
#         (from the repository root)

using TwoPhaseAMR
using Test, Random, Printf
const AMR = TwoPhaseAMR
isdefined(Main, :VofAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "vof_audit.jl"))
using .VofAudit
const D = VofAudit.D

"""Random smooth periodic streamfunction on `[0,L]^2` (a few modes) plus a high-frequency mode."""
function random_psi(rng, Lx, Ly)
    coeff = randn(rng, 4); kx = rand(rng, 1:3, 4); ky = rand(rng, 1:3, 4); ph = 2 * rand(rng, 8)
    return (x, y) -> sum(coeff[k] * Lx / (2pi) * sinpi(2kx[k] * x / Lx + ph[k]) * sinpi(2ky[k] * y / Ly + ph[4 + k])
        for k in 1:4)
end

"""Split the face velocity `u` (mixture flux density a^o u) into phase velocities with random jumps."""
function split_phases(rng, field, u; jump = 1.0)
    mesh = field.mesh
    reconstruct!(field)
    work = WYAdvectionWorkspace(field, (u, copy(u)); interface_transport = FrozenInterfaceFluxes())
    AMR._freeze_phase_apertures!(field, work)
    al, ag = copy(work.liquid_aperture), copy(work.gas_aperture)
    ul, ug = copy(u), copy(u)
    jumps = 0
    for index in eachindex(mesh.faces)
        if al[index] >= 0.05 && ag[index] >= 0.05
            dl = jump * (2rand(rng) - 1)
            dg = -al[index] * dl / ag[index]
            ul.interior[index] = u.interior[index] * (1 + dl)
            ug.interior[index] = u.interior[index] * (1 + dg)
            jumps += 1
        end
    end
    return ul, ug, al, ag, jumps
end

"""Mixture divergence |c| (D <a u>)_c = sum_f o |f| (a^l u^l + a^g u^g) per cell (independent of the package operators)."""
function mixture_divergence(field, ul, ug, al, ag)
    mesh = field.mesh
    incidence = D.cell_faces(mesh)
    out = Dict{CellRef, Float64}()
    for ref in D.all_cells(mesh)
        out[ref] = sum((o * mesh.faces[index].area * (al[index] * ul.interior[index] + ag[index] * ug.interior[index])
            for (index, o) in get(incidence, ref, Tuple{Int, Int}[])); init = 0.0)
    end
    return out
end

"""Step with the frozen carrier; halves dt if the subface strip is too small (see the audit of vof-boundedness)."""
function frozen_step(field, ul, ug; order)
    for attempt in 0:5
        trial = copy(field)
        work = WYAdvectionWorkspace(trial, (ul, ug); interface_transport = FrozenInterfaceFluxes())
        dt = stable_timestep((ul, ug), work) / 2^attempt
        apertures = (copy(work.liquid_aperture), copy(work.gas_aperture))
        carrier = D.frozen_carrier((ul, ug), apertures)
        replica = try
            D.wy_replica(trial, carrier, dt, order)
        catch err
            (err isa AssertionError || err isa ErrorException) || rethrow()
            continue
        end
        before = D.liquid_volume(trial)
        try
            advect!(trial, (ul, ug), dt, work; sweep_order = order, check_timestep = false)
        catch err
            err isa ErrorException || rethrow()
            continue
        end
        return (; dt, change = D.liquid_volume(trial) - before, before, replica, attempt, field = trial)
    end
    return nothing
end

@testset "frozen carrier: dilation and the mixture constraint" begin
    rng = MersenneTwister(20261111)
    @testset "(i)/(ii) mixture-solenoidal phase velocities with jumps (composite periodic meshes)" begin
        worst_constraint = worst_dilation = worst_change = worst_cellwise = 0.0
        jumps_total = opposite = cases = 0
        for rep in 1:8, order in ((1, 2), (2, 1))
            field, _ = VofAudit.random_field(rng; cut = false, levels = rep <= 4 ? 1 : 2, aniso = rep % 3 == 0)
            u = D.streamfunction_velocity(field.mesh, random_psi(rng, field.mesh.extent...))
            ul, ug, al, ag, jumps = split_phases(rng, field, u; jump = rep % 2 == 0 ? 2.0 : 0.8)
            jumps_total += jumps
            opposite += count(i -> ul.interior[i] * ug.interior[i] < 0, eachindex(ul.interior))
            m = mixture_divergence(field, ul, ug, al, ag)
            worst_constraint = max(worst_constraint, maximum(abs, values(m)))
            res = frozen_step(field, ul, ug; order)
            res === nothing && continue
            cases += 1
            chi = res.replica.chi
            # per-cell dilation of the replica = chi * dt * m_c  (both sweeps)
            for ref in D.all_cells(field.mesh)
                d = res.replica.states[2].dilation[ref] + res.replica.states[3].dilation[ref]
                worst_cellwise = max(worst_cellwise, abs(d - chi[ref] * res.dt * m[ref]))
            end
            worst_dilation = max(worst_dilation, abs(res.replica.dilation_total))
            worst_change = max(worst_change, abs(res.change) / res.before)
            @test res.replica.snaps == 0
        end
        println("(i)/(ii): ", cases, " steps, ", jumps_total, " faces with a velocity jump (", opposite, " with opposite signs)")
        println("          max |mixture divergence| ", worst_constraint, "; per-cell dilation vs chi dt m_c ", worst_cellwise,
            "; |total dilation| ", worst_dilation, "; relative volume change ", worst_change)
        @test cases >= 12 && opposite > 50
        @test worst_constraint <= 1.0e-13
        @test worst_cellwise <= 1.0e-14
        @test worst_change <= 1.0e-13
    end
    @testset "(i)/(ii) cut meshes (open streamfunction)" begin
        worst_constraint = worst_change = 0.0; cases = 0; split_faces = 0
        for kind in (:cut, :film), order in ((1, 2), (2, 1))
            field, _ = D.vof_fixture(kind)
            u = D.open_streamfunction_velocity(field, D.vof_psi_walls)
            ul, ug, al, ag, jumps = split_phases(rng, field, u; jump = 0.8)
            split_faces += jumps
            m = mixture_divergence(field, ul, ug, al, ag)
            worst_constraint = max(worst_constraint, maximum(abs, values(m)))
            res = frozen_step(field, ul, ug; order)
            res === nothing && continue
            cases += 1
            worst_change = max(worst_change, abs(res.change) / res.before)
            @test res.replica.snaps == 0
        end
        println("(i)/(ii) cut: ", cases, " steps, ", split_faces, " split faces; max |mixture divergence| ", worst_constraint,
            "; relative volume change ", worst_change)
        @test cases >= 3
        @test worst_constraint <= 1.0e-13
        @test worst_change <= 1.0e-13
    end
    @testset "(iii)/(iv)/(v) where the constraint is needed" begin
        field, _ = VofAudit.random_field(rng; cut = false, levels = 1)
        u = D.streamfunction_velocity(field.mesh, random_psi(rng, field.mesh.extent...))
        ul, ug, al, ag, _ = split_phases(rng, field, u; jump = 0.8)
        mesh = field.mesh
        reconstruct!(field)
        chi_cell = Dict(ref => D.cond(field, ref) > 0.5 for ref in D.all_cells(mesh))
        # (iii) violate the constraint on faces between two chi = 0 cells (change the gas velocity there)
        ul3, ug3 = copy(ul), copy(ug)
        touched = 0
        for (index, face) in pairs(mesh.faces)
            if !chi_cell[face.minus] && !chi_cell[face.plus]
                ug3.interior[index] += 0.3 * randn(rng); ul3.interior[index] += 0.3 * randn(rng); touched += 1
            end
        end
        m3 = mixture_divergence(field, ul3, ug3, al, ag)
        violated_gas = count(ref -> !chi_cell[ref] && abs(m3[ref]) > 1.0e-6, keys(m3))
        violated_liquid = count(ref -> chi_cell[ref] && abs(m3[ref]) > 1.0e-12, keys(m3))
        res3 = frozen_step(field, ul3, ug3; order = (1, 2))
        println("(iii) ", touched, " faces perturbed, mixture constraint violated in ", violated_gas,
            " chi=0 cells and in ", violated_liquid, " chi=1 cells; relative volume change ", abs(res3.change) / res3.before)
        @test touched > 50 && violated_gas > 10 && violated_liquid == 0
        @test abs(res3.change) <= 1.0e-13 * res3.before
        # (iv) one face next to a chi = 1 cell
        index = findfirst(i -> chi_cell[mesh.faces[i].minus] != chi_cell[mesh.faces[i].plus] &&
            al[i] > 0.1 && ag[i] > 0.1, eachindex(mesh.faces))
        ul4 = copy(ul); ug4 = copy(ug); ul4.interior[index] += 0.05
        m4 = mixture_divergence(field, ul4, ug4, al, ag)
        res4 = frozen_step(field, ul4, ug4; order = (1, 2))
        face = mesh.faces[index]
        expected = res4.dt * sum((chi_cell[ref] ? o * face.area * al[index] * 0.05 : 0.0
            for (ref, o) in ((face.minus, 1), (face.plus, -1))))
        println("(iv) a violation of ", maximum(abs, values(m4)), " in one face: volume change ", res4.change,
            ", dilation predicted by the formula ", expected, ", by the replica ", res4.replica.dilation_total)
        @test abs(res4.change - res4.replica.dilation_total) <= 1.0e-13 * res4.before
        @test abs(res4.change - expected) <= 1.0e-13 * res4.before
        @test abs(res4.change) > 1.0e-4 * res4.dt
        # (v) per-sweep dilation of the mixture-solenoidal carrier
        res5 = frozen_step(field, ul, ug; order = (1, 2))
        d1 = res5.replica.states[2].dilation; d2 = res5.replica.states[3].dilation
        singles = maximum(abs(d1[ref]) for ref in keys(d1) if res5.replica.chi[ref] == 1)
        sums = maximum(abs(d1[ref] + d2[ref]) for ref in keys(d1) if res5.replica.chi[ref] == 1)
        println("(v) max single-sweep dilation ", singles, "; max |sum of the two sweeps| ", sums)
        @test singles > 1.0e-3 && sums <= 1.0e-14
    end
end

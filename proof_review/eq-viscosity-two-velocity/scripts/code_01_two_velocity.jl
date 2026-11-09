# Code check 01: eq:viscosity:two-velocity -- the mass-weighted mixture, the common increment, the unchanged jump, the recombination
# and the combined momentum of the two-velocity diffusion substep.
#
# Label:  eq:viscosity:two-velocity
# Source: docs/techref/chapters/09_viscosity.tex:424-460; src/two_phase/two_velocity/stepping.jl
#         (_mass_weighted_one_velocity!, _apply_two_velocity_diffusion_update!), src/two_phase/two_velocity/velocity_fields.jl
#         (_mark_inactive_phase_velocities!, _phase_product)
# Checks (walled composite, cut, periodic composite drop; phase velocities random, absent phases NaN exactly where the package
# marks them inactive):
#   T1  mixture  ubar_f = (rho_l Chat_f u^l + rho_g (1-Chat_f) u^g)/rho_f with absent = 0, 0 on faces without open volume,
#   T2  u^{p,2} = u^{p,1} + (ubar2 - ubar1) on active faces, NaN stays NaN; the jump u^g - u^l changes by 0 where both are active,
#   T3  the mixture of the updated phases equals ubar2 on every face (also where only one phase is active, and covered faces),
#   T4  combined momentum change = sum rho_f V_f (ubar2 - ubar1) with the OPEN dual volume (V_f of the chapter must be the open
#       one with a solid), and = 1_k' r on the periodic mesh,
#   T5  continuous velocity u^l = u^g: both updated phases equal the one-velocity solve ubar2 (and are bitwise equal to each other),
#   T6  unclamped vs clamped fraction: size of the difference (clamp active only through round-off).
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-viscosity-two-velocity/scripts/code_01_two_velocity.jl
using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ViscosityAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "viscosity_audit.jl"))
using .ViscosityAudit
using TwoPhaseAMR
const AMR = TwoPhaseAMR

function setup(field; rng)
    mesh = field.mesh
    geometry = field.geometry
    physics = TwoPhase(field; density = FIX_DENSITY, dynamic_viscosity = FIX_VISC)
    model = AMR.TwoPhaseModel(physics)
    nf = length(mesh.faces)
    liquid = FaceVelocity(mesh); gas = FaceVelocity(mesh)
    liquid.interior .= randn(rng, nf); gas.interior .= randn(rng, nf)
    AMR._mark_inactive_phase_velocities!((liquid, gas), field)
    return model, (liquid, gas)
end

@testset "eq:viscosity:two-velocity" begin
    rng = MersenneTwister(4711)
    for (name, fx) in ((:walled, fixture_walled), (:cut, fixture_cut), (:periodic, fixture_periodic))
        field = fx(); mesh = field.mesh; geometry = field.geometry
        model, velocity = setup(field; rng)
        nf = length(mesh.faces)
        op = StaggeredDiffusionOperator(field; viscosity = FIX_VISC)
        AMR._update_dynamic_viscosity!(op, field, FIX_VISC)
        mixture = FaceVelocity(mesh); density = zeros(nf)
        AMR._mass_weighted_one_velocity!(mixture, density, velocity, model)
        # T1 independent mixture
        worst1 = 0.0; unclamped_gap = 0.0
        for (i, face) in pairs(mesh.faces)
            c = AMR._staggered_liquid_fraction(field, geometry, face)
            ml, mg = FIX_DENSITY[1] * c, FIX_DENSITY[2] * (1 - c)
            ul = ml == 0 ? 0.0 : ml * velocity[1].interior[i]
            ug = mg == 0 ? 0.0 : mg * velocity[2].interior[i]
            covered = !isnothing(geometry) && geometry.face_dual_volumes[i] == 0
            expected = covered ? 0.0 : (ul + ug) / (ml + mg)
            worst1 = max(worst1, abs(mixture.interior[i] - expected))
            @test density[i] ≈ ml + mg atol = 1.0e-12
        end
        @test worst1 <= 1.0e-13
        dt = FIX_DT
        solver = AMR.VariableStaggeredDiffusionSolver(op, density, dt)
        diffused = FaceVelocity(mesh)
        implicit_midpoint_diffusion!(diffused, mixture, op, dt, solver; density, tolerance = 1.0e-13, max_cycles = 4000)
        updated = (FaceVelocity(mesh), FaceVelocity(mesh))
        AMR._apply_two_velocity_diffusion_update!(updated, velocity, diffused, mixture)
        # T2 jump, NaN
        jump_change = 0.0; nan_kept = true; nboth = 0
        for i in 1:nf
            a = velocity[1].interior[i]; b = velocity[2].interior[i]
            na, nb = updated[1].interior[i], updated[2].interior[i]
            nan_kept &= isnan(a) == isnan(na) && isnan(b) == isnan(nb)
            if !isnan(a) && !isnan(b)
                nboth += 1
                jump_change = max(jump_change, abs((nb - na) - (b - a)))
            end
        end
        @test nan_kept
        @test jump_change <= 1.0e-14
        # T3 recombination
        recomb = FaceVelocity(mesh); rd = zeros(nf)
        AMR._mass_weighted_one_velocity!(recomb, rd, updated, model)
        openface = [isnothing(geometry) || geometry.face_dual_volumes[i] > 0 for i in 1:nf]
        r3 = maximum(abs(recomb.interior[i] - diffused.interior[i]) for i in 1:nf if openface[i]) / maximum(abs, diffused.interior)
        @test r3 <= 1.0e-13
        # covered faces (no open volume): the phase mixture is 0 by construction, ubar2 is the penalty-tied value (not exactly 0)
        covered_u2 = maximum((abs(diffused.interior[i]) for i in 1:nf if !openface[i]); init = 0.0) / maximum(abs, diffused.interior)
        name == :cut && println("cut: max |ubar2| on covered faces / max|ubar2| = ", covered_u2, " (recombination there fails by exactly this)")
        @test (name == :cut) == (covered_u2 > 1.0e-8)
        # T4 combined momentum
        Vopen = isnothing(geometry) ? mesh.face_dual_volumes : geometry.face_dual_volumes
        Valg = op.volumes
        fr = [AMR._staggered_liquid_fraction(field, geometry, face) for face in mesh.faces]
        combined(u, V) = sum(
            (
                FIX_DENSITY[1] * fr[i] * (isnan(u[1].interior[i]) ? 0.0 : u[1].interior[i]) +
                    FIX_DENSITY[2] * (1 - fr[i]) * (isnan(u[2].interior[i]) ? 0.0 : u[2].interior[i])
            ) * V[i] for i in 1:nf
        )
        for k in 1:2
            sel = [mesh.faces[i].axis == k for i in 1:nf]
            comb(u, V) = sum(
                (
                    FIX_DENSITY[1] * fr[i] * (isnan(u[1].interior[i]) ? 0.0 : u[1].interior[i]) +
                        FIX_DENSITY[2] * (1 - fr[i]) * (isnan(u[2].interior[i]) ? 0.0 : u[2].interior[i])
                ) * V[i] for i in 1:nf if sel[i]
            )
            one_vel = sum(density[i] * Valg[i] * (diffused.interior[i] - mixture.interior[i]) for i in 1:nf if sel[i])
            change_open = comb(updated, Vopen) - comb(velocity, Vopen)
            scale = sum(abs(density[i] * Valg[i] * mixture.interior[i]) for i in 1:nf if sel[i])
            gap_open = abs(change_open - sum(density[i] * Vopen[i] * (diffused.interior[i] - mixture.interior[i]) for i in 1:nf if sel[i])) / scale
            @test gap_open <= 1.0e-12
            if name == :periodic
                M = density .* Valg
                ones_k = [mesh.faces[i].axis == k ? 1.0 : 0.0 for i in 1:nf]
                r = M .* (diffused.interior .- mixture.interior) .+ dt / 2 .* (op.stiffness * (mixture.interior .+ diffused.interior))
                @test abs(change_open - dot(ones_k, r)) / scale <= 1.0e-12
            end
            if name == :cut
                # the uncut volume would be wrong with a solid
                change_uncut = comb(updated, mesh.face_dual_volumes) - comb(velocity, mesh.face_dual_volumes)
                gap_uncut = abs(change_uncut - sum(density[i] * Vopen[i] * (diffused.interior[i] - mixture.interior[i]) for i in 1:nf if sel[i])) / scale
                println("cut: combined-momentum gap with uncut dual volume = ", gap_uncut, " (open volume: ", gap_open, ")")
                @test gap_uncut > 1.0e-9
            end
        end
        # T5 continuous velocity
        cont_l = FaceVelocity(mesh); cont_l.interior .= velocity[1].interior; cont_g = FaceVelocity(mesh)
        for i in 1:nf
            cont_g.interior[i] = isnan(velocity[1].interior[i]) ? velocity[2].interior[i] : velocity[1].interior[i]
        end
        # continuous: both phases equal to the same value on faces where both are active; where one is absent keep the NaN
        cl = FaceVelocity(mesh); cg = FaceVelocity(mesh)
        base = randn(rng, nf)
        cl.interior .= base; cg.interior .= base
        AMR._mark_inactive_phase_velocities!((cl, cg), field)
        mix_c = FaceVelocity(mesh); den_c = zeros(nf)
        AMR._mass_weighted_one_velocity!(mix_c, den_c, (cl, cg), model)
        # mixture of equal velocities = that velocity on open faces
        eq_err = maximum(abs(mix_c.interior[i] - (isnothing(geometry) || geometry.face_dual_volumes[i] > 0 ? base[i] : 0.0)) for i in 1:nf)
        @test eq_err <= 1.0e-12
        dif_c = FaceVelocity(mesh)
        implicit_midpoint_diffusion!(dif_c, mix_c, op, dt, solver; density = den_c, tolerance = 1.0e-13, max_cycles = 4000)
        upd_c = (FaceVelocity(mesh), FaceVelocity(mesh))
        AMR._apply_two_velocity_diffusion_update!(upd_c, (cl, cg), dif_c, mix_c)
        both = [i for i in 1:nf if !isnan(cl.interior[i]) && !isnan(cg.interior[i])]
        T5 = maximum(abs(upd_c[1].interior[i] - upd_c[2].interior[i]) for i in both)
        T5b = maximum(abs(upd_c[1].interior[i] - dif_c.interior[i]) for i in both if isnothing(geometry) || geometry.face_dual_volumes[i] > 0)
        # T6 clamp
        T6 = maximum(abs(AMR._staggered_liquid_fraction(field, geometry, face) - clamp(AMR._staggered_liquid_fraction(field, geometry, face), 0, 1)) for face in mesh.faces)
        println(rpad(name, 9), "mixture ", worst1, "  jump ", jump_change, " (", nboth, " faces with both phases)  recomb ", r3, "  continuous: phases differ ", T5, ", vs one-velocity ", T5b)
        @test T5 == 0
        @test T5b <= 1.0e-13
    end
end

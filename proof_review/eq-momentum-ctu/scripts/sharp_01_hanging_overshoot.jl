# Sharpness experiments 01 for the boundedness of the code's term-wise upwinded (CTU) update at hanging segments.
#
# Label:  eq:momentum:ctu  (the sentence "whose switch bounds the update of a draining control volume [thesis Lemmas 6.2, 6.3]")
# Source: docs/techref/chapters/06_momentum_transport.tex:652-690; src/two_phase/phase_face_transport.jl (_ctu_phase_transport)
# Checks:
#   P4  (hard)  the extremal states at a hanging segment (all phase of the donor cell in its donating rectangle, the other cell empty) give,
#               through the package's weights, the own-value coefficient gamma_f = -(2 - varpi^- - varpi^+)/varpi^+ = -1/2 (coarse -> fine)
#               and -1 (fine -> coarse); 0 on a conforming patch.  These are the extrema of the update formula;
#   P5  (hard)  the package's own staggered-scalar transport reproduces them: on deterministic drop fixtures of the chapter's composite mesh
#               (uniform x-velocity, second sweep inactive, Upwind bulk) a staggered value 0 with neighbours 1 becomes 1.98 and 1.5;
#               the same experiment away from the hanging sides keeps the range [0, 1];
#   P5m (hard)  the same experiment through the one-velocity MOMENTUM update (advect!(...; momentum, density, momentum_source)): the
#               transported component is 1 except 0 at the unknown, carrier uniform and solenoidal, densities (1000, 1): the new value at the
#               unknown is 1.95 and 1.37 (> 1); with densities (1, 1) and (10, 1) the range [0, 1] is kept (reported, not asserted).
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-momentum-ctu/scripts/sharp_01_hanging_overshoot.jl

using Test, Random, LinearAlgebra
isdefined(Main, :MomentumAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "momentum_audit.jl"))
using .MomentumAudit
using TwoPhaseAMR
const MA = MomentumAudit
const D = MA.Docs
const AMR = TwoPhaseAMR

@testset "eq:momentum:ctu: extremal states at a hanging segment through the package weights (P4)" begin
    mesh = D.vof_amr_mesh()
    inc = MA.component_incidence(mesh, 1)
    s = D.phase_schedule(mesh, nothing, 1)
    f1 = D.face_index_at(mesh, (4.0, 5.25))
    l1 = inc.local_of[f1]
    face = mesh.faces[f1]
    c_minus, c_plus = face.minus, face.plus            # coarse cell on the left (minus), fine cell on the right (plus)
    varpi(c) = face.area / mesh.blocks[c.block].spacing[2]
    @test varpi(c_minus) == 1 / 2 && varpi(c_plus) == 1
    vol(c) = mesh.blocks[c.block].volume
    for (direction, donor, receiver, expected) in ((+1, c_minus, c_plus, -1 / 2), (-1, c_plus, c_minus, -1))
        x = 0.01
        flux = zeros(length(mesh.faces))
        flux[f1] = direction * x
        O, I, On, In = MA.partial_in_out(s, flux, 1)
        # all phase of the donor (x) inside its strip, the receiver empty; V rho = (1/2) sum_c varpi |c| theta_c
        thetas = Dict(donor => x / vol(donor), receiver => 0.0)
        Vold = sum(varpi(c) * vol(c) * thetas[c] / 2 for c in (c_minus, c_plus))
        Vnew_ = Vold - O[l1] + I[l1]
        gamma = (Vold - O[l1]) / Vnew_
        @test gamma ≈ expected atol = 1.0e-13
        @test Vnew_ ≈ varpi(receiver) * x / 2 atol = 1.0e-15
        @test gamma ≈ -(2 - varpi(c_minus) - varpi(c_plus)) / varpi(receiver) atol = 1.0e-13
        # the net-volume upwind coefficient is not better: the own flux crosses different dual faces
        gamma_net = (Vold - On[l1]) / Vnew_
        @test gamma_net <= 0
    end
    # conforming patch of the same mesh: gamma = 0
    g = D.face_index_at(mesh, (6.0, 5.25))
    lg = inc.local_of[g]
    @test mesh.faces[g].area == 0.5 && varpi(mesh.faces[g].minus) == 1 && varpi(mesh.faces[g].plus) == 1
    flux = zeros(length(mesh.faces)); flux[g] = 0.01
    O, I, On, In = MA.partial_in_out(s, flux, 1)
    Vold = sum(vol(c) * (c == mesh.faces[g].minus ? 0.01 / vol(c) : 0.0) / 2 for c in (mesh.faces[g].minus, mesh.faces[g].plus))
    @test (Vold - O[lg]) / (Vold - O[lg] + I[lg]) ≈ 0 atol = 1.0e-13
end

@testset "eq:momentum:ctu: the code's staggered scalar overshoots at such an unknown (P5)" begin
    # Deterministic fixtures on the composite fixture of the chapter: a liquid drop next to the hanging side x = 4, uniform x-velocity
    # (solenoidal, second sweep inactive), WY step at nu = 1/2, Upwind bulk.  Initial staggered values: 1 everywhere except 0 at the unknown
    # of the hanging segment; one sweep gives I_f / (V rho^new) = 1 - gamma_f there (all neighbours 1, own value 0).
    for (cx, cy, r, usign, at, expected_gamma) in (
            (4.15, 5.2, 0.16, -1.0, (4.0, 5.25), nothing),            # fluid enters the coarse cell through a fine segment
            (4.05, 5.2, 0.1, 1.0, (4.0, 5.75), -0.5),                # fluid leaves the coarse cell through a fine segment
        )
        function build()
            mesh = D.vof_amr_mesh()
            field = InterfaceField(mesh, (x, y) -> hypot(x - cx, y - cy) - r; lipschitz = 1)
            velocity = FaceVelocity(mesh, (x, y, t, a) -> a == 1 ? usign : 0.0)
            reconstruct!(field)
            work = WYAdvectionWorkspace(field)
            return field, velocity, work, stable_timestep(velocity, work)
        end
        field, velocity, work, dt = build()
        mesh = field.mesh
        C0 = Dict(ref => D.cond(field, ref) for ref in D.all_cells(mesh))
        advect!(field, velocity, dt, work)
        sweeps, _ = MA.replay_mass_identity(field, work, C0, (1, 2), (1000.0, 1.0))
        sweep = sweeps[1]
        comp = sweep.comps[1]
        s = comp.schedule
        O, I, On, In = MA.partial_in_out(s, work.liquid_flux, 1)
        old = D.independent_interpolation(s.interpolation, r -> sweep.C_old[r])
        new = D.independent_interpolation(s.interpolation, r -> sweep.pred[r])
        i = findfirst(c -> abs(c[1] - at[1]) < 1.0e-12 && abs(c[2] - at[2]) < 1.0e-12, s.dofs.centres)
        gamma = (s.V[i] * old[i] - O[i]) / (s.V[i] * new[i])
        predicted = I[i] / (s.V[i] * new[i])
        @test new[i] < 0.5                                                # the unknown is a draining control volume: all dual faces CTU
        @test gamma < -0.3
        expected_gamma === nothing || @test gamma ≈ expected_gamma atol = 1.0e-9
        @test predicted ≈ 1 - gamma rtol = 1.0e-9
        field, velocity, work, dt = build()
        q = PhaseFaceScalar{1}(field; liquid = (x, y) -> (abs(x - at[1]) < 1.0e-9 && abs(y - at[2]) < 1.0e-9) ? 0.0 : 1.0, gas = 1.0)
        advect!(field, velocity, dt, work; staggered_scalars = (q,), scheme = Upwind(), amr_interface_scheme = Upwind())
        value = q.liquid.interior[i]
        println(
            "P5 demonstration at ", at, ": new fraction ", new[i], ", gamma = ", gamma, ", code value after one step = ", value,
            " (predicted 1 - gamma = ", 1 - gamma, ")"
        )
        @test value ≈ predicted rtol = 1.0e-9
        @test value > 1.45                                                # overshoot of the initial range [0, 1]
        @test maximum(q.liquid.interior) >= value - 1.0e-12               # the global maximum principle fails
    end
end

@testset "eq:momentum:ctu: control, the same experiment away from the hanging sides keeps the range (P5)" begin
    rng = MersenneTwister(9)
    worst_low = worst_high = 0.0
    for (cx, cy, r, usign) in ((5.9, 5.2, 0.16, -1.0), (6.1, 5.3, 0.16, 1.0), (5.7, 6.4, 0.12, -1.0), (6.3, 6.1, 0.14, 1.0))
        mesh = D.vof_amr_mesh()
        field = InterfaceField(mesh, (x, y) -> hypot(x - cx, y - cy) - r; lipschitz = 1)
        velocity = FaceVelocity(mesh, (x, y, t, a) -> a == 1 ? usign : 0.0)
        reconstruct!(field)
        work = WYAdvectionWorkspace(field)
        dt = stable_timestep(velocity, work)
        pattern = Dict{Tuple{Float64, Float64}, Float64}()
        q = PhaseFaceScalar{1}(field; liquid = (x, y) -> get!(() -> rand(rng, Bool) ? 1.0 : 0.0, pattern, (x, y)), gas = 1.0)
        advect!(field, velocity, dt, work; staggered_scalars = (q,), scheme = Upwind(), amr_interface_scheme = Upwind())
        worst_low = min(worst_low, minimum(q.liquid.interior))
        worst_high = max(worst_high, maximum(q.liquid.interior) - 1)
    end
    println("control: range of the liquid values after one step: [", worst_low, ", ", 1 + worst_high, "]")
    @test worst_low >= -1.0e-12
    @test worst_high <= 1.0e-12
end

@testset "eq:momentum:ctu: the one-velocity momentum update overshoots at such an unknown (P5m)" begin
    function momentum_value(cx, cy, r, usign, at, density)
        mesh = D.vof_amr_mesh()
        field = InterfaceField(mesh, (x, y) -> hypot(x - cx, y - cy) - r; lipschitz = 1)
        velocity = FaceVelocity(mesh, (x, y, t, a) -> a == 1 ? usign : 0.0)
        reconstruct!(field)
        work = WYAdvectionWorkspace(field)
        dt = stable_timestep(velocity, work)
        source = FaceVelocity(
            mesh, (x, y, t, a) -> a == 1 ? ((abs(x - at[1]) < 1.0e-9 && abs(y - at[2]) < 1.0e-9) ? 0.0 : 1.0) : 0.0
        )
        destination = FaceVelocity(mesh, (x, y, t, a) -> 0.0)
        advect!(
            field, velocity, dt, work; momentum = destination, density,
            momentum_workspace = TwoPhaseAMR.MomentumAdvectionWorkspace(field), momentum_source = source,
            scheme = Upwind(), amr_interface_scheme = Upwind(),
        )
        xfaces = [g for g in eachindex(mesh.faces) if mesh.faces[g].axis == 1]
        return destination.interior[D.face_index_at(mesh, at)], extrema(destination.interior[xfaces])
    end
    for (cx, cy, r, usign, at) in ((4.15, 5.2, 0.16, -1.0, (4.0, 5.25)), (4.05, 5.2, 0.1, 1.0, (4.0, 5.75)))
        value, range = momentum_value(cx, cy, r, usign, at, (1000.0, 1.0))
        println("P5m at ", at, ", densities (1000, 1): value at the unknown ", value, ", range of all values ", range)
        @test value > 1.2                                                 # the initial range is [0, 1]
        @test range[2] >= value - 1.0e-12
        for density in ((1.0, 1.0), (10.0, 1.0))
            control, control_range = momentum_value(cx, cy, r, usign, at, density)
            println("   control densities ", density, ": value ", control, ", range ", control_range)
        end
    end
end

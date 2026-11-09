# Code check 01: the dual Courant numbers of the two-phase staggered transport under the volume-of-fluid time-step bound.
#
# Label:  sec:momentum:cfl  (the last paragraph: "On a uniform mesh without solid this bounds the dual Courant numbers ... so
#         nu_g <= nu^dagger <= 1/2 with V_f = |c|"; eq:momentum:cfl and the convexity sentence)
# Source: docs/techref/chapters/06_momentum_transport.tex:535-575; docs/techref/chapters/05_vof_transport.tex (eq:vof:cfl, eq:vof:dtwy);
#         src/vof/workspace.jl (stable_timestep), src/vof/fluxes.jl (_compute_directional_wy_fluxes!),
#         src/two_phase/phase_face_transport.jl (_phase_bulk_state), src/operators/advection/staggered_update.jl (stable_timestep)
# Checks (the sharpness experiments, B2 and the exact linear programs A1, A2, are in sharp_01_courant_extremizer.jl):
#   B1  (hard) uniform periodic mesh, random (non-solenoidal) face velocities scaled to the WY step for nu^dagger in {0.5, 0.45, 0.3}
#       and random interfaces: the dual Courant numbers |Upsilon^pi_g| / min(V) of the CARRIER, the liquid and the gas volumes
#       (the code's primal volumes, independent dual schedule) are at most nu^dagger, and |liquid volume| <= |carrier volume| face by face;
#   D1  (hard) two velocities (frozen and projected carriers) on the uniform periodic mesh: the code's liquid and gas phase volumes at the
#       WY step of the tuple satisfy nu_g <= nu^dagger;
#   E1  (hard) eq:momentum:cfl: the unsplit Upwind update of a solenoidal velocity at dt = c dt^stag(cfl = 1) is a convex combination of old
#       values for every c <= 2 (not only c <= 1), has a negative diagonal coefficient for c > 2, and its row sums are one;
#   E2  (hard) the same with NoSlip walls (boundary dual faces excluded from the bound, as printed) on uncut walled composite meshes, solenoidal
#       velocity, unknowns outside limitation (i): the coefficients stay non-negative for c <= 1 (the boundary dual face carries at most
#       as much as the interior ones, so the factor 2 of the bound absorbs it).
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/sec-momentum-cfl/scripts/code_01_courant.jl

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

@testset "sec:momentum:cfl: uniform periodic mesh, random fields and translations (B1, B2)" begin
    rng = MersenneTwister(2026)
    mesh = D.vof_uniform_mesh()
    worst = 0.0
    face_check = 0.0
    for nu in (0.5, 0.45, 0.3), trial in 1:6
        levelset, lipschitz = MA.random_interface(rng, mesh.extent)
        field = InterfaceField(mesh, levelset; lipschitz)
        work = WYAdvectionWorkspace(field)
        velocity = trial <= 3 ? MA.random_velocity(rng, mesh) : MA.smooth_velocity(rng, mesh, nothing, nothing; potential = 0.6)
        dt = stable_timestep(velocity, work; cfl = nu)
        carrier, liquid, gas = sweep_volumes(field, velocity, dt, work)
        for vols in (carrier, liquid, gas)
            ratio = max_dual_courant(mesh, nothing, vols) / nu
            worst = max(worst, ratio)
            @test ratio <= 1 + 1.0e-12
        end
        face_check = max(face_check, maximum(abs.(liquid) .- abs.(carrier)), maximum(abs.(gas) .- abs.(carrier)))
        @test all(abs.(liquid) .<= abs.(carrier) .+ 1.0e-15) && all(abs.(gas) .<= abs.(carrier) .+ 1.0e-15)
    end
    println("B1: largest dual Courant number / nu^dagger over carrier, liquid and gas volumes: ", worst)
    @test worst <= 1 + 1.0e-12
end

@testset "sec:momentum:cfl: two velocities on the uniform periodic mesh (D1)" begin
    rng = MersenneTwister(31)
    mesh = D.vof_uniform_mesh()
    projected = ProjectedLiquidVelocity(; tolerance = 1.0e-12, max_cycles = 1000)
    worst = Dict(:frozen => 0.0, :projected => 0.0)
    for (name, method) in ((:frozen, FrozenInterfaceFluxes()), (:projected, projected)), trial in 1:3
        levelset, lipschitz = MA.random_interface(rng, mesh.extent)
        field = InterfaceField(mesh, levelset; lipschitz)
        velocity = (
            MA.smooth_velocity(rng, mesh, nothing, nothing; potential = 0.3),
            MA.smooth_velocity(rng, mesh, nothing, nothing; potential = 0.3),
        )
        work = WYAdvectionWorkspace(field, velocity; interface_transport = method)
        nu = 0.5
        dt = try
            stable_timestep(velocity, work; cfl = nu)
        catch err
            (err isa ErrorException || err isa ArgumentError) || rethrow()
            continue
        end
        isfinite(dt) || continue
        try
            advect!(field, velocity, dt, work)
        catch err
            (err isa ErrorException || err isa ArgumentError) || rethrow()
            continue
        end
        for vols in (copy(work.liquid_flux), copy(work.gas_flux))
            ratio = max_dual_courant(mesh, nothing, vols) / nu
            worst[name] = max(worst[name], ratio)
            @test ratio <= 1 + 1.0e-12
        end
    end
    println("D1: largest nu_g / nu^dagger of the phase volumes: ", worst)
    @test worst[:frozen] > 0 && worst[:projected] > 0
end

@testset "sec:momentum:cfl: convexity of the unsplit upwind update (E1)" begin
    rng = MersenneTwister(5)
    for (kind, periodic) in ((:uniform, (true, true)), (:amr, (true, true)), (:cut, (true, true)))
        for trial in 1:2
            mesh = kind == :uniform ? D.vof_uniform_mesh() : MA.random_mesh(rng; periodic, levels = 1, refine_probability = 0.5)
            solid = kind == :cut ? MA.random_solid(rng, mesh.extent; kind = :disc) : nothing
            geometry = solid === nothing ? nothing : GeometryField(mesh, solid; lipschitz = 2)
            velocity = try
                psi = MA.random_streamfunction(rng, mesh.extent)
                geometry === nothing ? MA.stream_velocity(mesh, psi) : D.solid_streamfunction_velocity(mesh, geometry, solid, psi)
            catch err
                err isa AssertionError || rethrow()
                continue
            end
            geometry === nothing || enforce_geometry_no_slip!(velocity, geometry)
            for k in (1, 2)
                C, _, ws = MA.assemble_convection(mesh, geometry, velocity, k; scheme = Upwind(), amr_interface_scheme = Upwind())
                n = size(C, 1)
                base = stable_timestep(velocity, ws; cfl = 1.0)
                for c in (0.5, 1.0, 1.9, 2.0)
                    A = I - c * base * C
                    @test all(A[i, j] >= -1.0e-13 for i in 1:n, j in 1:n if i != j)           # off-diagonal coefficients
                    @test minimum(diag(A)) >= -1.0e-9                                        # diagonal: convex up to c = 2
                    @test maximum(abs, A * ones(n) .- 1) <= 1.0e-9                           # solenoidal: row sums one
                end
                A = I - 2.05 * base * C
                @test minimum(diag(A)) < -0.01                                                # c > 2: a negative coefficient
                @test minimum(diag(I - 2.0 * base * C)) <= 1.0e-9                             # and c = 2 is attained: not sharper than 2
            end
        end
    end
end

@testset "sec:momentum:cfl: convexity at walls, boundary dual faces outside the bound (E2)" begin
    rng = MersenneTwister(15)
    worst_diag = Inf
    checked = 0
    for trial in 1:8
        mesh = MA.random_mesh(rng; periodic = (false, false), levels = 1, refine_probability = 0.5)
        velocity = MA.stream_velocity(mesh, MA.random_streamfunction(rng, mesh.extent; walls = true))
        nf = length(mesh.faces)
        aper = ones(nf)
        cell_faces = MA.faces_of_cells(mesh)
        for k in (1, 2)
            inc = MA.component_incidence(mesh, k)
            C, offset, ws = MA.assemble_convection(
                mesh, nothing, velocity, k; scheme = Upwind(), amr_interface_scheme = Upwind(), boundary = NoSlip(), boundary_value = 0.0,
            )
            n = size(C, 1)
            base = stable_timestep(velocity, ws; cfl = 1.0)
            clean = [MA.unknown_class(mesh, inc, g, aper; cell_faces) != :tincomplete for g in inc.interior]   # outside limitation (i)
            for c in (0.5, 1.0)
                A = I - c * base * C
                for i in 1:n
                    clean[i] || continue
                    checked += 1
                    @test A[i, i] >= -1.0e-9
                    @test all(A[i, j] >= -1.0e-13 for j in 1:n if j != i)
                    c == 1.0 && (worst_diag = min(worst_diag, A[i, i]))
                end
            end
        end
    end
    println("E2: smallest diagonal coefficient at c = 1 over ", checked, " unknown-checks: ", worst_diag)
    @test worst_diag >= -1.0e-9
end

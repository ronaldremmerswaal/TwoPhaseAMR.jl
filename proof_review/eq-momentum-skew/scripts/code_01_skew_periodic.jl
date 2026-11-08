# Code check 01: skew-symmetry of the centred convection operator (periodic meshes), energy statements, scheme remarks.
#
# Label:  eq:momentum:skew  (eq:momentum:operator, eq:momentum:euler-energy)
# Source: docs/techref/chapters/06_momentum_transport.tex:486-512;
#         src/operators/advection/staggered_update.jl (convection!, advect!), src/operators/advection/staggered_fluxes.jl
# Checks, with the operator assembled from the package's convection! (Centred bulk and interface scheme, dt = 0) column by
# column on periodic uniform, composite and cut meshes, both components, random velocities (non-solenoidal) and streamfunction
# velocities (solenoidal):
#   K1  V C + (V C)^T = diag(V div Upsilon)  (the weighted skew-symmetry up to the diagonal), div Upsilon from the rules;
#   K2  2 diag(C) = div Upsilon;
#   K3  <psi, C q>_V + <q, C psi>_V = sum V q psi div Upsilon for random q, psi evaluated through convection! itself;
#   K4  solenoidal velocity: q^T V C q = 0 (energy of dq/dt = -C q conserved), also on cut meshes with open-solenoidal velocity;
#   K5  one forward-Euler step with the all-centred operator: E1 - E0 = (dt^2/2)|C q|_V^2 for div Upsilon = 0
#       (and E1 - E0 = -dt q^T V C q + (dt^2/2) |C q|_V^2 in general);
#   K6  the default Centred() (interface scheme MonotonizedCentral) is NOT skew on a composite mesh (size reported), but is skew on
#       a uniform mesh where no interface connection exists; Lax-Wendroff at dt = 0 with a centred interface scheme is the
#       centred operator exactly;
#   (the sharpness experiment K7 -- a non-arithmetic two-point state is not skew -- is in sharp_01_arithmetic_mean.jl)
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-momentum-skew/scripts/code_01_skew_periodic.jl

using Test, Random, LinearAlgebra
isdefined(Main, :MomentumAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "momentum_audit.jl"))
using .MomentumAudit
using TwoPhaseAMR
const MA = MomentumAudit
const D = MA.Docs
const AMR = TwoPhaseAMR

function periodic_fixture(rng, kind; solenoidal)
    for attempt in 1:50
        mesh = kind == :uniform ? D.vof_uniform_mesh() : MA.random_mesh(rng; levels = 1, refine_probability = 0.5)
        solid = kind == :cut ? MA.random_solid(rng, mesh.extent; kind = :disc) : nothing
        geometry = solid === nothing ? nothing : GeometryField(mesh, solid; lipschitz = 2)
        velocity = try
            if solenoidal
                psi = MA.random_streamfunction(rng, mesh.extent)
                geometry === nothing ? MA.stream_velocity(mesh, psi) : D.solid_streamfunction_velocity(mesh, geometry, solid, psi)
            else
                MA.random_velocity(rng, mesh)
            end
        catch err
            err isa AssertionError || rethrow()
            continue
        end
        geometry === nothing || enforce_geometry_no_slip!(velocity, geometry)
        return mesh, geometry, velocity
    end
    error("no admissible fixture")
end

energy(V, q) = sum(V .* q .^ 2) / 2

@testset "eq:momentum:skew: periodic meshes" begin
    rng = MersenneTwister(4242)
    worst = Dict(:skew => 0.0, :diag => 0.0, :bilinear => 0.0, :solenoidal => 0.0, :euler => 0.0, :lw_centred => 0.0)
    counts = Dict(:cases => 0, :default_skew_defect_amr => 0.0, :default_skew_defect_uniform => 0.0)
    for (kind, solenoidal) in ((:uniform, false), (:uniform, true), (:amr, false), (:amr, true), (:cut, false), (:cut, true)),
            trial in 1:(kind == :uniform ? 1 : 3)
        mesh, geometry, velocity = periodic_fixture(rng, kind; solenoidal)
        nf = length(mesh.faces)
        aper = geometry === nothing ? ones(nf) : Float64.(geometry.aperture)
        Phi = [mesh.faces[i].area * aper[i] * velocity.interior[i] for i in 1:nf]
        for k in (1, 2)
            counts[:cases] += 1
            inc = MA.component_incidence(mesh, k)
            C, offset, ws = MA.assemble_convection(mesh, geometry, velocity, k)
            @test maximum(abs, offset) == 0
            n = inc.n
            V = MA.schedule_volumes(ws, n)
            VdivU = MA.dual_divergence(mesh, inc, Phi)                       # V_f (div Upsilon)_f
            VC = V .* C
            scale = maximum(abs, VC)
            skew = maximum(abs, VC .+ VC' .- Diagonal(VdivU)) / scale
            worst[:skew] = max(worst[:skew], skew)
            diag_defect = maximum(abs, 2 .* diag(C) .- VdivU ./ V) / maximum(abs, C)
            worst[:diag] = max(worst[:diag], diag_defect)
            @test skew <= 1.0e-13
            @test diag_defect <= 1.0e-13
            # K3 through convection! applied to random vectors
            q = FaceField{k}(mesh); psi = FaceField{k}(mesh)
            q.interior .= randn(rng, n); psi.interior .= randn(rng, n)
            Cq = FaceField{k}(mesh); Cpsi = FaceField{k}(mesh)
            convection!(Cq, q, velocity, 0.0, ws; scheme = Centred(), amr_interface_scheme = Centred())
            convection!(Cpsi, psi, velocity, 0.0, ws; scheme = Centred(), amr_interface_scheme = Centred())
            lhs = sum(V .* psi.interior .* Cq.interior) + sum(V .* q.interior .* Cpsi.interior)
            rhs = sum(q.interior .* psi.interior .* VdivU)
            bilinear = abs(lhs - rhs) / (sum(abs.(V .* psi.interior .* Cq.interior)) + 1.0e-300)
            worst[:bilinear] = max(worst[:bilinear], bilinear)
            @test bilinear <= 1.0e-12
            if solenoidal
                @test maximum(abs, VdivU) <= 1.0e-12 * maximum(abs, Phi)
                e_rate = abs(sum(V .* q.interior .* Cq.interior)) / sum(V .* abs.(q.interior .* Cq.interior))
                worst[:solenoidal] = max(worst[:solenoidal], e_rate)
                @test e_rate <= 1.0e-12
            end
            # K5: one centred Euler step (unsplit)
            field = FaceField{k}(mesh); field.interior .= q.interior
            dt = 0.37 * stable_timestep(velocity, ws; cfl = 1.0)
            E0 = energy(V, field.interior)
            Cq0 = copy(Cq.interior)
            advect!(field, velocity, dt, ws; scheme = Centred(), amr_interface_scheme = Centred())
            E1 = energy(V, field.interior)
            predicted = -dt * sum(V .* q.interior .* Cq0) + dt^2 / 2 * sum(V .* Cq0 .^ 2)
            growth = dt^2 / 2 * sum(V .* Cq0 .^ 2)
            first_order = dt * sum(V .* q.interior .* Cq0)
            defect_E = abs((E1 - E0) - predicted)
            band = 100 * eps() * E0                                       # E1 - E0 is a difference of two O(E0) numbers
            worst[:euler] = max(worst[:euler], defect_E / (eps() * E0))   # in units of eps * E0
            @test defect_E <= 1.0e-9 * (growth + abs(first_order)) + band
            if solenoidal
                @test abs(first_order) <= 1.0e-12 * growth + band
                @test abs((E1 - E0) - growth) <= 1.0e-9 * growth + band
                @test E1 - E0 >= -band
            end
            # K6: default Centred() uses the MC interface scheme; LW at dt = 0 with a centred interface scheme is centred
            Cd, _, _ = MA.assemble_convection(mesh, geometry, velocity, k; scheme = Centred(), amr_interface_scheme = AMR.default_amr_interface_scheme(Centred()))
            VCd = V .* Cd
            defect = maximum(abs, VCd .+ VCd' .- Diagonal(VdivU)) / scale
            if kind == :uniform
                counts[:default_skew_defect_uniform] = max(counts[:default_skew_defect_uniform], defect)
            else
                counts[:default_skew_defect_amr] = max(counts[:default_skew_defect_amr], defect)
            end
            Clw, _, _ = MA.assemble_convection(mesh, geometry, velocity, k; scheme = LaxWendroff(), amr_interface_scheme = Centred())
            gap = maximum(abs, Clw .- C) / maximum(abs, C)
            worst[:lw_centred] = max(worst[:lw_centred], gap)
            @test gap <= 1.0e-14
        end
    end
    println("skew audit (relative to the largest entry of V C; :euler in units of eps*E0): ", worst)
    println("default Centred() skew defect: uniform ", counts[:default_skew_defect_uniform], ", composite/cut ", counts[:default_skew_defect_amr])
    @test counts[:default_skew_defect_uniform] <= 1.0e-13
    @test counts[:default_skew_defect_amr] > 1.0e-3                       # not skew on a composite mesh
end

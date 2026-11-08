# Sharpness experiment 01 for the centred operator: the arithmetic mean is the only two-point state with the skew identity.
#
# Label:  eq:momentum:skew
# Source: docs/techref/chapters/06_momentum_transport.tex:486-512
# Check (K7, hard): replacing the centred state of every dual face of the uniform periodic mesh by alpha q_tail + (1 - alpha) q_head
#   (alpha = 0.4) changes the symmetric part of the assembled V C off the diagonal by exactly U_g (1 - 2 alpha) per dual face, so the
#   weighted skew-symmetry of eq:momentum:skew is lost: V C + (V C)^T has off-diagonal entries of the size of the largest entry of V C.
#   (Symbolic version: step01 S2.  Each dual face of non-zero rate forces alpha = 1/2 unless two dual faces joining the same pair cancel.)
# Run:    julia --heap-size-hint=3G -t 1 --project=. proof_review/eq-momentum-skew/scripts/sharp_01_arithmetic_mean.jl

using Test, Random, LinearAlgebra
isdefined(Main, :MomentumAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "momentum_audit.jl"))
using .MomentumAudit
using TwoPhaseAMR
const MA = MomentumAudit
const D = MA.Docs

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

@testset "eq:momentum:skew: a non-arithmetic two-point state is not skew (K7)" begin
    rng = MersenneTwister(1)
    mesh, geometry, velocity = periodic_fixture(rng, :uniform; solenoidal = true)
    k = 1
    inc = MA.component_incidence(mesh, k)
    ws = D.staggered_workspace(FaceField{k}(mesh), geometry)
    C, _, _ = MA.assemble_convection(mesh, geometry, velocity, k)
    V = MA.schedule_volumes(ws, inc.n)
    weights = MA.rule_weights(mesh, inc)
    nf = length(mesh.faces)
    Phi = [mesh.faces[i].area * velocity.interior[i] for i in 1:nf]
    alpha = 0.4
    VCw = V .* C
    # replace the centred state of every dual face by alpha q_tail + (1 - alpha) q_head
    for ((i, j, axis), d) in weights
        U = sum(w * Phi[h] for (h, w) in d)
        VCw[i, i] += U * (alpha - 1 / 2); VCw[i, j] += U * ((1 - alpha) - 1 / 2)
        VCw[j, i] -= U * (alpha - 1 / 2); VCw[j, j] -= U * ((1 - alpha) - 1 / 2)
    end
    off = VCw .+ VCw'
    expected = zeros(inc.n, inc.n)
    for ((i, j, axis), d) in weights
        U = sum(w * Phi[h] for (h, w) in d)
        expected[i, j] += U * (1 - 2alpha); expected[j, i] += U * (1 - 2alpha)
    end
    for i in 1:inc.n
        off[i, i] = 0.0
    end
    @test maximum(abs, off .- expected) <= 1.0e-13 * maximum(abs, VCw)
    @test maximum(abs, expected) > 1.0e-3 * maximum(abs, VCw)
end

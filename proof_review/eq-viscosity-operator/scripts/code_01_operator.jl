# Code check 01: eq:viscosity:operator -- the integrated stiffness E'WE and the operator -V^{-1} E'WE returned by diffusion!;
# V = diag(dual volumes), or open dual volumes with 1 where they vanish (solid).
#
# Label:  eq:viscosity:operator
# Source: docs/techref/chapters/09_viscosity.tex:296-310; src/two_phase/diffusion/variable_diffusion_operator.jl (diffusion!,
#         _factorized_variable_stiffness_mul!), src/embedded/operators.jl (_geometry_face_dual_volumes)
# Checks:
#   O1  op.stiffness = E'WE (independent E, W; plus the code's wall rows of a solid) entrywise,
#   O2  diffusion!(rate, u, op) = -V^{-1} S u, V = mesh.face_dual_volumes (no solid), open dual volumes with 1 on covered faces (solid),
#       boundary rates zero; the factorised action = S u,
#   O3  dimensional: constant-viscosity limit of the operator on a uniform periodic mesh with div u = 0 gives (mu/rho)-free mu*Laplacian:
#       -V^{-1} S u -> mu Delta u (second order; the thesis/chapter remark "which for a constant viscosity and div u = 0 is mu Delta u"),
#       observed order with the streamfunction velocity u = curl psi on 16, 32, 64.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-viscosity-operator/scripts/code_01_operator.jl
using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ViscosityAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "viscosity_audit.jl"))
using .ViscosityAudit
using TwoPhaseAMR
const AMR = TwoPhaseAMR

@testset "O1/O2 operator" begin
    rng = MersenneTwister(12)
    for (name, fx) in ((:walled, fixture_walled), (:cut, fixture_cut), (:periodic, fixture_periodic))
        field = fx(); mesh = field.mesh; geometry = field.geometry
        op = StaggeredDiffusionOperator(field; viscosity = FIX_VISC)
        E, meas, logmu, _ = independent_strain(field, FIX_VISC)
        n = size(E, 1)
        S = E' * Diagonal(meas .* exp.(logmu)) * E
        if size(op.strain, 1) > n
            Ew = op.strain[(n + 1):end, :]
            S = S + Ew' * Diagonal(op.stress_weights[(n + 1):end]) * Ew
        end
        @test maximum(abs, S - op.stiffness) <= 1.0e-12 * maximum(abs, S)
        V = isnothing(geometry) ? copy(mesh.face_dual_volumes) : [geometry.face_dual_volumes[i] > 0 ? geometry.face_dual_volumes[i] : 1.0 for i in eachindex(mesh.faces)]
        @test op.volumes == V
        u = FaceVelocity(mesh); u.interior .= randn(rng, length(mesh.faces))
        rate = FaceVelocity(mesh)
        diffusion!(rate, u, op)
        expect = -(op.stiffness * u.interior) ./ V
        @test maximum(abs, rate.interior .- expect) <= 1.0e-12 * maximum(abs, expect)
        @test all(iszero, rate.boundary)
        work = zeros(length(op.stress_weights)); out = zeros(length(V))
        AMR._factorized_variable_stiffness_mul!(out, op, u.interior, work)
        @test maximum(abs, out .- op.stiffness * u.interior) <= 1.0e-11 * maximum(abs, out)
        println(name, ": stiffness defect ", maximum(abs, S - op.stiffness) / maximum(abs, S), ", operator defect ", maximum(abs, rate.interior .- expect) / maximum(abs, expect))
    end
end

@testset "O3: constant viscosity, div u = 0: -V^-1 S u -> mu Laplacian u" begin
    L = 8.0
    prev = nothing
    for n in (16, 32, 64)
        mesh = BlockTreeMesh(; base_mesh = (n, n), block_size = (4, 4), extent = (L, L), periodic = (true, true), max_level = 0)
        field = InterfaceField(mesh, (x, y) -> 1.0; lipschitz = 1)        # all gas, mu = FIX_VISC[2]
        op = StaggeredDiffusionOperator(field; viscosity = FIX_VISC)
        k = 2pi / L
        # solenoidal field: d_x u1 + d_y u2 = -k^2 cos sin + k^2 cos sin = 0, and Delta u = -2 k^2 u
        u1(x, y) = -k * sin(k * x + 0.3) * sin(k * y + 0.1)
        u2(x, y) = -k * cos(k * x + 0.3) * cos(k * y + 0.1)
        vel = FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? u1(x, y) : u2(x, y))
        rate = FaceVelocity(mesh)
        diffusion!(rate, vel, op)
        mu = FIX_VISC[2]
        exact = FaceVelocity(mesh, (x, y, t, axis) -> -2k^2 * mu * (axis == 1 ? u1(x, y) : u2(x, y)))
        err = maximum(abs, rate.interior .- exact.interior) / maximum(abs, exact.interior)
        order = prev === nothing ? NaN : log2(prev / err)
        println("n=", n, " relative max error ", err, " order ", order)
        prev = err
        n == 64 && @test order > 1.9
    end
end

# Code check 01: eq:viscosity:single-operator -- the operator returned by diffusion! for the single phase.
#
# Label:  eq:viscosity:single-operator
# Source: docs/techref/chapters/09_viscosity.tex:154-166; src/operators/staggered_diffusion.jl (_apply_staggered_diffusion!, diffusion!)
# Checks (walled composite mesh, x-periodic channel with moving walls, uniform periodic mesh):
#   S1  diffusion!(rate, u, op) = nu * ( L u + V^{-1} sum_{w: f(w)=f} cond_w u_w^{wall}(t) )  on the unknowns, assembled independently
#       from component.laplacian, component.volumes and the wall list (value rules of eq:viscosity:walls), zero on the boundary faces;
#   S2  the wall data are sampled at the time passed to diffusion! (time-dependent lid), and `viscosity` scales both parts;
#   S3  uniform periodic mesh: equals nu times the five-point Laplacian of each component (independent stencil).
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-viscosity-single-operator/scripts/code_01_single_operator.jl
using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ViscosityAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "viscosity_audit.jl"))
using .ViscosityAudit
using TwoPhaseAMR
const AMR = TwoPhaseAMR

wallvalue(boundary, c, time, axis) = begin
    # eq:viscosity:walls value rules, written out independently of AMR._wall_value
    bc = boundary isa VelocityBoundaryConditions ? AMR._side_boundary(boundary, c.boundary_axis, c.boundary_side) : boundary
    bc isa NoSlip && return c.tangential ? bc.velocity(c.point[1], c.point[2], time, axis) : 0.0
    bc isa Inflow && return bc.velocity(c.point[1], c.point[2], time, axis)
    return 0.0
end
included(boundary, c) = begin
    bc = boundary isa VelocityBoundaryConditions ? AMR._side_boundary(boundary, c.boundary_axis, c.boundary_side) : boundary
    bc isa FreeSlip && return !c.tangential
    bc isa Union{Outflow, ZeroGradientVelocity} && return false
    return true
end

@testset "S1/S2 diffusion! = nu (L u + V^-1 sum cond u_w)" begin
    rng = MersenneTwister(77)
    mesh = BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic = (false, false), max_level = 1, refine = (key, o, e) -> key.level == 0 && key.i >= 2)
    lid(x, y, t, axis) = axis == 1 ? 1.0 + sin(3t) + 0.2x : 0.1 * cos(t)
    inflow(x, y, t, axis) = axis == 1 ? 0.5 * t + y : 0.0
    boundaries = (NoSlip(lid), FreeSlip(), VelocityBoundaryConditions(left = Inflow(inflow), right = Outflow(), bottom = NoSlip(lid), top = FreeSlip()), NoSlip())
    for boundary in boundaries, nu in (1.0, 0.37), time in (0.0, 0.8)
        op = StaggeredDiffusionOperator(mesh; viscosity = nu, boundary)
        u = FaceVelocity(mesh); u.interior .= randn(rng, length(mesh.faces))
        rate = FaceVelocity(mesh); rate.boundary .= 5.0
        diffusion!(rate, u, op; time)
        expect = zeros(length(mesh.faces))
        for (component, axis) in ((op.x, 1), (op.y, 2))
            g = component.local_to_global
            r = component.laplacian * u.interior[g]
            for c in component.wall_connections
                included(boundary, c) || continue
                r[c.degree_of_freedom] += c.conductance * wallvalue(boundary, c, time, axis) / component.volumes[c.degree_of_freedom]
            end
            expect[g] .= nu .* r
        end
        @test maximum(abs, rate.interior .- expect) <= 1.0e-12 * maximum(abs, expect)
        @test all(iszero, rate.boundary)
    end
end

@testset "S3: uniform periodic five-point" begin
    mesh = BlockTreeMesh(; base_mesh = (16, 16), block_size = (4, 4), extent = (8.0, 8.0), periodic = (true, true))
    nu = 0.25
    op = StaggeredDiffusionOperator(mesh; viscosity = nu)
    u = FaceVelocity(mesh, (x, y, t, axis) -> axis == 1 ? sin(2pi * x / 8) * cos(2pi * y / 8 + 0.2) : cos(2pi * x / 8 + 0.9) * sin(4pi * y / 8))
    rate = FaceVelocity(mesh); diffusion!(rate, u, op)
    h = 0.5
    key(f) = (round(Int, 4f.center[1]), round(Int, 4f.center[2]))
    index = Dict((f.axis, key(f)) => i for (i, f) in pairs(mesh.faces))
    worst = 0.0
    for (i, f) in pairs(mesh.faces)
        k = key(f)
        nb = [index[(f.axis, (mod(k[1] + dx, 32), mod(k[2] + dy, 32)))] for (dx, dy) in ((2, 0), (-2, 0), (0, 2), (0, -2))]
        worst = max(worst, abs(rate.interior[i] - nu * (sum(u.interior[nb]) - 4 * u.interior[i]) / h^2))
    end
    @test worst <= 1.0e-12 * maximum(abs, rate.interior)
end

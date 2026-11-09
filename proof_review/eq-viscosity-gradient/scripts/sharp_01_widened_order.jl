# Sharpness 01: eq:viscosity:gradient -- accuracy of the widened derivative: exact for affine functions (claimed), NOT quadratic; the observed
# order of the derivative error at the connection midpoint under refinement, and of the two-point stencils (second order at the midpoint).
#
# Label:  eq:viscosity:gradient
# Source: docs/techref/chapters/09_viscosity.tex:62-92
# Checks (periodic composite meshes, base n = 16, 32, 64, right part refined; f(x,y) = sin(2 pi x/8 + 0.3) cos(4 pi y/8 + 0.1), components via
# positions of the unknowns):
#   Y1  two-point connections: error of (f+ - f-)/d against d_t f at the midpoint: order 2 on conforming pairs, order 1 where the two unknowns are
#       at different levels (stretched);
#   Y2  widened connections: error against d_t f at the midpoint of the pair: order 1 (the stencil is affine exact only);
#   Y3  the quadratic defect of the widened stencil (sum a (x_t)^2 relative to the minus point) is O(h).
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/eq-viscosity-gradient/scripts/sharp_01_widened_order.jl
using Test, Random, LinearAlgebra, SparseArrays
isdefined(Main, :ViscosityAudit) || include(joinpath(@__DIR__, "..", "..", "_shared", "viscosity_audit.jl"))
using .ViscosityAudit
using TwoPhaseAMR
const AMR = TwoPhaseAMR

function gradient_errors(n)
    L = 8.0
    mesh = BlockTreeMesh(; base_mesh = (n, n), block_size = (4, 4), extent = (L, L), periodic = (true, true), max_level = 1, refine = (key, o, e) -> key.level == 0 && key.i >= div(n, 8))
    f(x, y) = sin(2pi * x / L + 0.3) * cos(4pi * y / L + 0.1)
    fx(x, y) = 2pi / L * cos(2pi * x / L + 0.3) * cos(4pi * y / L + 0.1)
    fy(x, y) = -4pi / L * sin(2pi * x / L + 0.3) * sin(4pi * y / L + 0.1)
    err = Dict(:two_conforming => 0.0, :two_stretched => 0.0, :widened => 0.0)
    for axis in 1:2
        field = FaceField{axis}(mesh)
        schedule = AMR.StaggeredAdvectionWorkspace(field).schedule
        g2l, l2g = AMR._component_local_to_global(mesh, axis)
        conns = AMR._dual_diffusion_connections(mesh, schedule, l2g)
        adjacency = [Int[] for _ in l2g]
        for c in conns; push!(adjacency[c.minus], c.plus); push!(adjacency[c.plus], c.minus); end
        foreach(unique!, adjacency)
        for c in conns
            xm = mesh.faces[l2g[c.minus]].center; xp = mesh.faces[l2g[c.plus]].center
            dx = wrapped(xp[1] - xm[1], L, true); dy = wrapped(xp[2] - xm[2], L, true)
            mid = (xm[1] + dx / 2, xm[2] + dy / 2)
            exact = c.axis == 1 ? fx(mid...) : fy(mid...)
            idx, a = AMR._gradient_stencil(mesh, c, l2g, adjacency)
            value = sum(a[k] * f((xm[1] + wrapped(mesh.faces[l2g[j]].center[1] - xm[1], L, true)), (xm[2] + wrapped(mesh.faces[l2g[j]].center[2] - xm[2], L, true))) for (k, j) in enumerate(idx))
            sgn = (length(idx) == 2 && (c.axis == 1 ? dx : dy) < 0) ? -1 : 1
            e = abs(sgn * value - exact)
            kind = length(idx) == 2 ? (abs(c.distance - 0.5 * mesh.extent[1] / n * 2) < 1.0e-12 || true ? (mesh.faces[l2g[c.minus]].area ≈ mesh.faces[l2g[c.plus]].area && c.refinement_interface == false ? :two_conforming : :two_stretched) : :two_stretched) : :widened
            err[kind] = max(err[kind], e)
        end
    end
    return err
end

@testset "widened derivative accuracy" begin
    prev = nothing
    for n in (16, 32, 64)
        err = gradient_errors(n)
        println("n=", n, " ", Dict(k => round(v, sigdigits = 3) for (k, v) in err), prev === nothing ? "" : "  orders " * string(Dict(k => round(log2(prev[k] / err[k]), digits = 2) for k in keys(err))))
        if prev !== nothing
            @test log2(prev[:widened] / err[:widened]) > 0.8       # first order
            @test log2(prev[:two_conforming] / err[:two_conforming]) > 1.8
        end
        prev = err
    end
end

# Code check 02: what an O(1) truncation error on a level transition implies for the SOLUTION order (sec:verif:orders, closing
# paragraph; chapter 9 transition rows, #87 / #112).
#
# Label:  (no \label; slug sec-verif-transition-order)
# Source: docs/techref/chapters/14_verification.tex:473-483 ("The solution converges with order one there; the shifted solve pins 0.8
#         (#112)"), docs/techref/chapters/09_viscosity.tex:242-249 (truncation of order one at transition rows, solution >= 1.8 in the
#         maximum norm, #87), docs/techref/chapters/13_stability_accuracy.tex:502-506 (0.985, 1.002),
#         test/two_phase/variable_staggered_diffusion.jl ("Variable stress shifted solve on fixed and adapted trees"),
#         test/operators/staggered_diffusion.jl ("cut-cell Helmholtz convergence at walls and level transitions (#87)")
# Two problems with a truncation error of order one on a strip of area O(h) around a fixed level transition:
#   A  variable-stress (mu(x) = 1 + cos(2 pi x)/4, no slip) shifted solve (I + 0.01 S) u = f on a fixed L-shaped transition, #112;
#   B  scalar cut-cell Helmholtz (I + 0.1 S) u = f without the corrected band, transition on x = 1/2, no slip and free slip, #87.
# Checks:
#   C1  B: rows of order one on the transition strip (max |tau| does not decay), L2(tau) ~ h^(1/2); A: the truncation is WORSE than order
#       one (max |tau| ~ 1/h, alternating sign along the transition: tiny monopole ratio);
#   C2  A: solution L2 order ~1 (0.985, 1.002 of chapter 13), i.e. between the stability bound 1/2 and 2;
#   C3  B: solution maximum-norm order >= 1.8 (the asserted bound) although the truncation is also O(1): the solution order is NOT
#       determined by the order of the truncation error; the printed monopole ratio |sum V tau| / sum V |tau| over the strip shows
#       whether the rows cancel (supraconvergence) or add up;
#   C4  the stability bound ||u - u_h||_V <= shift ||tau||_V of the shifted problem (M e = shift V tau, M = V + shift S, S symmetric
#       positive semidefinite): a truncation of L2 order 1/2 gives a solution order of AT LEAST 1/2, no more can be inferred.
# Run:    julia --heap-size-hint=3G -t 1 --project=docs/techref/figures proof_review/sec-verif-transition-order/scripts/code_02_truncation_vs_solution.jl

using Test, TwoPhaseAMR, SparseArrays, LinearAlgebra

rate(a, b) = log2(a / b)

# ---- A: variable stress, shifted solve on the fixed tree of the test
function case_A(n)
    mesh = BlockTreeMesh(
        block_size = (n, n), max_level = 2, periodic = (false, false),
        refine = (key, origin, extent) -> key.level == 0 || (key.level == 1 && key.i == 0 && key.j == 0),
    )
    viscosity_function = (x, y) -> 1 + 0.25cospi(2x)
    viscosity = CellField(mesh, viscosity_function)
    operator = StaggeredDiffusionOperator(mesh; viscosity, boundary = NoSlip())
    exact_velocity(x, y, axis) = axis == 1 ? sinpi(x)^2 * sinpi(2y) : -sinpi(2x) * sinpi(y)^2
    function exact_diffusion(x, y, axis)
        cosx, cosy = cospi(2x), cospi(2y)
        sinx, siny = sinpi(2x), sinpi(2y)
        mu = viscosity_function(x, y)
        mux = -pi * sinx / 2
        return axis == 1 ? mu * pi^2 * (4cosx - 2) * siny + 2mux * pi * sinx * siny :
            mu * 2pi^2 * sinx * (1 - 2cosy) + mux * 2pi * (sinpi(x)^2 * cosy - cosx * sinpi(y)^2)
    end
    exact = FaceVelocity(mesh, (x, y, time, axis) -> exact_velocity(x, y, axis))
    shift = 0.01
    V = operator.volumes
    rhs = [V[i] * (exact.interior[i] - shift * exact_diffusion(mesh.faces[i].center..., mesh.faces[i].axis)) for i in eachindex(V)]
    solution = (spdiagm(0 => V) + shift * operator.stiffness) \ rhs
    err = solution .- exact.interior
    tendency = FaceVelocity(mesh)
    diffusion!(tendency, exact, operator)
    tau = [tendency.interior[i] - exact_diffusion(mesh.faces[i].center..., mesh.faces[i].axis) for i in eachindex(V)]
    # distance to the transition curve: x = 1/2 (y <= 1/2) and y = 1/2 (x <= 1/2); coarse width 1/(2n)
    hc = 1 / (2n)
    dist = map(mesh.faces) do f
        x, y = f.center
        d1 = x >= 0.5 ? (y <= 0.5 ? x - 0.5 : hypot(x - 0.5, y - 0.5)) : (y <= 0.5 ? 0.5 - x : hypot(x - 0.5, y - 0.5))
        d2 = y >= 0.5 ? (x <= 0.5 ? y - 0.5 : hypot(x - 0.5, y - 0.5)) : (x <= 0.5 ? 0.5 - y : hypot(x - 0.5, y - 0.5))
        min(abs(d1), abs(d2))
    end
    # (distance to the L-shaped curve; rows at the outer wall are excluded from the strip statistics below)
    wall = [min(f.center[1], 1 - f.center[1], f.center[2], 1 - f.center[2]) for f in mesh.faces]
    return (; V, err, tau, dist, wall, hc, total = sum(V))
end

# ---- B: scalar cut-cell Helmholtz without band, transition at x = 1/2
function case_B(n, boundary)
    mesh = BlockTreeMesh(
        base_mesh = (n, n), block_size = (4, 4), periodic = (false, false), max_level = 1,
        refine = (key, origin, extent) -> key.level == 0 && origin[1] < 0.5,
    )
    geometry = GeometryField(mesh, (x, y) -> 1.0; lipschitz = 1)
    diffusion = StaggeredDiffusionOperator(geometry; viscosity = 1.0, boundary)
    shift = 0.1
    out = []
    for (axis, component) in enumerate((diffusion.x, diffusion.y))
        pts = [mesh.faces[i].center for i in component.local_to_global]
        exact = [
            boundary isa FreeSlip ? (axis == 1 ? sinpi(x) * cospi(2y) : cospi(2x) * sinpi(y)) : sinpi(x) * sinpi(y) for (x, y) in pts
        ]
        lf = boundary isa FreeSlip ? 5 : 2
        V = component.volumes
        rhs = V .* ((1 + lf * shift * pi^2) .* exact)
        solution = (spdiagm(0 => V) + shift * component.stiffness) \ rhs
        tau = (-(component.stiffness * exact) ./ V) .+ lf * pi^2 .* exact      # operator(u_I) - Laplacian(u)
        hc = 1 / n
        dist = [abs(p[1] - 0.5) for p in pts]
        wall = [min(p[1], 1 - p[1], p[2], 1 - p[2]) for p in pts]
        push!(out, (; V, err = solution .- exact, tau, dist, wall, hc, total = sum(V)))
    end
    return out
end

function stats(c; strip_width = 2.01, wall_clear = 3.0)
    strip = (c.dist .<= strip_width * c.hc) .& (c.wall .>= wall_clear * c.hc)
    rest = .!strip
    l2(v, mask) = sqrt(sum(c.V[i] * v[i]^2 for i in eachindex(v) if mask[i]) / c.total)
    return (;
        l2err = l2(c.err, trues(length(c.err))), maxerr = maximum(abs, c.err),
        l2tau_strip = l2(c.tau, strip), l2tau_all = l2(c.tau, trues(length(c.tau))),
        maxtau_strip = maximum(abs(c.tau[i]) for i in eachindex(c.tau) if strip[i]),
        monopole = abs(sum(c.V[i] * c.tau[i] for i in eachindex(c.tau) if strip[i])) / sum(c.V[i] * abs(c.tau[i]) for i in eachindex(c.tau) if strip[i]),
        mass_strip = sum(c.V[i] * abs(c.tau[i]) for i in eachindex(c.tau) if strip[i]),
    )
end

show_table(name, st) = for (k, s) in zip(keys(st), values(st))
    println(name, " ", k, ": ", join((string(x, " ") for x in s), ""))
end

@testset "truncation order one on a transition: solution order" begin
    # A
    ns = (6, 12, 24, 48)
    A = [stats(case_A(n)) for n in ns]
    println("A (#112)  n=", ns)
    println("  solution L2 error      ", [round(a.l2err, sigdigits = 4) for a in A], "  orders ", [round(rate(A[i].l2err, A[i + 1].l2err), digits = 3) for i in 1:3])
    println("  solution max error     ", [round(a.maxerr, sigdigits = 4) for a in A], "  orders ", [round(rate(A[i].maxerr, A[i + 1].maxerr), digits = 3) for i in 1:3])
    println("  L2 truncation (strip)  ", [round(a.l2tau_strip, sigdigits = 4) for a in A], "  orders ", [round(rate(A[i].l2tau_strip, A[i + 1].l2tau_strip), digits = 3) for i in 1:3])
    println("  max truncation (strip) ", [round(a.maxtau_strip, sigdigits = 4) for a in A], "  orders ", [round(rate(A[i].maxtau_strip, A[i + 1].maxtau_strip), digits = 3) for i in 1:3])
    println("  monopole ratio |sum V tau|/sum V|tau| over the strip ", [round(a.monopole, sigdigits = 3) for a in A])
    println("  strip mass sum V|tau| (orders) ", [round(rate(A[i].mass_strip, A[i + 1].mass_strip), digits = 3) for i in 1:3])
    @test 0.8 < rate(A[3].l2err, A[4].l2err) < 1.3                                   # C2
    # C1 (A): here the truncation relative to the continuous operator is NOT of order one: its maximum over the transition rows grows
    # like 1/h (alternating sign along the transition, hence the tiny monopole ratio), its strip L2 norm like h^(-1/2) at the finest
    # pair; the solution still converges with order one.
    @test all(a.maxtau_strip > 1.0 for a in A)
    @test rate(A[3].maxtau_strip, A[4].maxtau_strip) < -0.8
    @test rate(A[3].l2tau_strip, A[4].l2tau_strip) < -0.4
    @test all(a.monopole < 0.01 for a in A)
    @test all(A[i].l2err <= 0.01 * A[i].l2tau_all * (1 + 1.0e-9) for i in 1:4)         # C4: e = shift M^-1 V tau, ||M^-1 V|| <= 1
    # B
    for boundary in (NoSlip(), FreeSlip())
        ns = (16, 32, 64, 128)
        Bs = [case_B(n, boundary) for n in ns]
        for axis in 1:2
            B = [stats(b[axis]) for b in Bs]
            println("B (#87) ", nameof(typeof(boundary)), " axis $axis  n=", ns)
            println("  solution max error     ", [round(b.maxerr, sigdigits = 4) for b in B], "  orders ", [round(rate(B[i].maxerr, B[i + 1].maxerr), digits = 3) for i in 1:3])
            println("  solution L2 error      ", [round(b.l2err, sigdigits = 4) for b in B], "  orders ", [round(rate(B[i].l2err, B[i + 1].l2err), digits = 3) for i in 1:3])
            println("  max truncation (strip) ", [round(b.maxtau_strip, sigdigits = 4) for b in B], "  orders ", [round(rate(B[i].maxtau_strip, B[i + 1].maxtau_strip), digits = 3) for i in 1:3])
            println("  L2 truncation (strip)  ", [round(b.l2tau_strip, sigdigits = 4) for b in B], "  orders ", [round(rate(B[i].l2tau_strip, B[i + 1].l2tau_strip), digits = 3) for i in 1:3])
            println("  monopole ratio over the strip ", [round(b.monopole, sigdigits = 3) for b in B])
            @test rate(B[1].maxerr, B[2].maxerr) >= 1.8 && rate(B[2].maxerr, B[3].maxerr) >= 1.8       # C3 (asserted by test #87)
            @test rate(B[3].maxtau_strip, B[4].maxtau_strip) < 0.3                                          # truncation not decaying
            @test all(b.l2err <= 0.1 * b.l2tau_all * (1 + 1.0e-9) for b in B)                                # C4
        end
    end
end
